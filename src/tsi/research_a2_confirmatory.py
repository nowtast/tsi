"""Seed-driven cohort construction for one-shot Research A2 confirmation."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from .paper34_resolution_benchmark import (
    TransitionCase,
    center_accuracy,
    coordinate_nll,
)
from .research_a2_contract import (
    NOISE_ADVANTAGE_PROBABILITIES,
    NOISE_BOUNDARY_STRESS_PROBABILITY,
    NOISE_PROBABILITIES,
    NOISE_SAMPLE_SIZES,
    OOD_NOISE,
    SCOPE_SAMPLE_SIZE,
    TEST_CASE_COUNT,
    TRAIN_NOISE_FOR_WIDTH_AND_SCOPE,
    WIDTH_SAMPLE_SIZES,
    WORLD_COUNT_PER_AXIS_OR_SCOPE_CONDITION,
)
from .research_a2_design import (
    exact_catalog_support_recovered,
    fit_catalog_generic,
    fit_typed_catalog,
    fit_width_generic,
    typed_parameters_recovered,
)
from .research_a2_features import (
    ALTERNATIVE_FAMILY_CATALOG,
    TYPED_FAMILY_CATALOG,
    WIDTH_POSITION_COUNTS,
)
from .research_a2_populations import (
    MATCHED,
    A2WorldSpec,
    balanced_world_specs,
    generate_a2_cases,
    generate_coupled_noise_cases,
    paired_misspecification_specs,
)


def _case_payload(case: TransitionCase) -> dict[str, object]:
    return {
        "source": list(case.source),
        "action": list(case.action),
        "observed": list(case.observed),
        "center": list(case.center),
        "composition_stratum": case.composition_stratum,
    }


def _spec_payload(spec: A2WorldSpec) -> dict[str, object]:
    return {
        "world_index": spec.world_index,
        "condition": spec.condition,
        "graph": [spec.graph[0], list(spec.graph[1])],
        "families": list(spec.families),
        "multipliers": list(spec.multipliers),
        "coefficients": list(spec.coefficients),
    }


def _matched_record(
    spec: A2WorldSpec,
    train: Sequence[TransitionCase],
    test: Sequence[TransitionCase],
    sample_size: int,
    width: int,
) -> dict[str, object]:
    prefix = train[:sample_size]
    typed = fit_typed_catalog(prefix, spec.graph)
    generic = fit_width_generic(prefix, spec.graph, width)
    typed_exact = typed_parameters_recovered(typed, spec)
    generic_exact = exact_catalog_support_recovered(generic, spec, TYPED_FAMILY_CATALOG)
    typed_nll = coordinate_nll(typed, test)
    generic_nll = coordinate_nll(generic, test)
    return {
        "world_index": spec.world_index,
        "family_pair": list(spec.families),
        "sample_size": sample_size,
        "position_count": width,
        "typed_exact": typed_exact,
        "generic_exact": generic_exact,
        "typed_nll": typed_nll,
        "generic_nll": generic_nll,
        "generic_minus_typed_nll": generic_nll - typed_nll,
        "typed_minus_generic_exact": int(typed_exact) - int(generic_exact),
    }


def _scope_record(
    spec: A2WorldSpec,
    train: Sequence[TransitionCase],
    test: Sequence[TransitionCase],
    sample_size: int,
) -> dict[str, object]:
    catalog = (
        TYPED_FAMILY_CATALOG
        if spec.condition == MATCHED
        else ALTERNATIVE_FAMILY_CATALOG
    )
    prefix = train[:sample_size]
    typed = fit_typed_catalog(prefix, spec.graph)
    generic = fit_catalog_generic(prefix, spec.graph, family_catalog=catalog)
    typed_nll = coordinate_nll(typed, test)
    generic_nll = coordinate_nll(generic, test)
    return {
        "world_index": spec.world_index,
        "condition": spec.condition,
        "family_pair": list(spec.families),
        "sample_size": sample_size,
        "typed_exact": typed_parameters_recovered(typed, spec),
        "generic_exact": exact_catalog_support_recovered(generic, spec, catalog),
        "generic_minus_typed_nll": generic_nll - typed_nll,
        "typed_minus_generic_center_accuracy": center_accuracy(typed, test)
        - center_accuracy(generic, test),
    }


def _family_audit(specs: Sequence[A2WorldSpec]) -> dict[str, object]:
    counts = Counter(spec.families for spec in specs)
    graph_counts = Counter(spec.graph for spec in specs)
    return {
        "family_pair_count": len(counts),
        "family_pair_counts": {
            "|".join(pair): count for pair, count in sorted(counts.items())
        },
        "family_pairs_balanced": len(set(counts.values())) == 1,
        "graph_manifest_coverage": len(graph_counts),
        "graph_count_minimum": min(graph_counts.values()),
        "graph_count_maximum": max(graph_counts.values()),
    }


def run_a2_cohort(
    root_seed: bytes,
    *,
    world_count: int = WORLD_COUNT_PER_AXIS_OR_SCOPE_CONDITION,
    width_sample_sizes: Sequence[int] = WIDTH_SAMPLE_SIZES,
    noise_sample_sizes: Sequence[int] = NOISE_SAMPLE_SIZES,
    noise_probabilities: Sequence[float] = NOISE_PROBABILITIES,
    scope_sample_size: int = SCOPE_SAMPLE_SIZE,
    test_case_count: int = TEST_CASE_COUNT,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object], dict[str, object]]:
    """Generate A2 evidence rows, portable replay inputs, and derivation audits."""

    if len(root_seed) != 32:
        raise ValueError("Research A2 root seed must contain 32 bytes")
    if world_count < 45 or world_count % 45:
        raise ValueError("world_count must be a positive multiple of 45")
    width_sizes = tuple(int(value) for value in width_sample_sizes)
    noise_sizes = tuple(int(value) for value in noise_sample_sizes)
    noise_levels = tuple(float(value) for value in noise_probabilities)
    for values, name in (
        (width_sizes, "width sample sizes"),
        (noise_sizes, "noise sample sizes"),
        (noise_levels, "noise probabilities"),
    ):
        if not values or values != tuple(sorted(set(values))):
            raise ValueError(f"{name} must be unique and increasing")
    if scope_sample_size <= 0 or test_case_count <= 0:
        raise ValueError("scope and test case counts must be positive")

    root = np.random.SeedSequence(int.from_bytes(root_seed, "little"))
    width_root, noise_root, scope_root = root.spawn(3)
    records: dict[str, list[dict[str, object]]] = {
        "candidate_width": [],
        "training_noise": [],
        "misspecification": [],
    }
    portable: dict[str, object] = {
        "status": "confirmatory_portable_replay_with_answers",
        "world_count_per_axis_or_scope_condition": world_count,
        "test_case_count_per_world": test_case_count,
        "design": {
            "width_position_counts": list(WIDTH_POSITION_COUNTS),
            "width_sample_sizes": list(width_sizes),
            "noise_probabilities": list(noise_levels),
            "noise_advantage_probabilities": list(
                NOISE_ADVANTAGE_PROBABILITIES
            ),
            "noise_boundary_stress_probability": NOISE_BOUNDARY_STRESS_PROBABILITY,
            "noise_sample_sizes": list(noise_sizes),
            "scope_sample_size": scope_sample_size,
        },
        "axes": {
            "candidate_width": [],
            "training_noise": [],
            "misspecification": [],
        },
    }
    portable_axes = portable["axes"]
    assert isinstance(portable_axes, dict)

    width_assignment, *width_seeds = width_root.spawn(world_count + 1)
    width_specs = balanced_world_specs(
        world_count, MATCHED, np.random.default_rng(width_assignment)
    )
    for spec, world_seed in zip(width_specs, width_seeds, strict=True):
        rng = np.random.default_rng(world_seed)
        train = generate_a2_cases(
            spec,
            max(width_sizes),
            rng,
            composition=False,
            noise_probability=TRAIN_NOISE_FOR_WIDTH_AND_SCOPE,
        )
        test = generate_a2_cases(
            spec,
            test_case_count,
            rng,
            composition=True,
            noise_probability=OOD_NOISE,
        )
        for sample_size in width_sizes:
            for width in WIDTH_POSITION_COUNTS:
                records["candidate_width"].append(
                    _matched_record(spec, train, test, sample_size, width)
                )
        portable_axes["candidate_width"].append(
            {
                "spec": _spec_payload(spec),
                "train": [_case_payload(case) for case in train],
                "test": [_case_payload(case) for case in test],
            }
        )

    noise_assignment, *noise_seeds = noise_root.spawn(world_count + 1)
    noise_specs = balanced_world_specs(
        world_count, MATCHED, np.random.default_rng(noise_assignment)
    )
    for spec, world_seed in zip(noise_specs, noise_seeds, strict=True):
        rng = np.random.default_rng(world_seed)
        streams = generate_coupled_noise_cases(
            spec, max(noise_sizes), rng, noise_levels
        )
        test = generate_a2_cases(
            spec,
            test_case_count,
            rng,
            composition=True,
            noise_probability=OOD_NOISE,
        )
        for probability in noise_levels:
            train = streams[probability]
            for sample_size in noise_sizes:
                record = _matched_record(
                    spec,
                    train,
                    test,
                    sample_size,
                    WIDTH_POSITION_COUNTS[0],
                )
                record["train_noise_probability"] = probability
                records["training_noise"].append(record)
        portable_axes["training_noise"].append(
            {
                "spec": _spec_payload(spec),
                "train_by_noise_probability": {
                    str(probability): [
                        _case_payload(case) for case in streams[probability]
                    ]
                    for probability in noise_levels
                },
                "test": [_case_payload(case) for case in test],
            }
        )

    spawned = scope_root.spawn(2 + 2 * world_count)
    cubic_assignment, matched_assignment = spawned[:2]
    paired_seeds = spawned[2 : 2 + world_count]
    matched_seeds = spawned[2 + world_count :]
    cubic_specs, quadratic_specs = paired_misspecification_specs(
        world_count, np.random.default_rng(cubic_assignment)
    )
    scope_matched_specs = balanced_world_specs(
        world_count, MATCHED, np.random.default_rng(matched_assignment)
    )
    for spec, world_seed in zip(scope_matched_specs, matched_seeds, strict=True):
        rng = np.random.default_rng(world_seed)
        train = generate_a2_cases(
            spec,
            scope_sample_size,
            rng,
            composition=False,
            noise_probability=TRAIN_NOISE_FOR_WIDTH_AND_SCOPE,
        )
        test = generate_a2_cases(
            spec,
            test_case_count,
            rng,
            composition=True,
            noise_probability=OOD_NOISE,
        )
        records["misspecification"].append(
            _scope_record(spec, train, test, scope_sample_size)
        )
        portable_axes["misspecification"].append(
            {
                "spec": _spec_payload(spec),
                "train": [_case_payload(case) for case in train],
                "test": [_case_payload(case) for case in test],
            }
        )
    for cubic, quadratic, world_seed in zip(
        cubic_specs, quadratic_specs, paired_seeds, strict=True
    ):
        stream_seed = int(world_seed.generate_state(1, dtype=np.uint64)[0])
        for spec in (cubic, quadratic):
            rng = np.random.default_rng(stream_seed)
            train = generate_a2_cases(
                spec,
                scope_sample_size,
                rng,
                composition=False,
                noise_probability=TRAIN_NOISE_FOR_WIDTH_AND_SCOPE,
            )
            test = generate_a2_cases(
                spec,
                test_case_count,
                rng,
                composition=True,
                noise_probability=OOD_NOISE,
            )
            records["misspecification"].append(
                _scope_record(spec, train, test, scope_sample_size)
            )
            portable_axes["misspecification"].append(
                {
                    "spec": _spec_payload(spec),
                    "train": [_case_payload(case) for case in train],
                    "test": [_case_payload(case) for case in test],
                }
            )

    audit = {
        "world_count_per_axis_or_scope_condition": world_count,
        "candidate_width": _family_audit(width_specs),
        "training_noise": _family_audit(noise_specs),
        "scope_matched": _family_audit(scope_matched_specs),
        "scope_typed_misspecified": _family_audit(cubic_specs),
        "scope_generic_misspecified": _family_audit(quadratic_specs),
        "paired_scope_nonfamily_parameters_equal": all(
            cubic.graph == quadratic.graph
            and cubic.multipliers == quadratic.multipliers
            and cubic.coefficients == quadratic.coefficients
            for cubic, quadratic in zip(cubic_specs, quadratic_specs, strict=True)
        ),
    }
    return records, portable, audit
