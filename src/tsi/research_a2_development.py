"""Development-only execution for the three prespecified Research A2 axes."""

from __future__ import annotations

from hashlib import sha256
from math import sqrt
from statistics import mean, stdev
from typing import Mapping, Sequence

import numpy as np

from .paper34_resolution_benchmark import center_accuracy, coordinate_nll
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
    audit_fourth_family_separation,
    audit_misspecification_catalogs,
    audit_width_feature_libraries,
)
from .research_a2_populations import (
    GENERIC_MISSPECIFIED,
    MATCHED,
    TYPED_MISSPECIFIED,
    A2WorldSpec,
    balanced_world_specs,
    generate_a2_cases,
    generate_coupled_noise_cases,
    paired_misspecification_specs,
)


DEVELOPMENT_SEED_LABEL = "TSI-RESEARCH-A2-DEVELOPMENT-v1"
WIDTH_SAMPLE_SIZES = (10, 15, 20, 25, 30, 40)
NOISE_SAMPLE_SIZES = (15, 20, 30, 40, 80, 160)
NOISE_PROBABILITIES = (0.08, 0.3, 0.6, 0.8)
MISSPECIFICATION_SAMPLE_SIZES = (20, 40, 80, 160, 320)
DEFAULT_MATCHED_WORLD_COUNT = 36
DEFAULT_MISSPECIFICATION_WORLD_COUNT = 45
DEFAULT_TEST_CASE_COUNT = 600
OOD_NOISE_PROBABILITY = 0.12


def _root_seed(label: str) -> int:
    return int.from_bytes(sha256(label.encode()).digest()[:8], "little")


def _validate_sizes(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or result != tuple(sorted(set(result))) or result[0] <= 0:
        raise ValueError(f"{name} must be positive, unique, and increasing")
    return result


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    sd = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean(values),
        "world_sd": sd,
        "standard_error": sd / sqrt(len(values)),
        "minimum": min(values),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": max(values),
    }


def _evaluate_matched(
    spec: A2WorldSpec,
    train: Sequence[object],
    test: Sequence[object],
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
        "sample_size": sample_size,
        "position_count": width,
        "typed_exact": typed_exact,
        "generic_exact": generic_exact,
        "typed_nll": typed_nll,
        "generic_nll": generic_nll,
        "generic_minus_typed_nll": generic_nll - typed_nll,
        "typed_minus_generic_exact": int(typed_exact) - int(generic_exact),
    }


def _summarize_records(
    records: Sequence[Mapping[str, object]], group_keys: Sequence[str]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for record in records:
        key = tuple(record[name] for name in group_keys)
        groups.setdefault(key, []).append(record)
    summaries = []
    for key, rows in sorted(groups.items(), key=lambda item: item[0]):
        summary: dict[str, object] = dict(zip(group_keys, key, strict=True))
        for metric in (
            "generic_minus_typed_nll",
            "typed_minus_generic_exact",
            "typed_minus_generic_center_accuracy",
        ):
            if metric in rows[0]:
                summary[metric] = _summary([float(row[metric]) for row in rows])
        for metric in ("typed_exact", "generic_exact"):
            if metric in rows[0]:
                summary[f"{metric}_rate"] = mean(
                    float(bool(row[metric])) for row in rows
                )
        summaries.append(summary)
    return summaries


def _run_width_axis(
    specs: Sequence[A2WorldSpec],
    world_seeds: Sequence[np.random.SeedSequence],
    sample_sizes: Sequence[int],
    test_case_count: int,
) -> dict[str, object]:
    records = []
    for spec, world_seed in zip(specs, world_seeds, strict=True):
        rng = np.random.default_rng(world_seed)
        train = generate_a2_cases(
            spec,
            max(sample_sizes),
            rng,
            composition=False,
            noise_probability=0.08,
        )
        test = generate_a2_cases(
            spec,
            test_case_count,
            rng,
            composition=True,
            noise_probability=OOD_NOISE_PROBABILITY,
        )
        for sample_size in sample_sizes:
            for width in WIDTH_POSITION_COUNTS:
                record = _evaluate_matched(spec, train, test, sample_size, width)
                record["world_index"] = spec.world_index
                record["family_pair"] = list(spec.families)
                records.append(record)
    return {
        "axis": "candidate_width",
        "sample_sizes": list(sample_sizes),
        "position_counts": list(WIDTH_POSITION_COUNTS),
        "multiplicity_member_count_candidate": len(sample_sizes)
        * len(WIDTH_POSITION_COUNTS)
        * 2,
        "records": records,
        "summaries": _summarize_records(records, ("position_count", "sample_size")),
    }


def _run_noise_axis(
    specs: Sequence[A2WorldSpec],
    world_seeds: Sequence[np.random.SeedSequence],
    sample_sizes: Sequence[int],
    test_case_count: int,
) -> dict[str, object]:
    records = []
    for spec, world_seed in zip(specs, world_seeds, strict=True):
        rng = np.random.default_rng(world_seed)
        streams = generate_coupled_noise_cases(
            spec, max(sample_sizes), rng, NOISE_PROBABILITIES
        )
        test = generate_a2_cases(
            spec,
            test_case_count,
            rng,
            composition=True,
            noise_probability=OOD_NOISE_PROBABILITY,
        )
        for noise_probability in NOISE_PROBABILITIES:
            train = streams[noise_probability]
            for sample_size in sample_sizes:
                record = _evaluate_matched(
                    spec, train, test, sample_size, WIDTH_POSITION_COUNTS[0]
                )
                record["world_index"] = spec.world_index
                record["family_pair"] = list(spec.families)
                record["train_noise_probability"] = noise_probability
                records.append(record)
    return {
        "axis": "training_noise",
        "sample_sizes": list(sample_sizes),
        "train_noise_probabilities": list(NOISE_PROBABILITIES),
        "noise_masks_nested_within_world": True,
        "multiplicity_member_count_candidate": len(sample_sizes)
        * len(NOISE_PROBABILITIES)
        * 2,
        "records": records,
        "summaries": _summarize_records(
            records, ("train_noise_probability", "sample_size")
        ),
    }


def _misspecification_record(
    spec: A2WorldSpec,
    train: Sequence[object],
    test: Sequence[object],
    sample_size: int,
) -> dict[str, object]:
    prefix = train[:sample_size]
    generic_catalog = (
        TYPED_FAMILY_CATALOG
        if spec.condition == MATCHED
        else ALTERNATIVE_FAMILY_CATALOG
    )
    typed = fit_typed_catalog(prefix, spec.graph, family_catalog=TYPED_FAMILY_CATALOG)
    generic = fit_catalog_generic(
        prefix,
        spec.graph,
        family_catalog=generic_catalog,
    )
    typed_nll = coordinate_nll(typed, test)
    generic_nll = coordinate_nll(generic, test)
    typed_accuracy = center_accuracy(typed, test)
    generic_accuracy = center_accuracy(generic, test)
    return {
        "world_index": spec.world_index,
        "condition": spec.condition,
        "sample_size": sample_size,
        "family_pair": list(spec.families),
        "typed_exact": typed_parameters_recovered(typed, spec),
        "generic_exact": exact_catalog_support_recovered(
            generic, spec, generic_catalog
        ),
        "generic_minus_typed_nll": generic_nll - typed_nll,
        "typed_minus_generic_center_accuracy": typed_accuracy - generic_accuracy,
    }


def _run_misspecification_axis(
    matched_specs: Sequence[A2WorldSpec],
    cubic_specs: Sequence[A2WorldSpec],
    quadratic_specs: Sequence[A2WorldSpec],
    matched_seeds: Sequence[np.random.SeedSequence],
    paired_seeds: Sequence[np.random.SeedSequence],
    sample_sizes: Sequence[int],
    test_case_count: int,
) -> dict[str, object]:
    records = []
    for spec, world_seed in zip(matched_specs, matched_seeds, strict=True):
        rng = np.random.default_rng(world_seed)
        train = generate_a2_cases(
            spec, max(sample_sizes), rng, composition=False, noise_probability=0.08
        )
        test = generate_a2_cases(
            spec,
            test_case_count,
            rng,
            composition=True,
            noise_probability=OOD_NOISE_PROBABILITY,
        )
        for sample_size in sample_sizes:
            records.append(_misspecification_record(spec, train, test, sample_size))
    for cubic, quadratic, world_seed in zip(
        cubic_specs, quadratic_specs, paired_seeds, strict=True
    ):
        stream_seed = int(world_seed.generate_state(1, dtype=np.uint64)[0])
        for spec in (cubic, quadratic):
            rng = np.random.default_rng(stream_seed)
            train = generate_a2_cases(
                spec,
                max(sample_sizes),
                rng,
                composition=False,
                noise_probability=0.08,
            )
            test = generate_a2_cases(
                spec,
                test_case_count,
                rng,
                composition=True,
                noise_probability=OOD_NOISE_PROBABILITY,
            )
            for sample_size in sample_sizes:
                records.append(_misspecification_record(spec, train, test, sample_size))
    return {
        "axis": "bidirectional_misspecification_scope_audit",
        "sample_sizes": list(sample_sizes),
        "conditions": [MATCHED, TYPED_MISSPECIFIED, GENERIC_MISSPECIFIED],
        "catalog_position_count_each_arm": 55,
        "paired_cubic_quadratic_graph_parameters_and_raw_streams": True,
        "confirmatory_role": "scope_and_falsification_only_cannot_rescue_efficiency",
        "records": records,
        "summaries": _summarize_records(records, ("condition", "sample_size")),
    }


def run_a2_development(
    *,
    matched_world_count: int = DEFAULT_MATCHED_WORLD_COUNT,
    misspecification_world_count: int = DEFAULT_MISSPECIFICATION_WORLD_COUNT,
    width_sample_sizes: Sequence[int] = WIDTH_SAMPLE_SIZES,
    noise_sample_sizes: Sequence[int] = NOISE_SAMPLE_SIZES,
    misspecification_sample_sizes: Sequence[int] = MISSPECIFICATION_SAMPLE_SIZES,
    test_case_count: int = DEFAULT_TEST_CASE_COUNT,
) -> dict[str, object]:
    """Run the reusable A2 pilot without creating or consuming a sealed seed."""

    width_sizes = _validate_sizes(width_sample_sizes, "width_sample_sizes")
    noise_sizes = _validate_sizes(noise_sample_sizes, "noise_sample_sizes")
    misspec_sizes = _validate_sizes(
        misspecification_sample_sizes, "misspecification_sample_sizes"
    )
    if matched_world_count <= 1 or misspecification_world_count <= 1:
        raise ValueError("each development population requires at least two worlds")
    if test_case_count <= 0:
        raise ValueError("test_case_count must be positive")

    root = np.random.SeedSequence(_root_seed(DEVELOPMENT_SEED_LABEL))
    width_root, noise_root, misspec_root = root.spawn(3)

    width_assignment, *width_seeds = width_root.spawn(matched_world_count + 1)
    width_specs = balanced_world_specs(
        matched_world_count, MATCHED, np.random.default_rng(width_assignment)
    )
    width_axis = _run_width_axis(width_specs, width_seeds, width_sizes, test_case_count)

    noise_assignment, *noise_seeds = noise_root.spawn(matched_world_count + 1)
    noise_specs = balanced_world_specs(
        matched_world_count, MATCHED, np.random.default_rng(noise_assignment)
    )
    noise_axis = _run_noise_axis(noise_specs, noise_seeds, noise_sizes, test_case_count)

    misspec_assignment, matched_assignment, *misspec_seeds = misspec_root.spawn(
        misspecification_world_count + 2
    )
    cubic_specs, quadratic_specs = paired_misspecification_specs(
        misspecification_world_count,
        np.random.default_rng(misspec_assignment),
    )
    matched_specs = balanced_world_specs(
        misspecification_world_count,
        MATCHED,
        np.random.default_rng(matched_assignment),
    )
    matched_seed_root = np.random.SeedSequence(
        _root_seed(DEVELOPMENT_SEED_LABEL + "-MISSPEC-MATCHED")
    )
    misspec_axis = _run_misspecification_axis(
        matched_specs,
        cubic_specs,
        quadratic_specs,
        matched_seed_root.spawn(misspecification_world_count),
        misspec_seeds,
        misspec_sizes,
        test_case_count,
    )

    audits = {
        "width_feature_library": audit_width_feature_libraries(),
        "fourth_family_separation": audit_fourth_family_separation(),
        "misspecification_catalogs": audit_misspecification_catalogs(),
    }
    if not all(bool(audit["passed"]) for audit in audits.values()):
        raise RuntimeError("an A2 mathematical or implementation audit failed")
    return {
        "status": "development_only_not_confirmatory",
        "confirmatory_seed_created": False,
        "seed_label": DEVELOPMENT_SEED_LABEL,
        "seed_label_sha256": sha256(DEVELOPMENT_SEED_LABEL.encode()).hexdigest(),
        "matched_world_count_per_efficiency_axis": matched_world_count,
        "world_count_per_misspecification_condition": misspecification_world_count,
        "test_case_count_per_world": test_case_count,
        "held_out_test_used_for_fit_or_selection": False,
        "audits": audits,
        "axes": {
            "candidate_width": width_axis,
            "training_noise": noise_axis,
            "misspecification": misspec_axis,
        },
    }
