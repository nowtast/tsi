"""Cohort construction for the one-shot Research A1 confirmation."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from .paper34_resolution_benchmark import (
    TransitionCase,
    WorldSpec,
    coordinate_nll,
    generate_cases,
)
from .research_a_contract import (
    OOD_NOISE,
    PRIMARY_SAMPLE_SIZES,
    TEST_CASE_COUNT,
    TRAIN_NOISE,
    WORLD_COUNT,
)
from .research_a_design import (
    exact_generic_support_recovered,
    fit_isomorphic_generic,
    fit_typed_structured,
    fit_unstructured_generic,
    isomorphic_prediction_audit,
)
from .research_a_development import _balanced_assignments


def _case_payload(case: TransitionCase) -> dict[str, object]:
    return {
        "source": list(case.source),
        "action": list(case.action),
        "observed": list(case.observed),
        "center": list(case.center),
        "composition_stratum": case.composition_stratum,
    }


def _typed_exact(model: object, spec: WorldSpec) -> bool:
    return bool(
        getattr(model, "families") == spec.families
        and getattr(model, "multipliers") == spec.multipliers
        and getattr(model, "coefficients") == spec.coefficients
    )


def run_cohort(
    root_seed: bytes,
    *,
    world_count: int = WORLD_COUNT,
    sample_sizes: Sequence[int] = PRIMARY_SAMPLE_SIZES,
    test_case_count: int = TEST_CASE_COUNT,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Generate evidence rows, portable replay data, and derivation audits."""

    if len(root_seed) != 32:
        raise ValueError("Research A1 root seed must contain 32 bytes")
    if world_count % 9:
        raise ValueError("world_count must balance nine family-pair strata")
    sizes = tuple(sample_sizes)
    if sizes != tuple(sorted(set(sizes))) or not sizes:
        raise ValueError("sample_sizes must be nonempty, unique, and increasing")
    root = np.random.SeedSequence(int.from_bytes(root_seed, "little"))
    assignment_seed, *world_seeds = root.spawn(world_count + 1)
    assignment_rng = np.random.default_rng(assignment_seed)
    graphs, families = _balanced_assignments(world_count, assignment_rng)
    rows = []
    portable_worlds = []

    for world_index, (graph, family_pair, world_seed) in enumerate(
        zip(graphs, families, world_seeds, strict=True)
    ):
        rng = np.random.default_rng(world_seed)
        spec = WorldSpec(
            world_index=world_index,
            graph=graph,
            families=family_pair,
            multipliers=tuple(int(value) for value in rng.integers(1, 4, 5)),
            coefficients=tuple(int(value) for value in rng.integers(1, 4, 2)),
        )
        train = generate_cases(
            spec,
            sizes[-1],
            rng,
            composition=False,
            noise_probability=TRAIN_NOISE,
        )
        test = generate_cases(
            spec,
            test_case_count,
            rng,
            composition=True,
            noise_probability=OOD_NOISE,
        )
        estimates = []
        for sample_size in sizes:
            prefix = train[:sample_size]
            typed = fit_typed_structured(prefix, spec.graph)
            isomorphic = fit_isomorphic_generic(prefix, spec.graph)
            generic = fit_unstructured_generic(prefix, spec.graph)
            notation = isomorphic_prediction_audit(typed, isomorphic, test)
            if not notation["passed"]:
                raise RuntimeError("typed/isomorphic notation invariant failed")
            typed_nll = coordinate_nll(typed, test)
            isomorphic_nll = coordinate_nll(isomorphic, test)
            generic_nll = coordinate_nll(generic, test)
            typed_exact = _typed_exact(typed, spec)
            generic_exact = exact_generic_support_recovered(generic, spec)
            estimates.append(
                {
                    "sample_size": sample_size,
                    "typed_exact": typed_exact,
                    "isomorphic_exact": exact_generic_support_recovered(
                        isomorphic, spec
                    ),
                    "generic_exact": generic_exact,
                    "typed_nll": typed_nll,
                    "isomorphic_nll": isomorphic_nll,
                    "generic_nll": generic_nll,
                    "generic_minus_typed_nll": generic_nll - typed_nll,
                    "typed_minus_generic_exact": int(typed_exact)
                    - int(generic_exact),
                    "typed_minus_isomorphic_nll": typed_nll - isomorphic_nll,
                }
            )
        rows.append(
            {
                "world_index": world_index,
                "graph": [graph[0], list(graph[1])],
                "families": list(family_pair),
                "estimates": estimates,
            }
        )
        portable_worlds.append(
            {
                "world_index": world_index,
                "spec": {
                    "graph": [graph[0], list(graph[1])],
                    "families": list(family_pair),
                    "multipliers": list(spec.multipliers),
                    "coefficients": list(spec.coefficients),
                },
                "train": [_case_payload(case) for case in train],
                "test": [_case_payload(case) for case in test],
            }
        )

    family_counts = Counter(tuple(pair) for pair in families)
    graph_counts = Counter(graph for graph in graphs)
    audit = {
        "world_count": world_count,
        "family_pair_count": len(family_counts),
        "family_pair_counts": {
            "|".join(pair): count for pair, count in sorted(family_counts.items())
        },
        "family_pairs_balanced": len(set(family_counts.values())) == 1,
        "graph_manifest_coverage": len(graph_counts),
        "graph_count_minimum": min(graph_counts.values()),
        "graph_count_maximum": max(graph_counts.values()),
        "notation_invariant_passed": all(
            all(estimate["typed_minus_isomorphic_nll"] == 0.0 for estimate in row["estimates"])  # type: ignore[index]
            for row in rows
        ),
    }
    portable = {
        "status": "confirmatory_portable_replay_with_answers",
        "world_count": world_count,
        "sample_sizes": list(sizes),
        "test_case_count_per_world": test_case_count,
        "worlds": portable_worlds,
    }
    return rows, portable, audit
