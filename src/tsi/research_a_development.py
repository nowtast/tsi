"""Development-only pilot for calibrating the prospective Research A study."""

from __future__ import annotations

from hashlib import sha256
from itertools import product
from math import sqrt
from statistics import mean, stdev
from typing import Sequence

import numpy as np

from .paper34_resolution_benchmark import (
    GRAPH_MANIFEST,
    WorldSpec,
    coordinate_nll,
    generate_cases,
)
from .paper34_resolution_contract import HEAD_FAMILIES
from .research_a_design import (
    exact_generic_support_recovered,
    fit_isomorphic_generic,
    fit_typed_structured,
    fit_unstructured_generic,
    isomorphic_prediction_audit,
)


DEVELOPMENT_SEED_LABEL = "TSI-RESEARCH-A-DEVELOPMENT-v1"
DEFAULT_SAMPLE_SIZES = (50, 100, 200, 300, 400, 800, 1600, 3200, 6400, 12800)
DEFAULT_DEVELOPMENT_WORLD_COUNT = 36
DEFAULT_TEST_CASE_COUNT = 1200


def _root_seed(label: str) -> int:
    return int.from_bytes(sha256(label.encode()).digest()[:8], "little")


def _balanced_assignments(
    world_count: int, rng: np.random.Generator
) -> tuple[list[tuple[int, tuple[int, int]]], list[tuple[str, str]]]:
    family_pairs = tuple(product(HEAD_FAMILIES, repeat=2))
    graphs = [GRAPH_MANIFEST[index % len(GRAPH_MANIFEST)] for index in range(world_count)]
    families = [family_pairs[index % len(family_pairs)] for index in range(world_count)]
    rng.shuffle(graphs)
    rng.shuffle(families)
    return graphs, families


def _typed_exact(model: object, spec: WorldSpec) -> bool:
    return bool(
        getattr(model, "families") == spec.families
        and getattr(model, "multipliers") == spec.multipliers
        and getattr(model, "coefficients") == spec.coefficients
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "world_sd": stdev(values) if len(values) > 1 else 0.0,
        "standard_error": stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def run_development(
    *,
    world_count: int = DEFAULT_DEVELOPMENT_WORLD_COUNT,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    test_case_count: int = DEFAULT_TEST_CASE_COUNT,
) -> dict[str, object]:
    """Run a reusable pilot; this function must never consume a sealed seed."""

    if world_count < 9 or world_count % 9:
        raise ValueError("development world_count must be a positive multiple of nine")
    ordered_sizes = tuple(int(value) for value in sample_sizes)
    if not ordered_sizes or ordered_sizes != tuple(sorted(set(ordered_sizes))):
        raise ValueError("sample_sizes must be nonempty, unique, and increasing")
    if ordered_sizes[0] <= 0 or test_case_count <= 0:
        raise ValueError("case counts must be positive")

    root = np.random.SeedSequence(_root_seed(DEVELOPMENT_SEED_LABEL))
    assignment_seed, *world_seeds = root.spawn(world_count + 1)
    assignment_rng = np.random.default_rng(assignment_seed)
    graphs, families = _balanced_assignments(world_count, assignment_rng)
    rows = []
    maximum_size = ordered_sizes[-1]

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
            maximum_size,
            rng,
            composition=False,
            noise_probability=0.08,
        )
        test = generate_cases(
            spec,
            test_case_count,
            rng,
            composition=True,
            noise_probability=0.12,
        )
        estimates = []
        for sample_size in ordered_sizes:
            prefix = train[:sample_size]
            typed = fit_typed_structured(prefix, spec.graph)
            isomorphic = fit_isomorphic_generic(prefix, spec.graph)
            generic = fit_unstructured_generic(prefix, spec.graph)
            notation_audit = isomorphic_prediction_audit(typed, isomorphic, test)
            if not notation_audit["passed"]:
                raise RuntimeError("typed/isomorphic prediction invariant failed")
            typed_nll = coordinate_nll(typed, test)
            isomorphic_nll = coordinate_nll(isomorphic, test)
            generic_nll = coordinate_nll(generic, test)
            estimates.append(
                {
                    "sample_size": sample_size,
                    "typed_exact": _typed_exact(typed, spec),
                    "isomorphic_exact": exact_generic_support_recovered(
                        isomorphic, spec
                    ),
                    "generic_exact": exact_generic_support_recovered(generic, spec),
                    "typed_nll": typed_nll,
                    "isomorphic_nll": isomorphic_nll,
                    "generic_nll": generic_nll,
                    "generic_minus_typed_nll": generic_nll - typed_nll,
                    "typed_minus_generic_exact": int(_typed_exact(typed, spec))
                    - int(exact_generic_support_recovered(generic, spec)),
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

    summaries = []
    for position, sample_size in enumerate(ordered_sizes):
        records = [row["estimates"][position] for row in rows]  # type: ignore[index]
        summaries.append(
            {
                "sample_size": sample_size,
                "typed_exact_rate": mean(float(record["typed_exact"]) for record in records),
                "isomorphic_exact_rate": mean(
                    float(record["isomorphic_exact"]) for record in records
                ),
                "generic_exact_rate": mean(
                    float(record["generic_exact"]) for record in records
                ),
                "generic_minus_typed_nll": _summary(
                    [float(record["generic_minus_typed_nll"]) for record in records]
                ),
                "typed_minus_generic_exact": _summary(
                    [float(record["typed_minus_generic_exact"]) for record in records]
                ),
                "maximum_absolute_notation_nll_difference": max(
                    abs(float(record["typed_minus_isomorphic_nll"]))
                    for record in records
                ),
            }
        )

    return {
        "status": "development_only_not_confirmatory",
        "seed_label": DEVELOPMENT_SEED_LABEL,
        "seed_label_sha256": sha256(DEVELOPMENT_SEED_LABEL.encode()).hexdigest(),
        "world_count": world_count,
        "sample_sizes": list(ordered_sizes),
        "test_case_count_per_world": test_case_count,
        "train_prefixes_nested": True,
        "test_used_for_fit_or_selection": False,
        "summaries": summaries,
        "rows": rows,
    }
