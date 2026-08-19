"""Factorized estimator for the independent P3-5B replication family."""

from __future__ import annotations

from dataclasses import dataclass

from .paper3_replication_family import (
    CARDINALITIES,
    COEFFICIENT_OPTIONS,
    GRAPH_EDGES,
    LAYER_NAMES,
    ReplicationCase,
    ReplicationDataset,
)


@dataclass(frozen=True)
class ReplicationSignature:
    graph: str
    multipliers: tuple[int, ...]
    coefficient: int
    matched_case_count: int


def _delta(case: ReplicationCase, layer: int) -> int:
    return (case.target[layer] - case.source[layer]) % CARDINALITIES[layer]


def _unique(candidates: tuple[int, ...], equations: list[tuple[int, int, int]]) -> int:
    matches = tuple(
        candidate
        for candidate in candidates
        if all(
            (candidate * value) % modulus == observed
            for value, observed, modulus in equations
        )
    )
    if len(matches) != 1:
        raise ValueError(f"replication parameter is not unique: {matches}")
    return matches[0]


def factorize(dataset: ReplicationDataset) -> ReplicationSignature:
    cases = dataset.partitions["train"]
    source_name, target_name = dict(GRAPH_EDGES)[dataset.graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)
    multipliers = []
    for layer, modulus in enumerate(CARDINALITIES):
        equations = []
        for case in cases:
            if (
                sum(case.action) == 1
                and case.action[layer] == 1
                and (layer != target_index or case.source[target_index] == 0)
            ):
                equations.append((1, _delta(case, layer), modulus))
        multipliers.append(_unique((1, 2), equations))
    coefficient_equations = [
        (
            1 + case.source[target_index],
            _delta(case, target_index),
            CARDINALITIES[target_index],
        )
        for case in cases
        if sum(case.action) == 1 and case.action[source_index] == 1
    ]
    coefficient = _unique(COEFFICIENT_OPTIONS, coefficient_equations)
    return ReplicationSignature(
        dataset.graph, tuple(multipliers), coefficient, len(cases)
    )


def evaluate(
    dataset: ReplicationDataset,
    signature: ReplicationSignature | None = None,
    *,
    split: str = "test",
) -> dict[str, object]:
    signature = signature or factorize(dataset)
    correct = 0
    cases = dataset.partitions[split]
    source_name, target_name = dict(GRAPH_EDGES)[dataset.graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)
    for case in cases:
        delta = [
            signature.multipliers[index] * case.action[index] for index in range(5)
        ]
        delta[target_index] += (
            signature.coefficient
            * case.action[source_index]
            * (1 + case.source[target_index])
        )
        prediction = tuple(
            (case.source[index] + delta[index]) % CARDINALITIES[index]
            for index in range(5)
        )
        correct += prediction == case.target
    return {
        "graph": dataset.graph,
        "combination_index": dataset.combination_index,
        "case_count": len(cases),
        "exact_accuracy": correct / len(cases),
        "signature": signature.__dict__,
    }
