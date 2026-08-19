"""Stage 4 comparative validation on the independent replication family."""

from __future__ import annotations

from dataclasses import dataclass

from .paper3_learned_structure import discover_replication_graph
from .paper3_replication_family import (
    CARDINALITIES,
    GRAPH_NAMES,
    ReplicationCase,
    ReplicationDataset,
)
from .paper3_replication_factorized import factorize


def _diagonal_prediction(
    case: ReplicationCase, multipliers: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        (case.source[index] + multipliers[index] * case.action[index])
        % CARDINALITIES[index]
        for index in range(5)
    )


def _estimate_diagonal(dataset: ReplicationDataset) -> tuple[int, ...]:
    multipliers = []
    for layer, modulus in enumerate(CARDINALITIES):
        values = {
            (case.target[layer] - case.source[layer]) % modulus
            for case in dataset.partitions["train"]
            if sum(case.action) == 1 and case.action[layer] == 1
        }
        multipliers.append(
            next((value for value in (1, 2) if value % modulus in values), 1)
        )
    return tuple(multipliers)


@dataclass(frozen=True)
class ComparativeModel:
    name: str
    predict: object


def fit_vector_only(dataset: ReplicationDataset) -> ComparativeModel:
    multipliers = _estimate_diagonal(dataset)
    return ComparativeModel(
        "vector_only_diagonal", lambda case: _diagonal_prediction(case, multipliers)
    )


def fit_unstructured_lookup(dataset: ReplicationDataset) -> ComparativeModel:
    table = {
        (case.source, case.action): case.target for case in dataset.partitions["train"]
    }
    return ComparativeModel(
        "unstructured_lookup",
        lambda case: table.get((case.source, case.action), case.source),
    )


def fit_tsi_factorized(dataset: ReplicationDataset) -> ComparativeModel:
    discovered = discover_replication_graph(dataset)
    from dataclasses import replace

    discovered_dataset = replace(dataset, graph=discovered["identified_graph"])
    signature = factorize(discovered_dataset)
    source_name, target_name = {
        "topology_to_metric": ("topology", "metric"),
        "metric_to_relation": ("metric", "relation"),
        "relation_to_order": ("relation", "order"),
    }[discovered["identified_graph"]]
    names = ("signal", "topology", "metric", "relation", "order")
    source_index = names.index(source_name)
    target_index = names.index(target_name)
    multipliers, coefficient = signature.multipliers, signature.coefficient

    def predict(case: ReplicationCase) -> tuple[int, ...]:
        delta = [multipliers[index] * case.action[index] for index in range(5)]
        delta[target_index] += (
            coefficient * case.action[source_index] * (1 + case.source[target_index])
        )
        return tuple(
            (case.source[index] + delta[index]) % CARDINALITIES[index]
            for index in range(5)
        )

    return ComparativeModel("tsi_graph_discovered_factorized", predict)


def fit_wrong_routed_factorized(dataset: ReplicationDataset) -> ComparativeModel:
    from dataclasses import replace

    wrong_graph = GRAPH_NAMES[(GRAPH_NAMES.index(dataset.graph) + 1) % len(GRAPH_NAMES)]
    try:
        signature = factorize(replace(dataset, graph=wrong_graph))
    except ValueError:
        return ComparativeModel("wrong_routed_factorized", lambda case: case.source)
    source_name, target_name = dict(
        zip(
            GRAPH_NAMES,
            (("topology", "metric"), ("metric", "relation"), ("relation", "order")),
            strict=True,
        )
    )[wrong_graph]
    names = ("signal", "topology", "metric", "relation", "order")
    source_index, target_index = names.index(source_name), names.index(target_name)

    def predict(case: ReplicationCase) -> tuple[int, ...]:
        delta = [
            signature.multipliers[index] * case.action[index] for index in range(5)
        ]
        delta[target_index] += (
            signature.coefficient
            * case.action[source_index]
            * (1 + case.source[target_index])
        )
        return tuple(
            (case.source[index] + delta[index]) % CARDINALITIES[index]
            for index in range(5)
        )

    return ComparativeModel("wrong_routed_factorized", predict)


def evaluate_model(
    model: ComparativeModel, dataset: ReplicationDataset
) -> dict[str, object]:
    cases = dataset.partitions["test"]
    correct = sum(model.predict(case) == case.target for case in cases)
    intervention_cases = tuple(case for case in cases if case.intervention)
    intervention_correct = sum(
        model.predict(case) == case.target for case in intervention_cases
    )
    return {
        "model": model.name,
        "case_count": len(cases),
        "exact_accuracy": correct / len(cases),
        "intervention_case_count": len(intervention_cases),
        "intervention_exact_accuracy": intervention_correct / len(intervention_cases),
    }
