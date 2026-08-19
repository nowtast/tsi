"""Observation-only graph discovery for the structural factorization benchmark."""

from __future__ import annotations

from dataclasses import replace

from .paper3_learned_v2_mechanism import (
    ObservableMechanismSignature,
    predict_target_code,
)
from .paper3_learned_v2_generator import GRAPH_VARIANT_MANIFEST
from .paper3_learned_v3_factorized_head import factorize_training_signature
from .paper3_learned_v3_generator import V3WorldDataset
from .paper3_replication_family import GRAPH_NAMES as REPLICATION_GRAPH_NAMES
from .paper3_replication_family import ReplicationDataset
from .paper3_replication_factorized import evaluate as evaluate_replication
from .paper3_replication_factorized import factorize as factorize_replication


def _v3_training_accuracy(
    dataset: V3WorldDataset, graph_name: str
) -> tuple[float, ObservableMechanismSignature]:
    signature = factorize_training_signature(dataset.partitions["train"], graph_name)
    observable = ObservableMechanismSignature(
        graph_name,
        signature.layer_multipliers,
        signature.bridge_coefficient,
        signature.context_coefficient,
        len(dataset.partitions["train"]),
        1,
    )
    correct = sum(
        predict_target_code(case.source_code, case.action, observable)
        == case.target_code
        for case in dataset.partitions["train"]
    )
    return correct / len(dataset.partitions["train"]), observable


def discover_v3_graph(dataset: V3WorldDataset) -> dict[str, object]:
    candidates = []
    for graph in GRAPH_VARIANT_MANIFEST:
        try:
            accuracy, signature = _v3_training_accuracy(dataset, graph.identifier)
        except ValueError:
            accuracy, signature = 0.0, None
        candidates.append((graph.identifier, accuracy, signature))
    winners = [candidate for candidate in candidates if candidate[1] == 1.0]
    if len(winners) != 1:
        raise ValueError(
            f"graph is not uniquely discovered: {[candidate[0] for candidate in winners]}"
        )
    winner = winners[0]
    return {
        "identified_graph": winner[0],
        "true_graph": dataset.graph.identifier,
        "training_exact": winner[1],
        "graph_exact": winner[0] == dataset.graph.identifier,
        "candidate_training_accuracies": {
            name: accuracy for name, accuracy, _ in candidates
        },
    }


def discover_replication_graph(dataset: ReplicationDataset) -> dict[str, object]:
    candidates = []
    for graph in REPLICATION_GRAPH_NAMES:
        candidate_dataset = replace(dataset, graph=graph)
        try:
            signature = factorize_replication(candidate_dataset)
            score = evaluate_replication(candidate_dataset, signature, split="train")
            accuracy = score["exact_accuracy"]
        except ValueError:
            signature, accuracy = None, 0.0
        candidates.append((graph, accuracy, signature))
    winners = [candidate for candidate in candidates if candidate[1] == 1.0]
    if len(winners) != 1:
        raise ValueError(
            f"replication graph is not uniquely discovered: {[candidate[0] for candidate in winners]}"
        )
    return {
        "identified_graph": winners[0][0],
        "true_graph": dataset.graph,
        "training_exact": winners[0][1],
        "graph_exact": winners[0][0] == dataset.graph,
        "candidate_training_accuracies": {
            name: accuracy for name, accuracy, _ in candidates
        },
    }
