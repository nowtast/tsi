"""Explicit graph-conditioned parameterized transition head for v3."""

from __future__ import annotations

from dataclasses import dataclass

from .paper3_learned_v2_mechanism import identify_observable_mechanism
from .paper3_learned_v2_generator import GRAPH_VARIANT_MANIFEST
from .paper3_learned_v3_generator import V3WorldDataset
from .paper3_multiworld import LAYER_ORDER, MultiworldStateCode


@dataclass(frozen=True)
class StructuredHeadTrace:
    training_world_count: int
    training_case_count: int
    training_exact_accuracy: float
    active_signature_count: int


def _successor_from_signature(source, action, signature) -> MultiworldStateCode:
    values = dict(zip(LAYER_ORDER, source.as_tuple(), strict=True))
    actions = action.mapping
    multipliers = dict(zip(LAYER_ORDER, signature.layer_multipliers, strict=True))
    delta = {layer: multipliers[layer] * actions[layer] for layer in LAYER_ORDER}
    graph = next(item for item in GRAPH_VARIANT_MANIFEST if item.identifier == signature.graph_variant)
    for source_layer, target_layer in graph.source_cross_edges:
        if (source_layer, target_layer) == ("topology", "relation"):
            delta[target_layer] += signature.bridge_coefficient * values[source_layer] * actions[target_layer]
        elif (source_layer, target_layer) == ("order", "metric"):
            delta[target_layer] += signature.context_coefficient * values[source_layer] * actions[target_layer]
        elif (source_layer, target_layer) == ("metric", "relation"):
            delta[target_layer] += signature.bridge_coefficient * values[source_layer] * actions[target_layer]
        elif (source_layer, target_layer) == ("relation", "topology"):
            delta[target_layer] += signature.bridge_coefficient * values[source_layer] * actions[target_layer]
    for source_layer, target_layer in graph.action_cross_edges:
        coefficient = signature.context_coefficient if target_layer == "metric" else signature.bridge_coefficient
        delta[target_layer] += coefficient * actions[source_layer] * (1 + values[target_layer])
    return MultiworldStateCode(
        label_phase=(values["label"] + delta["label"]) % 3,
        topology_mode=(values["topology"] + delta["topology"]) % 3,
        metric_mode=(values["metric"] + delta["metric"]) % 3,
        influence_mode=(values["relation"] + delta["relation"]) % 4,
        order_mode=(values["order"] + delta["order"]) % 3,
    )


@dataclass(frozen=True)
class StructuredParameterizedTransitionHead:
    """Explicit parameterized transition law conditioned on an observable signature."""

    signatures: tuple[object, ...]
    trace: StructuredHeadTrace

    @classmethod
    def fit(cls, datasets: tuple[V3WorldDataset, ...] | list[V3WorldDataset]) -> "StructuredParameterizedTransitionHead":
        if not datasets:
            raise ValueError("structured head requires training datasets")
        signatures = []
        correct = 0
        total = 0
        for dataset in datasets:
            signature = identify_observable_mechanism(dataset.partitions["train"])
            signatures.append(signature)
            for case in dataset.partitions["train"]:
                correct += _successor_from_signature(case.source_code, case.action, signature) == case.target_code
                total += 1
        trace = StructuredHeadTrace(
            training_world_count=len(datasets),
            training_case_count=total,
            training_exact_accuracy=correct / total,
            active_signature_count=len({(s.graph_variant, s.mechanism_tuple) for s in signatures}),
        )
        return cls(tuple(signatures), trace)

    def evaluate(self, dataset: V3WorldDataset, *, partition: str = "test") -> dict[str, object]:
        signature = identify_observable_mechanism(dataset.partitions["train"])
        cases = dataset.partitions[partition]
        correct = sum(
            _successor_from_signature(case.source_code, case.action, signature) == case.target_code
            for case in cases
        )
        return {
            "world_index": dataset.world_index,
            "mechanism_combination_index": dataset.mechanism_combination_index,
            "graph_variant": dataset.graph.identifier,
            "partition": partition,
            "case_count": len(cases),
            "exact_accuracy": correct / len(cases) if cases else float("nan"),
            "active_signature": signature.as_dict(),
        }
