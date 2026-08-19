"""Graph-randomized four-way development generator for P3-5A-v2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping

from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_learned_v2_contract import GRAPH_VARIANTS, SPLITS
from .paper3_multiworld import (
    MECHANISM_PARAMETER_ACTIONS,
    PRIMITIVE_ACTIONS,
    MultiworldStateCode,
    StructuredAction,
    WorldMechanism,
    all_multiworld_state_codes,
    build_world_mechanism,
)


P3_LEARNED_V2_GENERATOR_ID = "P3-5A-LEARNED-GRAPH-RANDOMIZED-GENERATOR-v1"
DEVELOPMENT_ROOT = "tsi:p3-5a:v2:development:2026-08-07"


@dataclass(frozen=True)
class V2GraphVariant:
    identifier: str
    source_cross_edges: tuple[tuple[str, str], ...]
    action_cross_edges: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "source_cross_edges": [list(edge) for edge in self.source_cross_edges],
            "action_cross_edges": [list(edge) for edge in self.action_cross_edges],
        }


GRAPH_VARIANT_MANIFEST = (
    V2GraphVariant(
        "bridge_topology_to_relation",
        (("topology", "relation"),),
        (("topology", "relation"),),
    ),
    V2GraphVariant(
        "context_order_to_metric",
        (("order", "metric"),),
        (("order", "metric"),),
    ),
    V2GraphVariant(
        "independent_relation",
        (("metric", "relation"),),
        (("metric", "relation"),),
    ),
    V2GraphVariant(
        "wrong_direction_negative_control",
        (("relation", "topology"),),
        (("relation", "topology"),),
    ),
)

if tuple(item.identifier for item in GRAPH_VARIANT_MANIFEST) != GRAPH_VARIANTS:
    raise RuntimeError("v2 graph manifest does not match the frozen contract")


@dataclass(frozen=True)
class V2TransitionCase:
    partition: str
    graph_variant: str
    source_code: MultiworldStateCode
    action: StructuredAction
    target_code: MultiworldStateCode
    intervention: bool

    def __post_init__(self) -> None:
        if self.partition not in SPLITS:
            raise ValueError("unknown v2 partition")
        if self.graph_variant not in GRAPH_VARIANTS:
            raise ValueError("unknown v2 graph variant")
        if self.partition != "test" and self.intervention:
            raise ValueError("intervention cases belong only to the test split")

    @property
    def input_key(self) -> tuple[MultiworldStateCode, tuple[int, ...]]:
        return self.source_code, self.action.components


@dataclass(frozen=True)
class V2WorldDataset:
    mechanism: WorldMechanism
    graph: V2GraphVariant
    partitions: Mapping[str, tuple[V2TransitionCase, ...]]
    digest: str

    def __post_init__(self) -> None:
        if tuple(self.partitions) != SPLITS:
            raise ValueError("v2 partitions must follow the frozen order")
        object.__setattr__(
            self,
            "partitions",
            MappingProxyType(
                {name: tuple(cases) for name, cases in self.partitions.items()}
            ),
        )


def graph_variant_for_world(world_index: int) -> V2GraphVariant:
    if type(world_index) is not int or world_index < 0:
        raise ValueError("world_index must be a nonnegative integer")
    return GRAPH_VARIANT_MANIFEST[world_index % len(GRAPH_VARIANT_MANIFEST)]


def _partition_for(
    source: MultiworldStateCode,
    action: StructuredAction,
    world_index: int,
) -> str:
    payload = f"{P3_LEARNED_V2_GENERATOR_ID}:{world_index}:{source.as_tuple()}:{action.components}"
    bucket = int.from_bytes(sha256(payload.encode("utf-8")).digest()[:4], "little") % 100
    if bucket < 50:
        return "train"
    if bucket < 70:
        return "routing_selection"
    if bucket < 85:
        return "downstream_evaluation"
    return "test"


def _successor(
    source: MultiworldStateCode,
    action: StructuredAction,
    mechanism: WorldMechanism,
    graph: V2GraphVariant,
) -> MultiworldStateCode:
    values = dict(zip(("label", "topology", "metric", "relation", "order"), source.as_tuple(), strict=True))
    actions = action.mapping
    multipliers = dict(
        zip(("label", "topology", "metric", "relation", "order"), mechanism.layer_multipliers, strict=True)
    )
    delta = {
        layer: multipliers[layer] * actions[layer]
        for layer in ("label", "topology", "metric", "relation", "order")
    }
    for edge in graph.source_cross_edges:
        source_layer, target_layer = edge
        if edge == ("topology", "relation"):
            delta[target_layer] += mechanism.bridge_coefficient * values[source_layer] * actions[target_layer]
        elif edge == ("order", "metric"):
            delta[target_layer] += mechanism.context_coefficient * values[source_layer] * actions[target_layer]
        elif edge == ("metric", "relation"):
            delta[target_layer] += mechanism.bridge_coefficient * values[source_layer] * actions[target_layer]
        elif edge == ("relation", "topology"):
            delta[target_layer] += mechanism.bridge_coefficient * values[source_layer] * actions[target_layer]
    for source_layer, target_layer in graph.action_cross_edges:
        coefficient = (
            mechanism.context_coefficient
            if target_layer == "metric"
            else mechanism.bridge_coefficient
        )
        delta[target_layer] += coefficient * actions[source_layer] * (1 + values[target_layer])
    return MultiworldStateCode(
        label_phase=(values["label"] + delta["label"]) % 3,
        topology_mode=(values["topology"] + delta["topology"]) % 3,
        metric_mode=(values["metric"] + delta["metric"]) % 3,
        influence_mode=(values["relation"] + delta["relation"]) % 4,
        order_mode=(values["order"] + delta["order"]) % 3,
    )


def _build_v2_dataset(
    *,
    dataset_world_index: int,
    mechanism: WorldMechanism,
    graph: V2GraphVariant,
) -> V2WorldDataset:
    partitions: dict[str, list[V2TransitionCase]] = {name: [] for name in SPLITS}
    for source in all_multiworld_state_codes():
        for action in PRIMITIVE_ACTIONS:
            partition = _partition_for(source, action, dataset_world_index)
            partitions[partition].append(
                V2TransitionCase(
                    partition=partition,
                    graph_variant=graph.identifier,
                    source_code=source,
                    action=action,
                    target_code=_successor(source, action, mechanism, graph),
                    intervention=False,
                )
            )
        for action in MECHANISM_PARAMETER_ACTIONS:
            partitions["test"].append(
                V2TransitionCase(
                    partition="test",
                    graph_variant=graph.identifier,
                    source_code=source,
                    action=action,
                    target_code=_successor(source, action, mechanism, graph),
                    intervention=True,
                )
            )
    frozen = {name: tuple(cases) for name, cases in partitions.items()}
    payload = {
        "identifier": P3_LEARNED_V2_GENERATOR_ID,
        "dataset_world_index": dataset_world_index,
        "mechanism": mechanism.as_dict(),
        "graph": graph.as_dict(),
        "partitions": {
            name: [
                {
                    "source": list(case.source_code.as_tuple()),
                    "action": list(case.action.components),
                    "target": list(case.target_code.as_tuple()),
                    "intervention": case.intervention,
                }
                for case in cases
            ]
            for name, cases in frozen.items()
        },
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return V2WorldDataset(mechanism, graph, frozen, digest)


def build_v2_world_dataset(world_index: int) -> V2WorldDataset:
    mechanism = build_world_mechanism(
        family=WorldFamily.CONTEXT_DEPENDENT,
        cohort=BenchmarkSplit.DEVELOPMENT,
        world_index=world_index,
    )
    return _build_v2_dataset(
        dataset_world_index=world_index,
        mechanism=mechanism,
        graph=graph_variant_for_world(world_index),
    )


def build_balanced_v2_world_dataset(
    world_index: int,
    mechanism_slot: int,
) -> V2WorldDataset:
    if type(world_index) is not int or world_index < 0:
        raise ValueError("world_index must be a nonnegative integer")
    if type(mechanism_slot) is not int or mechanism_slot < 0:
        raise ValueError("mechanism_slot must be a nonnegative integer")
    mechanism = build_world_mechanism(
        family=WorldFamily.CONTEXT_DEPENDENT,
        cohort=BenchmarkSplit.DEVELOPMENT,
        world_index=mechanism_slot,
    )
    return _build_v2_dataset(
        dataset_world_index=world_index,
        mechanism=mechanism,
        graph=graph_variant_for_world(world_index),
    )


def mechanism_parameter_candidates() -> tuple[tuple[tuple[int, int, int, int, int], int, int], ...]:
    """Return the public finite hypothesis class for observable mechanisms."""
    return tuple(
        (multipliers, bridge, context)
        for multipliers in product(
            (1, 2), (1, 2), (1, 2), (1, 2, 3), (1, 2)
        )
        for bridge in (1, 3)
        for context in (1, 2)
    )


def audit_v2_dataset(dataset: V2WorldDataset) -> dict[str, object]:
    errors: list[str] = []
    keys_by_split = {
        name: {case.input_key for case in cases}
        for name, cases in dataset.partitions.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if keys_by_split[left] & keys_by_split[right]:
                errors.append(f"input overlap: {left}/{right}")
    if not dataset.partitions["test"]:
        errors.append("test partition is empty")
    if not any(case.intervention for case in dataset.partitions["test"]):
        errors.append("test partition has no intervention cases")
    return {
        "identifier": P3_LEARNED_V2_GENERATOR_ID,
        "world_index": dataset.mechanism.world_index,
        "graph_variant": dataset.graph.identifier,
        "partition_counts": {name: len(cases) for name, cases in dataset.partitions.items()},
        "errors": errors,
        "passed": not errors,
    }
