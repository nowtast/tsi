"""P3-5A-v3 generator with held-out mechanism combinations and five splits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping

from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_learned_v2_generator import (
    GRAPH_VARIANT_MANIFEST,
    V2GraphVariant,
    _successor,
)
from .paper3_learned_v3_contract import (
    GRAPH_VARIANTS,
    SPLITS,
    mechanism_combinations,
    mechanism_split_for_combination,
)
from .paper3_multiworld import (
    MECHANISM_PARAMETER_ACTIONS,
    PRIMITIVE_ACTIONS,
    MultiworldStateCode,
    StructuredAction,
    WorldMechanism,
    all_multiworld_state_codes,
)


P3_LEARNED_V3_GENERATOR_ID = "P3-5A-LEARNED-GRAPH-MECHANISM-HOLDOUT-GENERATOR-v1"


@dataclass(frozen=True)
class V3TransitionCase:
    partition: str
    graph_variant: str
    mechanism_combination_index: int
    source_code: MultiworldStateCode
    action: StructuredAction
    target_code: MultiworldStateCode
    intervention: bool

    def __post_init__(self) -> None:
        if self.partition not in SPLITS:
            raise ValueError("unknown v3 partition")
        if self.graph_variant not in GRAPH_VARIANTS:
            raise ValueError("unknown v3 graph variant")
        if self.partition != "test" and self.intervention:
            raise ValueError("intervention cases belong only to the test split")

    @property
    def input_key(self) -> tuple[MultiworldStateCode, tuple[int, ...]]:
        return self.source_code, self.action.components


@dataclass(frozen=True)
class V3WorldDataset:
    world_index: int
    graph: V2GraphVariant
    mechanism_combination_index: int
    mechanism: WorldMechanism
    partitions: Mapping[str, tuple[V3TransitionCase, ...]]
    digest: str

    def __post_init__(self) -> None:
        if tuple(self.partitions) != SPLITS:
            raise ValueError("v3 partitions must follow the frozen order")
        object.__setattr__(
            self,
            "partitions",
            MappingProxyType({name: tuple(cases) for name, cases in self.partitions.items()}),
        )


def graph_variant_for_v3_world(world_index: int, graph_index: int | None = None) -> V2GraphVariant:
    if type(world_index) is not int or world_index < 0:
        raise ValueError("world_index must be a nonnegative integer")
    index = world_index if graph_index is None else graph_index
    if type(index) is not int or not 0 <= index < len(GRAPH_VARIANT_MANIFEST):
        raise ValueError("graph index is out of range")
    return GRAPH_VARIANT_MANIFEST[index % len(GRAPH_VARIANT_MANIFEST)]


def _partition_for(source: MultiworldStateCode, action: StructuredAction, world_index: int) -> str:
    payload = f"{P3_LEARNED_V3_GENERATOR_ID}:{world_index}:{source.as_tuple()}:{action.components}"
    bucket = int.from_bytes(sha256(payload.encode()).digest()[:4], "little") % 100
    if bucket < 45:
        return "train"
    if bucket < 65:
        return "routing_selection"
    if bucket < 82:
        return "calibration"
    if bucket < 92:
        return "downstream_evaluation"
    return "test"


def _mechanism(combination_index: int) -> WorldMechanism:
    combinations = mechanism_combinations()
    if type(combination_index) is not int or not 0 <= combination_index < len(combinations):
        raise ValueError("mechanism combination index is out of range")
    multipliers, bridge, context = combinations[combination_index]
    digest = sha256(json.dumps(combinations[combination_index], separators=(",", ":")).encode()).hexdigest()
    return WorldMechanism(
        family=WorldFamily.CONTEXT_DEPENDENT,
        cohort=BenchmarkSplit.DEVELOPMENT,
        world_index=combination_index,
        layer_multipliers=multipliers,
        bridge_coefficient=bridge,
        context_coefficient=context,
        root_commitment=sha256(P3_LEARNED_V3_GENERATOR_ID.encode()).hexdigest(),
        mechanism_digest=digest,
    )


def build_v3_world_dataset(
    world_index: int,
    mechanism_combination_index: int,
    *,
    graph_index: int | None = None,
) -> V3WorldDataset:
    graph = graph_variant_for_v3_world(world_index, graph_index)
    mechanism = _mechanism(mechanism_combination_index)
    partitions: dict[str, list[V3TransitionCase]] = {name: [] for name in SPLITS}
    for source in all_multiworld_state_codes():
        for action in PRIMITIVE_ACTIONS:
            partition = _partition_for(source, action, world_index)
            partitions[partition].append(
                V3TransitionCase(
                    partition=partition,
                    graph_variant=graph.identifier,
                    mechanism_combination_index=mechanism_combination_index,
                    source_code=source,
                    action=action,
                    target_code=_successor(source, action, mechanism, graph),
                    intervention=False,
                )
            )
        for action in MECHANISM_PARAMETER_ACTIONS:
            partitions["test"].append(
                V3TransitionCase(
                    partition="test",
                    graph_variant=graph.identifier,
                    mechanism_combination_index=mechanism_combination_index,
                    source_code=source,
                    action=action,
                    target_code=_successor(source, action, mechanism, graph),
                    intervention=True,
                )
            )
    frozen = {name: tuple(cases) for name, cases in partitions.items()}
    payload = {
        "identifier": P3_LEARNED_V3_GENERATOR_ID,
        "world_index": world_index,
        "graph": graph.as_dict(),
        "mechanism_combination_index": mechanism_combination_index,
        "mechanism_split": mechanism_split_for_combination(mechanism_combination_index),
        "mechanism": mechanism.as_dict(),
        "partitions": {
            name: [
                {"source": list(case.source_code.as_tuple()), "action": list(case.action.components), "target": list(case.target_code.as_tuple()), "intervention": case.intervention}
                for case in cases
            ]
            for name, cases in frozen.items()
        },
    }
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return V3WorldDataset(world_index, graph, mechanism_combination_index, mechanism, frozen, digest)


def audit_v3_dataset(dataset: V3WorldDataset) -> dict[str, object]:
    errors: list[str] = []
    keys_by_split = {name: {case.input_key for case in cases} for name, cases in dataset.partitions.items()}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            if keys_by_split[left] & keys_by_split[right]:
                errors.append(f"input overlap: {left}/{right}")
    if not dataset.partitions["test"]:
        errors.append("test partition is empty")
    if not any(case.intervention for case in dataset.partitions["test"]):
        errors.append("test partition has no intervention cases")
    return {
        "identifier": P3_LEARNED_V3_GENERATOR_ID,
        "world_index": dataset.world_index,
        "graph_variant": dataset.graph.identifier,
        "mechanism_combination_index": dataset.mechanism_combination_index,
        "mechanism_split": mechanism_split_for_combination(dataset.mechanism_combination_index),
        "partition_counts": {name: len(cases) for name, cases in dataset.partitions.items()},
        "errors": errors,
        "passed": not errors,
    }
