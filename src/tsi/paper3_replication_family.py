"""Independent cross-family benchmark for the TSI structural factorization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product


CARDINALITIES = (4, 3, 5, 3, 4)
LAYER_NAMES = ("signal", "topology", "metric", "relation", "order")
GRAPH_EDGES = (
    ("topology_to_metric", ("topology", "metric")),
    ("metric_to_relation", ("metric", "relation")),
    ("relation_to_order", ("relation", "order")),
)
GRAPH_NAMES = tuple(name for name, _ in GRAPH_EDGES)
MULTIPLIER_OPTIONS = tuple((1, 2) for _ in CARDINALITIES)
COEFFICIENT_OPTIONS = (1, 2, 3)
COMBINATIONS = tuple(
    (multipliers, coefficient)
    for multipliers in product(*MULTIPLIER_OPTIONS)
    for coefficient in COEFFICIENT_OPTIONS
)
PRIMITIVE_ACTIONS = tuple(
    tuple(1 if index == layer else 0 for index in range(5)) for layer in range(5)
)
INTERVENTION_ACTIONS = (
    (0, 2, 0, 0, 0),
    (0, 0, 2, 0, 0),
)


@dataclass(frozen=True)
class ReplicationCase:
    split: str
    graph: str
    combination_index: int
    source: tuple[int, ...]
    action: tuple[int, ...]
    target: tuple[int, ...]
    intervention: bool


@dataclass(frozen=True)
class ReplicationDataset:
    graph: str
    combination_index: int
    partitions: dict[str, tuple[ReplicationCase, ...]]


def successor(
    source: tuple[int, ...], action: tuple[int, ...], graph: str, combination_index: int
) -> tuple[int, ...]:
    multipliers, coefficient = COMBINATIONS[combination_index]
    values = list(source)
    delta = [multipliers[index] * action[index] for index in range(5)]
    source_name, target_name = dict(GRAPH_EDGES)[graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)
    delta[target_index] += (
        coefficient * action[source_index] * (1 + source[target_index])
    )
    return tuple(
        (values[index] + delta[index]) % CARDINALITIES[index] for index in range(5)
    )


def _split(
    source: tuple[int, ...], action: tuple[int, ...], graph: str, combination_index: int
) -> str:
    payload = f"TSI-P3-5B-REPLICATION-v1:{graph}:{combination_index}:{source}:{action}"
    bucket = int.from_bytes(sha256(payload.encode()).digest()[:4], "little") % 100
    return "train" if bucket < 70 else "test"


def build_replication_dataset(graph: str, combination_index: int) -> ReplicationDataset:
    if graph not in GRAPH_NAMES:
        raise ValueError("unknown replication graph")
    if not 0 <= combination_index < len(COMBINATIONS):
        raise ValueError("unknown replication combination")
    partitions = {"train": [], "test": []}
    for source in product(*(range(cardinality) for cardinality in CARDINALITIES)):
        for action in PRIMITIVE_ACTIONS:
            split = _split(source, action, graph, combination_index)
            partitions[split].append(
                ReplicationCase(
                    split,
                    graph,
                    combination_index,
                    source,
                    action,
                    successor(source, action, graph, combination_index),
                    False,
                )
            )
        for action in INTERVENTION_ACTIONS:
            partitions["test"].append(
                ReplicationCase(
                    "test",
                    graph,
                    combination_index,
                    source,
                    action,
                    successor(source, action, graph, combination_index),
                    True,
                )
            )
    return ReplicationDataset(
        graph,
        combination_index,
        {name: tuple(cases) for name, cases in partitions.items()},
    )
