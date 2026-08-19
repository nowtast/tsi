"""Pre-registered variable-cardinality factorization benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product


CARDINALITY_PANELS = ((3, 3, 3, 3, 3), (4, 3, 5, 3, 4), (5, 4, 6, 4, 5))
LAYER_NAMES = ("signal", "topology", "metric", "relation", "order")
GRAPH_EDGES = (
    ("topology_to_metric", ("topology", "metric")),
    ("metric_to_relation", ("metric", "relation")),
    ("relation_to_order", ("relation", "order")),
)
GRAPH_NAMES = tuple(name for name, _ in GRAPH_EDGES)
COMBINATIONS = tuple(
    (multipliers, coefficient)
    for multipliers in product((1, 2), repeat=5)
    for coefficient in (1, 2)
)
PRIMITIVE_ACTIONS = tuple(
    tuple(1 if index == layer else 0 for index in range(5)) for layer in range(5)
)
INTERVENTION_ACTIONS = ((0, 2, 0, 0, 0), (0, 0, 2, 0, 0))


@dataclass(frozen=True)
class VariableCase:
    split: str
    panel_index: int
    cardinalities: tuple[int, ...]
    graph: str
    combination_index: int
    source: tuple[int, ...]
    action: tuple[int, ...]
    target: tuple[int, ...]
    intervention: bool


@dataclass(frozen=True)
class VariableDataset:
    panel_index: int
    cardinalities: tuple[int, ...]
    graph: str
    combination_index: int
    partitions: dict[str, tuple[VariableCase, ...]]


def successor(source, action, cardinalities, graph, combination_index):
    multipliers, coefficient = COMBINATIONS[combination_index]
    delta = [multipliers[index] * action[index] for index in range(5)]
    source_name, target_name = dict(GRAPH_EDGES)[graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)
    delta[target_index] += (
        coefficient * action[source_index] * (1 + source[target_index])
    )
    return tuple(
        (source[index] + delta[index]) % cardinalities[index] for index in range(5)
    )


def _split(source, action, panel_index, graph, combination_index):
    payload = f"TSI-P3-VARIABLE-CARDINALITY-v1:{panel_index}:{graph}:{combination_index}:{source}:{action}"
    return (
        "train"
        if int.from_bytes(sha256(payload.encode()).digest()[:4], "little") % 100 < 70
        else "test"
    )


def build_variable_dataset(panel_index, graph, combination_index):
    cardinalities = CARDINALITY_PANELS[panel_index]
    partitions = {"train": [], "test": []}
    for source in product(*(range(size) for size in cardinalities)):
        for action in PRIMITIVE_ACTIONS:
            split = _split(source, action, panel_index, graph, combination_index)
            partitions[split].append(
                VariableCase(
                    split,
                    panel_index,
                    cardinalities,
                    graph,
                    combination_index,
                    source,
                    action,
                    successor(source, action, cardinalities, graph, combination_index),
                    False,
                )
            )
        for action in INTERVENTION_ACTIONS:
            partitions["test"].append(
                VariableCase(
                    "test",
                    panel_index,
                    cardinalities,
                    graph,
                    combination_index,
                    source,
                    action,
                    successor(source, action, cardinalities, graph, combination_index),
                    True,
                )
            )
    return VariableDataset(
        panel_index,
        cardinalities,
        graph,
        combination_index,
        {name: tuple(cases) for name, cases in partitions.items()},
    )


def _delta(case, layer):
    return (case.target[layer] - case.source[layer]) % case.cardinalities[layer]


def _unique(candidates, equations):
    matches = tuple(
        candidate
        for candidate in candidates
        if all(
            (candidate * value) % modulus == observed
            for value, observed, modulus in equations
        )
    )
    if len(matches) != 1:
        raise ValueError(f"variable-cardinality parameter is not unique: {matches}")
    return matches[0]


def factorize(dataset):
    source_name, target_name = dict(GRAPH_EDGES)[dataset.graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)
    multipliers = []
    for layer, modulus in enumerate(dataset.cardinalities):
        equations = [
            (1, _delta(case, layer), modulus)
            for case in dataset.partitions["train"]
            if sum(case.action) == 1
            and case.action[layer] == 1
            and (layer != target_index or case.source[target_index] == 0)
        ]
        multipliers.append(_unique((1, 2), equations))
    equations = [
        (
            1 + case.source[target_index],
            _delta(case, target_index),
            dataset.cardinalities[target_index],
        )
        for case in dataset.partitions["train"]
        if sum(case.action) == 1 and case.action[source_index] == 1
    ]
    return tuple(multipliers), _unique((1, 2), equations)


def evaluate(dataset, signature=None):
    signature = signature or factorize(dataset)
    multipliers, coefficient = signature
    source_name, target_name = dict(GRAPH_EDGES)[dataset.graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)
    correct = 0
    for case in dataset.partitions["test"]:
        delta = [multipliers[index] * case.action[index] for index in range(5)]
        delta[target_index] += (
            coefficient * case.action[source_index] * (1 + case.source[target_index])
        )
        correct += (
            tuple(
                (case.source[index] + delta[index]) % dataset.cardinalities[index]
                for index in range(5)
            )
            == case.target
        )
    return correct / len(dataset.partitions["test"])
