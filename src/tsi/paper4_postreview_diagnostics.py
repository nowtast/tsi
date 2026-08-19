"""Post-review diagnostics for the attribution limits in Paper 4."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations, product

import numpy as np

from .paper3_replication_factorized import factorize
from .paper3_replication_family import (
    CARDINALITIES,
    COMBINATIONS,
    GRAPH_EDGES,
    GRAPH_NAMES,
    LAYER_NAMES,
    ReplicationCase,
    ReplicationDataset,
    build_replication_dataset,
    successor,
)
from .paper4_capacity_matched import fit_capacity_matched
from .paper4_comparative_validation import (
    ComparativeModel,
    fit_tsi_factorized,
    fit_wrong_routed_factorized,
)


MAGNITUDE_TWO_ACTIONS = tuple(
    tuple(2 if index == layer else 0 for index in range(5))
    for layer in range(5)
)
COMPOSITION_ACTIONS = tuple(
    tuple(1 if index in pair else 0 for index in range(5))
    for pair in combinations(range(5), 2)
)


def _factorized_with_declared_graph(
    dataset: ReplicationDataset, graph: str, name: str
) -> ComparativeModel:
    declared = replace(dataset, graph=graph)
    try:
        signature = factorize(declared)
    except ValueError:
        return ComparativeModel(name, lambda case: case.source)
    source_name, target_name = dict(GRAPH_EDGES)[graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)

    def predict(case: ReplicationCase) -> tuple[int, ...]:
        delta = [
            signature.multipliers[index] * case.action[index]
            for index in range(5)
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

    return ComparativeModel(name, predict)


def _graph_features(
    cases: tuple[ReplicationCase, ...], graph: str
) -> np.ndarray:
    source_name, target_name = dict(GRAPH_EDGES)[graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)
    return np.asarray(
        [
            [
                1.0,
                *case.action,
                case.action[source_index] * (1 + case.source[target_index]),
            ]
            for case in cases
        ],
        dtype=np.float64,
    )


def fit_graph_conditioned_dense(
    dataset: ReplicationDataset, graph: str, name: str
) -> ComparativeModel:
    """Fit an untied multi-output polynomial head given a declared graph edge."""

    cases = dataset.partitions["train"]
    x = _graph_features(cases, graph)
    y = np.asarray(
        [
            [
                (case.target[index] - case.source[index]) % CARDINALITIES[index]
                for index in range(5)
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    def predict(case: ReplicationCase) -> tuple[int, ...]:
        delta = np.rint(_graph_features((case,), graph) @ coefficients)[0].astype(int)
        return tuple(
            (case.source[index] + int(delta[index])) % CARDINALITIES[index]
            for index in range(5)
        )

    return ComparativeModel(name, predict)


def _cases(
    graph: str,
    combination_index: int,
    actions: tuple[tuple[int, ...], ...],
    *,
    misspecified: bool,
) -> tuple[ReplicationCase, ...]:
    source_name, target_name = dict(GRAPH_EDGES)[graph]
    source_index = LAYER_NAMES.index(source_name)
    target_index = LAYER_NAMES.index(target_name)
    cases = []
    for source in product(*(range(cardinality) for cardinality in CARDINALITIES)):
        for action in actions:
            target = list(successor(source, action, graph, combination_index))
            if misspecified:
                curvature = (
                    action[source_index]
                    * (action[source_index] - 1)
                    * (1 + source[target_index])
                )
                target[target_index] = (
                    target[target_index] + curvature
                ) % CARDINALITIES[target_index]
            cases.append(
                ReplicationCase(
                    "post_review_test",
                    graph,
                    combination_index,
                    source,
                    action,
                    tuple(target),
                    True,
                )
            )
    return tuple(cases)


def _accuracy(model: ComparativeModel, cases: tuple[ReplicationCase, ...]) -> float:
    return sum(model.predict(case) == case.target for case in cases) / len(cases)


def run_cell(graph: str, combination_index: int) -> dict[str, object]:
    dataset = build_replication_dataset(graph, combination_index)
    wrong_graph = GRAPH_NAMES[(GRAPH_NAMES.index(graph) + 1) % len(GRAPH_NAMES)]
    models = {
        "correct_graph_factorized": _factorized_with_declared_graph(
            dataset, graph, "correct_graph_factorized"
        ),
        "wrong_graph_factorized": fit_wrong_routed_factorized(dataset),
        "correct_graph_dense_head": fit_graph_conditioned_dense(
            dataset, graph, "correct_graph_dense_head"
        ),
        "wrong_graph_dense_head": fit_graph_conditioned_dense(
            dataset, wrong_graph, "wrong_graph_dense_head"
        ),
        "graph_unaware_dense": fit_capacity_matched(
            dataset, seed=0, include_interactions=True
        ),
        "graph_discovered_factorized": fit_tsi_factorized(dataset),
    }
    original_cases = tuple(
        case for case in dataset.partitions["test"] if case.intervention
    )
    composition_cases = _cases(
        graph, combination_index, COMPOSITION_ACTIONS, misspecified=False
    )
    misspecified_cases = _cases(
        graph, combination_index, MAGNITUDE_TWO_ACTIONS, misspecified=True
    )
    return {
        "graph": graph,
        "combination_index": combination_index,
        "case_counts": {
            "original_magnitude_two": len(original_cases),
            "two_coordinate_composition": len(composition_cases),
            "misspecified_curvature": len(misspecified_cases),
        },
        "panels": {
            "original_magnitude_two": {
                name: _accuracy(model, original_cases) for name, model in models.items()
            },
            "two_coordinate_composition": {
                name: _accuracy(model, composition_cases)
                for name, model in models.items()
            },
            "misspecified_curvature": {
                name: _accuracy(model, misspecified_cases)
                for name, model in models.items()
            },
        },
    }


def _panel_summary(
    rows: list[dict[str, object]], panel: str
) -> dict[str, object]:
    model_names = tuple(rows[0]["panels"][panel])
    summary = {}
    for model in model_names:
        values = [float(row["panels"][panel][model]) for row in rows]
        distribution = Counter(round(value, 12) for value in values)
        summary[model] = {
            "mean_cell_accuracy": float(np.mean(values)),
            "sd_cell_accuracy": float(np.std(values)),
            "exact_cell_count": sum(value == 1.0 for value in values),
            "min_cell_accuracy": min(values),
            "max_cell_accuracy": max(values),
            "cell_accuracy_distribution": {
                f"{key:.12g}": count for key, count in sorted(distribution.items())
            },
        }
    return summary


def run_full_diagnostic() -> dict[str, object]:
    rows = [
        run_cell(graph, combination_index)
        for graph in GRAPH_NAMES
        for combination_index in range(len(COMBINATIONS))
    ]
    panels = {
        panel: _panel_summary(rows, panel)
        for panel in (
            "original_magnitude_two",
            "two_coordinate_composition",
            "misspecified_curvature",
        )
    }
    original = panels["original_magnitude_two"]
    factorial = {
        "graph_effect_with_factorized_head": (
            original["correct_graph_factorized"]["mean_cell_accuracy"]
            - original["wrong_graph_factorized"]["mean_cell_accuracy"]
        ),
        "graph_effect_with_dense_head": (
            original["correct_graph_dense_head"]["mean_cell_accuracy"]
            - original["wrong_graph_dense_head"]["mean_cell_accuracy"]
        ),
        "head_effect_with_correct_graph": (
            original["correct_graph_factorized"]["mean_cell_accuracy"]
            - original["correct_graph_dense_head"]["mean_cell_accuracy"]
        ),
        "head_effect_with_wrong_graph": (
            original["wrong_graph_factorized"]["mean_cell_accuracy"]
            - original["wrong_graph_dense_head"]["mean_cell_accuracy"]
        ),
    }
    return {
        "status": "post_review_diagnostic_not_preregistered",
        "independent_unit": "graph_mechanism_cell",
        "cell_count": len(rows),
        "training_policy": "all models fit the original primitive-action training partition only",
        "panels": panels,
        "factorial_contrasts": factorial,
        "panel_definitions": {
            "original_magnitude_two": (
                "the original two single-coordinate magnitude-2 actions"
            ),
            "two_coordinate_composition": (
                "all ten binary actions with two distinct nonzero coordinates"
            ),
            "misspecified_curvature": (
                "five single-coordinate magnitude-2 actions with an unmodeled "
                "a_source(a_source-1)(1+x_target) transition term"
            ),
        },
        "interpretation_boundary": (
            "The original panel remains an exact-representability diagnostic. "
            "The factorial panel separates declared graph choice from two specified "
            "head families but does not establish universal causal superiority. "
            "The misspecified and composition panels are post-review diagnostics, "
            "not confirmatory replication."
        ),
        "rows": rows,
    }
