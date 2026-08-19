#!/usr/bin/env python3
"""Reproducible oracle and learned feasibility checks for Stage 2-I0."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from tsi.coherent import (
    CoherenceSignature,
    CoherentStructuralState,
    coherent_structural_discrepancy,
)
from tsi.dynamical import IntegratedStructuralState
from tsi.order_topology import FinitePreorder
from tsi.relational import (
    ArrowSpec,
    FiniteRelation,
    FiniteRelationAssignment,
    FiniteRelationalSchema,
)


SEED = 20_260_727
PAIR_INDEX = ((0, 1), (0, 2), (1, 2))
SCHEMA = FiniteRelationalSchema(
    objects=("entity",),
    arrows=(ArrowSpec("rel", "entity", "entity"),),
)


def make_state(
    size: int,
    *,
    spacing: float = 1.0,
    edge: bool = False,
    relation: bool = False,
    linear_order: bool = False,
    first_label: str = "same",
) -> CoherentStructuralState:
    entities = tuple(range(size))
    labels = (first_label,) + tuple("same" for _ in range(size - 1))
    relation_pairs = frozenset({(0, 1)}) if relation and size >= 2 else frozenset()
    relational = FiniteRelationAssignment(
        SCHEMA,
        {"entity": entities},
        {"entity": labels},
        {"rel": FiniteRelation(entities, entities, relation_pairs)},
    )
    tagged = tuple(("entity", entity) for entity in entities)
    simplices = {
        frozenset(),
        *(frozenset((vertex,)) for vertex in tagged),
    }
    if edge and size >= 2:
        simplices.add(frozenset((tagged[0], tagged[1])))
    distances = tuple(
        tuple(abs(i - j) * spacing for j in range(size))
        for i in range(size)
    )
    core = IntegratedStructuralState(
        relational,
        frozenset(simplices),
        distances,
    )
    order = FinitePreorder(
        tagged,
        frozenset(
            (left, right)
            for i, left in enumerate(tagged)
            for j, right in enumerate(tagged)
            if i == j or (linear_order and i <= j)
        ),
        core.tagged_labels,
    )
    return CoherentStructuralState(core, order, CoherenceSignature())


def oracle_audit() -> dict[str, int | float | bool]:
    states = (
        make_state(1),
        make_state(1, first_label="red"),
        make_state(2),
        make_state(2, spacing=2.0),
        make_state(2, edge=True, relation=True),
        make_state(2, linear_order=True),
    )
    distances = {
        (i, j): coherent_structural_discrepancy(left, right)
        for i, left in enumerate(states)
        for j, right in enumerate(states)
    }
    identity_checks = sum(
        abs(distances[(i, i)]) <= 1e-12 for i in range(len(states))
    )
    symmetry_checks = 0
    triangle_checks = 0
    maximum_triangle_slack = 0.0
    for i, j in product(range(len(states)), repeat=2):
        if abs(distances[(i, j)] - distances[(j, i)]) <= 1e-12:
            symmetry_checks += 1
    for i, j, k in product(range(len(states)), repeat=3):
        slack = distances[(i, k)] - distances[(i, j)] - distances[(j, k)]
        maximum_triangle_slack = max(maximum_triangle_slack, slack)
        if slack <= 1e-12:
            triangle_checks += 1
    unequal_distance = coherent_structural_discrepancy(states[0], states[2])
    return {
        "states": len(states),
        "identity_checks_passed": identity_checks,
        "symmetry_checks_passed": symmetry_checks,
        "triangle_checks_passed": triangle_checks,
        "maximum_triangle_violation": maximum_triangle_slack,
        "unequal_cardinality_distance": unequal_distance,
        "unequal_cardinality_is_finite": bool(np.isfinite(unequal_distance)),
    }


def sample_scenes(
    rng: np.random.Generator,
    count: int,
    observation_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = rng.uniform(0.0, 1.0, size=(count, 3))
    translation = rng.normal(0.0, 0.25, size=(count, 1))
    observations = positions + translation + rng.normal(
        0.0,
        observation_noise,
        size=positions.shape,
    )
    edges = np.stack(
        [
            np.abs(positions[:, left] - positions[:, right]) <= 0.25
            for left, right in PAIR_INDEX
        ],
        axis=1,
    ).astype(float)
    return positions, observations, edges


def features(observations: np.ndarray) -> np.ndarray:
    pair_distances = np.stack(
        [
            np.abs(observations[:, left] - observations[:, right])
            for left, right in PAIR_INDEX
        ],
        axis=1,
    )
    return np.concatenate(
        [np.ones((len(observations), 1)), observations, pair_distances],
        axis=1,
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float = 1e-3) -> np.ndarray:
    regularizer = penalty * np.eye(x.shape[1])
    regularizer[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 2_500,
    learning_rate: float = 0.15,
    penalty: float = 1e-3,
) -> np.ndarray:
    weights = np.zeros((x.shape[1], y.shape[1]))
    for _ in range(steps):
        logits = np.clip(x @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ (probabilities - y) / len(x) + penalty * weights
        gradient[0] -= penalty * weights[0]
        weights -= learning_rate * gradient
    return weights


def binary_f1(predicted: np.ndarray, target: np.ndarray) -> float:
    true_positive = np.sum((predicted == 1) & (target == 1))
    false_positive = np.sum((predicted == 1) & (target == 0))
    false_negative = np.sum((predicted == 0) & (target == 1))
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 1.0


def learned_pilot() -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    train_positions, train_observations, train_edges = sample_scenes(
        rng,
        4_000,
        observation_noise=0.03,
    )
    test_positions, test_observations, test_edges = sample_scenes(
        rng,
        1_000,
        observation_noise=0.08,
    )
    train_x = features(train_observations)
    test_x = features(test_observations)

    geometry_weights = fit_ridge(train_x, train_positions)
    predicted_positions = test_x @ geometry_weights
    predicted_distances = np.stack(
        [
            np.abs(predicted_positions[:, left] - predicted_positions[:, right])
            for left, right in PAIR_INDEX
        ],
        axis=1,
    )
    true_distances = np.stack(
        [
            np.abs(test_positions[:, left] - test_positions[:, right])
            for left, right in PAIR_INDEX
        ],
        axis=1,
    )
    structured_edges = (predicted_distances <= 0.25).astype(int)

    baseline_weights = fit_logistic(train_x, train_edges)
    baseline_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(test_x @ baseline_weights, -30.0, 30.0))
    )
    baseline_edges = (baseline_probabilities >= 0.5).astype(int)
    target_edges = test_edges.astype(int)

    structured_bridge_defect = np.mean(
        structured_edges != (predicted_distances <= 0.25)
    )
    baseline_bridge_defect = np.mean(baseline_edges != structured_edges)
    baseline_coherent_scene_rate = np.mean(
        np.all(baseline_edges == structured_edges, axis=1)
    )
    structured_coherent_scene_rate = np.mean(
        np.all(
            structured_edges == (predicted_distances <= 0.25),
            axis=1,
        )
    )
    return {
        "seed": SEED,
        "train_scenes": len(train_positions),
        "test_scenes": len(test_positions),
        "train_noise_std": 0.03,
        "test_noise_std": 0.08,
        "structured_relation_accuracy": float(
            np.mean(structured_edges == target_edges)
        ),
        "baseline_relation_accuracy": float(
            np.mean(baseline_edges == target_edges)
        ),
        "structured_relation_f1": binary_f1(structured_edges, target_edges),
        "baseline_relation_f1": binary_f1(baseline_edges, target_edges),
        "decoded_metric_mae": float(
            np.mean(np.abs(predicted_distances - true_distances))
        ),
        "structured_bridge_defect": float(structured_bridge_defect),
        "baseline_bridge_defect": float(baseline_bridge_defect),
        "structured_coherent_scene_rate": float(structured_coherent_scene_rate),
        "baseline_coherent_scene_rate": float(baseline_coherent_scene_rate),
    }


def main() -> None:
    result = {
        "status": "completed",
        "interpretation": (
            "Oracle checks falsify finite implementation errors. The learned "
            "pilot is a feasibility test and is not evidence of identifiability "
            "or a proof of the metric theorem."
        ),
        "oracle": oracle_audit(),
        "learned_pilot": learned_pilot(),
    }
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
