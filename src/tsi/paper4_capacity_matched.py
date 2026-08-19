"""Capacity-matched trainable baselines for Paper 4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .paper3_replication_family import (
    CARDINALITIES,
    ReplicationCase,
    ReplicationDataset,
)
from .paper4_comparative_validation import evaluate_model, fit_tsi_factorized


BOOTSTRAP_SEEDS = (0, 1, 2, 3, 4)
FEATURE_WIDTH = 1 + 5 + 5 + 25


def _features(
    cases: tuple[ReplicationCase, ...], *, include_interactions: bool
) -> np.ndarray:
    rows = []
    for case in cases:
        source = np.asarray(case.source, dtype=np.float64) / np.asarray(
            CARDINALITIES, dtype=np.float64
        )
        action = np.asarray(case.action, dtype=np.float64) / 2.0
        values = [1.0, *source, *action]
        if include_interactions:
            values.extend(np.outer(source, action).reshape(-1))
        else:
            values.extend([0.0] * 25)
        rows.append(values)
    return np.asarray(rows)


def _targets(cases: tuple[ReplicationCase, ...]) -> np.ndarray:
    return np.asarray(
        [
            (case.target[index] - case.source[index]) % CARDINALITIES[index]
            for case in cases
            for index in range(5)
        ],
        dtype=np.float64,
    ).reshape(len(cases), 5)


def _seed(seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"TSI-PAPER4:{seed}".encode()).digest()[:8], "little"
    )


@dataclass(frozen=True)
class CapacityMatchedModel:
    name: str
    coefficients: np.ndarray
    include_interactions: bool

    def predict(self, case: ReplicationCase) -> tuple[int, ...]:
        features = _features((case,), include_interactions=self.include_interactions)
        delta = np.rint(features @ self.coefficients)[0].astype(int)
        return tuple(
            (case.source[index] + int(delta[index])) % CARDINALITIES[index]
            for index in range(5)
        )


def fit_capacity_matched(
    dataset: ReplicationDataset, *, seed: int, include_interactions: bool
) -> CapacityMatchedModel:
    cases = dataset.partitions["train"]
    rng = np.random.default_rng(_seed(seed))
    order = rng.permutation(len(cases))
    x = _features(
        tuple(cases[index] for index in order),
        include_interactions=include_interactions,
    )
    y = _targets(tuple(cases[index] for index in order))
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    name = (
        "dense_polynomial_trainable" if include_interactions else "diagonal_trainable"
    )
    return CapacityMatchedModel(name, coefficients, include_interactions)


def evaluate_capacity_model(
    model: CapacityMatchedModel, dataset: ReplicationDataset
) -> dict[str, object]:
    cases = dataset.partitions["test"]
    features = _features(cases, include_interactions=model.include_interactions)
    deltas = np.rint(features @ model.coefficients).astype(int)
    predictions = [
        tuple(
            (case.source[index] + int(deltas[row, index])) % CARDINALITIES[index]
            for index in range(5)
        )
        for row, case in enumerate(cases)
    ]
    correct = sum(
        prediction == case.target
        for prediction, case in zip(predictions, cases, strict=True)
    )
    intervention_indices = tuple(
        index for index, case in enumerate(cases) if case.intervention
    )
    intervention_correct = sum(
        predictions[index] == cases[index].target for index in intervention_indices
    )
    return {
        "model": model.name,
        "exact_accuracy": correct / len(cases),
        "intervention_exact_accuracy": intervention_correct / len(intervention_indices),
    }


def run_capacity_matched_cell(dataset: ReplicationDataset) -> list[dict[str, object]]:
    rows = []
    for seed in BOOTSTRAP_SEEDS:
        for include_interactions in (False, True):
            model = fit_capacity_matched(
                dataset, seed=seed, include_interactions=include_interactions
            )
            rows.append(evaluate_capacity_model(model, dataset) | {"seed": seed})
    rows.append(
        evaluate_model(fit_tsi_factorized(dataset), dataset)
        | {"seed": None, "model": "tsi_graph_discovered_factorized"}
    )
    return rows
