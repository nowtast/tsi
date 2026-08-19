"""Development-only fitting and frozen application of P3-4B predictors."""

from __future__ import annotations

from hashlib import sha256
import json
from math import log
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .paper3_independence_contract import MODEL_CONTROLS
from .paper3_multiworld import LAYER_ORDER
from .paper3_validity_contract import (
    LOGISTIC_MAX_ITERATIONS,
    LOGISTIC_TOLERANCE,
    OPTIMIZER_SEEDS,
    PRIMARY_PREDICTIVE_MODELS,
    RIDGE_PENALTY,
    TASKS_PER_UNIT,
    UNITS_PER_WORLD,
    validity_contract_digest,
)
from .paper3_validity_experiment import P3_VALIDITY_DEVELOPMENT_ID


P3_VALIDITY_PREDICTOR_ID = "P3-4B-FROZEN-PREDICTORS-v2"
MODEL_IDS = tuple(model.identifier for model in MODEL_CONTROLS)
REFERENCE_MODEL = MODEL_IDS[0]
FIXED_LAYER_ORDER = ("label", "simplicial", "metric", "relation", "order")

SCALAR_GENERIC_FEATURE_NAMES = (
    "training_final_nll",
    "probe_teacher_one_step_mse",
    "probe_open_loop_latent_mse",
    "probe_terminal_exactness",
    "oracle_mean_reward_gap",
    "mean_plan_horizon",
    "stratum_unseen_structural_mode",
    *(f"task_goal_weight_{layer}" for layer in LAYER_ORDER),
    *(f"model_is_{model}" for model in MODEL_IDS[1:]),
    "local_candidate_latent_mse_mean",
    "local_candidate_latent_mse_max",
    "local_candidate_latent_mse_selected_minus_alternative",
    "local_candidate_endpoint_exact_rate",
    "local_candidate_endpoint_exact_selected_minus_alternative",
    "local_predicted_utility_gap",
    "local_oracle_utility_gap",
    "local_normalized_plan_horizon",
    *(f"local_goal_weight_{layer}" for layer in LAYER_ORDER),
)
LAYER_AWARE_ADDITIONAL_FEATURE_NAMES = (
    "probe_generic_task_aligned_mismatch",
    *(f"probe_raw_layer_mismatch_{layer}" for layer in LAYER_ORDER),
    *(f"local_candidate_raw_mismatch_{layer}" for layer in LAYER_ORDER),
    *(
        f"local_candidate_raw_mismatch_selected_minus_alternative_{layer}"
        for layer in LAYER_ORDER
    ),
    "local_goal_aligned_raw_mismatch",
    "local_goal_aligned_raw_selected_minus_alternative",
)
LAYER_AWARE_FEATURE_NAMES = (
    *SCALAR_GENERIC_FEATURE_NAMES,
    *LAYER_AWARE_ADDITIONAL_FEATURE_NAMES,
)
TSI_ADDITIONAL_FEATURE_NAMES = (
    "local_candidate_i0_mean",
    "local_candidate_i0_max",
    "local_candidate_i0_selected_minus_alternative",
    "local_candidate_fixed_total_mean",
    "local_candidate_fixed_total_max",
    "local_candidate_fixed_total_selected_minus_alternative",
    *(f"local_candidate_fixed_{layer}" for layer in FIXED_LAYER_ORDER),
    *(
        f"local_candidate_fixed_selected_minus_alternative_{layer}"
        for layer in FIXED_LAYER_ORDER
    ),
    "local_goal_aligned_fixed_discrepancy",
    "local_goal_aligned_fixed_selected_minus_alternative",
)
TSI_FEATURE_NAMES = (*SCALAR_GENERIC_FEATURE_NAMES, *TSI_ADDITIONAL_FEATURE_NAMES)
LAYER_AWARE_TSI_FEATURE_NAMES = (
    *LAYER_AWARE_FEATURE_NAMES,
    *TSI_ADDITIONAL_FEATURE_NAMES,
)
HAZARD_TIME_FEATURE_NAMES = tuple(
    f"failure_round_{time}" for time in range(2, TASKS_PER_UNIT + 1)
)


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _finite_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _completed_runs(
    result: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    runs = result.get("runs")
    if not isinstance(runs, list):
        raise ValueError("validity result runs must be a list")
    return tuple(
        run
        for run in runs
        if isinstance(run, dict) and run.get("status") == "completed"
    )


def flatten_validity_records(
    result: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, str, int, int]] = set()
    for run in _completed_runs(result):
        world = run.get("world_index")
        model = run.get("model")
        seed = run.get("optimizer_seed")
        training = run.get("training")
        unit_records = run.get("unit_records")
        if type(world) is not int or model not in MODEL_IDS or type(seed) is not int:
            raise ValueError("validity run identity is malformed")
        if not isinstance(training, dict) or not isinstance(unit_records, list):
            raise ValueError("validity run payload is malformed")
        final_nll = _finite_float(training.get("final_nll"), "training_final_nll")
        if len(unit_records) != UNITS_PER_WORLD:
            raise ValueError("validity run unit count changed")
        for record in unit_records:
            if not isinstance(record, dict):
                raise ValueError("validity unit record must be a mapping")
            unit = record.get("unit_index")
            generic = record.get("generic_predictors")
            outcomes = record.get("outcomes")
            task_records = record.get("task_records")
            if (
                type(unit) is not int
                or not isinstance(generic, dict)
                or not isinstance(outcomes, dict)
                or not isinstance(task_records, list)
                or len(task_records) != TASKS_PER_UNIT
            ):
                raise ValueError("validity unit record is malformed")
            key = (world, str(model), seed, unit)
            if key in seen:
                raise ValueError("duplicate validity unit record")
            seen.add(key)
            probe_mismatch = generic.get("probe_open_loop_layer_mismatch_rate")
            aggregate_weights = generic.get("task_goal_layer_weights")
            if not isinstance(probe_mismatch, dict) or not isinstance(
                aggregate_weights, dict
            ):
                raise ValueError("validity probe layer predictor is malformed")
            scalar_base = [
                final_nll,
                _finite_float(
                    generic.get("probe_teacher_one_step_mse"),
                    "probe_teacher_one_step_mse",
                ),
                _finite_float(
                    generic.get("probe_open_loop_latent_mse"),
                    "probe_open_loop_latent_mse",
                ),
                _finite_float(
                    generic.get("probe_terminal_exactness"),
                    "probe_terminal_exactness",
                ),
                _finite_float(
                    generic.get("oracle_mean_reward_gap"),
                    "oracle_mean_reward_gap",
                ),
                _finite_float(
                    generic.get("mean_plan_horizon"),
                    "mean_plan_horizon",
                ),
                float(record.get("stratum") == "unseen_structural_mode"),
                *[
                    _finite_float(
                        aggregate_weights.get(layer),
                        f"task_goal_weight_{layer}",
                    )
                    for layer in LAYER_ORDER
                ],
                *[float(model == candidate) for candidate in MODEL_IDS[1:]],
            ]
            probe_layer_aware = [
                _finite_float(
                    generic.get("generic_task_aligned_mismatch"),
                    "probe_generic_task_aligned_mismatch",
                ),
                *[
                    _finite_float(
                        probe_mismatch.get(layer),
                        f"probe_raw_layer_mismatch_{layer}",
                    )
                    for layer in LAYER_ORDER
                ],
            ]
            scalar_task_features: list[list[float]] = []
            tsi_task_features: list[list[float]] = []
            layer_task_features: list[list[float]] = []
            layer_tsi_task_features: list[list[float]] = []
            observed_failures: list[int] = []
            for task in task_records:
                if not isinstance(task, dict):
                    raise ValueError("validity task record must be a mapping")
                task_generic = task.get("generic_task_predictors")
                task_tsi = task.get("tsi_task_predictors")
                if not isinstance(task_generic, dict) or not isinstance(task_tsi, dict):
                    raise ValueError("validity task predictors are malformed")
                local_weights = task_generic.get("goal_layer_weights")
                local_mismatch = task_generic.get("candidate_layer_mismatch_rate")
                local_mismatch_contrast = task_generic.get(
                    "candidate_layer_mismatch_selected_minus_alternative"
                )
                fixed_layers = task_tsi.get("candidate_fixed_layer_mean")
                fixed_layer_contrast = task_tsi.get(
                    "candidate_fixed_layer_selected_minus_alternative"
                )
                if not all(
                    isinstance(value, dict)
                    for value in (
                        local_weights,
                        local_mismatch,
                        local_mismatch_contrast,
                        fixed_layers,
                        fixed_layer_contrast,
                    )
                ):
                    raise ValueError("validity task layer vectors are malformed")
                local_scalar = [
                    _finite_float(
                        task_generic.get("candidate_latent_mse_mean"),
                        "local_candidate_latent_mse_mean",
                    ),
                    _finite_float(
                        task_generic.get("candidate_latent_mse_max"),
                        "local_candidate_latent_mse_max",
                    ),
                    _finite_float(
                        task_generic.get(
                            "candidate_latent_mse_selected_minus_alternative"
                        ),
                        (
                            "local_candidate_latent_mse_"
                            "selected_minus_alternative"
                        ),
                    ),
                    _finite_float(
                        task_generic.get("candidate_endpoint_exact_rate"),
                        "local_candidate_endpoint_exact_rate",
                    ),
                    _finite_float(
                        task_generic.get(
                            "candidate_endpoint_exact_selected_minus_alternative"
                        ),
                        (
                            "local_candidate_endpoint_exact_"
                            "selected_minus_alternative"
                        ),
                    ),
                    _finite_float(
                        task_generic.get("predicted_utility_gap"),
                        "local_predicted_utility_gap",
                    ),
                    _finite_float(
                        task_generic.get("oracle_utility_gap"),
                        "local_oracle_utility_gap",
                    ),
                    _finite_float(
                        task_generic.get("normalized_plan_horizon"),
                        "local_normalized_plan_horizon",
                    ),
                    *[
                        _finite_float(
                            local_weights.get(layer),
                            f"local_goal_weight_{layer}",
                        )
                        for layer in LAYER_ORDER
                    ],
                ]
                local_layer_aware = [
                    *[
                        _finite_float(
                            local_mismatch.get(layer),
                            f"local_candidate_raw_mismatch_{layer}",
                        )
                        for layer in LAYER_ORDER
                    ],
                    *[
                        _finite_float(
                            local_mismatch_contrast.get(layer),
                            (
                                "local_candidate_raw_mismatch_"
                                f"selected_minus_alternative_{layer}"
                            ),
                        )
                        for layer in LAYER_ORDER
                    ],
                    _finite_float(
                        task_generic.get("goal_aligned_raw_mismatch"),
                        "local_goal_aligned_raw_mismatch",
                    ),
                    _finite_float(
                        task_generic.get(
                            "goal_aligned_raw_selected_minus_alternative"
                        ),
                        (
                            "local_goal_aligned_raw_"
                            "selected_minus_alternative"
                        ),
                    ),
                ]
                local_tsi = [
                    _finite_float(
                        task_tsi.get("candidate_i0_mean"),
                        "local_candidate_i0_mean",
                    ),
                    _finite_float(
                        task_tsi.get("candidate_i0_max"),
                        "local_candidate_i0_max",
                    ),
                    _finite_float(
                        task_tsi.get(
                            "candidate_i0_selected_minus_alternative"
                        ),
                        "local_candidate_i0_selected_minus_alternative",
                    ),
                    _finite_float(
                        task_tsi.get("candidate_fixed_total_mean"),
                        "local_candidate_fixed_total_mean",
                    ),
                    _finite_float(
                        task_tsi.get("candidate_fixed_total_max"),
                        "local_candidate_fixed_total_max",
                    ),
                    _finite_float(
                        task_tsi.get(
                            "candidate_fixed_total_selected_minus_alternative"
                        ),
                        (
                            "local_candidate_fixed_total_"
                            "selected_minus_alternative"
                        ),
                    ),
                    *[
                        _finite_float(
                            fixed_layers.get(layer),
                            f"local_candidate_fixed_{layer}",
                        )
                        for layer in FIXED_LAYER_ORDER
                    ],
                    *[
                        _finite_float(
                            fixed_layer_contrast.get(layer),
                            (
                                "local_candidate_fixed_"
                                f"selected_minus_alternative_{layer}"
                            ),
                        )
                        for layer in FIXED_LAYER_ORDER
                    ],
                    _finite_float(
                        task_tsi.get("goal_aligned_fixed_discrepancy"),
                        "local_goal_aligned_fixed_discrepancy",
                    ),
                    _finite_float(
                        task_tsi.get(
                            "goal_aligned_fixed_selected_minus_alternative"
                        ),
                        (
                            "local_goal_aligned_fixed_"
                            "selected_minus_alternative"
                        ),
                    ),
                ]
                scalar = [*scalar_base, *local_scalar]
                layer_aware = [*scalar, *probe_layer_aware, *local_layer_aware]
                scalar_task_features.append(scalar)
                tsi_task_features.append([*scalar, *local_tsi])
                layer_task_features.append(layer_aware)
                layer_tsi_task_features.append([*layer_aware, *local_tsi])
                failure = task.get("task_failure")
                if failure not in (0, 1):
                    raise ValueError("task_failure must be binary")
                observed_failures.append(int(failure))
            any_failure = outcomes.get("any_task_failure")
            first_failure = outcomes.get("first_failure_time")
            if any_failure not in (0, 1):
                raise ValueError("any_task_failure must be binary")
            if (
                type(first_failure) is not int
                or not 1 <= first_failure <= TASKS_PER_UNIT + 1
            ):
                raise ValueError("first_failure_time is outside follow-up")
            if int(any(observed_failures)) != any_failure:
                raise ValueError("unit and task failure labels disagree")
            rows.append(
                {
                    "world_index": world,
                    "model": model,
                    "optimizer_seed": seed,
                    "unit_index": unit,
                    "is_primary_predictive_model": model in PRIMARY_PREDICTIVE_MODELS,
                    "scalar_task_features": scalar_task_features,
                    "tsi_task_features": tsi_task_features,
                    "layer_task_features": layer_task_features,
                    "layer_tsi_task_features": layer_tsi_task_features,
                    "task_failures": observed_failures,
                    "any_task_failure": any_failure,
                    "first_failure_time": first_failure,
                }
            )
    return rows


def _matrix(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> np.ndarray:
    matrix = np.asarray([row[field] for row in rows], dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field} does not form a finite matrix")
    return matrix


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _ridge_loss(
    design: np.ndarray,
    targets: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    probabilities = np.clip(
        _sigmoid(design @ coefficients),
        1.0e-12,
        1.0 - 1.0e-12,
    )
    data = -np.mean(
        targets * np.log(probabilities)
        + (1.0 - targets) * np.log(1.0 - probabilities)
    )
    return float(data + 0.5 * RIDGE_PENALTY * np.sum(coefficients[1:] ** 2))


def fit_ridge_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, object]:
    values = np.asarray(features, dtype=np.float64)
    outcomes = np.asarray(targets, dtype=np.float64)
    names = tuple(feature_names)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("logistic feature shape does not match names")
    if outcomes.shape != (len(values),) or not np.all(
        np.logical_or(outcomes == 0.0, outcomes == 1.0)
    ):
        raise ValueError("logistic targets must be binary")
    if len(values) < 2 or len(np.unique(outcomes)) != 2:
        raise ValueError("logistic fitting requires events and non-events")
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0)
    scales[scales < 1.0e-12] = 1.0
    standardized = (values - means) / scales
    design = np.column_stack((np.ones(len(values)), standardized))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    event_rate = float(np.mean(outcomes))
    coefficients[0] = log(event_rate / (1.0 - event_rate))
    penalty = np.zeros(design.shape[1], dtype=np.float64)
    penalty[1:] = RIDGE_PENALTY
    converged = False
    iteration = 0
    for iteration in range(1, LOGISTIC_MAX_ITERATIONS + 1):
        probabilities = _sigmoid(design @ coefficients)
        gradient = design.T @ (probabilities - outcomes) / len(values)
        gradient += penalty * coefficients
        weights = np.maximum(
            probabilities * (1.0 - probabilities),
            1.0e-8,
        )
        hessian = (
            design.T @ (design * weights[:, np.newaxis]) / len(values)
            + np.diag(penalty)
        )
        hessian += np.eye(len(coefficients)) * 1.0e-12
        step = np.linalg.solve(hessian, gradient)
        current_loss = _ridge_loss(design, outcomes, coefficients)
        scale = 1.0
        candidate = coefficients - step
        while (
            _ridge_loss(design, outcomes, candidate) > current_loss
            and scale > 1.0e-8
        ):
            scale *= 0.5
            candidate = coefficients - scale * step
        maximum_change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        if maximum_change < LOGISTIC_TOLERANCE:
            converged = True
            break
    payload: dict[str, object] = {
        "model": "ridge_logistic",
        "feature_names": list(names),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_penalty": RIDGE_PENALTY,
        "iterations": iteration,
        "converged": converged,
        "training_rows": len(values),
        "training_event_rate": event_rate,
        "penalized_training_loss": _ridge_loss(design, outcomes, coefficients),
    }
    return {**payload, "model_digest": _canonical_digest(payload)}


def predict_ridge_logistic(
    model: Mapping[str, object],
    features: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    names = tuple(feature_names)
    if model.get("model") != "ridge_logistic":
        raise ValueError("unexpected frozen predictor model")
    if model.get("feature_names") != list(names):
        raise ValueError("frozen predictor feature order changed")
    payload = {key: value for key, value in model.items() if key != "model_digest"}
    if model.get("model_digest") != _canonical_digest(payload):
        raise ValueError("frozen predictor model digest mismatch")
    values = np.asarray(features, dtype=np.float64)
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("prediction feature shape changed")
    standardized = (values - means) / scales
    design = np.column_stack((np.ones(len(values)), standardized))
    return _sigmoid(design @ coefficients)


def _hazard_training_data(
    rows: Sequence[Mapping[str, object]],
    field: str,
    base_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    features: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        time_features = row.get(field)
        task_failures = row.get("task_failures")
        if not isinstance(time_features, list) or len(time_features) != TASKS_PER_UNIT:
            raise ValueError(f"{field} does not contain the frozen task panel")
        if not isinstance(task_failures, list) or len(task_failures) != TASKS_PER_UNIT:
            raise ValueError("task failure panel changed")
        last = min(int(row["first_failure_time"]), TASKS_PER_UNIT)
        for time in range(1, last + 1):
            values = list(time_features[time - 1])
            if len(values) != len(base_names):
                raise ValueError(f"{field} feature width changed")
            indicators = [
                float(time == candidate)
                for candidate in range(2, TASKS_PER_UNIT + 1)
            ]
            features.append([*values, *indicators])
            targets.append(float(task_failures[time - 1]))
    names = (*tuple(base_names), *HAZARD_TIME_FEATURE_NAMES)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        names,
    )


def _fit_model_set(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows or any(
        row.get("model") not in PRIMARY_PREDICTIVE_MODELS for row in rows
    ):
        raise ValueError("hazard fitting requires the primary predictive population")
    scalar, scalar_targets, scalar_names = _hazard_training_data(
        rows,
        "scalar_task_features",
        SCALAR_GENERIC_FEATURE_NAMES,
    )
    tsi, tsi_targets, tsi_names = _hazard_training_data(
        rows,
        "tsi_task_features",
        TSI_FEATURE_NAMES,
    )
    layer, layer_targets, layer_names = _hazard_training_data(
        rows,
        "layer_task_features",
        LAYER_AWARE_FEATURE_NAMES,
    )
    layer_tsi, layer_tsi_targets, layer_tsi_names = _hazard_training_data(
        rows,
        "layer_tsi_task_features",
        LAYER_AWARE_TSI_FEATURE_NAMES,
    )
    if not (
        np.array_equal(scalar_targets, tsi_targets)
        and np.array_equal(scalar_targets, layer_targets)
        and np.array_equal(scalar_targets, layer_tsi_targets)
    ):
        raise RuntimeError("P3-4B hazard outcomes diverged across feature sets")
    return {
        "hazard_baseline": fit_ridge_logistic(
            scalar,
            scalar_targets,
            scalar_names,
        ),
        "hazard_tsi": fit_ridge_logistic(
            tsi,
            tsi_targets,
            tsi_names,
        ),
        "hazard_layer_aware_baseline": fit_ridge_logistic(
            layer,
            layer_targets,
            layer_names,
        ),
        "hazard_layer_aware_tsi": fit_ridge_logistic(
            layer_tsi,
            layer_tsi_targets,
            layer_tsi_names,
        ),
    }


def _hazard_probabilities(
    model: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    field: str,
    base_names: Sequence[str],
) -> np.ndarray:
    expanded: list[list[float]] = []
    for row in rows:
        time_features = row.get(field)
        if not isinstance(time_features, list) or len(time_features) != TASKS_PER_UNIT:
            raise ValueError(f"{field} does not contain the frozen task panel")
        for time in range(1, TASKS_PER_UNIT + 1):
            values = list(time_features[time - 1])
            if len(values) != len(base_names):
                raise ValueError(f"{field} feature width changed")
            indicators = [
                float(time == candidate)
                for candidate in range(2, TASKS_PER_UNIT + 1)
            ]
            expanded.append([*values, *indicators])
    names = (*tuple(base_names), *HAZARD_TIME_FEATURE_NAMES)
    hazards = predict_ridge_logistic(
        model,
        np.asarray(expanded, dtype=np.float64),
        names,
    ).reshape(len(rows), TASKS_PER_UNIT)
    return hazards


def score_frozen_predictors(
    rows: Sequence[Mapping[str, object]],
    models: Mapping[str, object],
) -> list[dict[str, object]]:
    baseline_hazards = _hazard_probabilities(
        models["hazard_baseline"],
        rows,
        "scalar_task_features",
        SCALAR_GENERIC_FEATURE_NAMES,
    )
    tsi_hazards = _hazard_probabilities(
        models["hazard_tsi"],
        rows,
        "tsi_task_features",
        TSI_FEATURE_NAMES,
    )
    layer_hazards = _hazard_probabilities(
        models["hazard_layer_aware_baseline"],
        rows,
        "layer_task_features",
        LAYER_AWARE_FEATURE_NAMES,
    )
    layer_tsi_hazards = _hazard_probabilities(
        models["hazard_layer_aware_tsi"],
        rows,
        "layer_tsi_task_features",
        LAYER_AWARE_TSI_FEATURE_NAMES,
    )
    baseline_failure = 1.0 - np.cumprod(1.0 - baseline_hazards, axis=1)
    tsi_failure = 1.0 - np.cumprod(1.0 - tsi_hazards, axis=1)
    layer_failure = 1.0 - np.cumprod(1.0 - layer_hazards, axis=1)
    layer_tsi_failure = 1.0 - np.cumprod(1.0 - layer_tsi_hazards, axis=1)
    scored: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        first_failure = int(row["first_failure_time"])
        event_by_time = np.asarray(
            [
                float(first_failure <= time)
                for time in range(1, TASKS_PER_UNIT + 1)
            ],
            dtype=np.float64,
        )
        baseline_brier = (event_by_time - baseline_failure[index]) ** 2
        tsi_brier = (event_by_time - tsi_failure[index]) ** 2
        layer_brier = (event_by_time - layer_failure[index]) ** 2
        layer_tsi_brier = (event_by_time - layer_tsi_failure[index]) ** 2
        scored.append(
            {
                "world_index": int(row["world_index"]),
                "model": str(row["model"]),
                "optimizer_seed": int(row["optimizer_seed"]),
                "unit_index": int(row["unit_index"]),
                "is_primary_predictive_model": bool(
                    row["is_primary_predictive_model"]
                ),
                "any_task_failure": int(row["any_task_failure"]),
                "first_failure_time": first_failure,
                "baseline_cumulative_failure_probability": (
                    baseline_failure[index].tolist()
                ),
                "tsi_cumulative_failure_probability": (
                    tsi_failure[index].tolist()
                ),
                "binary_baseline_brier": float(baseline_brier[-1]),
                "binary_tsi_brier": float(tsi_brier[-1]),
                "binary_brier_improvement": float(
                    baseline_brier[-1] - tsi_brier[-1]
                ),
                "first_failure_baseline_integrated_brier": float(
                    np.mean(baseline_brier)
                ),
                "first_failure_tsi_integrated_brier": float(
                    np.mean(tsi_brier)
                ),
                "first_failure_integrated_brier_improvement": float(
                    np.mean(baseline_brier - tsi_brier)
                ),
                "layer_aware_binary_brier_improvement": float(
                    layer_brier[-1] - layer_tsi_brier[-1]
                ),
                "layer_aware_integrated_brier_improvement": float(
                    np.mean(layer_brier - layer_tsi_brier)
                ),
            }
        )
    return scored


def _world_seed_effects(
    scored: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in scored:
        if row.get("model") not in PRIMARY_PREDICTIVE_MODELS:
            continue
        key = int(row["world_index"]), int(row["optimizer_seed"])
        grouped.setdefault(key, []).append(row)
    return [
        {
            "world_index": world,
            "optimizer_seed": seed,
            "record_count": len(rows),
            "binary_brier_improvement": float(
                np.mean([row["binary_brier_improvement"] for row in rows])
            ),
            "first_failure_integrated_brier_improvement": float(
                np.mean(
                    [
                        row["first_failure_integrated_brier_improvement"]
                        for row in rows
                    ]
                )
            ),
            "layer_aware_binary_brier_improvement": float(
                np.mean(
                    [row["layer_aware_binary_brier_improvement"] for row in rows]
                )
            ),
            "layer_aware_integrated_brier_improvement": float(
                np.mean(
                    [
                        row["layer_aware_integrated_brier_improvement"]
                        for row in rows
                    ]
                )
            ),
        }
        for (world, seed), rows in sorted(grouped.items())
    ]


def fit_frozen_validity_predictors(
    development: Mapping[str, object],
    *,
    perform_lowo: bool = True,
) -> dict[str, object]:
    if development.get("identifier") != P3_VALIDITY_DEVELOPMENT_ID:
        raise ValueError("unexpected P3-4B development result")
    if development.get("test_output_used") is not False:
        raise ValueError("predictors cannot be fitted with test output")
    rows = flatten_validity_records(development)
    expected = (
        int(development["world_count"])
        * len(OPTIMIZER_SEEDS)
        * len(MODEL_IDS)
        * UNITS_PER_WORLD
    )
    if len(rows) != expected:
        raise ValueError("development validity record count is incomplete")
    primary_rows = [row for row in rows if row["is_primary_predictive_model"]]
    expected_primary = (
        int(development["world_count"])
        * len(OPTIMIZER_SEEDS)
        * len(PRIMARY_PREDICTIVE_MODELS)
        * UNITS_PER_WORLD
    )
    if len(primary_rows) != expected_primary:
        raise ValueError("primary predictive population is incomplete")

    cross_validated: list[dict[str, object]] = []
    if perform_lowo:
        worlds = sorted({int(row["world_index"]) for row in primary_rows})
        for held_out in worlds:
            training = [
                row for row in primary_rows if row["world_index"] != held_out
            ]
            testing = [
                row for row in primary_rows if row["world_index"] == held_out
            ]
            fold_models = _fit_model_set(training)
            cross_validated.extend(score_frozen_predictors(testing, fold_models))
    final_models = _fit_model_set(primary_rows)
    frozen_payload = {
        "identifier": P3_VALIDITY_PREDICTOR_ID,
        "contract_digest": validity_contract_digest(),
        "development_digest": development.get("report_digest"),
        "primary_predictive_models": list(PRIMARY_PREDICTIVE_MODELS),
        "scalar_generic_feature_names": list(SCALAR_GENERIC_FEATURE_NAMES),
        "layer_aware_additional_feature_names": list(
            LAYER_AWARE_ADDITIONAL_FEATURE_NAMES
        ),
        "tsi_additional_feature_names": list(TSI_ADDITIONAL_FEATURE_NAMES),
        "hazard_time_feature_names": list(HAZARD_TIME_FEATURE_NAMES),
        "models": final_models,
        "sealed_refitting_permitted": False,
    }
    frozen = {
        **frozen_payload,
        "frozen_predictor_digest": _canonical_digest(frozen_payload),
    }
    event_rate = float(
        np.mean([float(row["any_task_failure"]) for row in primary_rows])
    )
    all_event_rate = float(
        np.mean([float(row["any_task_failure"]) for row in rows])
    )
    exact_rows = [row for row in rows if not row["is_primary_predictive_model"]]
    exact_event_rate = float(
        np.mean([float(row["any_task_failure"]) for row in exact_rows])
    )
    cv_effects = _world_seed_effects(cross_validated)
    cv_means = {
        name: float(np.mean([row[name] for row in cv_effects]))
        for name in (
            "binary_brier_improvement",
            "first_failure_integrated_brier_improvement",
            "layer_aware_binary_brier_improvement",
            "layer_aware_integrated_brier_improvement",
        )
    }
    converged = all(model["converged"] for model in final_models.values())
    payload: dict[str, object] = {
        "identifier": P3_VALIDITY_PREDICTOR_ID,
        "test_output_used": False,
        "development_worlds": int(development["world_count"]),
        "development_records_all_controls": len(rows),
        "development_records_primary_population": len(primary_rows),
        "development_event_rate_primary": event_rate,
        "development_event_rate_all_controls": all_event_rate,
        "development_event_rate_exact_controls": exact_event_rate,
        "development_first_failure_distribution_primary": {
            str(time): sum(
                int(row["first_failure_time"] == time) for row in primary_rows
            )
            for time in range(1, TASKS_PER_UNIT + 2)
        },
        "all_final_models_converged": converged,
        "development_lowo_performed": perform_lowo,
        "development_lowo_mean_effects": cv_means,
        "development_lowo_world_seed_effects": cv_effects,
        "layer_aware_sensitivity_is_non_gating": True,
        "frozen_predictors": frozen,
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def validate_frozen_predictors(
    report: Mapping[str, object],
) -> Mapping[str, object]:
    if report.get("identifier") != P3_VALIDITY_PREDICTOR_ID:
        raise ValueError("unexpected frozen predictor report")
    report_payload = {
        key: value for key, value in report.items() if key != "report_digest"
    }
    if report.get("report_digest") != _canonical_digest(report_payload):
        raise ValueError("frozen predictor report digest mismatch")
    frozen = report.get("frozen_predictors")
    if not isinstance(frozen, dict):
        raise ValueError("frozen predictor report has no predictor bundle")
    payload = {
        key: value for key, value in frozen.items() if key != "frozen_predictor_digest"
    }
    if frozen.get("frozen_predictor_digest") != _canonical_digest(payload):
        raise ValueError("frozen predictor bundle digest mismatch")
    if frozen.get("sealed_refitting_permitted") is not False:
        raise ValueError("sealed predictor refitting policy changed")
    if frozen.get("primary_predictive_models") != list(
        PRIMARY_PREDICTIVE_MODELS
    ):
        raise ValueError("primary predictive population changed")
    if frozen.get("scalar_generic_feature_names") != list(
        SCALAR_GENERIC_FEATURE_NAMES
    ):
        raise ValueError("scalar generic feature order changed")
    if frozen.get("layer_aware_additional_feature_names") != list(
        LAYER_AWARE_ADDITIONAL_FEATURE_NAMES
    ):
        raise ValueError("layer-aware sensitivity feature order changed")
    if frozen.get("tsi_additional_feature_names") != list(
        TSI_ADDITIONAL_FEATURE_NAMES
    ):
        raise ValueError("TSI predictor feature order changed")
    models = frozen.get("models")
    if not isinstance(models, dict) or set(models) != {
        "hazard_baseline",
        "hazard_tsi",
        "hazard_layer_aware_baseline",
        "hazard_layer_aware_tsi",
    }:
        raise ValueError("frozen predictor model set changed")
    return frozen


def score_validity_result(
    result: Mapping[str, object],
    predictor_report: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frozen = validate_frozen_predictors(predictor_report)
    rows = flatten_validity_records(result)
    scored = score_frozen_predictors(rows, frozen["models"])
    return scored, _world_seed_effects(scored)


def write_validity_predictors(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
