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
    RIDGE_PENALTY,
    TASKS_PER_UNIT,
    UNITS_PER_WORLD,
    validity_contract_digest,
)
from .paper3_validity_experiment import P3_VALIDITY_DEVELOPMENT_ID


P3_VALIDITY_PREDICTOR_ID = "P3-4B-FROZEN-PREDICTORS-v1"
MODEL_IDS = tuple(model.identifier for model in MODEL_CONTROLS)
REFERENCE_MODEL = MODEL_IDS[0]
FIXED_LAYER_ORDER = ("label", "simplicial", "metric", "relation", "order")

GENERIC_FEATURE_NAMES = (
    "training_final_nll",
    "probe_teacher_one_step_mse",
    "probe_open_loop_latent_mse",
    "probe_terminal_exactness",
    "generic_task_aligned_mismatch",
    "oracle_mean_reward_gap",
    "mean_plan_horizon",
    "stratum_unseen_structural_mode",
    *(f"raw_layer_mismatch_{layer}" for layer in LAYER_ORDER),
    *(f"task_goal_weight_{layer}" for layer in LAYER_ORDER),
    *(f"model_is_{model}" for model in MODEL_IDS[1:]),
)
TSI_ADDITIONAL_FEATURE_NAMES = (
    "probe_i0_correspondence_auc",
    "probe_fixed_total_auc",
    *(f"probe_fixed_layer_{layer}" for layer in FIXED_LAYER_ORDER),
    "tsi_task_aligned_fixed_auc",
)
TSI_FEATURE_NAMES = (*GENERIC_FEATURE_NAMES, *TSI_ADDITIONAL_FEATURE_NAMES)
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
            tsi = record.get("tsi_predictors")
            outcomes = record.get("outcomes")
            if (
                type(unit) is not int
                or not isinstance(generic, dict)
                or not isinstance(tsi, dict)
                or not isinstance(outcomes, dict)
            ):
                raise ValueError("validity unit record is malformed")
            key = (world, str(model), seed, unit)
            if key in seen:
                raise ValueError("duplicate validity unit record")
            seen.add(key)
            mismatch = generic.get("probe_open_loop_layer_mismatch_rate")
            weights = generic.get("task_goal_layer_weights")
            fixed = tsi.get("probe_fixed_layer_auc")
            if not all(isinstance(value, dict) for value in (mismatch, weights, fixed)):
                raise ValueError("validity layer predictor is malformed")
            generic_values = [
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
                    generic.get("generic_task_aligned_mismatch"),
                    "generic_task_aligned_mismatch",
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
                    _finite_float(mismatch.get(layer), f"mismatch_{layer}")
                    for layer in LAYER_ORDER
                ],
                *[
                    _finite_float(weights.get(layer), f"goal_weight_{layer}")
                    for layer in LAYER_ORDER
                ],
                *[float(model == candidate) for candidate in MODEL_IDS[1:]],
            ]
            tsi_values = [
                _finite_float(
                    tsi.get("probe_i0_correspondence_auc"),
                    "probe_i0_correspondence_auc",
                ),
                _finite_float(
                    tsi.get("probe_fixed_total_auc"),
                    "probe_fixed_total_auc",
                ),
                *[
                    _finite_float(fixed.get(layer), f"fixed_layer_{layer}")
                    for layer in FIXED_LAYER_ORDER
                ],
                _finite_float(
                    tsi.get("tsi_task_aligned_fixed_auc"),
                    "tsi_task_aligned_fixed_auc",
                ),
            ]
            any_failure = outcomes.get("any_task_failure")
            first_failure = outcomes.get("first_failure_time")
            if any_failure not in (0, 1):
                raise ValueError("any_task_failure must be binary")
            if (
                type(first_failure) is not int
                or not 1 <= first_failure <= TASKS_PER_UNIT + 1
            ):
                raise ValueError("first_failure_time is outside follow-up")
            rows.append(
                {
                    "world_index": world,
                    "model": model,
                    "optimizer_seed": seed,
                    "unit_index": unit,
                    "generic_features": generic_values,
                    "tsi_features": [*generic_values, *tsi_values],
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
    base = _matrix(rows, field)
    features: list[np.ndarray] = []
    targets: list[float] = []
    for row_index, row in enumerate(rows):
        failure_time = int(row["first_failure_time"])
        last = min(failure_time, TASKS_PER_UNIT)
        for time in range(1, last + 1):
            time_features = np.zeros(TASKS_PER_UNIT - 1, dtype=np.float64)
            if time >= 2:
                time_features[time - 2] = 1.0
            features.append(np.concatenate((base[row_index], time_features)))
            targets.append(float(failure_time == time))
    names = (*tuple(base_names), *HAZARD_TIME_FEATURE_NAMES)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        names,
    )


def _fit_model_set(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    generic = _matrix(rows, "generic_features")
    tsi = _matrix(rows, "tsi_features")
    binary_targets = np.asarray(
        [row["any_task_failure"] for row in rows],
        dtype=np.float64,
    )
    generic_hazard, hazard_targets, generic_hazard_names = _hazard_training_data(
        rows,
        "generic_features",
        GENERIC_FEATURE_NAMES,
    )
    tsi_hazard, tsi_hazard_targets, tsi_hazard_names = _hazard_training_data(
        rows,
        "tsi_features",
        TSI_FEATURE_NAMES,
    )
    if not np.array_equal(hazard_targets, tsi_hazard_targets):
        raise RuntimeError("baseline and TSI hazard outcomes diverged")
    return {
        "binary_baseline": fit_ridge_logistic(
            generic,
            binary_targets,
            GENERIC_FEATURE_NAMES,
        ),
        "binary_tsi": fit_ridge_logistic(
            tsi,
            binary_targets,
            TSI_FEATURE_NAMES,
        ),
        "hazard_baseline": fit_ridge_logistic(
            generic_hazard,
            hazard_targets,
            generic_hazard_names,
        ),
        "hazard_tsi": fit_ridge_logistic(
            tsi_hazard,
            hazard_targets,
            tsi_hazard_names,
        ),
    }


def _hazard_probabilities(
    model: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    field: str,
    base_names: Sequence[str],
) -> np.ndarray:
    base = _matrix(rows, field)
    expanded: list[np.ndarray] = []
    for row in base:
        for time in range(1, TASKS_PER_UNIT + 1):
            time_features = np.zeros(TASKS_PER_UNIT - 1, dtype=np.float64)
            if time >= 2:
                time_features[time - 2] = 1.0
            expanded.append(np.concatenate((row, time_features)))
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
    generic = _matrix(rows, "generic_features")
    tsi = _matrix(rows, "tsi_features")
    baseline_binary = predict_ridge_logistic(
        models["binary_baseline"],
        generic,
        GENERIC_FEATURE_NAMES,
    )
    tsi_binary = predict_ridge_logistic(
        models["binary_tsi"],
        tsi,
        TSI_FEATURE_NAMES,
    )
    baseline_hazards = _hazard_probabilities(
        models["hazard_baseline"],
        rows,
        "generic_features",
        GENERIC_FEATURE_NAMES,
    )
    tsi_hazards = _hazard_probabilities(
        models["hazard_tsi"],
        rows,
        "tsi_features",
        TSI_FEATURE_NAMES,
    )
    baseline_failure = 1.0 - np.cumprod(1.0 - baseline_hazards, axis=1)
    tsi_failure = 1.0 - np.cumprod(1.0 - tsi_hazards, axis=1)
    scored: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        outcome = float(row["any_task_failure"])
        first_failure = int(row["first_failure_time"])
        event_by_time = np.asarray(
            [
                float(first_failure <= time)
                for time in range(1, TASKS_PER_UNIT + 1)
            ],
            dtype=np.float64,
        )
        binary_baseline_brier = (outcome - baseline_binary[index]) ** 2
        binary_tsi_brier = (outcome - tsi_binary[index]) ** 2
        time_baseline_brier = float(
            np.mean((event_by_time - baseline_failure[index]) ** 2)
        )
        time_tsi_brier = float(
            np.mean((event_by_time - tsi_failure[index]) ** 2)
        )
        scored.append(
            {
                "world_index": int(row["world_index"]),
                "model": str(row["model"]),
                "optimizer_seed": int(row["optimizer_seed"]),
                "unit_index": int(row["unit_index"]),
                "any_task_failure": int(outcome),
                "first_failure_time": first_failure,
                "binary_baseline_probability": float(baseline_binary[index]),
                "binary_tsi_probability": float(tsi_binary[index]),
                "binary_baseline_brier": float(binary_baseline_brier),
                "binary_tsi_brier": float(binary_tsi_brier),
                "binary_brier_improvement": float(
                    binary_baseline_brier - binary_tsi_brier
                ),
                "baseline_cumulative_failure_probability": (
                    baseline_failure[index].tolist()
                ),
                "tsi_cumulative_failure_probability": tsi_failure[index].tolist(),
                "first_failure_baseline_integrated_brier": time_baseline_brier,
                "first_failure_tsi_integrated_brier": time_tsi_brier,
                "first_failure_integrated_brier_improvement": (
                    time_baseline_brier - time_tsi_brier
                ),
            }
        )
    return scored


def _world_seed_effects(
    scored: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in scored:
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

    cross_validated: list[dict[str, object]] = []
    if perform_lowo:
        worlds = sorted({int(row["world_index"]) for row in rows})
        for held_out in worlds:
            training = [row for row in rows if row["world_index"] != held_out]
            testing = [row for row in rows if row["world_index"] == held_out]
            fold_models = _fit_model_set(training)
            cross_validated.extend(score_frozen_predictors(testing, fold_models))
    final_models = _fit_model_set(rows)
    frozen_payload = {
        "identifier": P3_VALIDITY_PREDICTOR_ID,
        "contract_digest": validity_contract_digest(),
        "development_digest": development.get("report_digest"),
        "generic_feature_names": list(GENERIC_FEATURE_NAMES),
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
        np.mean([float(row["any_task_failure"]) for row in rows])
    )
    cv_effects = _world_seed_effects(cross_validated)
    cv_means = {
        "binary_brier_improvement": (
            float(np.mean([row["binary_brier_improvement"] for row in cv_effects]))
            if cv_effects
            else None
        ),
        "first_failure_integrated_brier_improvement": (
            float(
                np.mean(
                    [
                        row["first_failure_integrated_brier_improvement"]
                        for row in cv_effects
                    ]
                )
            )
            if cv_effects
            else None
        ),
    }
    converged = all(model["converged"] for model in final_models.values())
    payload: dict[str, object] = {
        "identifier": P3_VALIDITY_PREDICTOR_ID,
        "test_output_used": False,
        "development_worlds": int(development["world_count"]),
        "development_records": len(rows),
        "development_event_rate": event_rate,
        "development_first_failure_distribution": {
            str(time): sum(
                int(row["first_failure_time"] == time) for row in rows
            )
            for time in range(1, TASKS_PER_UNIT + 2)
        },
        "all_final_models_converged": converged,
        "development_lowo_performed": perform_lowo,
        "development_lowo_mean_effects": cv_means,
        "development_lowo_world_seed_effects": cv_effects,
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
    if frozen.get("generic_feature_names") != list(GENERIC_FEATURE_NAMES):
        raise ValueError("generic predictor feature order changed")
    if frozen.get("tsi_additional_feature_names") != list(
        TSI_ADDITIONAL_FEATURE_NAMES
    ):
        raise ValueError("TSI predictor feature order changed")
    models = frozen.get("models")
    if not isinstance(models, dict) or set(models) != {
        "binary_baseline",
        "binary_tsi",
        "hazard_baseline",
        "hazard_tsi",
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
