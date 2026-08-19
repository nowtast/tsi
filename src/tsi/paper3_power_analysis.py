"""World-level variance estimation and Holm power simulation for P3-3A."""

from __future__ import annotations

from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from typing import Mapping

import numpy as np

from .paper3_analysis_plan import (
    DENSE_NONINFERIORITY_MARGIN,
    POWER_TARGET,
    PRIMARY_CONTROLS,
    PRIMARY_FAMILY,
    PRIMARY_MODEL,
    PRIMARY_OOD_SLICE,
    SMALLEST_EFFECT_OF_INTEREST,
    analysis_plan_digest,
)
from .paper3_development_experiment import (
    OPTIMIZER_SEEDS,
    P3_DEVELOPMENT_EXPERIMENT_ID,
)
from .paper3_independence_contract import (
    FROZEN_STATISTICAL_PLAN,
    planned_test_world_count,
)


P3_POWER_REPORT_ID = "P3-3A-POWER-v1"
POWER_SIMULATION_ID = "P3-3A-HOLM-NORMAL-SIM-v1"
POWER_SIMULATION_SEED = "tsi:p3-3a:power:2026-07-29:v1"
SIMULATION_ITERATIONS = 20_000
HOLM_ONE_SIDED_NORMAL_CRITICALS = (
    2.128045234184983,
    1.959963984540054,
    1.6448536269514722,
)
POSITIVE_CONTROL_FAMILY = "separable"
POSITIVE_CONTROL_MODEL = "strict_factorized_action"
POSITIVE_CONTROL_SLICE = "unseen_action_composition"
POSITIVE_CONTROL_MAX_ERROR = 0.05
POSITIVE_CONTROL_MIN_EXACT = 0.90


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _completed_runs(
    pilot: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    runs = pilot.get("runs")
    if not isinstance(runs, list):
        raise ValueError("development pilot runs must be a list")
    return tuple(
        run
        for run in runs
        if isinstance(run, dict) and run.get("status") == "completed"
    )


def _slice_metric(
    run: Mapping[str, object],
    slice_name: str,
    field: str,
) -> float:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("completed run has no metric mapping")
    selected = metrics.get(slice_name)
    if not isinstance(selected, dict):
        raise ValueError(f"completed run misses slice {slice_name}")
    value = selected.get(field)
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError(f"slice metric {field} must be finite")
    return float(value)


def world_seed_errors(
    pilot: Mapping[str, object],
    *,
    family: str,
    slice_name: str,
) -> dict[int, dict[str, dict[int, float]]]:
    """Index the primary endpoint by independent world, model, and nested seed."""

    indexed: dict[int, dict[str, dict[int, float]]] = {}
    for run in _completed_runs(pilot):
        if run.get("family") != family:
            continue
        world_index = run.get("world_index")
        model = run.get("model")
        seed = run.get("optimizer_seed")
        if type(world_index) is not int or not isinstance(model, str):
            raise ValueError("run world/model identity is invalid")
        if type(seed) is not int:
            raise ValueError("run optimizer seed is invalid")
        error = _slice_metric(
            run,
            slice_name,
            "mean_normalized_i0_quotient_error",
        )
        model_errors = indexed.setdefault(world_index, {}).setdefault(model, {})
        if seed in model_errors:
            raise ValueError("duplicate world/model/seed result")
        model_errors[seed] = error
    return indexed


def primary_world_effects(
    pilot: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    indexed = world_seed_errors(
        pilot,
        family=PRIMARY_FAMILY,
        slice_name=PRIMARY_OOD_SLICE,
    )
    required_models = (PRIMARY_MODEL, *PRIMARY_CONTROLS)
    rows: list[tuple[float, float, float]] = []
    raw_rows: list[dict[str, object]] = []
    expected_seeds = set(OPTIMIZER_SEEDS)
    for world_index in sorted(indexed):
        model_errors = indexed[world_index]
        missing = [model for model in required_models if model not in model_errors]
        if missing:
            raise ValueError(
                f"world {world_index} misses primary models: {', '.join(missing)}"
            )
        means: dict[str, float] = {}
        for model in required_models:
            by_seed = model_errors[model]
            if set(by_seed) != expected_seeds:
                raise ValueError(
                    f"world {world_index}/{model} has unmatched optimizer seeds"
                )
            means[model] = float(np.mean(tuple(by_seed.values())))
        signature = means[PRIMARY_MODEL]
        dense_difference = signature - means["dense_active_matched"]
        random_difference = means["random_routed_matched_sparsity"] - signature
        wrong_difference = means["permuted_or_wrong_routed"] - signature
        transformed = (
            DENSE_NONINFERIORITY_MARGIN - dense_difference,
            random_difference,
            wrong_difference,
        )
        rows.append(transformed)
        raw_rows.append(
            {
                "world_index": world_index,
                "seed_averaged_errors": means,
                "raw_contrasts": {
                    "signature_minus_dense": dense_difference,
                    "random_minus_signature": random_difference,
                    "wrong_minus_signature": wrong_difference,
                },
                "positive_success_transforms": {
                    "dense_noninferiority_margin_minus_difference": (transformed[0]),
                    "random_superiority": transformed[1],
                    "wrong_superiority": transformed[2],
                },
            }
        )
    if not rows:
        raise ValueError("no primary development worlds were found")
    return np.asarray(rows, dtype=np.float64), {"worlds": raw_rows}


def positive_control_summary(
    pilot: Mapping[str, object],
) -> dict[str, object]:
    selected = tuple(
        run
        for run in _completed_runs(pilot)
        if run.get("family") == POSITIVE_CONTROL_FAMILY
        and run.get("model") == POSITIVE_CONTROL_MODEL
    )
    if not selected:
        raise ValueError("separable positive-control runs are absent")
    errors = tuple(
        _slice_metric(
            run,
            POSITIVE_CONTROL_SLICE,
            "mean_normalized_i0_quotient_error",
        )
        for run in selected
    )
    exact = tuple(
        _slice_metric(
            run,
            POSITIVE_CONTROL_SLICE,
            "fixed_joint_exact_rate",
        )
        for run in selected
    )
    mean_error = float(np.mean(errors))
    mean_exact = float(np.mean(exact))
    passed = (
        mean_error <= POSITIVE_CONTROL_MAX_ERROR
        and mean_exact >= POSITIVE_CONTROL_MIN_EXACT
    )
    return {
        "family": POSITIVE_CONTROL_FAMILY,
        "model": POSITIVE_CONTROL_MODEL,
        "slice": POSITIVE_CONTROL_SLICE,
        "run_count": len(selected),
        "mean_normalized_i0_quotient_error": mean_error,
        "mean_fixed_joint_exact_rate": mean_exact,
        "maximum_allowed_error": POSITIVE_CONTROL_MAX_ERROR,
        "minimum_required_exact_rate": POSITIVE_CONTROL_MIN_EXACT,
        "passed": passed,
    }


def planning_covariance(
    effects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(effects, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("effects must have shape (at least 2, 3)")
    observed_sd = np.std(values, axis=0, ddof=1)
    planning_sd = np.maximum(0.10, 1.25 * observed_sd)
    correlation = np.eye(3, dtype=np.float64)
    for left in range(3):
        for right in range(left + 1, 3):
            if observed_sd[left] == 0.0 or observed_sd[right] == 0.0:
                coefficient = 0.0
            else:
                coefficient = float(
                    np.corrcoef(values[:, left], values[:, right])[0, 1]
                )
                if not np.isfinite(coefficient):
                    coefficient = 0.0
            correlation[left, right] = coefficient
            correlation[right, left] = coefficient
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1.0e-8)
    correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(correlation))
    correlation = correlation / np.outer(diagonal, diagonal)
    covariance = np.outer(planning_sd, planning_sd) * correlation
    return observed_sd, planning_sd, covariance


def _wilson_lower(successes: int, trials: int) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = proportion + z**2 / (2.0 * trials)
    radius = z * sqrt(
        proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)
    )
    return (center - radius) / denominator


def simulate_holm_power(
    covariance: np.ndarray,
    *,
    iterations: int = SIMULATION_ITERATIONS,
    minimum_worlds: int | None = None,
    maximum_worlds: int | None = None,
) -> dict[str, object]:
    plan = FROZEN_STATISTICAL_PLAN
    minimum = plan.minimum_test_worlds if minimum_worlds is None else minimum_worlds
    maximum = plan.maximum_test_worlds if maximum_worlds is None else maximum_worlds
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be positive")
    if minimum < 2 or maximum < minimum:
        raise ValueError("invalid simulation world range")
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("planning covariance must be 3 by 3")

    seed = int.from_bytes(
        sha256(POWER_SIMULATION_SEED.encode("utf-8")).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(
        mean=np.full(3, SMALLEST_EFFECT_OF_INTEREST, dtype=np.float64),
        cov=matrix,
        size=(iterations, maximum),
        check_valid="raise",
    )
    cumulative = np.cumsum(draws, axis=1)
    cumulative_squares = np.cumsum(draws**2, axis=1)
    power_curve: list[dict[str, object]] = []
    selected_worlds: int | None = None
    for count in range(minimum, maximum + 1):
        total = cumulative[:, count - 1, :]
        total_squares = cumulative_squares[:, count - 1, :]
        means = total / count
        variances = np.maximum(
            (total_squares - count * means**2) / (count - 1),
            np.finfo(np.float64).tiny,
        )
        statistics = means / np.sqrt(variances / count)
        ordered = np.sort(statistics, axis=1)[:, ::-1]
        success = (
            (ordered[:, 0] > HOLM_ONE_SIDED_NORMAL_CRITICALS[0])
            & (ordered[:, 1] > HOLM_ONE_SIDED_NORMAL_CRITICALS[1])
            & (ordered[:, 2] > HOLM_ONE_SIDED_NORMAL_CRITICALS[2])
        )
        successes = int(np.sum(success))
        power = successes / iterations
        lower = _wilson_lower(successes, iterations)
        power_curve.append(
            {
                "worlds": count,
                "conjunctive_holm_power": power,
                "monte_carlo_95pct_lower_bound": lower,
            }
        )
        if selected_worlds is None and lower >= POWER_TARGET:
            selected_worlds = count
    selected = (
        None if selected_worlds is None else power_curve[selected_worlds - minimum]
    )
    return {
        "identifier": POWER_SIMULATION_ID,
        "iterations": iterations,
        "seed_commitment": sha256(POWER_SIMULATION_SEED.encode("utf-8")).hexdigest(),
        "alternative_success_transform_mean": [SMALLEST_EFFECT_OF_INTEREST] * 3,
        "holm_one_sided_normal_criticals": list(HOLM_ONE_SIDED_NORMAL_CRITICALS),
        "selection_rule": (
            "smallest_n_with_95pct_monte_carlo_lower_bound_at_least_0.90"
        ),
        "selected_worlds": selected_worlds,
        "selected_result": selected,
        "power_curve": power_curve,
    }


def build_power_report(
    pilot: Mapping[str, object],
    *,
    iterations: int = SIMULATION_ITERATIONS,
) -> dict[str, object]:
    if pilot.get("identifier") != P3_DEVELOPMENT_EXPERIMENT_ID:
        raise ValueError("unexpected development pilot identifier")
    if pilot.get("test_output_used") is not False:
        raise ValueError("development power cannot use test output")
    effects, world_detail = primary_world_effects(pilot)
    observed_sd, planning_sd, covariance = planning_covariance(effects)
    simulation = simulate_holm_power(covariance, iterations=iterations)
    simulation_worlds = simulation["selected_worlds"]
    maximum_observed_sd = float(np.max(observed_sd))
    analytic_worlds = planned_test_world_count(maximum_observed_sd)
    if not isinstance(simulation_worlds, int):
        planned_worlds = FROZEN_STATISTICAL_PLAN.maximum_test_worlds
        selected_power = 0.0
        selected_lower = 0.0
    else:
        planned_worlds = max(analytic_worlds, simulation_worlds)
        selected = simulation["power_curve"][
            planned_worlds - FROZEN_STATISTICAL_PLAN.minimum_test_worlds
        ]
        selected_power = float(selected["conjunctive_holm_power"])
        selected_lower = float(selected["monte_carlo_95pct_lower_bound"])

    effect_names = (
        "dense_noninferiority_success_transform",
        "random_superiority",
        "wrong_superiority",
    )
    observed = {
        name: {
            "mean": float(np.mean(effects[:, index])),
            "world_sd": float(observed_sd[index]),
            "planning_sd": float(planning_sd[index]),
            "development_readiness_threshold": (SMALLEST_EFFECT_OF_INTEREST),
        }
        for index, name in enumerate(effect_names)
    }
    development_effects_passed = all(
        detail["mean"] + 1.0e-12 >= SMALLEST_EFFECT_OF_INTEREST
        for detail in observed.values()
    )
    positive = positive_control_summary(pilot)
    no_failures = pilot.get("failure_count") == 0
    full_development_size = (
        effects.shape[0] == FROZEN_STATISTICAL_PLAN.development_worlds_per_family
    )
    simulation_passed = (
        isinstance(simulation_worlds, int)
        and planned_worlds <= FROZEN_STATISTICAL_PLAN.maximum_test_worlds
        and selected_power >= POWER_TARGET
        and selected_lower >= POWER_TARGET
    )
    passed = bool(
        no_failures
        and full_development_size
        and positive["passed"]
        and development_effects_passed
        and simulation_passed
    )
    payload: dict[str, object] = {
        "identifier": P3_POWER_REPORT_ID,
        "passed": passed,
        "test_output_used": False,
        "development_pilot_digest": pilot.get("report_digest"),
        "analysis_plan_digest": analysis_plan_digest(),
        "independent_unit": "world",
        "nested_replicate": "optimizer_seed",
        "development_worlds": int(effects.shape[0]),
        "optimizer_seeds_per_world": len(OPTIMIZER_SEEDS),
        "primary_family": PRIMARY_FAMILY,
        "primary_ood_slice": PRIMARY_OOD_SLICE,
        "observed_success_transforms": observed,
        "development_effects_passed": development_effects_passed,
        "positive_control": positive,
        "failure_count": pilot.get("failure_count"),
        "planning_covariance": covariance.tolist(),
        "analytic_holm_world_floor": analytic_worlds,
        "planned_test_worlds": planned_worlds,
        "minimum_simulation_power": selected_power,
        "simulation_power_95pct_lower_bound": selected_lower,
        "simulation": simulation,
        "world_level_detail": world_detail,
        "decision_requirements": {
            "zero_failed_runs": no_failures,
            "24_primary_development_worlds": full_development_size,
            "positive_control_passed": positive["passed"],
            "all_development_effects_at_least_0.05": (development_effects_passed),
            "conjunctive_holm_power_and_mc_lower_bound_at_least_0.90": (
                simulation_passed
            ),
        },
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def write_power_report(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
