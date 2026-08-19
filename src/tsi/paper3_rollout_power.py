"""World-level variance estimation and Holm power planning for P3-4A."""

from __future__ import annotations

from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .paper3_rollout_contract import (
    DENSE_NONINFERIORITY_MARGIN,
    DEVELOPMENT_WORLDS,
    EXPOSURE_GAP_MARGIN,
    LOCAL_LAW_VIOLATION_MAXIMUM,
    MAXIMUM_TEST_WORLDS,
    MINIMUM_TEST_WORLDS,
    OPEN_LOOP_AUC_MAXIMUM,
    OPTIMIZER_SEEDS,
    PLANNING_SD_FLOOR,
    PLANNING_SD_INFLATION,
    POWER_ALTERNATIVE_EFFECT,
    POWER_SIMULATION_ITERATIONS,
    POWER_TARGET,
    PRIMARY_MODEL,
    PRIMARY_MODEL_SET,
    SMALLEST_ROUTING_EFFECT,
    SUCCESS_EFFECT_NAMES,
    TERMINAL_I0_MAXIMUM,
    TRACKING_ERROR_MAXIMUM,
    analytic_world_floor,
    holm_normal_criticals,
    rollout_contract_digest,
)
from .paper3_rollout_experiment import P3_ROLLOUT_DEVELOPMENT_ID


P3_ROLLOUT_POWER_ID = "P3-4A-ROLLOUT-POWER-v1"
P3_ROLLOUT_ANALYSIS_PLAN_ID = "P3-4A-ROLLOUT-ANALYSIS-PLAN-v1"
POWER_SIMULATION_ID = "P3-4A-HOLM-NORMAL-SIM-v1"
POWER_SIMULATION_SEED = "tsi:p3-4a:rollout-power:2026-07-29:v1"
CONFIRMATORY_ANALYSIS_SEED = "tsi:p3-4a:confirmatory-analysis:2026-07-29:v1"
CLUSTER_BOOTSTRAP_ITERATIONS = 20_000


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _completed_runs(
    result: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    runs = result.get("runs")
    if not isinstance(runs, list):
        raise ValueError("rollout result runs must be a list")
    return tuple(
        run
        for run in runs
        if isinstance(run, dict) and run.get("status") == "completed"
    )


def _run_metric(run: Mapping[str, object], field: str) -> float:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("completed rollout run has no metrics")
    value = metrics.get(field)
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError(f"rollout metric {field} must be finite")
    return float(value)


def indexed_rollout_metrics(
    result: Mapping[str, object],
) -> dict[int, dict[str, dict[int, Mapping[str, object]]]]:
    indexed: dict[int, dict[str, dict[int, Mapping[str, object]]]] = {}
    for run in _completed_runs(result):
        world = run.get("world_index")
        model = run.get("model")
        seed = run.get("optimizer_seed")
        metrics = run.get("metrics")
        if type(world) is not int or not isinstance(model, str):
            raise ValueError("rollout run has an invalid world/model identity")
        if type(seed) is not int or not isinstance(metrics, dict):
            raise ValueError("rollout run has an invalid seed/metrics payload")
        by_seed = indexed.setdefault(world, {}).setdefault(model, {})
        if seed in by_seed:
            raise ValueError("duplicate rollout world/model/seed result")
        by_seed[seed] = metrics
    return indexed


def _success_effects_from_metrics(
    by_model: Mapping[str, Mapping[str, object]],
) -> tuple[float, ...]:
    missing = [model for model in PRIMARY_MODEL_SET if model not in by_model]
    if missing:
        raise ValueError(f"rollout effects miss models: {', '.join(missing)}")
    signature = by_model[PRIMARY_MODEL]
    dense = by_model["dense_active_matched"]
    random = by_model["random_routed_matched_sparsity"]
    wrong = by_model["permuted_or_wrong_routed"]
    signature_auc = float(signature["open_loop_i0_auc"])
    return (
        OPEN_LOOP_AUC_MAXIMUM - signature_auc,
        TERMINAL_I0_MAXIMUM - float(signature["terminal_open_loop_i0_error"]),
        EXPOSURE_GAP_MARGIN - float(signature["exposure_gap_i0_auc"]),
        TRACKING_ERROR_MAXIMUM - float(signature["terminal_open_loop_tracking_error"]),
        LOCAL_LAW_VIOLATION_MAXIMUM
        - float(signature["self_conditioned_local_law_violation_rate"]),
        DENSE_NONINFERIORITY_MARGIN
        - (signature_auc - float(dense["open_loop_i0_auc"])),
        float(random["open_loop_i0_auc"]) - signature_auc,
        float(wrong["open_loop_i0_auc"]) - signature_auc,
    )


def seed_level_success_effects(
    result: Mapping[str, object],
    *,
    expected_world_count: int,
    expected_seeds: Sequence[int] = OPTIMIZER_SEEDS,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    indexed = indexed_rollout_metrics(result)
    if set(indexed) != set(range(expected_world_count)):
        raise ValueError("rollout result does not contain the expected world indices")
    seeds = tuple(expected_seeds)
    effects = np.zeros(
        (expected_world_count, len(seeds), len(SUCCESS_EFFECT_NAMES)),
        dtype=np.float64,
    )
    detail: list[dict[str, object]] = []
    for world in range(expected_world_count):
        by_model_seed = indexed[world]
        if any(model not in by_model_seed for model in PRIMARY_MODEL_SET):
            raise ValueError(f"rollout world {world} misses a primary model")
        world_detail: dict[str, object] = {"world_index": world, "seeds": []}
        for seed_position, seed in enumerate(seeds):
            if any(seed not in by_model_seed[model] for model in PRIMARY_MODEL_SET):
                raise ValueError(
                    f"rollout world {world} has unmatched optimizer seed {seed}"
                )
            by_model = {
                model: by_model_seed[model][seed] for model in PRIMARY_MODEL_SET
            }
            row = _success_effects_from_metrics(by_model)
            effects[world, seed_position] = row
            world_detail["seeds"].append(
                {
                    "optimizer_seed": seed,
                    "success_effects": dict(
                        zip(SUCCESS_EFFECT_NAMES, row, strict=True)
                    ),
                }
            )
        detail.append(world_detail)
    return effects, detail


def planning_covariance(
    world_effects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(world_effects, dtype=np.float64)
    effect_count = len(SUCCESS_EFFECT_NAMES)
    if values.ndim != 2 or values.shape[1] != effect_count or len(values) < 2:
        raise ValueError("world effects have the wrong shape")
    observed_sd = np.std(values, axis=0, ddof=1)
    planning_sd = np.maximum(
        PLANNING_SD_FLOOR,
        PLANNING_SD_INFLATION * observed_sd,
    )
    correlation = np.eye(effect_count, dtype=np.float64)
    for left in range(effect_count):
        for right in range(left + 1, effect_count):
            if observed_sd[left] <= 1.0e-12 or observed_sd[right] <= 1.0e-12:
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
    if not 0 <= successes <= trials or trials <= 0:
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
    iterations: int = POWER_SIMULATION_ITERATIONS,
    minimum_worlds: int = MINIMUM_TEST_WORLDS,
    maximum_worlds: int = MAXIMUM_TEST_WORLDS,
    batch_size: int = 1_000,
) -> dict[str, object]:
    matrix = np.asarray(covariance, dtype=np.float64)
    effect_count = len(SUCCESS_EFFECT_NAMES)
    if matrix.shape != (effect_count, effect_count):
        raise ValueError("rollout planning covariance has the wrong shape")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("power iterations must be positive")
    if not 2 <= minimum_worlds <= maximum_worlds:
        raise ValueError("invalid rollout power world range")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")

    seed = int.from_bytes(
        sha256(POWER_SIMULATION_SEED.encode("utf-8")).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    criticals = np.asarray(holm_normal_criticals(), dtype=np.float64)
    counts = np.arange(minimum_worlds, maximum_worlds + 1)
    successes = np.zeros(len(counts), dtype=np.int64)
    remaining = iterations
    while remaining:
        current = min(batch_size, remaining)
        draws = rng.multivariate_normal(
            mean=np.full(
                effect_count,
                POWER_ALTERNATIVE_EFFECT,
                dtype=np.float64,
            ),
            cov=matrix,
            size=(current, maximum_worlds),
            check_valid="raise",
        )
        cumulative = np.cumsum(draws, axis=1)
        cumulative_squares = np.cumsum(draws**2, axis=1)
        for position, world_count in enumerate(counts):
            total = cumulative[:, world_count - 1, :]
            total_squares = cumulative_squares[:, world_count - 1, :]
            means = total / world_count
            variances = np.maximum(
                (total_squares - world_count * means**2) / (world_count - 1),
                np.finfo(np.float64).tiny,
            )
            statistics = means / np.sqrt(variances / world_count)
            ordered = np.sort(statistics, axis=1)[:, ::-1]
            successes[position] += int(np.sum(np.all(ordered > criticals, axis=1)))
        remaining -= current

    power_curve: list[dict[str, object]] = []
    selected_worlds: int | None = None
    for world_count, success_count in zip(counts, successes, strict=True):
        power = int(success_count) / iterations
        lower = _wilson_lower(int(success_count), iterations)
        power_curve.append(
            {
                "worlds": int(world_count),
                "conjunctive_holm_power": power,
                "monte_carlo_95pct_lower_bound": lower,
            }
        )
        if selected_worlds is None and lower >= POWER_TARGET:
            selected_worlds = int(world_count)
    selected = (
        None
        if selected_worlds is None
        else power_curve[selected_worlds - minimum_worlds]
    )
    return {
        "identifier": POWER_SIMULATION_ID,
        "iterations": iterations,
        "seed_commitment": sha256(POWER_SIMULATION_SEED.encode("utf-8")).hexdigest(),
        "alternative_success_transform_mean": [POWER_ALTERNATIVE_EFFECT] * effect_count,
        "holm_one_sided_normal_criticals": criticals.tolist(),
        "selection_rule": (
            "smallest_n_with_95pct_monte_carlo_lower_bound_at_least_0.90"
        ),
        "selected_worlds": selected_worlds,
        "selected_result": selected,
        "power_curve": power_curve,
    }


def _analysis_plan(
    development_digest: str,
    planned_worlds: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "identifier": P3_ROLLOUT_ANALYSIS_PLAN_ID,
        "contract_digest": rollout_contract_digest(),
        "development_pilot_digest": development_digest,
        "planned_test_worlds": planned_worlds,
        "primary_success_effects": list(SUCCESS_EFFECT_NAMES),
        "primary_models": list(PRIMARY_MODEL_SET),
        "smallest_routing_effect": SMALLEST_ROUTING_EFFECT,
        "student_test": "one_sided_world_level_student_t",
        "multiplicity": "holm_fwer_eight_coprimary_effects",
        "world_cluster_bootstrap_iterations": CLUSTER_BOOTSTRAP_ITERATIONS,
        "world_cluster_bootstrap_tail": (
            "bonferroni_alpha_over_eight_simultaneous_lower"
        ),
        "hierarchical_model": (
            "world_random_intercept_seed_nested_variance_decomposition"
        ),
        "confirmatory_analysis_seed_commitment": sha256(
            CONFIRMATORY_ANALYSIS_SEED.encode("utf-8")
        ).hexdigest(),
        "test_output_used": False,
    }
    return {**payload, "analysis_plan_digest": _canonical_digest(payload)}


def build_rollout_power_report(
    pilot: Mapping[str, object],
    *,
    iterations: int = POWER_SIMULATION_ITERATIONS,
) -> dict[str, object]:
    if pilot.get("identifier") != P3_ROLLOUT_DEVELOPMENT_ID:
        raise ValueError("unexpected rollout development result")
    if pilot.get("test_output_used") is not False:
        raise ValueError("rollout power cannot use test output")
    seed_effects, world_detail = seed_level_success_effects(
        pilot,
        expected_world_count=DEVELOPMENT_WORLDS,
    )
    world_effects = np.mean(seed_effects, axis=1)
    observed_sd, planning_sd, covariance = planning_covariance(world_effects)
    simulation = simulate_holm_power(covariance, iterations=iterations)
    analytic = analytic_world_floor(float(np.max(planning_sd)))
    simulated = simulation["selected_worlds"]
    if not isinstance(simulated, int):
        planned_worlds = MAXIMUM_TEST_WORLDS
        selected_power = 0.0
        selected_lower = 0.0
    else:
        planned_worlds = max(analytic, simulated)
        selected = simulation["power_curve"][planned_worlds - MINIMUM_TEST_WORLDS]
        selected_power = float(selected["conjunctive_holm_power"])
        selected_lower = float(selected["monte_carlo_95pct_lower_bound"])

    observed = {
        name: {
            "mean": float(np.mean(world_effects[:, index])),
            "world_sd": float(observed_sd[index]),
            "planning_sd": float(planning_sd[index]),
            "development_readiness_threshold": POWER_ALTERNATIVE_EFFECT,
        }
        for index, name in enumerate(SUCCESS_EFFECT_NAMES)
    }
    development_effects_passed = all(
        item["mean"] + 1.0e-12 >= POWER_ALTERNATIVE_EFFECT for item in observed.values()
    )
    completed = _completed_runs(pilot)
    bound_passed = all(
        _run_metric(run, "recursive_bound_violation_count") == 0.0 for run in completed
    )
    design_complete = bool(
        pilot.get("world_count") == DEVELOPMENT_WORLDS
        and pilot.get("optimizer_seeds") == list(OPTIMIZER_SEEDS)
        and pilot.get("failure_count") == 0
        and len(completed) == DEVELOPMENT_WORLDS * len(OPTIMIZER_SEEDS) * 6
    )
    power_passed = bool(
        isinstance(simulated, int)
        and planned_worlds <= MAXIMUM_TEST_WORLDS
        and selected_power >= POWER_TARGET
        and selected_lower >= POWER_TARGET
    )
    analysis_plan = _analysis_plan(
        str(pilot["report_digest"]),
        planned_worlds,
    )
    passed = bool(
        design_complete and development_effects_passed and bound_passed and power_passed
    )
    payload: dict[str, object] = {
        "identifier": P3_ROLLOUT_POWER_ID,
        "passed": passed,
        "test_output_used": False,
        "contract_digest": rollout_contract_digest(),
        "development_pilot_digest": pilot.get("report_digest"),
        "development_worlds": DEVELOPMENT_WORLDS,
        "optimizer_seeds_per_world": len(OPTIMIZER_SEEDS),
        "observed_success_transforms": observed,
        "development_effects_passed": development_effects_passed,
        "recursive_bound_audit_passed": bound_passed,
        "design_complete": design_complete,
        "planning_covariance": covariance.tolist(),
        "analytic_holm_world_floor": analytic,
        "power_simulation": simulation,
        "planned_test_worlds": planned_worlds,
        "selected_conjunctive_power": selected_power,
        "selected_monte_carlo_95pct_lower_bound": selected_lower,
        "analysis_plan": analysis_plan,
        "world_effect_detail": world_detail,
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def write_rollout_power_report(
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


def write_rollout_analysis_plan(
    path: Path,
    power_report: Mapping[str, object],
) -> None:
    plan = power_report.get("analysis_plan")
    if not isinstance(plan, dict):
        raise ValueError("rollout power report has no analysis plan")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(plan, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
