"""World-level development variance and power planning for P3-4B."""

from __future__ import annotations

from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from typing import Mapping

import numpy as np

from .paper3_validity_contract import (
    DEVELOPMENT_WORLDS,
    MAXIMUM_TEST_WORLDS,
    MINIMUM_TEST_WORLDS,
    PLANNING_SD_FLOOR,
    PLANNING_SD_INFLATION,
    POWER_ALTERNATIVE_EFFECT,
    POWER_SIMULATION_ITERATIONS,
    POWER_TARGET,
    PREDICTIVE_SESOI,
    PRIMARY_EFFECT_NAMES,
    analytic_world_floor,
    holm_normal_criticals,
    validity_contract_digest,
)
from .paper3_validity_predictor import (
    P3_VALIDITY_PREDICTOR_ID,
    validate_frozen_predictors,
)


P3_VALIDITY_POWER_ID = "P3-4B-VALIDITY-POWER-v1"
P3_VALIDITY_ANALYSIS_PLAN_ID = "P3-4B-VALIDITY-ANALYSIS-PLAN-v1"
POWER_SIMULATION_ID = "P3-4B-HOLM-NORMAL-SIM-v1"
POWER_SIMULATION_SEED = "tsi:p3-4b:validity-power:2026-07-29:v1"
CONFIRMATORY_ANALYSIS_SEED = "tsi:p3-4b:confirmatory-analysis:2026-07-29:v1"
CLUSTER_BOOTSTRAP_ITERATIONS = 20_000


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def development_seed_effects(
    predictor_report: Mapping[str, object],
) -> np.ndarray:
    rows = predictor_report.get("development_lowo_world_seed_effects")
    if not isinstance(rows, list):
        raise ValueError("predictor report has no LOWO world/seed effects")
    indexed: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("development LOWO effect row is malformed")
        world = row.get("world_index")
        seed = row.get("optimizer_seed")
        if type(world) is not int or type(seed) is not int:
            raise ValueError("development LOWO effect identity is malformed")
        key = world, seed
        if key in indexed:
            raise ValueError("duplicate development LOWO effect")
        indexed[key] = row
    expected = {
        (world, seed)
        for world in range(DEVELOPMENT_WORLDS)
        for seed in range(3)
    }
    if set(indexed) != expected:
        raise ValueError("development LOWO effect panel is incomplete")
    effects = np.zeros(
        (DEVELOPMENT_WORLDS, 3, len(PRIMARY_EFFECT_NAMES)),
        dtype=np.float64,
    )
    for world, seed in sorted(indexed):
        row = indexed[(world, seed)]
        effects[world, seed] = [
            float(row[name]) for name in PRIMARY_EFFECT_NAMES
        ]
    if not np.all(np.isfinite(effects)):
        raise ValueError("development LOWO effects must be finite")
    return effects


def planning_covariance(
    world_effects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(world_effects, dtype=np.float64)
    effect_count = len(PRIMARY_EFFECT_NAMES)
    if values.shape != (DEVELOPMENT_WORLDS, effect_count):
        raise ValueError("validity world effects have the wrong shape")
    observed_sd = np.std(values, axis=0, ddof=1)
    planning_sd = np.maximum(
        PLANNING_SD_FLOOR,
        PLANNING_SD_INFLATION * observed_sd,
    )
    correlation = np.eye(effect_count, dtype=np.float64)
    if np.all(observed_sd > 1.0e-12):
        coefficient = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
        if np.isfinite(coefficient):
            correlation[0, 1] = coefficient
            correlation[1, 0] = coefficient
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
    effect_count = len(PRIMARY_EFFECT_NAMES)
    if matrix.shape != (effect_count, effect_count):
        raise ValueError("validity planning covariance has the wrong shape")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("power iterations must be positive")
    if not 2 <= minimum_worlds <= maximum_worlds:
        raise ValueError("invalid validity power world range")
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
            mean=np.full(effect_count, POWER_ALTERNATIVE_EFFECT),
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

    selected_worlds: int | None = None
    power_curve: list[dict[str, object]] = []
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
        "alternative_success_transform_mean": [
            POWER_ALTERNATIVE_EFFECT
        ] * effect_count,
        "holm_one_sided_normal_criticals": criticals.tolist(),
        "selection_rule": (
            "smallest_n_with_95pct_monte_carlo_lower_bound_at_least_0.90"
        ),
        "selected_worlds": selected_worlds,
        "selected_result": selected,
        "power_curve": power_curve,
    }


def _analysis_plan(
    predictor_report: Mapping[str, object],
    planned_worlds: int,
) -> dict[str, object]:
    frozen = validate_frozen_predictors(predictor_report)
    payload: dict[str, object] = {
        "identifier": P3_VALIDITY_ANALYSIS_PLAN_ID,
        "contract_digest": validity_contract_digest(),
        "development_predictor_report_digest": predictor_report["report_digest"],
        "frozen_predictor_digest": frozen["frozen_predictor_digest"],
        "planned_test_worlds": planned_worlds,
        "primary_success_effects": list(PRIMARY_EFFECT_NAMES),
        "predictive_sesoi": PREDICTIVE_SESOI,
        "student_test": "one_sided_world_level_student_t",
        "multiplicity": "holm_fwer_two_coprimary_effects",
        "world_cluster_bootstrap_iterations": CLUSTER_BOOTSTRAP_ITERATIONS,
        "world_cluster_bootstrap_tail": (
            "bonferroni_alpha_over_two_simultaneous_lower"
        ),
        "hierarchical_model": (
            "world_random_intercept_seed_nested_variance_decomposition"
        ),
        "sealed_predictor_refitting": False,
        "confirmatory_analysis_seed_commitment": sha256(
            CONFIRMATORY_ANALYSIS_SEED.encode("utf-8")
        ).hexdigest(),
        "test_output_used": False,
    }
    return {**payload, "analysis_plan_digest": _canonical_digest(payload)}


def build_validity_power_report(
    predictor_report: Mapping[str, object],
    *,
    iterations: int = POWER_SIMULATION_ITERATIONS,
) -> dict[str, object]:
    if predictor_report.get("identifier") != P3_VALIDITY_PREDICTOR_ID:
        raise ValueError("unexpected P3-4B predictor report")
    if predictor_report.get("test_output_used") is not False:
        raise ValueError("validity power cannot use test output")
    validate_frozen_predictors(predictor_report)
    seed_effects = development_seed_effects(predictor_report)
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
            "development_readiness_threshold": PREDICTIVE_SESOI,
        }
        for index, name in enumerate(PRIMARY_EFFECT_NAMES)
    }
    development_effects_passed = all(
        item["mean"] >= PREDICTIVE_SESOI for item in observed.values()
    )
    event_rate = float(predictor_report["development_event_rate"])
    event_balance_passed = 0.10 <= event_rate <= 0.90
    design_complete = bool(
        predictor_report.get("development_worlds") == DEVELOPMENT_WORLDS
        and predictor_report.get("development_lowo_performed") is True
        and predictor_report.get("all_final_models_converged") is True
        and len(predictor_report.get("development_lowo_world_seed_effects", []))
        == DEVELOPMENT_WORLDS * 3
    )
    power_passed = bool(
        isinstance(simulated, int)
        and planned_worlds <= MAXIMUM_TEST_WORLDS
        and selected_power >= POWER_TARGET
        and selected_lower >= POWER_TARGET
    )
    analysis_plan = _analysis_plan(predictor_report, planned_worlds)
    passed = bool(
        design_complete
        and event_balance_passed
        and development_effects_passed
        and power_passed
    )
    payload: dict[str, object] = {
        "identifier": P3_VALIDITY_POWER_ID,
        "passed": passed,
        "test_output_used": False,
        "contract_digest": validity_contract_digest(),
        "development_predictor_report_digest": predictor_report.get("report_digest"),
        "development_worlds": DEVELOPMENT_WORLDS,
        "development_event_rate": event_rate,
        "event_balance_passed": event_balance_passed,
        "observed_success_transforms": observed,
        "development_effects_passed": development_effects_passed,
        "design_complete": design_complete,
        "planning_covariance": covariance.tolist(),
        "analytic_holm_world_floor": analytic,
        "power_simulation": simulation,
        "planned_test_worlds": planned_worlds,
        "selected_conjunctive_power": selected_power,
        "selected_monte_carlo_95pct_lower_bound": selected_lower,
        "analysis_plan": analysis_plan,
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def write_validity_power_report(
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


def write_validity_analysis_plan(
    path: Path,
    power_report: Mapping[str, object],
) -> None:
    plan = power_report.get("analysis_plan")
    if not isinstance(plan, dict):
        raise ValueError("validity power report has no analysis plan")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(plan, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
