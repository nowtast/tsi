"""Frozen world-level confirmatory analysis for P3-4B."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from math import inf, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from .paper3_confirmatory_analysis import student_t_survival
from .paper3_validity_contract import (
    FAMILYWISE_ALPHA,
    OPTIMIZER_SEEDS,
    PREDICTIVE_SESOI,
    PRIMARY_EFFECT_NAMES,
    TASKS_PER_UNIT,
    UNITS_PER_WORLD,
)
from .paper3_validity_experiment import P3_VALIDITY_SEALED_RAW_ID
from .paper3_validity_power import (
    CLUSTER_BOOTSTRAP_ITERATIONS,
    CONFIRMATORY_ANALYSIS_SEED,
    P3_VALIDITY_ANALYSIS_PLAN_ID,
)
from .paper3_validity_predictor import (
    MODEL_IDS,
    score_validity_result,
    validate_frozen_predictors,
)


P3_VALIDITY_CONFIRMATORY_ANALYSIS_ID = "P3-4B-CONFIRMATORY-ANALYSIS-v1"


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_analysis_plan(plan: Mapping[str, object]) -> None:
    if plan.get("identifier") != P3_VALIDITY_ANALYSIS_PLAN_ID:
        raise ValueError("unexpected P3-4B analysis plan")
    payload = {
        key: value for key, value in plan.items() if key != "analysis_plan_digest"
    }
    if plan.get("analysis_plan_digest") != _canonical_digest(payload):
        raise ValueError("P3-4B analysis plan digest mismatch")
    world_count = plan.get("planned_test_worlds")
    if type(world_count) is not int or world_count < 2:
        raise ValueError("P3-4B analysis plan has an invalid world count")
    if plan.get("primary_success_effects") != list(PRIMARY_EFFECT_NAMES):
        raise ValueError("P3-4B co-primary effect order changed")
    if plan.get("sealed_predictor_refitting") is not False:
        raise ValueError("P3-4B sealed predictor-refitting policy changed")


def holm_adjusted_pvalues(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if (
        values.shape != (len(PRIMARY_EFFECT_NAMES),)
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("P3-4B needs exactly two valid p-values")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _seed_effect_array(
    world_seed_rows: Sequence[Mapping[str, object]],
    world_count: int,
) -> np.ndarray:
    indexed: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in world_seed_rows:
        key = int(row["world_index"]), int(row["optimizer_seed"])
        if key in indexed:
            raise ValueError("duplicate sealed validity world/seed effect")
        indexed[key] = row
    expected = {
        (world, seed)
        for world in range(world_count)
        for seed in OPTIMIZER_SEEDS
    }
    if set(indexed) != expected:
        raise ValueError("sealed validity world/seed effect panel is incomplete")
    effects = np.zeros(
        (world_count, len(OPTIMIZER_SEEDS), len(PRIMARY_EFFECT_NAMES)),
        dtype=np.float64,
    )
    for world in range(world_count):
        for seed_position, seed in enumerate(OPTIMIZER_SEEDS):
            row = indexed[(world, seed)]
            effects[world, seed_position] = [
                float(row[name]) for name in PRIMARY_EFFECT_NAMES
            ]
    if not np.all(np.isfinite(effects)):
        raise ValueError("sealed validity effects must be finite")
    return effects


def _student_tests(world_means: np.ndarray) -> dict[str, object]:
    count = len(world_means)
    means = np.mean(world_means, axis=0)
    standard_deviations = np.std(world_means, axis=0, ddof=1)
    statistics = np.empty(len(PRIMARY_EFFECT_NAMES), dtype=np.float64)
    pvalues = np.empty(len(PRIMARY_EFFECT_NAMES), dtype=np.float64)
    for index in range(len(PRIMARY_EFFECT_NAMES)):
        if standard_deviations[index] <= 1.0e-15:
            statistics[index] = (
                inf if means[index] > 0.0 else (-inf if means[index] < 0.0 else 0.0)
            )
        else:
            statistics[index] = means[index] / (
                standard_deviations[index] / sqrt(count)
            )
        pvalues[index] = student_t_survival(
            float(statistics[index]),
            count - 1,
        )
    adjusted = holm_adjusted_pvalues(pvalues)
    return {
        name: {
            "mean_brier_improvement": float(means[index]),
            "world_sd": float(standard_deviations[index]),
            "student_t": float(statistics[index]),
            "degrees_of_freedom": count - 1,
            "one_sided_p": float(pvalues[index]),
            "holm_adjusted_p": float(adjusted[index]),
            "holm_reject": bool(adjusted[index] < FAMILYWISE_ALPHA),
        }
        for index, name in enumerate(PRIMARY_EFFECT_NAMES)
    }


def _world_cluster_bootstrap(world_means: np.ndarray) -> dict[str, object]:
    seed = int.from_bytes(
        sha256(CONFIRMATORY_ANALYSIS_SEED.encode("utf-8")).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(world_means),
        size=(CLUSTER_BOOTSTRAP_ITERATIONS, len(world_means)),
    )
    replicates = np.mean(world_means[indices], axis=1)
    simultaneous_probability = FAMILYWISE_ALPHA / len(PRIMARY_EFFECT_NAMES)
    simultaneous = np.quantile(
        replicates,
        simultaneous_probability,
        axis=0,
    )
    marginal = np.quantile(replicates, FAMILYWISE_ALPHA, axis=0)
    return {
        "iterations": CLUSTER_BOOTSTRAP_ITERATIONS,
        "seed_commitment": sha256(
            CONFIRMATORY_ANALYSIS_SEED.encode("utf-8")
        ).hexdigest(),
        "cluster_unit": "world_with_all_nested_optimizer_seeds",
        "simultaneous_method": "bonferroni_alpha_over_2",
        "effects": {
            name: {
                "marginal_95pct_lower_bound": float(marginal[index]),
                "simultaneous_95pct_lower_bound": float(simultaneous[index]),
                "simultaneous_bound_positive": bool(simultaneous[index] > 0.0),
            }
            for index, name in enumerate(PRIMARY_EFFECT_NAMES)
        },
    }


def _hierarchical_variance(seed_effects: np.ndarray) -> dict[str, object]:
    world_count, seed_count, _effect_count = seed_effects.shape
    world_means = np.mean(seed_effects, axis=1)
    grand_means = np.mean(world_means, axis=0)
    within_seed_variance = np.mean(
        np.var(seed_effects, axis=1, ddof=1),
        axis=0,
    )
    observed_world_mean_variance = np.var(world_means, axis=0, ddof=1)
    between_world_variance = np.maximum(
        observed_world_mean_variance - within_seed_variance / seed_count,
        0.0,
    )
    standard_errors = np.sqrt(
        between_world_variance / world_count
        + within_seed_variance / (world_count * seed_count)
    )
    critical = NormalDist().inv_cdf(
        1.0 - FAMILYWISE_ALPHA / len(PRIMARY_EFFECT_NAMES)
    )
    lower_bounds = grand_means - critical * standard_errors
    return {
        "model": "world_random_intercept_seed_nested_variance_decomposition",
        "simultaneous_normal_critical": critical,
        "effects": {
            name: {
                "grand_mean": float(grand_means[index]),
                "between_world_variance": float(between_world_variance[index]),
                "within_world_seed_variance": float(within_seed_variance[index]),
                "standard_error": float(standard_errors[index]),
                "simultaneous_normal_lower_bound": float(lower_bounds[index]),
                "simultaneous_bound_positive": bool(lower_bounds[index] > 0.0),
            }
            for index, name in enumerate(PRIMARY_EFFECT_NAMES)
        },
    }


def _secondary_summary(
    scored: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_model: dict[str, list[Mapping[str, object]]] = {
        model: [] for model in MODEL_IDS
    }
    for row in scored:
        by_model[str(row["model"])].append(row)
    first_failure = Counter(int(row["first_failure_time"]) for row in scored)
    return {
        "overall": {
            "record_count": len(scored),
            "event_rate": float(
                np.mean([row["any_task_failure"] for row in scored])
            ),
            "binary_baseline_brier": float(
                np.mean([row["binary_baseline_brier"] for row in scored])
            ),
            "binary_tsi_brier": float(
                np.mean([row["binary_tsi_brier"] for row in scored])
            ),
            "first_failure_baseline_integrated_brier": float(
                np.mean(
                    [
                        row["first_failure_baseline_integrated_brier"]
                        for row in scored
                    ]
                )
            ),
            "first_failure_tsi_integrated_brier": float(
                np.mean(
                    [
                        row["first_failure_tsi_integrated_brier"]
                        for row in scored
                    ]
                )
            ),
            "first_failure_distribution": {
                str(time): first_failure.get(time, 0)
                for time in range(1, TASKS_PER_UNIT + 2)
            },
        },
        "by_model": {
            model: {
                "record_count": len(rows),
                "event_rate": float(
                    np.mean([row["any_task_failure"] for row in rows])
                ),
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
            for model, rows in by_model.items()
        },
    }


def analyze_validity_confirmatory(
    raw: Mapping[str, object],
    analysis_plan: Mapping[str, object],
    predictor_report: Mapping[str, object],
) -> dict[str, object]:
    _validate_analysis_plan(analysis_plan)
    frozen = validate_frozen_predictors(predictor_report)
    if raw.get("identifier") != P3_VALIDITY_SEALED_RAW_ID:
        raise ValueError("unexpected P3-4B sealed raw result")
    if raw.get("analysis_plan_digest") != analysis_plan.get("analysis_plan_digest"):
        raise ValueError("sealed validity result does not match its analysis plan")
    if raw.get("frozen_predictor_digest") != frozen.get(
        "frozen_predictor_digest"
    ):
        raise ValueError("sealed validity result changed the frozen predictors")
    if analysis_plan.get("frozen_predictor_digest") != frozen.get(
        "frozen_predictor_digest"
    ):
        raise ValueError("analysis plan changed the frozen predictors")

    planned_worlds = int(analysis_plan["planned_test_worlds"])
    scored, world_seed_rows = score_validity_result(raw, predictor_report)
    seed_effects = _seed_effect_array(world_seed_rows, planned_worlds)
    world_means = np.mean(seed_effects, axis=1)
    student = _student_tests(world_means)
    bootstrap = _world_cluster_bootstrap(world_means)
    hierarchical = _hierarchical_variance(seed_effects)
    mean_effects = np.mean(world_means, axis=0)

    runs = raw.get("runs")
    if not isinstance(runs, list):
        raise ValueError("sealed validity runs must be a list")
    completed = tuple(
        run
        for run in runs
        if isinstance(run, dict) and run.get("status") == "completed"
    )
    expected_records = (
        planned_worlds
        * len(OPTIMIZER_SEEDS)
        * len(MODEL_IDS)
        * UNITS_PER_WORLD
    )
    design_complete = bool(
        raw.get("world_count") == planned_worlds
        and raw.get("optimizer_seeds") == list(OPTIMIZER_SEEDS)
        and raw.get("run_count") == planned_worlds * len(OPTIMIZER_SEEDS) * 6
        and raw.get("failure_count") == 0
        and len(completed) == raw.get("run_count")
        and raw.get("unit_count_per_world") == UNITS_PER_WORLD
        and len(scored) == expected_records
        and raw.get("constructive_metric_cache", {}).get(
            "global_target_state_candidates"
        )
        == 0
    )
    outcomes_noncircular = all(
        record.get("outcome_uses_tsi_metric") is False
        and record.get("probe_task_domains_separated") is True
        for run in completed
        for record in run.get("unit_records", [])
    )
    event_rate = float(
        np.mean([row["any_task_failure"] for row in scored])
    )
    event_balance = 0.05 <= event_rate <= 0.95
    holm_passed = all(item["holm_reject"] for item in student.values())
    bootstrap_passed = all(
        item["simultaneous_bound_positive"] for item in bootstrap["effects"].values()
    )
    hierarchical_passed = all(
        item["simultaneous_bound_positive"] for item in hierarchical["effects"].values()
    )
    sesoi_passed = bool(np.all(mean_effects >= PREDICTIVE_SESOI))
    passed = bool(
        design_complete
        and outcomes_noncircular
        and event_balance
        and holm_passed
        and bootstrap_passed
        and hierarchical_passed
        and sesoi_passed
    )
    payload: dict[str, object] = {
        "identifier": P3_VALIDITY_CONFIRMATORY_ANALYSIS_ID,
        "analysis_plan_digest": analysis_plan["analysis_plan_digest"],
        "frozen_predictor_digest": frozen["frozen_predictor_digest"],
        "raw_result_digest": raw.get("report_digest"),
        "test_output_used": True,
        "world_count": planned_worlds,
        "optimizer_seeds_per_world": len(OPTIMIZER_SEEDS),
        "primary_success_effects": list(PRIMARY_EFFECT_NAMES),
        "student_t_holm": student,
        "world_cluster_bootstrap": bootstrap,
        "hierarchical_world_seed_analysis": hierarchical,
        "mean_success_effects": dict(
            zip(PRIMARY_EFFECT_NAMES, mean_effects.tolist(), strict=True)
        ),
        "predictive_sesoi": PREDICTIVE_SESOI,
        "decision_requirements": {
            "design_complete_without_failed_runs": design_complete,
            "outcomes_noncircular_and_probe_task_separated": outcomes_noncircular,
            "sealed_event_balance_passed": event_balance,
            "holm_two_tests_passed": holm_passed,
            "world_cluster_simultaneous_bounds_passed": bootstrap_passed,
            "hierarchical_simultaneous_bounds_passed": hierarchical_passed,
            "both_point_effects_meet_sesoi": sesoi_passed,
            "sealed_predictors_were_not_refit": True,
        },
        "sealed_event_rate": event_rate,
        "secondary_predictive_summary": _secondary_summary(scored),
        "world_seed_effects": world_seed_rows,
        "passed": passed,
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def write_validity_confirmatory_analysis(
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
