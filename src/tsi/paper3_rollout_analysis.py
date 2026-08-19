"""Frozen world-level confirmatory analysis for the sealed P3-4A run."""

from __future__ import annotations

from hashlib import sha256
import json
from math import inf, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Mapping

import numpy as np

from .paper3_confirmatory_analysis import student_t_survival
from .paper3_rollout_contract import (
    FAMILYWISE_ALPHA,
    MAX_HORIZON,
    OPTIMIZER_SEEDS,
    PRIMARY_MODEL,
    SMALLEST_ROUTING_EFFECT,
    SUCCESS_EFFECT_NAMES,
    TRAJECTORIES_PER_WORLD,
)
from .paper3_rollout_experiment import P3_ROLLOUT_SEALED_RAW_ID
from .paper3_rollout_power import (
    CLUSTER_BOOTSTRAP_ITERATIONS,
    CONFIRMATORY_ANALYSIS_SEED,
    P3_ROLLOUT_ANALYSIS_PLAN_ID,
    seed_level_success_effects,
)


P3_ROLLOUT_CONFIRMATORY_ANALYSIS_ID = "P3-4A-CONFIRMATORY-ANALYSIS-v1"


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_analysis_plan(plan: Mapping[str, object]) -> None:
    if plan.get("identifier") != P3_ROLLOUT_ANALYSIS_PLAN_ID:
        raise ValueError("unexpected P3-4A analysis plan")
    payload = {
        key: value for key, value in plan.items() if key != "analysis_plan_digest"
    }
    if plan.get("analysis_plan_digest") != _canonical_digest(payload):
        raise ValueError("P3-4A analysis plan digest mismatch")
    world_count = plan.get("planned_test_worlds")
    if type(world_count) is not int or world_count < 2:
        raise ValueError("P3-4A analysis plan has an invalid world count")
    if plan.get("primary_success_effects") != list(SUCCESS_EFFECT_NAMES):
        raise ValueError("P3-4A co-primary effect order changed")


def holm_adjusted_pvalues(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if (
        values.shape != (len(SUCCESS_EFFECT_NAMES),)
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("P3-4A needs exactly eight valid p-values")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _student_tests(world_means: np.ndarray) -> dict[str, object]:
    count = len(world_means)
    means = np.mean(world_means, axis=0)
    standard_deviations = np.std(world_means, axis=0, ddof=1)
    statistics = np.empty(len(SUCCESS_EFFECT_NAMES), dtype=np.float64)
    pvalues = np.empty(len(SUCCESS_EFFECT_NAMES), dtype=np.float64)
    for index in range(len(SUCCESS_EFFECT_NAMES)):
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
            "mean_success_effect": float(means[index]),
            "world_sd": float(standard_deviations[index]),
            "student_t": float(statistics[index]),
            "degrees_of_freedom": count - 1,
            "one_sided_p": float(pvalues[index]),
            "holm_adjusted_p": float(adjusted[index]),
            "holm_reject": bool(adjusted[index] < FAMILYWISE_ALPHA),
        }
        for index, name in enumerate(SUCCESS_EFFECT_NAMES)
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
    simultaneous_probability = FAMILYWISE_ALPHA / len(SUCCESS_EFFECT_NAMES)
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
        "simultaneous_method": "bonferroni_alpha_over_8",
        "effects": {
            name: {
                "marginal_95pct_lower_bound": float(marginal[index]),
                "simultaneous_95pct_lower_bound": float(simultaneous[index]),
                "simultaneous_bound_positive": bool(simultaneous[index] > 0.0),
            }
            for index, name in enumerate(SUCCESS_EFFECT_NAMES)
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
    critical = NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / len(SUCCESS_EFFECT_NAMES))
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
            for index, name in enumerate(SUCCESS_EFFECT_NAMES)
        },
    }


def _secondary_model_summary(
    raw: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "teacher_forced_i0_auc",
        "open_loop_i0_auc",
        "exposure_gap_i0_auc",
        "terminal_open_loop_i0_error",
        "terminal_open_loop_fixed_error",
        "terminal_open_loop_tracking_error",
        "self_conditioned_local_law_violation_rate",
        "state_coherence_bridge_violation_rate",
        "terminal_trajectory_survival_rate",
        "mean_first_structural_failure_time",
    )
    collected: dict[str, dict[str, list[float]]] = {}
    horizon_collected: dict[str, dict[str, dict[str, list[float]]]] = {}
    runs = raw.get("runs")
    if not isinstance(runs, list):
        raise ValueError("sealed rollout runs must be a list")
    for run in runs:
        if not isinstance(run, dict) or run.get("status") != "completed":
            continue
        model = str(run["model"])
        metrics = run["metrics"]
        destination = collected.setdefault(
            model,
            {field: [] for field in fields},
        )
        for field in fields:
            destination[field].append(float(metrics[field]))
        for horizon, values in metrics["horizon_summary"].items():
            horizon_destination = horizon_collected.setdefault(model, {}).setdefault(
                horizon,
                {
                    "teacher_forced_mean_i0_error": [],
                    "open_loop_mean_i0_error": [],
                    "open_loop_state_exact_rate": [],
                    "trajectory_survival_rate": [],
                },
            )
            for field in horizon_destination:
                horizon_destination[field].append(float(values[field]))
    return {
        model: {
            "overall": {
                field: float(np.mean(values)) for field, values in model_fields.items()
            },
            "horizons": {
                horizon: {
                    field: float(np.mean(values))
                    for field, values in horizon_fields.items()
                }
                for horizon, horizon_fields in horizon_collected[model].items()
            },
        }
        for model, model_fields in collected.items()
    }


def analyze_rollout_confirmatory(
    raw: Mapping[str, object],
    analysis_plan: Mapping[str, object],
) -> dict[str, object]:
    _validate_analysis_plan(analysis_plan)
    if raw.get("identifier") != P3_ROLLOUT_SEALED_RAW_ID:
        raise ValueError("unexpected P3-4A sealed raw result")
    if raw.get("analysis_plan_digest") != analysis_plan.get("analysis_plan_digest"):
        raise ValueError("sealed rollout result does not match its analysis plan")
    planned_worlds = int(analysis_plan["planned_test_worlds"])
    seed_effects, world_detail = seed_level_success_effects(
        raw,
        expected_world_count=planned_worlds,
    )
    world_means = np.mean(seed_effects, axis=1)
    student = _student_tests(world_means)
    bootstrap = _world_cluster_bootstrap(world_means)
    hierarchical = _hierarchical_variance(seed_effects)

    holm_passed = all(item["holm_reject"] for item in student.values())
    bootstrap_passed = all(
        item["simultaneous_bound_positive"] for item in bootstrap["effects"].values()
    )
    hierarchical_passed = all(
        item["simultaneous_bound_positive"] for item in hierarchical["effects"].values()
    )
    mean_effects = np.mean(world_means, axis=0)
    routing_sesoi_passed = bool(
        mean_effects[-2] >= SMALLEST_ROUTING_EFFECT
        and mean_effects[-1] >= SMALLEST_ROUTING_EFFECT
    )
    runs = raw.get("runs")
    if not isinstance(runs, list):
        raise ValueError("sealed rollout runs must be a list")
    completed = tuple(
        run
        for run in runs
        if isinstance(run, dict) and run.get("status") == "completed"
    )
    recursive_bounds_passed = all(
        run["metrics"]["recursive_bound_violation_count"] == 0 for run in completed
    )
    signature_bridge_valid = all(
        run["metrics"]["state_coherence_bridge_violation_rate"] == 0.0
        for run in completed
        if run.get("model") == PRIMARY_MODEL
    )
    design_complete = bool(
        raw.get("world_count") == planned_worlds
        and raw.get("optimizer_seeds") == list(OPTIMIZER_SEEDS)
        and raw.get("run_count") == planned_worlds * len(OPTIMIZER_SEEDS) * 6
        and raw.get("failure_count") == 0
        and len(completed) == raw.get("run_count")
        and raw.get("trajectory_count_per_world") == TRAJECTORIES_PER_WORLD
        and raw.get("constructive_metric_cache", {}).get(
            "global_target_state_candidates"
        )
        == 0
    )
    passed = bool(
        design_complete
        and recursive_bounds_passed
        and signature_bridge_valid
        and holm_passed
        and bootstrap_passed
        and hierarchical_passed
        and routing_sesoi_passed
    )
    payload: dict[str, object] = {
        "identifier": P3_ROLLOUT_CONFIRMATORY_ANALYSIS_ID,
        "analysis_plan_digest": analysis_plan["analysis_plan_digest"],
        "raw_result_digest": raw.get("report_digest"),
        "test_output_used": True,
        "world_count": planned_worlds,
        "optimizer_seeds_per_world": len(OPTIMIZER_SEEDS),
        "maximum_horizon": MAX_HORIZON,
        "primary_success_effects": list(SUCCESS_EFFECT_NAMES),
        "student_t_holm": student,
        "world_cluster_bootstrap": bootstrap,
        "hierarchical_world_seed_analysis": hierarchical,
        "mean_success_effects": dict(
            zip(SUCCESS_EFFECT_NAMES, mean_effects.tolist(), strict=True)
        ),
        "routing_point_effects": {
            "random_minus_signature_open_loop_auc": float(mean_effects[-2]),
            "wrong_minus_signature_open_loop_auc": float(mean_effects[-1]),
            "both_meet_0.05_sesoi": routing_sesoi_passed,
        },
        "decision_requirements": {
            "design_complete_without_failed_runs": design_complete,
            "recursive_rollout_bounds_passed": recursive_bounds_passed,
            "signature_state_bridge_validity_passed": signature_bridge_valid,
            "holm_eight_tests_passed": holm_passed,
            "world_cluster_simultaneous_bounds_passed": bootstrap_passed,
            "hierarchical_simultaneous_bounds_passed": hierarchical_passed,
            "random_and_wrong_point_effects_meet_sesoi": (routing_sesoi_passed),
        },
        "secondary_model_summary": _secondary_model_summary(raw),
        "world_seed_detail": world_detail,
        "passed": passed,
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def write_rollout_confirmatory_analysis(
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
