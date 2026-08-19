"""Frozen world-level confirmatory analysis for the P3-3B sealed run."""

from __future__ import annotations

from hashlib import sha256
import json
from math import exp, inf, lgamma, log, log1p, sqrt
from pathlib import Path
from typing import Mapping

import numpy as np

from .paper3_analysis_plan import (
    CONFIRMATORY_ANALYSIS_SEED,
    DENSE_NONINFERIORITY_MARGIN,
    FAMILYWISE_ALPHA,
    PLANNED_TEST_WORLDS,
    PRIMARY_CONTROLS,
    PRIMARY_FAMILY,
    PRIMARY_MODEL,
    PRIMARY_OOD_SLICE,
    SMALLEST_EFFECT_OF_INTEREST,
    analysis_plan_digest,
)
from .paper3_confirmatory_experiment import P3_CONFIRMATORY_EXPERIMENT_ID
from .paper3_development_experiment import OPTIMIZER_SEEDS
from .paper3_power_analysis import world_seed_errors


P3_CONFIRMATORY_ANALYSIS_ID = "P3-3B-CONFIRMATORY-ANALYSIS-v1"
CLUSTER_BOOTSTRAP_ITERATIONS = 20_000
BONFERRONI_ONE_SIDED_CRITICAL = 2.128045234184983
EFFECT_NAMES = (
    "dense_noninferiority_success_transform",
    "random_superiority",
    "wrong_superiority",
)


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3.0e-14
    minimum = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c
        coefficient = -(
            (a + iteration) * (qab + iteration) * x / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < epsilon:
            return result
    raise RuntimeError("incomplete beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Exact one-sided Student-t survival via the incomplete beta identity."""

    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if statistic == inf:
        return 0.0
    if statistic == -inf:
        return 1.0
    x = degrees_of_freedom / (degrees_of_freedom + float(statistic) ** 2)
    tail = 0.5 * _regularized_beta(
        x,
        degrees_of_freedom / 2.0,
        0.5,
    )
    return tail if statistic >= 0.0 else 1.0 - tail


def holm_adjusted_pvalues(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.shape != (3,) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("exactly three valid p-values are required")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def seed_level_success_effects(
    raw: Mapping[str, object],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    indexed = world_seed_errors(
        raw,
        family=PRIMARY_FAMILY,
        slice_name=PRIMARY_OOD_SLICE,
    )
    expected_worlds = set(range(PLANNED_TEST_WORLDS))
    if set(indexed) != expected_worlds:
        raise ValueError("sealed result does not contain the frozen world indices")
    required_models = (PRIMARY_MODEL, *PRIMARY_CONTROLS)
    effects = np.zeros(
        (PLANNED_TEST_WORLDS, len(OPTIMIZER_SEEDS), 3),
        dtype=np.float64,
    )
    details: list[dict[str, object]] = []
    for world_index in range(PLANNED_TEST_WORLDS):
        by_model = indexed[world_index]
        if any(model not in by_model for model in required_models):
            raise ValueError(f"sealed world {world_index} misses a primary model")
        world_detail: dict[str, object] = {
            "world_index": world_index,
            "seeds": [],
        }
        for seed_position, seed in enumerate(OPTIMIZER_SEEDS):
            if any(seed not in by_model[model] for model in required_models):
                raise ValueError(
                    f"sealed world {world_index} has unmatched seed {seed}"
                )
            signature = by_model[PRIMARY_MODEL][seed]
            dense_raw = signature - by_model["dense_active_matched"][seed]
            random_raw = by_model["random_routed_matched_sparsity"][seed] - signature
            wrong_raw = by_model["permuted_or_wrong_routed"][seed] - signature
            effects[world_index, seed_position] = (
                DENSE_NONINFERIORITY_MARGIN - dense_raw,
                random_raw,
                wrong_raw,
            )
            world_detail["seeds"].append(
                {
                    "optimizer_seed": seed,
                    "signature_minus_dense": dense_raw,
                    "random_minus_signature": random_raw,
                    "wrong_minus_signature": wrong_raw,
                }
            )
        details.append(world_detail)
    return effects, details


def _student_tests(world_means: np.ndarray) -> dict[str, object]:
    count = len(world_means)
    means = np.mean(world_means, axis=0)
    standard_deviations = np.std(world_means, axis=0, ddof=1)
    statistics = np.empty(3, dtype=np.float64)
    pvalues = np.empty(3, dtype=np.float64)
    for index in range(3):
        if standard_deviations[index] == 0.0:
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
        for index, name in enumerate(EFFECT_NAMES)
    }


def _cluster_bootstrap(world_means: np.ndarray) -> dict[str, object]:
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
    lower_probability = FAMILYWISE_ALPHA / len(EFFECT_NAMES)
    simultaneous_lower = np.quantile(
        replicates,
        lower_probability,
        axis=0,
    )
    marginal_lower = np.quantile(replicates, FAMILYWISE_ALPHA, axis=0)
    return {
        "iterations": CLUSTER_BOOTSTRAP_ITERATIONS,
        "seed_commitment": sha256(
            CONFIRMATORY_ANALYSIS_SEED.encode("utf-8")
        ).hexdigest(),
        "cluster_unit": "world_with_all_nested_optimizer_seeds",
        "simultaneous_method": "bonferroni_alpha_over_3_implies_holm_control",
        "effects": {
            name: {
                "marginal_95pct_lower_bound": float(marginal_lower[index]),
                "simultaneous_95pct_lower_bound": float(simultaneous_lower[index]),
                "simultaneous_bound_positive": bool(simultaneous_lower[index] > 0.0),
            }
            for index, name in enumerate(EFFECT_NAMES)
        },
    }


def _hierarchical_variance(seed_effects: np.ndarray) -> dict[str, object]:
    world_count, seed_count, _ = seed_effects.shape
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
    lower_bounds = grand_means - BONFERRONI_ONE_SIDED_CRITICAL * standard_errors
    return {
        "model": "world_random_intercept_seed_nested_variance_decomposition",
        "effects": {
            name: {
                "grand_mean": float(grand_means[index]),
                "between_world_variance": float(between_world_variance[index]),
                "within_world_seed_variance": float(within_seed_variance[index]),
                "standard_error": float(standard_errors[index]),
                "simultaneous_normal_lower_bound": float(lower_bounds[index]),
                "simultaneous_bound_positive": bool(lower_bounds[index] > 0.0),
            }
            for index, name in enumerate(EFFECT_NAMES)
        },
    }


def _secondary_slice_summary(
    raw: Mapping[str, object],
) -> dict[str, object]:
    runs = raw.get("runs")
    if not isinstance(runs, list):
        raise ValueError("raw confirmatory runs must be a list")
    fields = (
        "mean_normalized_i0_quotient_error",
        "fixed_joint_exact_rate",
        "bridge_violation_rate",
        "tracking_exact_rate",
    )
    collected: dict[str, dict[str, dict[str, list[float]]]] = {}
    for run in runs:
        if not isinstance(run, dict) or run.get("status") != "completed":
            continue
        model = run["model"]
        metrics = run["metrics"]
        for slice_name, slice_metrics in metrics.items():
            destination = collected.setdefault(model, {}).setdefault(
                slice_name, {field: [] for field in fields}
            )
            for field in fields:
                destination[field].append(float(slice_metrics[field]))
    return {
        model: {
            slice_name: {
                field: float(np.mean(values))
                for field, values in fields_by_name.items()
            }
            for slice_name, fields_by_name in slices.items()
        }
        for model, slices in collected.items()
    }


def analyze_confirmatory_experiment(
    raw: Mapping[str, object],
) -> dict[str, object]:
    if raw.get("identifier") != P3_CONFIRMATORY_EXPERIMENT_ID:
        raise ValueError("unexpected confirmatory raw-result identifier")
    if raw.get("analysis_plan_digest") != analysis_plan_digest():
        raise ValueError("raw results do not match the frozen analysis digest")
    seed_effects, detail = seed_level_success_effects(raw)
    world_means = np.mean(seed_effects, axis=1)
    student = _student_tests(world_means)
    bootstrap = _cluster_bootstrap(world_means)
    hierarchical = _hierarchical_variance(seed_effects)

    holm_passed = all(item["holm_reject"] for item in student.values())
    bootstrap_passed = all(
        item["simultaneous_bound_positive"] for item in bootstrap["effects"].values()
    )
    hierarchical_passed = all(
        item["simultaneous_bound_positive"] for item in hierarchical["effects"].values()
    )
    mean_effects = np.mean(world_means, axis=0)
    point_effects_passed = bool(
        mean_effects[1] >= SMALLEST_EFFECT_OF_INTEREST
        and mean_effects[2] >= SMALLEST_EFFECT_OF_INTEREST
    )
    design_complete = bool(
        raw.get("world_count") == PLANNED_TEST_WORLDS
        and raw.get("run_count") == PLANNED_TEST_WORLDS * len(OPTIMIZER_SEEDS) * 6
        and raw.get("failure_count") == 0
        and raw.get("constructive_metric_cache", {}).get(
            "global_target_state_candidates"
        )
        == 0
    )
    passed = bool(
        design_complete
        and holm_passed
        and bootstrap_passed
        and hierarchical_passed
        and point_effects_passed
    )
    payload: dict[str, object] = {
        "identifier": P3_CONFIRMATORY_ANALYSIS_ID,
        "analysis_plan_digest": analysis_plan_digest(),
        "raw_result_digest": raw.get("report_digest"),
        "test_output_used": True,
        "world_count": PLANNED_TEST_WORLDS,
        "optimizer_seeds_per_world": len(OPTIMIZER_SEEDS),
        "primary_family": PRIMARY_FAMILY,
        "primary_ood_slice": PRIMARY_OOD_SLICE,
        "primary_success_effects": list(EFFECT_NAMES),
        "student_t_holm": student,
        "world_cluster_bootstrap": bootstrap,
        "hierarchical_world_seed_analysis": hierarchical,
        "point_effects": {
            "signature_minus_dense": float(
                DENSE_NONINFERIORITY_MARGIN - mean_effects[0]
            ),
            "random_minus_signature": float(mean_effects[1]),
            "wrong_minus_signature": float(mean_effects[2]),
            "random_and_wrong_meet_0.05_sesoi": point_effects_passed,
        },
        "decision_requirements": {
            "design_complete_without_failed_runs": design_complete,
            "holm_three_tests_passed": holm_passed,
            "world_cluster_simultaneous_bounds_passed": bootstrap_passed,
            "hierarchical_simultaneous_bounds_passed": hierarchical_passed,
            "random_and_wrong_point_effects_meet_sesoi": point_effects_passed,
        },
        "secondary_slice_summary": _secondary_slice_summary(raw),
        "world_seed_detail": detail,
        "passed": passed,
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def write_confirmatory_analysis(
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
