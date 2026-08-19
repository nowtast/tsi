"""World-level analysis for the prospective Paper 3/4 resolution study."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from .paper34_resolution_benchmark import calibrated_brier, fit_logistic_calibrator
from .paper34_resolution_contract import (
    CRITERION_BRIER_SESOI,
    DENSE_GRAPH_NLL_SESOI,
    FACTORIZED_GRAPH_NLL_SESOI,
    FAMILYWISE_ALPHA,
    GENERIC_GRAPH_NLL_SESOI,
    LEARNED_ROUTING_NLL_SESOI,
    MATCHED_HEAD_EQUIVALENCE_MARGIN,
    OUTSIDE_FAMILY_NONINFERIORITY_MARGIN,
    PRIMARY_INFERENTIAL_EFFECT_COUNT,
    ROLLOUT_HAMMING_SESOI,
    ROUTING_IDENTIFICATION_RATE_MINIMUM,
)


def _mean_interval(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    standard_error = sd / sqrt(len(array)) if len(array) else float("nan")
    critical = NormalDist().inv_cdf(
        1.0 - FAMILYWISE_ALPHA / (2.0 * PRIMARY_INFERENTIAL_EFFECT_COUNT)
    )
    return {
        "mean": mean,
        "world_sd": sd,
        "standard_error": standard_error,
        "simultaneous_lower": mean - critical * standard_error,
        "simultaneous_upper": mean + critical * standard_error,
        "critical": critical,
    }


def _wilson_lower(successes: int, count: int) -> float:
    if count <= 0:
        raise ValueError("Wilson interval requires observations")
    z = NormalDist().inv_cdf(
        1.0 - FAMILYWISE_ALPHA / (2.0 * PRIMARY_INFERENTIAL_EFFECT_COUNT)
    )
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    radius = z * sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count))
    return (center - radius) / denominator


def _difference_interval(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    difference = float(np.mean(left_array) - np.mean(right_array))
    variance = float(np.var(left_array, ddof=1) / len(left_array)) if len(left_array) > 1 else 0.0
    variance += float(np.var(right_array, ddof=1) / len(right_array)) if len(right_array) > 1 else 0.0
    standard_error = sqrt(variance)
    critical = NormalDist().inv_cdf(
        1.0 - FAMILYWISE_ALPHA / (2.0 * PRIMARY_INFERENTIAL_EFFECT_COUNT)
    )
    return {
        "mean": difference,
        "standard_error": standard_error,
        "simultaneous_lower": difference - critical * standard_error,
        "simultaneous_upper": difference + critical * standard_error,
        "critical": critical,
    }


def fit_development_calibration(rows: Sequence[Mapping[str, object]]) -> tuple[float, float]:
    scores = []
    outcomes = []
    for row in rows:
        for record in row["rollouts"]:  # type: ignore[index]
            scores.append(float(record["correct_score"]))
            outcomes.append(float(record["terminal_failure"]))
    return fit_logistic_calibrator(scores, outcomes)


def _world_effects(
    row: Mapping[str, object], calibration: tuple[float, float]
) -> dict[str, float]:
    metrics = row["metrics"]  # type: ignore[assignment]
    rollouts = row["rollouts"]  # type: ignore[assignment]

    def nll(name: str) -> float:
        return float(metrics[name]["composition_nll"])  # type: ignore[index]

    correct_scores = [float(record["correct_score"]) for record in rollouts]
    wrong_scores = [float(record["wrong_score"]) for record in rollouts]
    outcomes = [float(record["terminal_failure"]) for record in rollouts]
    return {
        "learned_routing_nll": nll("wrong_graph_correct_head") - nll("learned_factorized"),
        "factorized_graph_nll": nll("wrong_graph_correct_head") - nll("correct_graph_correct_head"),
        "factorized_head_nll_correct_graph": nll("correct_graph_generic_7") - nll("correct_graph_correct_head"),
        "factorized_head_nll_wrong_graph": nll("wrong_graph_generic_7") - nll("wrong_graph_correct_head"),
        "generic_graph_nll": nll("wrong_graph_generic_7") - nll("correct_graph_generic_7"),
        "large_generic_graph_nll": nll("wrong_graph_generic_55") - nll("correct_graph_generic_55"),
        "graph_by_head_interaction": (
            nll("wrong_graph_correct_head") - nll("correct_graph_correct_head")
        ) - (
            nll("wrong_graph_generic_7") - nll("correct_graph_generic_7")
        ),
        "rollout_hamming": float(
            np.mean(
                [
                    record["wrong_hamming_auc"] - record["learned_hamming_auc"]
                    for record in rollouts
                ]
            )
        ),
        "criterion_brier": calibrated_brier(wrong_scores, outcomes, calibration)
        - calibrated_brier(correct_scores, outcomes, calibration),
        "learned_composition_nll": nll("learned_factorized"),
    }


def summarize_cohort(
    rows: Sequence[Mapping[str, object]],
    *,
    calibration: tuple[float, float] | None = None,
    confirmatory: bool = False,
) -> dict[str, object]:
    if not rows:
        raise ValueError("resolution analysis requires at least one world")
    frozen_calibration = calibration or fit_development_calibration(rows)
    effects = [_world_effects(row, frozen_calibration) for row in rows]
    names = tuple(effects[0])
    intervals = {
        name: _mean_interval([effect[name] for effect in effects]) for name in names
    }
    identification_rate = float(
        np.mean([bool(row["graph_exact"]) and bool(row["head_exact"]) for row in rows])
    )
    identification_successes = sum(
        bool(row["graph_exact"]) and bool(row["head_exact"]) for row in rows
    )
    identification_lower = _wilson_lower(identification_successes, len(rows))
    outside = [
        effect["learned_composition_nll"]
        for effect, row in zip(effects, rows, strict=True)
        if bool(row["outside_original_linear_family"])
    ]
    original = [
        effect["learned_composition_nll"]
        for effect, row in zip(effects, rows, strict=True)
        if not bool(row["outside_original_linear_family"])
    ]
    outside_interval = _difference_interval(outside, original) if original else {
        "mean": float("nan"), "simultaneous_upper": float("nan")
    }
    positive = "simultaneous_lower"
    upper = "simultaneous_upper"
    gates = {
        "learned_graph_and_head_identification": (
            identification_lower >= ROUTING_IDENTIFICATION_RATE_MINIMUM
            if confirmatory
            else identification_rate >= ROUTING_IDENTIFICATION_RATE_MINIMUM
        ),
        "learned_vs_wrong_graph_composition_nll": intervals["learned_routing_nll"][positive] > 0.0
        and intervals["learned_routing_nll"]["mean"] >= LEARNED_ROUTING_NLL_SESOI,
        "factorized_graph_effect": intervals["factorized_graph_nll"][positive] > 0.0
        and intervals["factorized_graph_nll"]["mean"] >= FACTORIZED_GRAPH_NLL_SESOI,
        "generic_head_graph_effect": intervals["generic_graph_nll"][positive] > 0.0
        and intervals["generic_graph_nll"]["mean"] >= GENERIC_GRAPH_NLL_SESOI,
        "dense_head_graph_effect": intervals["large_generic_graph_nll"][positive] > 0.0
        and intervals["large_generic_graph_nll"]["mean"] >= DENSE_GRAPH_NLL_SESOI,
        "matched_head_predictive_equivalence": abs(intervals["factorized_head_nll_correct_graph"]["mean"])
        <= MATCHED_HEAD_EQUIVALENCE_MARGIN
        and intervals["factorized_head_nll_correct_graph"][positive] >= -MATCHED_HEAD_EQUIVALENCE_MARGIN
        and intervals["factorized_head_nll_correct_graph"][upper] <= MATCHED_HEAD_EQUIVALENCE_MARGIN,
        "learned_vs_wrong_graph_rollout_hamming": intervals["rollout_hamming"][positive] > 0.0
        and intervals["rollout_hamming"]["mean"] >= ROLLOUT_HAMMING_SESOI,
        "correct_vs_wrong_routing_criterion_brier": intervals["criterion_brier"][positive] > 0.0
        and intervals["criterion_brier"]["mean"] >= CRITERION_BRIER_SESOI,
        "outside_original_linear_family_noninferiority": outside_interval[upper]
        <= OUTSIDE_FAMILY_NONINFERIORITY_MARGIN,
    }
    return {
        "world_count": len(rows),
        "criterion_calibration": list(frozen_calibration),
        "identification_rate": identification_rate,
        "identification_simultaneous_wilson_lower": identification_lower,
        "effect_intervals": intervals,
        "outside_family": {
            "outside_world_count": len(outside),
            "original_linear_world_count": len(original),
            "nll_difference_outside_minus_original": outside_interval,
            "noninferiority_margin": OUTSIDE_FAMILY_NONINFERIORITY_MARGIN,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "world_effects": effects,
    }
