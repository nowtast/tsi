"""Post-review multiplicity bookkeeping for the frozen Paper 3/4 cohort."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

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
    PRIMARY_INFERENTIAL_QUANTITIES,
    ROLLOUT_HAMMING_SESOI,
    ROUTING_IDENTIFICATION_RATE_MINIMUM,
)


REPORTED_EFFECT_NAMES = (
    "learned_routing_nll",
    "factorized_graph_nll",
    "factorized_head_nll_correct_graph",
    "factorized_head_nll_wrong_graph",
    "generic_graph_nll",
    "large_generic_graph_nll",
    "graph_by_head_interaction",
    "rollout_hamming",
    "criterion_brier",
    "learned_composition_nll",
)
EXACT_DUPLICATE_PAIR = ("learned_routing_nll", "factorized_graph_nll")
SIGN_FLIPPED_PAIR = ("factorized_head_nll_wrong_graph", "graph_by_head_interaction")
DETERMINISTIC_ZERO = "factorized_head_nll_correct_graph"


def bonferroni_critical(divisor: int) -> float:
    if divisor <= 0:
        raise ValueError("Bonferroni divisor must be positive")
    return NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / (2.0 * divisor))


def _interval_at_divisor(
    interval: Mapping[str, float], divisor: int
) -> dict[str, float]:
    critical = bonferroni_critical(divisor)
    mean = float(interval["mean"])
    standard_error = float(interval["standard_error"])
    return {
        "mean": mean,
        "standard_error": standard_error,
        "simultaneous_lower": mean - critical * standard_error,
        "simultaneous_upper": mean + critical * standard_error,
        "critical": critical,
    }


def _wilson_lower(successes: int, count: int, divisor: int) -> float:
    z = bonferroni_critical(divisor)
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    radius = z * sqrt(
        proportion * (1.0 - proportion) / count
        + z * z / (4.0 * count * count)
    )
    return (center - radius) / denominator


def _all_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=1e-15))


def audit_multiplicity(
    analysis: Mapping[str, object], *, sensitivity_divisor: int = 10
) -> dict[str, object]:
    """Audit the frozen divisor and recompute a transparent divisor sensitivity."""

    intervals = analysis["effect_intervals"]
    world_effects = analysis["world_effects"]
    if not isinstance(intervals, Mapping) or not isinstance(world_effects, Sequence):
        raise TypeError("analysis is missing effect intervals or world effects")
    if tuple(intervals) != REPORTED_EFFECT_NAMES:
        raise ValueError("reported effect names or order changed")

    vectors = {
        name: [float(row[name]) for row in world_effects]  # type: ignore[index]
        for name in REPORTED_EFFECT_NAMES
    }
    duplicate_verified = _all_close(
        vectors[EXACT_DUPLICATE_PAIR[0]], vectors[EXACT_DUPLICATE_PAIR[1]]
    )
    sign_flip_verified = _all_close(
        vectors[SIGN_FLIPPED_PAIR[0]],
        [-value for value in vectors[SIGN_FLIPPED_PAIR[1]]],
    )
    zero_verified = _all_close(
        vectors[DETERMINISTIC_ZERO], [0.0] * len(world_effects)
    )
    if not (duplicate_verified and sign_flip_verified and zero_verified):
        raise ValueError("declared multiplicity relationships do not hold")

    sensitivity_intervals = {
        name: _interval_at_divisor(intervals[name], sensitivity_divisor)  # type: ignore[arg-type]
        for name in REPORTED_EFFECT_NAMES
    }
    outside = analysis["outside_family"]  # type: ignore[assignment]
    outside_original = outside["nll_difference_outside_minus_original"]
    outside_sensitivity = _interval_at_divisor(
        outside_original, sensitivity_divisor
    )
    world_count = int(analysis["world_count"])
    identification_successes = round(float(analysis["identification_rate"]) * world_count)
    identification_lower = _wilson_lower(
        identification_successes, world_count, sensitivity_divisor
    )

    lower = "simultaneous_lower"
    upper = "simultaneous_upper"
    gates = {
        "learned_graph_and_head_identification": (
            identification_lower >= ROUTING_IDENTIFICATION_RATE_MINIMUM
        ),
        "learned_vs_wrong_graph_composition_nll": (
            sensitivity_intervals["learned_routing_nll"][lower] > 0.0
            and sensitivity_intervals["learned_routing_nll"]["mean"]
            >= LEARNED_ROUTING_NLL_SESOI
        ),
        "factorized_graph_effect": (
            sensitivity_intervals["factorized_graph_nll"][lower] > 0.0
            and sensitivity_intervals["factorized_graph_nll"]["mean"]
            >= FACTORIZED_GRAPH_NLL_SESOI
        ),
        "generic_head_graph_effect": (
            sensitivity_intervals["generic_graph_nll"][lower] > 0.0
            and sensitivity_intervals["generic_graph_nll"]["mean"]
            >= GENERIC_GRAPH_NLL_SESOI
        ),
        "dense_head_graph_effect": (
            sensitivity_intervals["large_generic_graph_nll"][lower] > 0.0
            and sensitivity_intervals["large_generic_graph_nll"]["mean"]
            >= DENSE_GRAPH_NLL_SESOI
        ),
        "matched_head_predictive_equivalence": (
            abs(sensitivity_intervals["factorized_head_nll_correct_graph"]["mean"])
            <= MATCHED_HEAD_EQUIVALENCE_MARGIN
            and sensitivity_intervals["factorized_head_nll_correct_graph"][lower]
            >= -MATCHED_HEAD_EQUIVALENCE_MARGIN
            and sensitivity_intervals["factorized_head_nll_correct_graph"][upper]
            <= MATCHED_HEAD_EQUIVALENCE_MARGIN
        ),
        "learned_vs_wrong_graph_rollout_hamming": (
            sensitivity_intervals["rollout_hamming"][lower] > 0.0
            and sensitivity_intervals["rollout_hamming"]["mean"]
            >= ROLLOUT_HAMMING_SESOI
        ),
        "correct_vs_wrong_routing_criterion_brier": (
            sensitivity_intervals["criterion_brier"][lower] > 0.0
            and sensitivity_intervals["criterion_brier"]["mean"]
            >= CRITERION_BRIER_SESOI
        ),
        "outside_original_linear_family_noninferiority": (
            outside_sensitivity[upper] <= OUTSIDE_FAMILY_NONINFERIORITY_MARGIN
        ),
    }

    return {
        "status": "postreview_retrospective_multiplicity_sensitivity",
        "not_preregistered": True,
        "frozen_divisor": PRIMARY_INFERENTIAL_EFFECT_COUNT,
        "sensitivity_divisor": sensitivity_divisor,
        "frozen_contract_named_members": False,
        "postfreeze_named_gate_quantities": list(PRIMARY_INFERENTIAL_QUANTITIES),
        "reported_interval_quantity_count": len(REPORTED_EFFECT_NAMES),
        "bookkeeping": {
            "exact_duplicate_pair": list(EXACT_DUPLICATE_PAIR),
            "exact_duplicate_verified_worldwise": duplicate_verified,
            "sign_flipped_pair": list(SIGN_FLIPPED_PAIR),
            "sign_flip_verified_worldwise": sign_flip_verified,
            "deterministic_zero": DETERMINISTIC_ZERO,
            "deterministic_zero_verified_worldwise": zero_verified,
            "distinct_stochastic_quantity_count": 8,
            "informationally_distinct_stochastic_quantity_count": 7,
        },
        "sensitivity_critical": bonferroni_critical(sensitivity_divisor),
        "sensitivity_effect_intervals": sensitivity_intervals,
        "sensitivity_identification_wilson_lower": identification_lower,
        "sensitivity_outside_family_interval": outside_sensitivity,
        "sensitivity_gates": gates,
        "all_sensitivity_gates_passed": all(gates.values()),
    }
