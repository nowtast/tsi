"""Frozen-candidate world-level analysis for Research A1."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from .research_a_contract import (
    FAMILYWISE_ALPHA,
    NLL_EQUIVALENCE_MARGIN,
    NLL_SESOI,
    PRIMARY_ENDPOINTS,
    PRIMARY_SAMPLE_SIZES,
    RECOVERY_EQUIVALENCE_MARGIN,
    RECOVERY_SESOI,
)


def _interval(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    standard_error = float(np.std(array, ddof=1) / sqrt(len(array)))
    critical = NormalDist().inv_cdf(
        1.0 - FAMILYWISE_ALPHA / (2.0 * len(PRIMARY_ENDPOINTS))
    )
    center = float(np.mean(array))
    return {
        "mean": center,
        "world_sd": float(np.std(array, ddof=1)),
        "standard_error": standard_error,
        "simultaneous_lower": center - critical * standard_error,
        "simultaneous_upper": center + critical * standard_error,
        "critical": critical,
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def analyze_confirmatory_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) < 2:
        raise ValueError("confirmatory analysis requires at least two worlds")
    summaries = []
    for position, sample_size in enumerate(PRIMARY_SAMPLE_SIZES):
        records = []
        for row in rows:
            estimates = row.get("estimates")
            if not isinstance(estimates, Sequence):
                raise TypeError("world row has no estimate sequence")
            record = estimates[position]
            if int(record["sample_size"]) != sample_size:  # type: ignore[index]
                raise ValueError("world estimate grid is misaligned")
            records.append(record)
        nll = _interval(
            [float(record["generic_minus_typed_nll"]) for record in records]  # type: ignore[index]
        )
        recovery = _interval(
            [float(record["typed_minus_generic_exact"]) for record in records]  # type: ignore[index]
        )
        advantage = (
            nll["simultaneous_lower"] > 0.0
            and nll["mean"] >= NLL_SESOI
            and recovery["simultaneous_lower"] > 0.0
            and recovery["mean"] >= RECOVERY_SESOI
        )
        equivalence = (
            nll["simultaneous_lower"] >= -NLL_EQUIVALENCE_MARGIN
            and nll["simultaneous_upper"] <= NLL_EQUIVALENCE_MARGIN
            and recovery["simultaneous_lower"] >= -RECOVERY_EQUIVALENCE_MARGIN
            and recovery["simultaneous_upper"] <= RECOVERY_EQUIVALENCE_MARGIN
        )
        notation_maximum = max(
            abs(float(record["typed_minus_isomorphic_nll"]))  # type: ignore[index]
            for record in records
        )
        summaries.append(
            {
                "sample_size": sample_size,
                "generic_minus_typed_nll": nll,
                "typed_minus_generic_exact_recovery": recovery,
                "typed_exact_rate": float(
                    np.mean([bool(record["typed_exact"]) for record in records])  # type: ignore[index]
                ),
                "generic_exact_rate": float(
                    np.mean([bool(record["generic_exact"]) for record in records])  # type: ignore[index]
                ),
                "maximum_absolute_typed_isomorphic_nll_difference": notation_maximum,
                "joint_advantage": advantage,
                "joint_equivalence": equivalence,
            }
        )

    advantage_sizes = [
        item["sample_size"] for item in summaries if item["joint_advantage"]
    ]
    transition = None
    if advantage_sizes:
        lower = max(advantage_sizes)
        later_equivalence = [
            item["sample_size"]
            for item in summaries
            if item["joint_equivalence"] and item["sample_size"] > lower
        ]
        if later_equivalence:
            transition = {"last_joint_advantage_n": lower, "first_later_equivalence_n": min(later_equivalence)}
    notation_invariant = all(
        item["maximum_absolute_typed_isomorphic_nll_difference"] == 0.0
        for item in summaries
    )
    return {
        "world_count": len(rows),
        "bonferroni_divisor": len(PRIMARY_ENDPOINTS),
        "sample_size_summaries": summaries,
        "notation_invariant_passed": notation_invariant,
        "efficiency_advantage_detected": bool(advantage_sizes),
        "transition_band": transition,
        "a1_supported": notation_invariant and transition is not None,
    }
