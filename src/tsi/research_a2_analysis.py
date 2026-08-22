"""Frozen-candidate world-level analysis for Research A2."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from .research_a2_contract import (
    FAMILYWISE_ALPHA,
    NOISE_ADVANTAGE_PROBABILITIES,
    NOISE_ENDPOINTS,
    SCOPE_ENDPOINTS,
    WIDTH_ENDPOINTS,
)
from .research_a2_development import (
    NOISE_PROBABILITIES,
    NOISE_SAMPLE_SIZES,
    WIDTH_SAMPLE_SIZES,
)
from .research_a2_features import WIDTH_POSITION_COUNTS
from .research_a2_populations import (
    GENERIC_MISSPECIFIED,
    MATCHED,
    TYPED_MISSPECIFIED,
)
from .research_a2_power import (
    NLL_EQUIVALENCE_MARGIN,
    NLL_SESOI,
    RECOVERY_EQUIVALENCE_MARGIN,
    RECOVERY_SESOI,
    SCOPE_ACCURACY_EQUIVALENCE_MARGIN,
    SCOPE_ACCURACY_SESOI,
    SCOPE_NLL_EQUIVALENCE_MARGIN,
    SCOPE_NLL_SESOI,
    SCOPE_SAMPLE_SIZE,
)


def _interval(values: Sequence[float], endpoint_count: int) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("A2 analysis requires at least two independent worlds")
    array = np.asarray(values, dtype=float)
    standard_error = float(np.std(array, ddof=1) / sqrt(len(array)))
    critical = NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / (2.0 * endpoint_count))
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


def _record_groups(
    records: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for record in records:
        key = tuple(record[name] for name in keys)
        groups.setdefault(key, []).append(record)
    return groups


def _efficiency_axis(
    records: Sequence[Mapping[str, object]],
    *,
    group_key: str,
    group_values: Sequence[object],
    sample_sizes: Sequence[int],
    endpoint_count: int,
    required_advantage_groups: Sequence[object],
) -> dict[str, object]:
    required = tuple(required_advantage_groups)
    if not required or any(value not in group_values for value in required):
        raise ValueError("required advantage groups must be a nonempty grid subset")
    groups = _record_groups(records, (group_key, "sample_size"))
    summaries = []
    advantage_by_group = {}
    equivalence_by_group = {}
    world_counts = set()
    for group_value in group_values:
        advantage_sizes = []
        equivalence_sizes = []
        for sample_size in sample_sizes:
            rows = groups.get((group_value, sample_size))
            if not rows:
                raise ValueError(
                    f"missing {group_key}={group_value}, n={sample_size} rows"
                )
            world_counts.add(len(rows))
            nll = _interval(
                [float(row["generic_minus_typed_nll"]) for row in rows],
                endpoint_count,
            )
            recovery = _interval(
                [float(row["typed_minus_generic_exact"]) for row in rows],
                endpoint_count,
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
            if advantage:
                advantage_sizes.append(sample_size)
            if equivalence:
                equivalence_sizes.append(sample_size)
            summaries.append(
                {
                    group_key: group_value,
                    "sample_size": sample_size,
                    "generic_minus_typed_nll": nll,
                    "typed_minus_generic_exact_recovery": recovery,
                    "typed_exact_rate": float(
                        np.mean([bool(row["typed_exact"]) for row in rows])
                    ),
                    "generic_exact_rate": float(
                        np.mean([bool(row["generic_exact"]) for row in rows])
                    ),
                    "joint_advantage": advantage,
                    "joint_equivalence": equivalence,
                }
            )
        advantage_by_group[str(group_value)] = advantage_sizes
        equivalence_by_group[str(group_value)] = equivalence_sizes
    if len(world_counts) != 1:
        raise ValueError("efficiency endpoint groups have unequal world counts")
    group_gate_values = {
        str(value): bool(advantage_by_group[str(value)]) for value in group_values
    }
    gate = all(group_gate_values[str(value)] for value in required)
    return {
        "world_count": world_counts.pop(),
        "bonferroni_endpoint_count": endpoint_count,
        "summaries": summaries,
        "joint_advantage_sample_sizes_by_group": advantage_by_group,
        "joint_equivalence_sample_sizes_by_group": equivalence_by_group,
        "required_advantage_groups": list(required),
        "descriptive_only_groups": [
            value for value in group_values if value not in required
        ],
        "group_gate_passed": group_gate_values,
        "gate_rule": "at_least_one_joint_advantage_for_each_required_group",
        "gate_passed": gate,
    }


def _scope_axis(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = [row for row in records if int(row["sample_size"]) == SCOPE_SAMPLE_SIZE]
    groups = _record_groups(selected, ("condition",))
    summaries = {}
    world_counts = set()
    for condition in (MATCHED, TYPED_MISSPECIFIED, GENERIC_MISSPECIFIED):
        rows = groups.get((condition,))
        if not rows:
            raise ValueError(f"scope condition is absent: {condition}")
        world_counts.add(len(rows))
        summaries[condition] = {
            "generic_minus_typed_nll": _interval(
                [float(row["generic_minus_typed_nll"]) for row in rows],
                len(SCOPE_ENDPOINTS),
            ),
            "typed_minus_generic_center_accuracy": _interval(
                [float(row["typed_minus_generic_center_accuracy"]) for row in rows],
                len(SCOPE_ENDPOINTS),
            ),
            "typed_exact_rate": float(
                np.mean([bool(row["typed_exact"]) for row in rows])
            ),
            "generic_exact_rate": float(
                np.mean([bool(row["generic_exact"]) for row in rows])
            ),
        }
    if len(world_counts) != 1:
        raise ValueError("scope conditions have unequal world counts")
    matched_nll = summaries[MATCHED]["generic_minus_typed_nll"]
    matched_accuracy = summaries[MATCHED]["typed_minus_generic_center_accuracy"]
    matched_equivalence = (
        matched_nll["simultaneous_lower"] >= -SCOPE_NLL_EQUIVALENCE_MARGIN
        and matched_nll["simultaneous_upper"] <= SCOPE_NLL_EQUIVALENCE_MARGIN
        and matched_accuracy["simultaneous_lower"] >= -SCOPE_ACCURACY_EQUIVALENCE_MARGIN
        and matched_accuracy["simultaneous_upper"] <= SCOPE_ACCURACY_EQUIVALENCE_MARGIN
    )
    typed_nll = summaries[TYPED_MISSPECIFIED]["generic_minus_typed_nll"]
    typed_accuracy = summaries[TYPED_MISSPECIFIED][
        "typed_minus_generic_center_accuracy"
    ]
    typed_direction = (
        typed_nll["simultaneous_upper"] < 0.0
        and typed_nll["mean"] <= -SCOPE_NLL_SESOI
        and typed_accuracy["simultaneous_upper"] < 0.0
        and typed_accuracy["mean"] <= -SCOPE_ACCURACY_SESOI
    )
    generic_nll = summaries[GENERIC_MISSPECIFIED]["generic_minus_typed_nll"]
    generic_accuracy = summaries[GENERIC_MISSPECIFIED][
        "typed_minus_generic_center_accuracy"
    ]
    generic_direction = (
        generic_nll["simultaneous_lower"] > 0.0
        and generic_nll["mean"] >= SCOPE_NLL_SESOI
        and generic_accuracy["simultaneous_lower"] > 0.0
        and generic_accuracy["mean"] >= SCOPE_ACCURACY_SESOI
    )
    gates = {
        "matched_equivalence": matched_equivalence,
        "typed_misspecified_favors_generic": typed_direction,
        "generic_misspecified_favors_typed": generic_direction,
    }
    return {
        "world_count_per_condition": world_counts.pop(),
        "sample_size": SCOPE_SAMPLE_SIZE,
        "bonferroni_endpoint_count": len(SCOPE_ENDPOINTS),
        "summaries": summaries,
        "gates": gates,
        "scope_gate_passed": all(gates.values()),
        "claim_boundary": "scope_and_falsification_only_cannot_rescue_efficiency",
    }


def analyze_a2_axes(
    axes: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    width = _efficiency_axis(
        axes["candidate_width"],
        group_key="position_count",
        group_values=WIDTH_POSITION_COUNTS,
        sample_sizes=WIDTH_SAMPLE_SIZES,
        endpoint_count=len(WIDTH_ENDPOINTS),
        required_advantage_groups=WIDTH_POSITION_COUNTS,
    )
    noise = _efficiency_axis(
        axes["training_noise"],
        group_key="train_noise_probability",
        group_values=NOISE_PROBABILITIES,
        sample_sizes=NOISE_SAMPLE_SIZES,
        endpoint_count=len(NOISE_ENDPOINTS),
        required_advantage_groups=NOISE_ADVANTAGE_PROBABILITIES,
    )
    scope = _scope_axis(axes["misspecification"])
    return {
        "candidate_width": width,
        "training_noise": noise,
        "misspecification_scope": scope,
        "efficiency_gate_passed": bool(width["gate_passed"])
        and bool(noise["gate_passed"]),
        "all_a2_gates_passed": bool(width["gate_passed"])
        and bool(noise["gate_passed"])
        and bool(scope["scope_gate_passed"]),
        "scope_cannot_change_efficiency_gate": True,
    }
