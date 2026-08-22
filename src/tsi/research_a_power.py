"""Prospective power calibration from Research A development worlds only."""

from __future__ import annotations

from hashlib import sha256
from math import sqrt
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np


POWER_SEED_LABEL = "TSI-RESEARCH-A-PROSPECTIVE-POWER-v1"
PRIMARY_SAMPLE_SIZES = (5, 10, 15, 20, 25, 30, 40, 50)
NLL_SESOI = 0.01
NLL_EQUIVALENCE_MARGIN = 0.01
RECOVERY_SESOI = 0.10
RECOVERY_EQUIVALENCE_MARGIN = 0.05
FAMILYWISE_ALPHA = 0.05
PRIMARY_ENDPOINT_COUNT = 2 * len(PRIMARY_SAMPLE_SIZES)


def _development_matrix(
    report: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    rows = report.get("rows")
    if not isinstance(rows, Sequence):
        raise TypeError("development report has no row sequence")
    matrices = []
    strata = []
    family_pairs: dict[tuple[str, str], int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("development row must be a mapping")
        estimates = row.get("estimates")
        families = row.get("families")
        if not isinstance(estimates, Sequence) or not isinstance(families, Sequence):
            raise TypeError("development row is incomplete")
        by_size = {
            int(item["sample_size"]): item
            for item in estimates
            if isinstance(item, Mapping)
        }
        if any(size not in by_size for size in PRIMARY_SAMPLE_SIZES):
            raise ValueError("development report does not cover the primary grid")
        nll = [float(by_size[size]["generic_minus_typed_nll"]) for size in PRIMARY_SAMPLE_SIZES]
        recovery = [float(by_size[size]["typed_minus_generic_exact"]) for size in PRIMARY_SAMPLE_SIZES]
        matrices.append(nll + recovery)
        pair = (str(families[0]), str(families[1]))
        if pair not in family_pairs:
            family_pairs[pair] = len(family_pairs)
        strata.append(family_pairs[pair])
    if len(family_pairs) != 9:
        raise ValueError("power calibration requires all nine family-pair strata")
    matrix = np.asarray(matrices, dtype=float)
    stratum_array = np.asarray(strata, dtype=np.int64)
    counts = [int(np.sum(stratum_array == index)) for index in range(9)]
    if len(set(counts)) != 1:
        raise ValueError("development family-pair strata must be balanced")
    return matrix, stratum_array


def estimate_prospective_power(
    report: Mapping[str, object],
    *,
    world_counts: Sequence[int] = (90, 126, 180, 270),
    iterations: int = 20_000,
    batch_size: int = 500,
) -> dict[str, object]:
    """Stratified bootstrap power without reading confirmatory data."""

    matrix, strata = _development_matrix(report)
    seed = int.from_bytes(sha256(POWER_SEED_LABEL.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    critical = NormalDist().inv_cdf(
        1.0 - FAMILYWISE_ALPHA / (2.0 * PRIMARY_ENDPOINT_COUNT)
    )
    curve = []
    grid_count = len(PRIMARY_SAMPLE_SIZES)

    for world_count in world_counts:
        if world_count < 18 or world_count % 9:
            raise ValueError("world counts must be multiples of nine and at least 18")
        per_stratum = world_count // 9
        successes = np.zeros(3, dtype=np.int64)
        advantage_counts = np.zeros(grid_count, dtype=np.int64)
        equivalence_counts = np.zeros(grid_count, dtype=np.int64)
        completed = 0
        while completed < iterations:
            current = min(batch_size, iterations - completed)
            draws = []
            for stratum in range(9):
                candidates = np.flatnonzero(strata == stratum)
                indices = rng.choice(
                    candidates, size=(current, per_stratum), replace=True
                )
                draws.append(matrix[indices])
            sample = np.concatenate(draws, axis=1)
            means = np.mean(sample, axis=1)
            standard_errors = np.std(sample, axis=1, ddof=1) / sqrt(world_count)
            lower = means - critical * standard_errors
            upper = means + critical * standard_errors
            nll_mean, recovery_mean = means[:, :grid_count], means[:, grid_count:]
            nll_lower, recovery_lower = lower[:, :grid_count], lower[:, grid_count:]
            nll_upper, recovery_upper = upper[:, :grid_count], upper[:, grid_count:]
            advantage = (
                (nll_lower > 0.0)
                & (nll_mean >= NLL_SESOI)
                & (recovery_lower > 0.0)
                & (recovery_mean >= RECOVERY_SESOI)
            )
            equivalence = (
                (nll_lower >= -NLL_EQUIVALENCE_MARGIN)
                & (nll_upper <= NLL_EQUIVALENCE_MARGIN)
                & (recovery_lower >= -RECOVERY_EQUIVALENCE_MARGIN)
                & (recovery_upper <= RECOVERY_EQUIVALENCE_MARGIN)
            )
            any_advantage = np.any(advantage, axis=1)
            any_equivalence = np.any(equivalence, axis=1)
            ordered_transition = np.zeros(current, dtype=bool)
            for left in range(grid_count - 1):
                ordered_transition |= advantage[:, left] & np.any(
                    equivalence[:, left + 1 :], axis=1
                )
            successes += np.asarray(
                [
                    np.sum(any_advantage),
                    np.sum(any_equivalence),
                    np.sum(ordered_transition),
                ],
                dtype=np.int64,
            )
            advantage_counts += np.sum(advantage, axis=0)
            equivalence_counts += np.sum(equivalence, axis=0)
            completed += current

        transition_power = successes[2] / iterations
        mc_se = sqrt(transition_power * (1.0 - transition_power) / iterations)
        curve.append(
            {
                "world_count": world_count,
                "per_family_pair_stratum": per_stratum,
                "any_joint_advantage_power": successes[0] / iterations,
                "any_joint_equivalence_power": successes[1] / iterations,
                "ordered_transition_power": transition_power,
                "ordered_transition_monte_carlo_se": mc_se,
                "joint_advantage_probability_by_n": dict(
                    zip(
                        map(str, PRIMARY_SAMPLE_SIZES),
                        (advantage_counts / iterations).tolist(),
                        strict=True,
                    )
                ),
                "joint_equivalence_probability_by_n": dict(
                    zip(
                        map(str, PRIMARY_SAMPLE_SIZES),
                        (equivalence_counts / iterations).tolist(),
                        strict=True,
                    )
                ),
            }
        )

    selected = next(item for item in curve if item["world_count"] == 126)
    return {
        "status": "prospective_power_from_development_only",
        "uses_confirmatory_results": False,
        "development_seed_label": report.get("seed_label"),
        "power_seed_label": POWER_SEED_LABEL,
        "bootstrap_iterations": iterations,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bonferroni_endpoint_count": PRIMARY_ENDPOINT_COUNT,
        "critical": critical,
        "primary_sample_sizes": list(PRIMARY_SAMPLE_SIZES),
        "thresholds": {
            "nll_sesoi": NLL_SESOI,
            "nll_equivalence_margin": NLL_EQUIVALENCE_MARGIN,
            "recovery_sesoi": RECOVERY_SESOI,
            "recovery_equivalence_margin": RECOVERY_EQUIVALENCE_MARGIN,
        },
        "power_curve": curve,
        "selected_world_count": 126,
        "selected_ordered_transition_power": selected["ordered_transition_power"],
        "limitations": [
            "Power is a stratified bootstrap of 36 development worlds.",
            "Normal simultaneous intervals are the frozen analysis target.",
            "The development grid was expanded below n=50 after a documented ceiling result.",
        ],
    }
