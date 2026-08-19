"""Retrospective sample-size justification from the frozen development worlds."""

from __future__ import annotations

from hashlib import sha256
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
    ROLLOUT_HAMMING_SESOI,
    ROUTING_IDENTIFICATION_RATE_MINIMUM,
)


POWER_SEED = "TSI-P34-retrospective-design-justification-v1"
POSITIVE_EFFECTS = (
    "learned_routing_nll",
    "factorized_graph_nll",
    "generic_graph_nll",
    "large_generic_graph_nll",
    "rollout_hamming",
    "criterion_brier",
)
SESOI = np.asarray(
    [
        LEARNED_ROUTING_NLL_SESOI,
        FACTORIZED_GRAPH_NLL_SESOI,
        GENERIC_GRAPH_NLL_SESOI,
        DENSE_GRAPH_NLL_SESOI,
        ROLLOUT_HAMMING_SESOI,
        CRITERION_BRIER_SESOI,
    ],
    dtype=float,
)


def _critical(divisor: int) -> float:
    return NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / (2.0 * divisor))


def _wilson_lower(successes: np.ndarray, count: int, divisor: int) -> np.ndarray:
    z = _critical(divisor)
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / count
        + z * z / (4.0 * count * count)
    )
    return (center - radius) / denominator


def _mean_and_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(values, axis=1)
    standard_errors = np.std(values, axis=1, ddof=1) / sqrt(values.shape[1])
    return means, standard_errors


def _development_arrays(
    development: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    analysis = development["analysis"]
    rows = development["rows"]
    if not isinstance(analysis, Mapping) or not isinstance(rows, Sequence):
        raise TypeError("development report is missing analysis or rows")
    effects = analysis["world_effects"]
    if not isinstance(effects, Sequence) or len(effects) != len(rows):
        raise ValueError("development rows and effects are misaligned")
    matrix = np.asarray(
        [[float(effect[name]) for name in POSITIVE_EFFECTS] for effect in effects],
        dtype=float,
    )
    identification = np.asarray(
        [
            bool(row["graph_exact"]) and bool(row["head_exact"])  # type: ignore[index]
            for row in rows
        ],
        dtype=bool,
    )
    outside = np.asarray(
        [bool(row["outside_original_linear_family"]) for row in rows],  # type: ignore[index]
        dtype=bool,
    )
    learned_nll = np.asarray(
        [float(effect["learned_composition_nll"]) for effect in effects],
        dtype=float,
    )
    packed = np.column_stack((matrix, identification.astype(float), learned_nll))
    return packed[~outside], packed[outside], matrix


def estimate_retrospective_power(
    development: Mapping[str, object],
    *,
    world_counts: Sequence[int] = (60, 90, 120, 150),
    iterations: int = 20_000,
    divisor: int = 8,
    batch_size: int = 500,
) -> dict[str, object]:
    """Bootstrap all gate decisions using only the 24 development worlds."""

    if iterations <= 0 or batch_size <= 0:
        raise ValueError("iterations and batch size must be positive")
    original, outside, matrix = _development_arrays(development)
    if len(original) < 2 or len(outside) < 2:
        raise ValueError("both development strata require at least two worlds")

    seed = int.from_bytes(sha256(POWER_SEED.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    curve: list[dict[str, object]] = []
    z = _critical(divisor)

    for world_count in world_counts:
        if world_count < 6 or world_count % 3:
            raise ValueError("world counts must be multiples of three and at least six")
        original_count = world_count // 3
        outside_count = world_count - original_count
        conjunctive_successes = 0
        gate_successes = np.zeros(9, dtype=np.int64)
        completed = 0
        while completed < iterations:
            current = min(batch_size, iterations - completed)
            original_indices = rng.integers(
                0, len(original), size=(current, original_count)
            )
            outside_indices = rng.integers(
                0, len(outside), size=(current, outside_count)
            )
            original_draws = original[original_indices]
            outside_draws = outside[outside_indices]
            all_draws = np.concatenate((original_draws, outside_draws), axis=1)

            effect_means, effect_se = _mean_and_se(all_draws[:, :, :6])
            positive_pass = (effect_means - z * effect_se > 0.0) & (
                effect_means >= SESOI
            )
            identification_successes = np.sum(all_draws[:, :, 6], axis=1)
            identification_pass = (
                _wilson_lower(identification_successes, world_count, divisor)
                >= ROUTING_IDENTIFICATION_RATE_MINIMUM
            )

            original_nll = original_draws[:, :, 7]
            outside_nll = outside_draws[:, :, 7]
            original_mean = np.mean(original_nll, axis=1)
            outside_mean = np.mean(outside_nll, axis=1)
            outside_se = np.sqrt(
                np.var(original_nll, axis=1, ddof=1) / original_count
                + np.var(outside_nll, axis=1, ddof=1) / outside_count
            )
            outside_pass = (
                outside_mean - original_mean + z * outside_se
                <= OUTSIDE_FAMILY_NONINFERIORITY_MARGIN
            )

            # The matched-head gate is analytic in this family and is zero in
            # every development world; retain it explicitly in the conjunction.
            matched_pass = np.ones(current, dtype=bool)
            ordered = np.column_stack(
                (
                    identification_pass,
                    positive_pass[:, 0],
                    positive_pass[:, 1],
                    positive_pass[:, 2],
                    positive_pass[:, 3],
                    matched_pass,
                    positive_pass[:, 4],
                    positive_pass[:, 5],
                    outside_pass,
                )
            )
            gate_successes += np.sum(ordered, axis=0)
            conjunctive_successes += int(np.sum(np.all(ordered, axis=1)))
            completed += current

        power = conjunctive_successes / iterations
        mc_se = sqrt(power * (1.0 - power) / iterations)
        curve.append(
            {
                "world_count": world_count,
                "original_linear_worlds": original_count,
                "outside_family_worlds": outside_count,
                "conjunctive_gate_power": power,
                "monte_carlo_standard_error": mc_se,
                "monte_carlo_95pct_interval": [
                    max(0.0, power - 1.959963984540054 * mc_se),
                    min(1.0, power + 1.959963984540054 * mc_se),
                ],
                "gate_pass_probabilities": dict(
                    zip(
                        (
                            "identification",
                            "learned_routing_nll",
                            "factorized_graph_nll",
                            "generic_graph_nll",
                            "large_generic_graph_nll",
                            "matched_head_equivalence",
                            "rollout_hamming",
                            "criterion_brier",
                            "outside_family_noninferiority",
                        ),
                        (gate_successes / iterations).tolist(),
                        strict=True,
                    )
                ),
            }
        )

    selected = next(item for item in curve if item["world_count"] == 120)
    observed_sd = dict(
        zip(POSITIVE_EFFECTS, np.std(matrix, axis=0, ddof=1).tolist(), strict=True)
    )
    return {
        "status": "retrospective_design_justification_not_preregistered",
        "uses_confirmatory_results": False,
        "development_world_count": len(original) + len(outside),
        "development_original_linear_worlds": len(original),
        "development_outside_family_worlds": len(outside),
        "bootstrap_iterations": iterations,
        "bootstrap_seed_label": POWER_SEED,
        "bonferroni_divisor": divisor,
        "critical": z,
        "positive_effect_names": list(POSITIVE_EFFECTS),
        "development_observed_sd": observed_sd,
        "power_curve": curve,
        "selected_world_count": 120,
        "selected_conjunctive_gate_power": selected["conjunctive_gate_power"],
        "limitations": [
            "This analysis was constructed after confirmation and is not a preregistration record.",
            "It bootstraps the 24 development worlds and therefore inherits their empirical support.",
            "The development-fitted criterion calibration is treated as fixed.",
            "All development worlds were identified correctly, so identification uncertainty may be understated.",
        ],
    }
