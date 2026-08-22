"""Prospective A2 power calibration from development worlds only."""

from __future__ import annotations

from hashlib import sha256
from math import sqrt
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from .research_a2_development import (
    MISSPECIFICATION_SAMPLE_SIZES,
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


POWER_SEED_LABEL = "TSI-RESEARCH-A2-PROSPECTIVE-POWER-v1"
FAMILYWISE_ALPHA = 0.05
NLL_SESOI = 0.01
RECOVERY_SESOI = 0.10
NLL_EQUIVALENCE_MARGIN = 0.01
RECOVERY_EQUIVALENCE_MARGIN = 0.05
SCOPE_NLL_SESOI = 0.10
SCOPE_ACCURACY_SESOI = 0.10
SCOPE_NLL_EQUIVALENCE_MARGIN = 0.01
SCOPE_ACCURACY_EQUIVALENCE_MARGIN = 0.025
SCOPE_SAMPLE_SIZE = MISSPECIFICATION_SAMPLE_SIZES[-1]
WIDTH_ENDPOINT_COUNT = len(WIDTH_POSITION_COUNTS) * len(WIDTH_SAMPLE_SIZES) * 2
NOISE_ENDPOINT_COUNT = len(NOISE_PROBABILITIES) * len(NOISE_SAMPLE_SIZES) * 2
SCOPE_ENDPOINT_COUNT = 6


def _axis_matrix(
    report: Mapping[str, object], axis_name: str, grid: Sequence[tuple[object, object]]
) -> tuple[np.ndarray, np.ndarray]:
    axes = report.get("axes")
    if not isinstance(axes, Mapping) or not isinstance(axes.get(axis_name), Mapping):
        raise TypeError(f"development report has no {axis_name} axis")
    records = axes[axis_name].get("records")  # type: ignore[union-attr]
    if not isinstance(records, Sequence):
        raise TypeError(f"{axis_name} has no records")
    by_world: dict[int, dict[tuple[object, object], Mapping[str, object]]] = {}
    families: dict[int, tuple[str, str]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("axis record is not a mapping")
        world = int(raw["world_index"])
        if axis_name == "candidate_width":
            key = (int(raw["position_count"]), int(raw["sample_size"]))
        else:
            key = (float(raw["train_noise_probability"]), int(raw["sample_size"]))
        by_world.setdefault(world, {})[key] = raw
        pair = raw.get("family_pair")
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise TypeError("axis record has no family-pair stratum")
        families[world] = (str(pair[0]), str(pair[1]))
    matrices = []
    pair_order = sorted(set(families.values()))
    if len(pair_order) != 9:
        raise ValueError("matched efficiency axes require nine family-pair strata")
    strata = []
    for world in sorted(by_world):
        if any(key not in by_world[world] for key in grid):
            raise ValueError(f"{axis_name} world does not cover the frozen grid")
        row = []
        for key in grid:
            record = by_world[world][key]
            row.extend(
                (
                    float(record["generic_minus_typed_nll"]),
                    float(record["typed_minus_generic_exact"]),
                )
            )
        matrices.append(row)
        strata.append(pair_order.index(families[world]))
    stratum_array = np.asarray(strata, dtype=np.int64)
    counts = [int(np.sum(stratum_array == index)) for index in range(9)]
    if len(set(counts)) != 1:
        raise ValueError("development family-pair strata are not balanced")
    return np.asarray(matrices, dtype=float), stratum_array


def _draw_stratified(
    matrix: np.ndarray,
    strata: np.ndarray,
    stratum_count: int,
    target_count: int,
    batch: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_count % stratum_count:
        raise ValueError("target world count does not balance the strata")
    per_stratum = target_count // stratum_count
    draws = []
    for stratum in range(stratum_count):
        candidates = np.flatnonzero(strata == stratum)
        indices = rng.choice(candidates, size=(batch, per_stratum), replace=True)
        draws.append(matrix[indices])
    return np.concatenate(draws, axis=1)


def _efficiency_power(
    matrix: np.ndarray,
    strata: np.ndarray,
    *,
    target_count: int,
    group_count: int,
    sizes_per_group: int,
    endpoint_count: int,
    iterations: int,
    batch_size: int,
    rng: np.random.Generator,
    require_every_group: bool,
) -> dict[str, object]:
    critical = NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / (2.0 * endpoint_count))
    group_successes = np.zeros(group_count, dtype=np.int64)
    gate_successes = 0
    completed = 0
    while completed < iterations:
        current = min(batch_size, iterations - completed)
        sample = _draw_stratified(matrix, strata, 9, target_count, current, rng)
        means = np.mean(sample, axis=1)
        standard_errors = np.std(sample, axis=1, ddof=1) / sqrt(target_count)
        lower = means - critical * standard_errors
        reshaped_means = means.reshape(current, group_count, sizes_per_group, 2)
        reshaped_lower = lower.reshape(current, group_count, sizes_per_group, 2)
        advantage = (
            (reshaped_lower[..., 0] > 0.0)
            & (reshaped_means[..., 0] >= NLL_SESOI)
            & (reshaped_lower[..., 1] > 0.0)
            & (reshaped_means[..., 1] >= RECOVERY_SESOI)
        )
        by_group = np.any(advantage, axis=2)
        group_successes += np.sum(by_group, axis=0)
        gate = (
            np.all(by_group, axis=1)
            if require_every_group
            else np.any(by_group, axis=1)
        )
        gate_successes += int(np.sum(gate))
        completed += current
    return {
        "world_count": target_count,
        "critical": critical,
        "gate_power": gate_successes / iterations,
        "group_power": (group_successes / iterations).tolist(),
    }


def _scope_matrices(
    report: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axes = report.get("axes")
    if not isinstance(axes, Mapping) or not isinstance(
        axes.get("misspecification"), Mapping
    ):
        raise TypeError("development report has no misspecification axis")
    records = axes["misspecification"].get("records")  # type: ignore[union-attr]
    if not isinstance(records, Sequence):
        raise TypeError("misspecification axis has no records")
    selected = [
        row
        for row in records
        if isinstance(row, Mapping) and int(row["sample_size"]) == SCOPE_SAMPLE_SIZE
    ]
    conditions: dict[str, dict[int, Mapping[str, object]]] = {}
    for row in selected:
        conditions.setdefault(str(row["condition"]), {})[int(row["world_index"])] = row
    if set(conditions) != {MATCHED, TYPED_MISSPECIFIED, GENERIC_MISSPECIFIED}:
        raise ValueError("scope power requires all three conditions")

    matched_pairs = sorted(
        {
            tuple(str(value) for value in row["family_pair"])
            for row in conditions[MATCHED].values()
        }
    )
    if len(matched_pairs) != 9:
        raise ValueError("matched scope population requires nine strata")
    matched_matrix = []
    matched_strata = []
    for world, row in sorted(conditions[MATCHED].items()):
        del world
        matched_matrix.append(
            [
                float(row["generic_minus_typed_nll"]),
                float(row["typed_minus_generic_center_accuracy"]),
            ]
        )
        pair = tuple(str(value) for value in row["family_pair"])
        matched_strata.append(matched_pairs.index(pair))

    paired_matrix = []
    paired_strata = []
    special_pairs = []
    for world, cubic in sorted(conditions[TYPED_MISSPECIFIED].items()):
        quadratic = conditions[GENERIC_MISSPECIFIED].get(world)
        if quadratic is None:
            raise ValueError("directional misspecification worlds are not paired")
        cubic_pair = tuple(str(value) for value in cubic["family_pair"])
        quadratic_pair = tuple(str(value) for value in quadratic["family_pair"])
        normalized_cubic = tuple(
            "special" if value == "cubic_target" else value for value in cubic_pair
        )
        normalized_quadratic = tuple(
            "special" if value == "quadratic_target" else value
            for value in quadratic_pair
        )
        if normalized_cubic != normalized_quadratic:
            raise ValueError("paired misspecification family templates differ")
        special_pairs.append(normalized_cubic)
        paired_matrix.append(
            [
                float(cubic["generic_minus_typed_nll"]),
                float(cubic["typed_minus_generic_center_accuracy"]),
                float(quadratic["generic_minus_typed_nll"]),
                float(quadratic["typed_minus_generic_center_accuracy"]),
            ]
        )
    special_order = sorted(set(special_pairs))
    if len(special_order) != 5:
        raise ValueError("directional scope population requires five special strata")
    paired_strata = [special_order.index(pair) for pair in special_pairs]
    return (
        np.asarray(matched_matrix, dtype=float),
        np.asarray(matched_strata, dtype=np.int64),
        np.asarray(paired_matrix, dtype=float),
        np.asarray(paired_strata, dtype=np.int64),
    )


def _scope_power(
    report: Mapping[str, object],
    *,
    target_count: int,
    iterations: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    matched, matched_strata, paired, paired_strata = _scope_matrices(report)
    critical = NormalDist().inv_cdf(
        1.0 - FAMILYWISE_ALPHA / (2.0 * SCOPE_ENDPOINT_COUNT)
    )
    component_successes = np.zeros(3, dtype=np.int64)
    gate_successes = 0
    completed = 0
    while completed < iterations:
        current = min(batch_size, iterations - completed)
        matched_sample = _draw_stratified(
            matched, matched_strata, 9, target_count, current, rng
        )
        paired_sample = _draw_stratified(
            paired, paired_strata, 5, target_count, current, rng
        )
        sample = np.concatenate((matched_sample, paired_sample), axis=2)
        means = np.mean(sample, axis=1)
        standard_errors = np.std(sample, axis=1, ddof=1) / sqrt(target_count)
        lower = means - critical * standard_errors
        upper = means + critical * standard_errors
        matched_equivalence = (
            (lower[:, 0] >= -SCOPE_NLL_EQUIVALENCE_MARGIN)
            & (upper[:, 0] <= SCOPE_NLL_EQUIVALENCE_MARGIN)
            & (lower[:, 1] >= -SCOPE_ACCURACY_EQUIVALENCE_MARGIN)
            & (upper[:, 1] <= SCOPE_ACCURACY_EQUIVALENCE_MARGIN)
        )
        typed_misspecified_direction = (
            (upper[:, 2] < 0.0)
            & (means[:, 2] <= -SCOPE_NLL_SESOI)
            & (upper[:, 3] < 0.0)
            & (means[:, 3] <= -SCOPE_ACCURACY_SESOI)
        )
        generic_misspecified_direction = (
            (lower[:, 4] > 0.0)
            & (means[:, 4] >= SCOPE_NLL_SESOI)
            & (lower[:, 5] > 0.0)
            & (means[:, 5] >= SCOPE_ACCURACY_SESOI)
        )
        components = np.column_stack(
            (
                matched_equivalence,
                typed_misspecified_direction,
                generic_misspecified_direction,
            )
        )
        component_successes += np.sum(components, axis=0)
        gate_successes += int(np.sum(np.all(components, axis=1)))
        completed += current
    return {
        "world_count_per_condition": target_count,
        "critical": critical,
        "scope_gate_power": gate_successes / iterations,
        "matched_equivalence_power": component_successes[0] / iterations,
        "typed_misspecified_direction_power": component_successes[1] / iterations,
        "generic_misspecified_direction_power": component_successes[2] / iterations,
    }


def estimate_a2_prospective_power(
    report: Mapping[str, object],
    *,
    world_counts: Sequence[int] = (90, 135, 180, 270),
    iterations: int = 20_000,
    batch_size: int = 500,
) -> dict[str, object]:
    """Estimate A2 operating characteristics without confirmatory data."""

    if report.get("status") != "development_only_not_confirmatory":
        raise ValueError("power input is not an A2 development report")
    width_grid = tuple(
        (width, size) for width in WIDTH_POSITION_COUNTS for size in WIDTH_SAMPLE_SIZES
    )
    noise_grid = tuple(
        (noise, size) for noise in NOISE_PROBABILITIES for size in NOISE_SAMPLE_SIZES
    )
    width_matrix, width_strata = _axis_matrix(report, "candidate_width", width_grid)
    noise_matrix, noise_strata = _axis_matrix(report, "training_noise", noise_grid)
    seed = int.from_bytes(sha256(POWER_SEED_LABEL.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    curve = []
    for world_count in world_counts:
        if world_count < 45 or world_count % 45:
            raise ValueError("A2 world counts must be multiples of 45 and at least 45")
        width = _efficiency_power(
            width_matrix,
            width_strata,
            target_count=world_count,
            group_count=len(WIDTH_POSITION_COUNTS),
            sizes_per_group=len(WIDTH_SAMPLE_SIZES),
            endpoint_count=WIDTH_ENDPOINT_COUNT,
            iterations=iterations,
            batch_size=batch_size,
            rng=rng,
            require_every_group=True,
        )
        noise = _efficiency_power(
            noise_matrix,
            noise_strata,
            target_count=world_count,
            group_count=len(NOISE_PROBABILITIES),
            sizes_per_group=len(NOISE_SAMPLE_SIZES),
            endpoint_count=NOISE_ENDPOINT_COUNT,
            iterations=iterations,
            batch_size=batch_size,
            rng=rng,
            require_every_group=False,
        )
        scope = _scope_power(
            report,
            target_count=world_count,
            iterations=iterations,
            batch_size=batch_size,
            rng=rng,
        )
        curve.append(
            {
                "world_count": world_count,
                "matched_worlds_per_family_pair": world_count // 9,
                "misspecified_worlds_per_special_pair": world_count // 5,
                "width_all_catalogs_any_joint_advantage_power": width["gate_power"],
                "width_power_by_position_count": dict(
                    zip(
                        map(str, WIDTH_POSITION_COUNTS),
                        width["group_power"],
                        strict=True,
                    )
                ),
                "noise_any_joint_advantage_power": noise["gate_power"],
                "noise_power_by_probability": dict(
                    zip(
                        map(str, NOISE_PROBABILITIES),
                        noise["group_power"],
                        strict=True,
                    )
                ),
                **scope,
            }
        )
    selected_world_count = 135
    selected = next(
        item for item in curve if item["world_count"] == selected_world_count
    )
    return {
        "status": "prospective_power_from_a2_development_only",
        "uses_confirmatory_results": False,
        "confirmatory_seed_created": False,
        "development_seed_label": report.get("seed_label"),
        "power_seed_label": POWER_SEED_LABEL,
        "bootstrap_iterations": iterations,
        "familywise_alpha_per_axis": FAMILYWISE_ALPHA,
        "multiplicity": {
            "candidate_width_endpoint_count": WIDTH_ENDPOINT_COUNT,
            "training_noise_endpoint_count": NOISE_ENDPOINT_COUNT,
            "scope_endpoint_count": SCOPE_ENDPOINT_COUNT,
            "families_are_separate": True,
        },
        "thresholds": {
            "efficiency_nll_sesoi": NLL_SESOI,
            "efficiency_recovery_sesoi": RECOVERY_SESOI,
            "efficiency_nll_equivalence_margin": NLL_EQUIVALENCE_MARGIN,
            "efficiency_recovery_equivalence_margin": RECOVERY_EQUIVALENCE_MARGIN,
            "scope_nll_sesoi": SCOPE_NLL_SESOI,
            "scope_accuracy_sesoi": SCOPE_ACCURACY_SESOI,
            "scope_nll_equivalence_margin": SCOPE_NLL_EQUIVALENCE_MARGIN,
            "scope_accuracy_equivalence_margin": SCOPE_ACCURACY_EQUIVALENCE_MARGIN,
        },
        "scope_sample_size": SCOPE_SAMPLE_SIZE,
        "power_curve": curve,
        "selected_world_count_per_axis_or_scope_condition": selected_world_count,
        "selected_operating_characteristics": selected,
        "selection_reason": (
            "135 is the smallest candidate at or above the pre-A1 minimum of 126 "
            "that exactly balances both nine matched family pairs and five special-family pairs."
        ),
        "limitations": [
            "Power is a stratified bootstrap of 36 matched and 45 scope development worlds.",
            "The width gate requires an advantage somewhere on the fixed n-grid for every width.",
            "The noise gate requires an advantage somewhere on the fixed probability-by-n grid; it does not require advantage at p=0.80.",
            "The scope gate is a falsification boundary and cannot rescue an efficiency failure.",
        ],
    }
