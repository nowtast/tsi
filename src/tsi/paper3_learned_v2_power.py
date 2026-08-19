"""Pre-sealed-style variance and power planning for v2 robustness endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np


SESOI = 0.05
FAMILYWISE_ALPHA = 0.05
POWER_TARGET = 0.90
POWER_ITERATIONS = 20_000
POWER_SEED = 20260807
PRIMARY_CONDITION = "gaussian_0.50"
PRIMARY_ENDPOINT = "source_logloss_degradation"
FAMILY_COUNT = 5


def _world_seed_values(
    rows: Sequence[dict[str, object]],
    *,
    condition: str,
    field: str,
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row.get("condition") == condition and field in row
    ]
    worlds = sorted({int(row["world_index"]) if "world_index" in row else int(row["world"]) for row in selected})
    seeds = sorted({int(row["seed"]) for row in selected})
    if not worlds or not seeds:
        raise ValueError("power input has no complete world/seed panel")
    values = np.full((len(worlds), len(seeds)), np.nan, dtype=np.float64)
    world_index = {world: index for index, world in enumerate(worlds)}
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    for row in selected:
        world = int(row["world_index"]) if "world_index" in row else int(row["world"])
        seed = int(row["seed"])
        values[world_index[world], seed_index[seed]] = float(row[field])
    if not np.isfinite(values).all():
        raise ValueError("power input has an incomplete world/seed panel")
    return values


def variance_decomposition(values: np.ndarray) -> dict[str, float]:
    panel = np.asarray(values, dtype=np.float64)
    if panel.ndim != 2 or panel.shape[0] < 2 or panel.shape[1] < 2:
        raise ValueError("variance decomposition requires at least two worlds and seeds")
    world_means = panel.mean(axis=1)
    within = float(np.mean((panel - world_means[:, None]) ** 2))
    between = float(np.var(world_means, ddof=1))
    total = float(np.var(panel, ddof=1))
    return {
        "world_count": int(panel.shape[0]),
        "seed_count": int(panel.shape[1]),
        "world_variance": between,
        "within_world_seed_variance": within,
        "total_variance": total,
        "world_mean": float(np.mean(world_means)),
    }


def simulate_one_sided_power(
    *,
    world_count: int,
    world_variance: float,
    seed_variance: float,
    effect: float = SESOI,
    family_count: int = FAMILY_COUNT,
    iterations: int = POWER_ITERATIONS,
    seed: int = POWER_SEED,
) -> float:
    if world_count <= 1 or family_count <= 0 or iterations <= 0:
        raise ValueError("power dimensions must be positive")
    if effect <= 0.0 or world_variance < 0.0 or seed_variance < 0.0:
        raise ValueError("power parameters are invalid")
    rng = np.random.default_rng(seed + world_count)
    world_effects = rng.normal(0.0, np.sqrt(world_variance), size=(iterations, world_count))
    seed_effects = rng.normal(
        0.0,
        np.sqrt(seed_variance),
        size=(iterations, world_count, 3),
    )
    observations = effect + world_effects[:, :, None] + seed_effects
    means = observations.mean(axis=(1, 2))
    standard_errors = observations.std(axis=(1, 2), ddof=1) / np.sqrt(world_count * 3)
    z_values = means / np.maximum(standard_errors, np.finfo(float).tiny)
    critical = NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / family_count)
    return float(np.mean(z_values >= critical))


def build_v2_power_report(
    development_path: str | Path,
    *,
    independent_path: str | Path | None = None,
) -> dict[str, object]:
    development = json.loads(Path(development_path).read_text())
    rows = list(development["results"])
    panel = _world_seed_values(rows, condition=PRIMARY_CONDITION, field=PRIMARY_ENDPOINT)
    decomposition = variance_decomposition(panel)
    curves = []
    for world_count in (24, 50, 64, 128):
        curves.append(
            {
                "world_count": world_count,
                "power": simulate_one_sided_power(
                    world_count=world_count,
                    world_variance=decomposition["world_variance"],
                    seed_variance=decomposition["within_world_seed_variance"],
                ),
            }
        )
    selected = next(
        (row for row in curves if row["power"] >= POWER_TARGET),
        curves[-1],
    )
    independent_summary = None
    if independent_path is not None:
        independent = json.loads(Path(independent_path).read_text())
        independent_panel = _world_seed_values(
            independent["results"],
            condition=PRIMARY_CONDITION,
            field=PRIMARY_ENDPOINT,
        )
        independent_summary = variance_decomposition(independent_panel)
    return {
        "status": "development_power_plan_not_sealed",
        "endpoint": PRIMARY_ENDPOINT,
        "condition": PRIMARY_CONDITION,
        "sesoi": SESOI,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "family_count": FAMILY_COUNT,
        "target_power": POWER_TARGET,
        "iterations": POWER_ITERATIONS,
        "development_variance": decomposition,
        "independent_validation_variance": independent_summary,
        "power_curve": curves,
        "selected_world_count": selected["world_count"],
        "power_passed": bool(selected["power"] >= POWER_TARGET),
    }
