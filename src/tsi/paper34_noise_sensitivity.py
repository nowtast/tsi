"""Post-review train/OOD noise sensitivity for the Paper 3/4 attribution panel."""

from __future__ import annotations

from hashlib import sha256
from math import sqrt
from statistics import NormalDist
from typing import Sequence

import numpy as np

from .paper34_resolution_benchmark import (
    GRAPH_MANIFEST,
    coordinate_nll,
    estimate_multipliers,
    fit_factorized,
    fit_generic_dense,
    fit_generic_sparse,
    generate_cases,
    learn_factorized,
    world_spec,
    wrong_graph,
)
from .paper34_resolution_contract import (
    HEAD_FAMILIES,
    OOD_CASES_PER_WORLD,
    SELECTION_CASES_PER_WORLD,
    TRAIN_CASES_PER_WORLD,
)


SENSITIVITY_SEED_LABEL = "TSI-P34-noise-sensitivity-v1"
TRAIN_NOISE_GRID = (0.04, 0.08, 0.16)
OOD_NOISE_GRID = (0.06, 0.12, 0.24)
EFFECT_NAMES = (
    "learned_routing_nll",
    "factorized_graph_nll",
    "generic_graph_nll",
    "large_generic_graph_nll",
)


def sensitivity_world_seed(world_index: int) -> int:
    material = f"{SENSITIVITY_SEED_LABEL}:{world_index}".encode()
    return int.from_bytes(sha256(material).digest()[:8], "little")


def _learn_at_noise(train: Sequence[object], selection: Sequence[object], noise: float):
    # The frozen learner used the fixed 0.08 likelihood. For p < 6/7, candidate
    # NLL ordering is identical to mismatch-count ordering; this explicit
    # parameterization makes the sensitivity implementation auditable.
    multipliers = estimate_multipliers(train)
    best = None
    for graph in GRAPH_MANIFEST:
        for first in HEAD_FAMILIES:
            for second in HEAD_FAMILIES:
                families = (first, second)
                model = fit_factorized(
                    train,
                    graph,
                    families,
                    name="learned_factorized",
                    multipliers=multipliers,
                )
                key = (
                    coordinate_nll(model, selection, noise),
                    graph,
                    families,
                    model,
                )
                if best is None or key[:3] < best[:3]:
                    best = key
    if best is None:
        raise RuntimeError("noise sensitivity search produced no model")
    return best[3]


def run_noise_world(
    world_index: int, train_noise: float, ood_noise: float
) -> dict[str, object]:
    if not 0.0 <= train_noise < 0.5 or not 0.0 <= ood_noise < 0.5:
        raise ValueError("noise probabilities must lie in [0, 0.5)")
    seed = sensitivity_world_seed(world_index)
    rng = np.random.default_rng(seed)
    spec = world_spec(world_index, rng)
    train = generate_cases(
        spec,
        TRAIN_CASES_PER_WORLD,
        rng,
        composition=False,
        noise_probability=train_noise,
    )
    selection = generate_cases(
        spec,
        SELECTION_CASES_PER_WORLD,
        rng,
        composition=False,
        noise_probability=train_noise,
    )
    test = generate_cases(
        spec,
        OOD_CASES_PER_WORLD,
        rng,
        composition=True,
        noise_probability=ood_noise,
    )
    learned = _learn_at_noise(train, selection, train_noise)
    shifted = wrong_graph(spec.graph)
    factorized_correct = fit_factorized(
        train,
        spec.graph,
        spec.families,
        name="factorized_correct",
    )
    factorized_wrong = fit_factorized(
        train,
        shifted,
        spec.families,
        name="factorized_wrong",
    )
    sparse_correct = fit_generic_sparse(
        train, spec.graph, 7, name="sparse_correct"
    )
    sparse_wrong = fit_generic_sparse(
        train, shifted, 7, name="sparse_wrong"
    )
    dense_correct = fit_generic_dense(train, spec.graph, name="dense_correct")
    dense_wrong = fit_generic_dense(train, shifted, name="dense_wrong")

    def nll(model: object) -> float:
        return coordinate_nll(model, test, ood_noise)

    learned_nll = nll(learned)
    correct_nll = nll(factorized_correct)
    wrong_nll = nll(factorized_wrong)
    effects = {
        "learned_routing_nll": wrong_nll - learned_nll,
        "factorized_graph_nll": wrong_nll - correct_nll,
        "generic_graph_nll": nll(sparse_wrong) - nll(sparse_correct),
        "large_generic_graph_nll": nll(dense_wrong) - nll(dense_correct),
    }
    return {
        "world_index": world_index,
        "seed": seed,
        "train_noise": train_noise,
        "ood_noise": ood_noise,
        "graph_exact": learned.graph == spec.graph,
        "head_exact": learned.families == spec.families,
        "effects": effects,
    }


def summarize_noise_cell(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if len(rows) < 2:
        raise ValueError("noise cell requires at least two worlds")
    z = NormalDist().inv_cdf(0.975)
    intervals = {}
    for name in EFFECT_NAMES:
        values = np.asarray(
            [float(row["effects"][name]) for row in rows],  # type: ignore[index]
            dtype=float,
        )
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        se = sd / sqrt(len(values))
        intervals[name] = {
            "mean": mean,
            "world_sd": sd,
            "standard_error": se,
            "descriptive_95pct_interval": [mean - z * se, mean + z * se],
        }
    identification = np.asarray(
        [bool(row["graph_exact"]) and bool(row["head_exact"]) for row in rows]
    )
    return {
        "world_count": len(rows),
        "train_noise": rows[0]["train_noise"],
        "ood_noise": rows[0]["ood_noise"],
        "identification_rate": float(np.mean(identification)),
        "effect_intervals": intervals,
    }


def build_noise_sensitivity(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (float(row["train_noise"]), float(row["ood_noise"]))
        grouped.setdefault(key, []).append(row)
    cells = [
        summarize_noise_cell(grouped[(train_noise, ood_noise)])
        for train_noise in TRAIN_NOISE_GRID
        for ood_noise in OOD_NOISE_GRID
    ]
    return {
        "status": "postreview_descriptive_noise_sensitivity_not_confirmatory",
        "seed_label": SENSITIVITY_SEED_LABEL,
        "train_noise_grid": list(TRAIN_NOISE_GRID),
        "ood_noise_grid": list(OOD_NOISE_GRID),
        "cells": cells,
        "all_cells_identification_rate": min(
            float(cell["identification_rate"]) for cell in cells
        ),
        "minimum_graph_effect_by_head": {
            name: min(
                float(cell["effect_intervals"][name]["mean"])  # type: ignore[index]
                for cell in cells
            )
            for name in EFFECT_NAMES
        },
        "limitations": [
            "The panel was designed after confirmation and is descriptive.",
            "It varies coordinate corruption probabilities only.",
            "It does not sample visual, continuous, neural, or real-world systems.",
        ],
    }
