"""Predeclared cell-level uncertainty analysis for Paper 4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .paper4_contract import FROZEN_PAPER4_CONTRACT


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260810


def _cell_means(
    rows: list[dict[str, object]], model: str
) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["model"] == model:
            grouped.setdefault(
                (int(row["combination_index"]), str(row["graph"])), []
            ).append(row)
    return {
        key: {
            "exact_accuracy": float(
                np.mean([float(row["exact_accuracy"]) for row in values])
            ),
            "intervention_exact_accuracy": float(
                np.mean([float(row["intervention_exact_accuracy"]) for row in values])
            ),
        }
        for key, values in grouped.items()
    }


def _bootstrap_interval(values: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    means = values[sample_indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _distribution_summary(
    cell_values: dict[tuple[int, str], dict[str, float]], metric: str
) -> dict[str, object]:
    values = np.asarray([value[metric] for value in cell_values.values()])
    unique, counts = np.unique(np.round(values, 12), return_counts=True)
    return {
        "cell_count": len(values),
        "mean": float(values.mean()),
        "zero_cell_count": int(np.sum(values == 0.0)),
        "exact_cell_count": int(np.sum(values == 1.0)),
        "distribution": {
            f"{value:.12g}": int(count)
            for value, count in zip(unique, counts, strict=True)
        },
    }


def analyze_final_audit(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["runs"]
    tsi = _cell_means(rows, "tsi_graph_discovered_factorized")
    dense = _cell_means(rows, "dense_polynomial_trainable")
    diagonal = _cell_means(rows, "diagonal_trainable")
    wrong = _cell_means(rows, "wrong_routed_factorized")
    lookup = _cell_means(rows, "unstructured_lookup")
    keys = sorted(tsi)
    intervention_differences = np.asarray(
        [
            tsi[key]["intervention_exact_accuracy"]
            - dense[key]["intervention_exact_accuracy"]
            for key in keys
        ]
    )
    exact_differences = np.asarray(
        [tsi[key]["exact_accuracy"] - dense[key]["exact_accuracy"] for key in keys]
    )
    return {
        "contract": FROZEN_PAPER4_CONTRACT.as_dict(),
        "independent_cell_count": len(keys),
        "nested_seed_policy": "dense cell means average five bootstrap seeds; TSI has one deterministic fit",
        "primary_metric": "intervention_exact_accuracy",
        "primary_contrast": "TSI minus dense trainable",
        "primary_contrast_interval": _bootstrap_interval(intervention_differences),
        "exact_accuracy_contrast_interval": _bootstrap_interval(exact_differences),
        "all_primary_differences_positive": bool(np.all(intervention_differences > 0)),
        "intervention_cell_distributions": {
            "tsi_graph_discovered_factorized": _distribution_summary(
                tsi, "intervention_exact_accuracy"
            ),
            "dense_polynomial_trainable": _distribution_summary(
                dense, "intervention_exact_accuracy"
            ),
            "diagonal_trainable": _distribution_summary(
                diagonal, "intervention_exact_accuracy"
            ),
            "wrong_routed_factorized": _distribution_summary(
                wrong, "intervention_exact_accuracy"
            ),
            "unstructured_lookup": _distribution_summary(
                lookup, "intervention_exact_accuracy"
            ),
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def write_analysis(audit_path: Path, output_path: Path) -> dict[str, object]:
    result = analyze_final_audit(audit_path)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
