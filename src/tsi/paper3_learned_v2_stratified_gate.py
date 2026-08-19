"""Stratum-aware confirmatory aggregation and power freeze for v2."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np


POSITIVE_GRAPH_VARIANTS = (
    "bridge_topology_to_relation",
    "context_order_to_metric",
    "independent_relation",
)
NEGATIVE_GRAPH_VARIANT = "wrong_direction_negative_control"
DEGRADATION_MAXIMUM = 0.15
SESOI = 0.05
FAMILYWISE_ALPHA = 0.05
POWER_TARGET = 0.90
STRATUM_COUNT = len(POSITIVE_GRAPH_VARIANTS) * 4


def _groups(rows: Sequence[dict[str, object]], condition: str) -> dict[tuple[str, int], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("condition") != condition:
            continue
        key = (str(row["graph_variant"]), int(row["mechanism_slot"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def _summary(rows: Sequence[dict[str, object]], field: str) -> dict[str, float | int | bool]:
    world_values: dict[int, list[float]] = {}
    for row in rows:
        world = int(row.get("world_index", row.get("world", -1)))
        world_values.setdefault(world, []).append(float(row[field]))
    values = np.asarray(
        [float(np.mean(observations)) for observations in world_values.values()],
        dtype=np.float64,
    )
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / sqrt(len(values))) if len(values) > 1 else float("inf")
    upper = mean + NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / STRATUM_COUNT) * standard_error
    return {
        "n": int(sum(len(observations) for observations in world_values.values())),
        "world_count": int(len(values)),
        "mean": mean,
        "world_variance": float(np.var(values, ddof=1)) if len(values) > 1 else float("inf"),
        "standard_error": standard_error,
        "simultaneous_upper_bound": upper,
        "passed": bool(upper <= DEGRADATION_MAXIMUM),
    }


def build_stratified_gate_report(
    artifact_path: str | Path,
    *,
    condition: str = "gaussian_0.50",
) -> dict[str, object]:
    payload = json.loads(Path(artifact_path).read_text())
    rows = payload["results"]
    grouped = _groups(rows, condition)
    strata = []
    for graph_variant in (*POSITIVE_GRAPH_VARIANTS, NEGATIVE_GRAPH_VARIANT):
        for mechanism_slot in range(4):
            selected = grouped.get((graph_variant, mechanism_slot), [])
            if not selected:
                strata.append({
                    "graph_variant": graph_variant,
                    "mechanism_slot": mechanism_slot,
                    "missing": True,
                    "passed": False,
                })
                continue
            source = _summary(selected, "source_logloss_degradation")
            target = _summary(selected, "target_logloss_degradation")
            strata.append({
                "graph_variant": graph_variant,
                "mechanism_slot": mechanism_slot,
                "missing": False,
                "source": source,
                "target": target,
                "passed": bool(source["passed"] and target["passed"]),
            })
    positive = [row for row in strata if row["graph_variant"] in POSITIVE_GRAPH_VARIANTS]
    return {
        "status": "development_stratified_gate_not_sealed",
        "condition": condition,
        "degradation_maximum": DEGRADATION_MAXIMUM,
        "simultaneous_alpha": FAMILYWISE_ALPHA,
        "stratum_count": STRATUM_COUNT,
        "aggregation_rule": "intersection_union_all_positive_graph_mechanism_strata",
        "positive_strata_passed": bool(positive) and all(row["passed"] for row in positive),
        "negative_control_reported_separately": True,
        "strata": strata,
    }


def build_stratified_power_freeze(
    gate_report: dict[str, object],
    *,
    world_counts: tuple[int, ...] = (50, 64, 128),
) -> dict[str, object]:
    strata = [
        row for row in gate_report["strata"]
        if row["graph_variant"] in POSITIVE_GRAPH_VARIANTS and not row.get("missing")
    ]
    variance_rows = []
    for row in strata:
        source = row["source"]
        target = row["target"]
        variance_rows.append({
            "graph_variant": row["graph_variant"],
            "mechanism_slot": row["mechanism_slot"],
            "source_se": source["standard_error"],
            "target_se": target["standard_error"],
            "source_world_variance": source["world_variance"],
            "target_world_variance": target["world_variance"],
            "world_count": source["world_count"],
        })
    critical = NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / STRATUM_COUNT)
    return {
        "status": "stratified_power_freeze_not_sealed",
        "aggregation_rule": gate_report["aggregation_rule"],
        "sesoi": SESOI,
        "degradation_maximum": DEGRADATION_MAXIMUM,
        "stratum_count": STRATUM_COUNT,
        "critical_normal": critical,
        "world_counts": list(world_counts),
        "variance_rows": variance_rows,
        "power_gate_passed": False,
        "power_gate_reason": (
            "confirmatory stratum gate failed; power freeze remains blocked"
            if not gate_report["positive_strata_passed"]
            else "power operating characteristic requires a predeclared margin alternative"
        ),
    }
