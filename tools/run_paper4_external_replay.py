"""Independent deterministic replay for the frozen Paper 4 contract."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from tsi.paper3_replication_family import COMBINATIONS, GRAPH_NAMES, build_replication_dataset
from tsi.paper4_capacity_matched import (
    BOOTSTRAP_SEEDS,
    evaluate_capacity_model,
    fit_capacity_matched,
)
from tsi.paper4_comparative_validation import (
    evaluate_model,
    fit_unstructured_lookup,
    fit_wrong_routed_factorized,
    fit_tsi_factorized,
)
from tsi.paper4_contract import FROZEN_PAPER4_CONTRACT


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def build_replay() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for graph in GRAPH_NAMES:
        for combination_index in range(len(COMBINATIONS)):
            dataset = build_replication_dataset(graph, combination_index)
            models = (
                fit_unstructured_lookup(dataset),
                fit_wrong_routed_factorized(dataset),
            )
            rows.extend(evaluate_model(model, dataset) | {
                "seed": None,
                "graph": graph,
                "combination_index": combination_index,
            } for model in models)
            # Least-squares is invariant to the seeded row permutation. Fit once,
            # then replicate the identical nested seed rows.
            for include_interactions in (False, True):
                model = fit_capacity_matched(
                    dataset, seed=BOOTSTRAP_SEEDS[0],
                    include_interactions=include_interactions,
                )
                result = evaluate_capacity_model(model, dataset)
                rows.extend(
                    result | {
                        "seed": seed,
                        "graph": graph,
                        "combination_index": combination_index,
                    }
                    for seed in BOOTSTRAP_SEEDS
                )
            rows.append(
                evaluate_model(fit_tsi_factorized(dataset), dataset)
                | {
                    "seed": None,
                    "graph": graph,
                    "combination_index": combination_index,
                    "model": "tsi_graph_discovered_factorized",
                }
            )
    payload = {
        "contract": FROZEN_PAPER4_CONTRACT.as_dict(),
        "cells": len(GRAPH_NAMES) * len(COMBINATIONS),
        "runs": len(rows),
        "runs_data": rows,
    }
    return payload | {"replay_digest": _digest(payload)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = build_replay()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("cells", "runs", "replay_digest")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
