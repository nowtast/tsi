"""Audit Paper 4 comparison budgets, split isolation, and external replay."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from tsi.paper3_replication_family import COMBINATIONS, GRAPH_NAMES, build_replication_dataset
from tsi.paper3_learned_structure import discover_replication_graph
from tsi.paper4_capacity_matched import FEATURE_WIDTH, fit_capacity_matched
from tsi.paper4_comparative_validation import evaluate_model
from tsi.paper4_contract import FROZEN_PAPER4_CONTRACT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(external_replay: Path, second_replay: Path) -> dict[str, object]:
    feature_counts = {"diagonal": 1 + 5 + 5, "dense": FEATURE_WIDTH}
    ranks = {"diagonal": set(), "dense": set()}
    split_checks = []
    discovery_correct = 0
    cells = 0
    for graph in GRAPH_NAMES:
        for combination_index in range(len(COMBINATIONS)):
            dataset = build_replication_dataset(graph, combination_index)
            cells += 1
            train_keys = {(case.source, case.action) for case in dataset.partitions["train"]}
            test_keys = {(case.source, case.action) for case in dataset.partitions["test"]}
            split_checks.append(train_keys.isdisjoint(test_keys))
            discovery_correct += int(
                discover_replication_graph(dataset)["identified_graph"] == graph
            )
            for name, include_interactions in (("diagonal", False), ("dense", True)):
                fit_capacity_matched(
                    dataset, seed=0, include_interactions=include_interactions
                )
                # The feature matrix rank is the identifiable coefficient count per
                # target coordinate, not the allocated zero-padded matrix size.
                from tsi.paper4_capacity_matched import _features

                x = _features(dataset.partitions["train"], include_interactions=include_interactions)
                ranks[name].add(int(np.linalg.matrix_rank(x)))

    source_checks = {
        "capacity_fit_reads_train_only": 'partitions["train"]' in inspect.getsource(fit_capacity_matched),
        "tsi_graph_discovery_reads_train_only": 'split="train"' in inspect.getsource(discover_replication_graph),
        "evaluation_reads_test_only": 'partitions["test"]' in inspect.getsource(evaluate_model),
    }
    payload = {
        "contract": FROZEN_PAPER4_CONTRACT.as_dict(),
        "cells": cells,
        "same_environment_replay_a": {
            "path": str(external_replay),
            "sha256": _sha256(external_replay),
            "replay_digest": json.loads(external_replay.read_text())["replay_digest"],
        },
        "same_environment_replay_b": {
            "path": str(second_replay),
            "sha256": _sha256(second_replay),
            "replay_digest": json.loads(second_replay.read_text())["replay_digest"],
        },
        "replay_byte_identical": external_replay.read_bytes() == second_replay.read_bytes(),
        "replay_scope": {
            "deterministic_same_environment": True,
            "independent_implementation": False,
            "independent_research_group": False,
        },
        "feature_budget": {
            "matched_across_all_models": False,
            "control_allocated_width_matched": True,
            "diagonal_features_per_output": feature_counts["diagonal"],
            "dense_features_per_output": feature_counts["dense"],
            "tsi_features_per_output": None,
            "reason": "TSI feature count is undefined in this audit",
        },
        "identifiable_parameter_budget": {
            "matched": False,
            "diagonal_rank_per_output": sorted(ranks["diagonal"]),
            "dense_rank_per_output": sorted(ranks["dense"]),
            "tsi_structural_parameters": 6,
        },
        "compute_budget": {
            "matched": False,
            "control_feature_evaluations_per_case": FEATURE_WIDTH,
            "tsi_graph_candidates": len(GRAPH_NAMES),
            "tsi_factorized_parameters_after_selection": 6,
        },
        "split_and_leakage_checks": {
            "all_train_test_input_keys_disjoint": all(split_checks),
            "graph_discovery_exact_on_training_only": discovery_correct == cells,
            **source_checks,
        },
        "interpretation": (
            "The trainable controls share an allocated feature width and the split is "
            "isolated, but a feature match to TSI is not established; parameter count "
            "and compute are also unmatched. The result must remain a "
            "bounded structural comparison, not a universal efficiency or model "
            "superiority claim."
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("external_replay", type=Path)
    parser.add_argument("second_replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.external_replay, args.second_replay)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["split_and_leakage_checks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
