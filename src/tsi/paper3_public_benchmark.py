"""Public, deterministic benchmark runner for the structural factorization claims."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from .paper3_learned_v3_contract import MECHANISM_HYPOTHESIS_COUNT
from .paper3_learned_v3_factorized_head import factorize_training_signature
from .paper3_learned_v3_generator import build_v3_world_dataset
from .paper3_learned_v3_contract import mechanism_combinations
from .paper3_replication_family import (
    COMBINATIONS,
    GRAPH_NAMES,
    build_replication_dataset,
)
from .paper3_replication_factorized import evaluate as evaluate_replication
from .paper3_replication_factorized import factorize as factorize_replication
from .paper3_variable_cardinality import (
    CARDINALITY_PANELS,
    COMBINATIONS as VARIABLE_COMBINATIONS,
    GRAPH_NAMES as VARIABLE_GRAPH_NAMES,
    build_variable_dataset,
    evaluate as evaluate_variable,
    factorize as factorize_variable,
)
from .paper3_learned_v2_mechanism import (
    ObservableMechanismSignature,
    predict_target_code,
)


BENCHMARK_ID = "TSI-P3-PUBLIC-STRUCTURAL-FACTORISATION-v1"
CODE_FILES = (
    "src/tsi/paper3_public_benchmark.py",
    "src/tsi/paper3_learned_v3_factorized_head.py",
    "src/tsi/paper3_learned_v3_generator.py",
    "src/tsi/paper3_replication_family.py",
    "src/tsi/paper3_replication_factorized.py",
    "src/tsi/paper3_variable_cardinality.py",
)


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def code_hashes(root: Path) -> dict[str, str]:
    return {
        relative: sha256((root / relative).read_bytes()).hexdigest()
        for relative in CODE_FILES
    }


def benchmark_manifest(root: Path) -> dict[str, object]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "v3_mechanism_count": MECHANISM_HYPOTHESIS_COUNT,
        "v3_graph_count": 4,
        "replication_mechanism_count": len(COMBINATIONS),
        "replication_graph_count": len(GRAPH_NAMES),
        "variable_cardinality_panel_count": len(CARDINALITY_PANELS),
        "variable_cardinality_combination_count": len(VARIABLE_COMBINATIONS),
        "v3_evaluation_partition": "test",
        "replication_evaluation_partition": "test",
        "code_hashes": code_hashes(root),
    }


def _v3_cells(
    indices: Iterable[int], graph_indices: Iterable[int]
) -> list[dict[str, object]]:
    rows = []
    combinations = mechanism_combinations()
    for combination_index in indices:
        expected = combinations[combination_index]
        for graph_index in graph_indices:
            dataset = build_v3_world_dataset(
                combination_index * 4 + graph_index,
                combination_index,
                graph_index=graph_index,
            )
            signature = factorize_training_signature(
                dataset.partitions["train"], dataset.graph.identifier
            )
            observable = ObservableMechanismSignature(
                dataset.graph.identifier,
                signature.layer_multipliers,
                signature.bridge_coefficient,
                signature.context_coefficient,
                len(dataset.partitions["train"]),
                1,
            )
            cases = dataset.partitions["test"]
            correct = sum(
                predict_target_code(case.source_code, case.action, observable)
                == case.target_code
                for case in cases
            )
            active_expected = expected[2] if graph_index == 1 else expected[1]
            active_observed = (
                signature.context_coefficient
                if graph_index == 1
                else signature.bridge_coefficient
            )
            rows.append(
                {
                    "combination_index": combination_index,
                    "graph_index": graph_index,
                    "parameter_exact": signature.layer_multipliers == expected[0]
                    and active_observed == active_expected,
                    "exact_accuracy": correct / len(cases),
                    "case_count": len(cases),
                }
            )
    return rows


def _replication_cells(
    indices: Iterable[int], graphs: Iterable[str]
) -> list[dict[str, object]]:
    rows = []
    for combination_index in indices:
        expected = COMBINATIONS[combination_index]
        for graph in graphs:
            dataset = build_replication_dataset(graph, combination_index)
            signature = factorize_replication(dataset)
            result = evaluate_replication(dataset, signature)
            rows.append(
                {
                    "combination_index": combination_index,
                    "graph": graph,
                    "parameter_exact": signature.multipliers == expected[0]
                    and signature.coefficient == expected[1],
                    "exact_accuracy": result["exact_accuracy"],
                    "case_count": result["case_count"],
                }
            )
    return rows


def _variable_cells(
    panel_indices: Iterable[int],
    combinations: Iterable[int],
    graphs: Iterable[str],
) -> list[dict[str, object]]:
    rows = []
    for panel_index in panel_indices:
        for combination_index in combinations:
            expected = VARIABLE_COMBINATIONS[combination_index]
            for graph in graphs:
                dataset = build_variable_dataset(panel_index, graph, combination_index)
                signature = factorize_variable(dataset)
                rows.append(
                    {
                        "panel_index": panel_index,
                        "cardinalities": CARDINALITY_PANELS[panel_index],
                        "graph": graph,
                        "combination_index": combination_index,
                        "parameter_exact": signature == expected,
                        "exact_accuracy": evaluate_variable(dataset, signature),
                        "case_count": len(dataset.partitions["test"]),
                    }
                )
    return rows


def run_public_benchmark(root: Path, *, smoke: bool = False) -> dict[str, object]:
    v3_indices = range(MECHANISM_HYPOTHESIS_COUNT) if not smoke else (0, 1, 2, 3)
    replication_indices = range(len(COMBINATIONS)) if not smoke else (0, 1, 2, 3)
    variable_panels = range(len(CARDINALITY_PANELS)) if not smoke else (0,)
    variable_indices = range(len(VARIABLE_COMBINATIONS)) if not smoke else (0, 1, 2, 3)
    v3_rows = _v3_cells(v3_indices, range(4))
    replication_rows = _replication_cells(replication_indices, GRAPH_NAMES)
    variable_rows = _variable_cells(
        variable_panels, variable_indices, VARIABLE_GRAPH_NAMES
    )
    result = {
        "benchmark_id": BENCHMARK_ID,
        "manifest": benchmark_manifest(root),
        "v3": {
            "cell_count": len(v3_rows),
            "parameter_exact_cells": sum(row["parameter_exact"] for row in v3_rows),
            "transition_exact_cells": sum(
                row["exact_accuracy"] == 1.0 for row in v3_rows
            ),
            "cells": v3_rows,
        },
        "replication": {
            "cell_count": len(replication_rows),
            "parameter_exact_cells": sum(
                row["parameter_exact"] for row in replication_rows
            ),
            "transition_exact_cells": sum(
                row["exact_accuracy"] == 1.0 for row in replication_rows
            ),
            "cells": replication_rows,
        },
        "variable_cardinality": {
            "cell_count": len(variable_rows),
            "parameter_exact_cells": sum(
                row["parameter_exact"] for row in variable_rows
            ),
            "transition_exact_cells": sum(
                row["exact_accuracy"] == 1.0 for row in variable_rows
            ),
            "cells": variable_rows,
        },
    }
    result["result_digest"] = _digest(result)
    return result


def write_public_benchmark(
    root: Path, output: Path, *, smoke: bool = False
) -> dict[str, object]:
    result = run_public_benchmark(root, smoke=smoke)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
