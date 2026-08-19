"""Public multi-world development pilot for P3-5A learned routing."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .paper3_development_experiment import ConstructiveMetricCache, evaluate_predictions
from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_learned_routing import edge_f1, run_learned_routing_pilot
from .paper3_multiworld import (
    DEVELOPMENT_WORLDS_PER_FAMILY,
    build_world_dataset,
    build_world_mechanism,
)
from .paper3_routing_controls import (
    correct_action_cross_edges,
    correct_source_cross_edges,
    TRAINING_UPDATES,
)


P3_LEARNED_DEVELOPMENT_ID = "P3-5A-LEARNED-DEVELOPMENT-v1"
PRIMARY_FAMILY = WorldFamily.CONTEXT_DEPENDENT
PRIMARY_OOD_SLICE = "bridge_consistent_shift"
DEFAULT_OPTIMIZER_SEEDS = (0, 1, 2)


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _score_model(cache: ConstructiveMetricCache, model: object, cases: Sequence[object]) -> dict[str, object]:
    predictions = model.predict_codes_precomputed(cases, model.basis.transform_cases(cases)[0])
    return evaluate_predictions(cache, cases, predictions).as_dict()


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values:
        raise ValueError("cannot average an empty field")
    return float(sum(values) / len(values))


def run_learned_development_pilot(
    *,
    worlds: int = DEVELOPMENT_WORLDS_PER_FAMILY,
    optimizer_seeds: Sequence[int] = DEFAULT_OPTIMIZER_SEEDS,
    updates: int = TRAINING_UPDATES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run only public context-dependent development worlds."""

    if type(worlds) is not int or not (1 <= worlds <= DEVELOPMENT_WORLDS_PER_FAMILY):
        raise ValueError("worlds must be in the public development range")
    seeds = tuple(optimizer_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("optimizer seeds must be nonempty and unique")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("optimizer seeds must be nonnegative integers")
    if type(updates) is not int or updates <= 0:
        raise ValueError("updates must be positive")

    cache = ConstructiveMetricCache()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for world_index in range(worlds):
        dataset = build_world_dataset(
            build_world_mechanism(
                PRIMARY_FAMILY,
                BenchmarkSplit.DEVELOPMENT,
                world_index,
            )
        )
        validation = dataset.partitions["validation"]
        ood = dataset.ood_by_slice[PRIMARY_OOD_SLICE]
        for seed in seeds:
            try:
                pilot, learned = run_learned_routing_pilot(
                    dataset.partitions["train"],
                    selection_cases=validation,
                    family=PRIMARY_FAMILY,
                    world_index=world_index,
                    optimizer_seed=seed,
                    updates=updates,
                )
                inferred_source = tuple(
                    edge for edge in pilot.source_edges if edge[0] != edge[1]
                )
                inferred_action = tuple(
                    edge for edge in pilot.action_edges if edge[0] != edge[1]
                )
                row = {
                    "family": PRIMARY_FAMILY.value,
                    "world_index": world_index,
                    "optimizer_seed": seed,
                    "source_cross_edge_f1": edge_f1(
                        inferred_source,
                        correct_source_cross_edges(PRIMARY_FAMILY),
                    ),
                    "action_cross_edge_f1": edge_f1(
                        inferred_action,
                        correct_action_cross_edges(PRIMARY_FAMILY),
                    ),
                    "dense_validation_i0_error": _score_model(
                        cache, pilot.dense_model, validation
                    )["mean_normalized_i0_quotient_error"],
                    "learned_validation_i0_error": _score_model(
                        cache, learned, validation
                    )["mean_normalized_i0_quotient_error"],
                    "dense_ood_i0_error": _score_model(cache, pilot.dense_model, ood)[
                        "mean_normalized_i0_quotient_error"
                    ],
                    "learned_ood_i0_error": _score_model(cache, learned, ood)[
                        "mean_normalized_i0_quotient_error"
                    ],
                    "dense_training_final_nll": pilot.dense_trace.final_nll,
                    "learned_training_final_nll": pilot.learned_trace.final_nll,
                    "status": "completed",
                }
                rows.append(row)
            except Exception as error:
                failures.append(
                    {
                        "world_index": world_index,
                        "optimizer_seed": seed,
                        "reason": repr(error),
                    }
                )
            if progress is not None:
                progress(f"world-{world_index}/seed-{seed}")

    world_rows: list[dict[str, object]] = []
    for world_index in range(worlds):
        current = [row for row in rows if row["world_index"] == world_index]
        if len(current) != len(seeds):
            continue
        world_rows.append(
            {
                "world_index": world_index,
                "source_cross_edge_f1": _mean(current, "source_cross_edge_f1"),
                "action_cross_edge_f1": _mean(current, "action_cross_edge_f1"),
                "dense_validation_i0_error": _mean(current, "dense_validation_i0_error"),
                "learned_validation_i0_error": _mean(current, "learned_validation_i0_error"),
                "dense_ood_i0_error": _mean(current, "dense_ood_i0_error"),
                "learned_ood_i0_error": _mean(current, "learned_ood_i0_error"),
            }
        )

    payload: dict[str, object] = {
        "identifier": P3_LEARNED_DEVELOPMENT_ID,
        "family": PRIMARY_FAMILY.value,
        "primary_ood_slice": PRIMARY_OOD_SLICE,
        "test_output_used": False,
        "worlds": worlds,
        "optimizer_seeds": list(seeds),
        "updates_per_run": updates,
        "run_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "world_count_with_complete_seed_panel": len(world_rows),
        "world_rows": world_rows,
        "runs": rows,
        "selection_rule": "validation_loss_increase_under_input_block_ablation",
        "oracle_graph_used_for_inference": False,
        "oracle_graph_used_for_external_audit": True,
    }
    return {**payload, "report_digest": _digest(payload)}


def write_learned_development_pilot(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
