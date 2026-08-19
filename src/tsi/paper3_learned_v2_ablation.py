"""Integrated development ablation runner for P3-5A-v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np

from .paper3_independence_contract import WorldFamily
from .paper3_learned_v2_contract import OPTIMIZER_SEEDS
from .paper3_learned_v2_generator import V2GraphVariant, V2TransitionCase, build_v2_world_dataset
from .paper3_learned_v2_model import EDGE_COUNT, JointGateRoutingModel
from .paper3_multiworld import LAYER_ORDER
from .paper3_routing_model import LOGIT_OFFSETS, encode_cases


CONTROL_ORDER = ("learned_joint_gate", "oracle_joint_gate", "dense_joint_gate", "random_joint_gate", "wrong_joint_gate")


@dataclass(frozen=True)
class V2ControlResult:
    world_index: int
    graph_variant: str
    seed: int
    control: str
    threshold: float | None
    edge_count: int
    source_edge_precision: float
    source_edge_recall: float
    action_edge_precision: float
    action_edge_recall: float
    downstream_nll: float
    test_nll: float
    intervention_target_logloss: float
    finite: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _edge_mask(graph: V2GraphVariant, control: str, seed: int) -> np.ndarray:
    mask = np.zeros((len(LAYER_ORDER), EDGE_COUNT), dtype=bool)
    if control == "dense_joint_gate":
        return np.ones_like(mask)
    if control == "oracle_joint_gate":
        source_edges = graph.source_cross_edges
        action_edges = graph.action_cross_edges
    elif control == "wrong_joint_gate":
        source_edges = tuple((target, source) for source, target in graph.source_cross_edges)
        action_edges = tuple((target, source) for source, target in graph.action_cross_edges)
    elif control == "random_joint_gate":
        cross_count = len(graph.source_cross_edges) + len(graph.action_cross_edges)
        candidates = [
            (target, edge)
            for target in range(len(LAYER_ORDER))
            for edge in range(len(LAYER_ORDER), EDGE_COUNT)
            if edge - len(LAYER_ORDER) != target
        ]
        ranked = sorted(
            candidates,
            key=lambda item: sha256(f"v2-random:{seed}:{item}".encode()).digest(),
        )
        source_edges = ()
        action_edges = tuple(
            (LAYER_ORDER[edge - len(LAYER_ORDER)], LAYER_ORDER[target])
            for target, edge in ranked[:cross_count]
        )
    else:
        raise ValueError(f"unsupported fixed control: {control}")
    for source, target in source_edges:
        mask[LAYER_ORDER.index(target), LAYER_ORDER.index(source)] = True
    for source, target in action_edges:
        mask[LAYER_ORDER.index(target), len(LAYER_ORDER) + LAYER_ORDER.index(source)] = True
    for target in range(len(LAYER_ORDER)):
        mask[target, len(LAYER_ORDER) + target] = True
    return mask


def _precision_recall(found: set[tuple[str, str]], truth: set[tuple[str, str]]) -> tuple[float, float]:
    if not truth:
        return (1.0 if not found else 0.0, 1.0)
    return (
        len(found & truth) / len(found) if found else 0.0,
        len(found & truth) / len(truth),
    )


def _target_logloss(model: JointGateRoutingModel, graph: V2GraphVariant, cases: Iterable[V2TransitionCase]) -> float:
    cases = tuple(cases)
    inputs, deltas = encode_cases(cases)
    logits = model._logits(model._features(inputs))
    values: list[float] = []
    for source_name, target_name in graph.action_cross_edges:
        source_index = LAYER_ORDER.index(source_name)
        target_index = LAYER_ORDER.index(target_name)
        selected = np.asarray(
            [case.action.components[source_index] > 0 for case in cases],
            dtype=bool,
        )
        if not np.any(selected):
            continue
        cardinality = (3, 3, 3, 4, 3)[target_index]
        offset = LOGIT_OFFSETS[target_index]
        probabilities = model._softmax(logits[selected, offset : offset + cardinality])
        rows = np.arange(int(np.sum(selected)))
        values.extend(
            -np.log(
                np.maximum(
                    probabilities[rows, deltas[selected, target_index]],
                    np.finfo(float).tiny,
                )
            )
        )
    if not values:
        raise ValueError("test cases do not contain any graph-sensitive interventions")
    return float(np.mean(values))


def _fit_fixed(
    train_cases: tuple[V2TransitionCase, ...],
    selection_cases: tuple[V2TransitionCase, ...],
    graph: V2GraphVariant,
    control: str,
    seed: int,
    *,
    updates: int,
) -> tuple[JointGateRoutingModel, float | None]:
    model = JointGateRoutingModel(WorldFamily.CONTEXT_DEPENDENT, seed)
    if control == "learned_joint_gate":
        model.fit(train_cases, updates=updates)
        selected = model.select_threshold(
            train_cases,
            selection_cases,
            updates=max(1, updates // 10),
            refit_updates=updates,
        )
        return selected.selected_model, selected.selected_threshold
    model.fit(train_cases, updates=updates, fixed_gate_mask=_edge_mask(graph, control, seed))
    return model, None


def run_v2_development_ablation(
    *,
    worlds: int = 4,
    seeds: tuple[int, ...] = OPTIMIZER_SEEDS,
    updates: int = 1_000,
) -> tuple[V2ControlResult, ...]:
    if worlds <= 0:
        raise ValueError("worlds must be positive")
    results: list[V2ControlResult] = []
    for world_index in range(worlds):
        dataset = build_v2_world_dataset(world_index)
        train = dataset.partitions["train"]
        selection = dataset.partitions["routing_selection"]
        downstream = dataset.partitions["downstream_evaluation"]
        test = dataset.partitions["test"]
        for seed in seeds:
            for control in CONTROL_ORDER:
                model, threshold = _fit_fixed(
                    train, selection, dataset.graph, control, seed, updates=updates
                )
                cutoff = threshold or 0.5
                source_found, action_found = (
                    set(model.inferred_edges(cutoff)[0]),
                    set(model.inferred_edges(cutoff)[1]),
                )
                source_precision, source_recall = _precision_recall(
                    source_found, set(dataset.graph.source_cross_edges)
                )
                action_precision, action_recall = _precision_recall(
                    action_found, set(dataset.graph.action_cross_edges)
                )
                downstream_inputs, downstream_deltas = encode_cases(downstream)
                test_inputs, test_deltas = encode_cases(test)
                results.append(
                    V2ControlResult(
                        world_index=world_index,
                        graph_variant=dataset.graph.identifier,
                        seed=seed,
                        control=control,
                        threshold=threshold,
                        edge_count=int(sum(len(edges) for edges in model.inferred_edges(cutoff))),
                        source_edge_precision=source_precision,
                        source_edge_recall=source_recall,
                        action_edge_precision=action_precision,
                        action_edge_recall=action_recall,
                        downstream_nll=model._nll(model._features(downstream_inputs), downstream_deltas),
                        test_nll=model._nll(model._features(test_inputs), test_deltas),
                        intervention_target_logloss=_target_logloss(model, dataset.graph, test),
                        finite=bool(model.trace and model.trace.finite),
                    )
                )
    return tuple(results)
