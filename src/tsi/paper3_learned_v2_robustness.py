"""Robustness and information-leakage diagnostics for v2 observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .paper3_independence_contract import WorldFamily
from .paper3_learned_v2_generator import (
    V2GraphVariant,
    build_balanced_v2_world_dataset,
    build_v2_world_dataset,
)
from .paper3_learned_v2_model import JointGateRoutingModel
from .paper3_multiworld import LAYER_ORDER
from .paper3_routing_model import LOGIT_OFFSETS
from .paper3_learned_v2_observation import (
    PixelTransitionCase,
    build_observed_partitions,
    corrupt_pixel_case,
    encode_pixel_cases,
    pixel_feature_leakage_audit,
)


@dataclass(frozen=True)
class PixelRobustnessResult:
    world_index: int
    graph_variant: str
    mechanism_slot: int | None
    bridge_coefficient: int
    layer_multipliers: tuple[int, ...]
    seed: int
    condition: str
    clean_nll: float
    corrupted_nll: float
    absolute_degradation: float
    relative_degradation: float
    feature_shift_l2: float
    posterior_kl: float
    argmax_flip_rate: float
    source_ablation_nll: float
    clean_target_logloss: float
    corrupted_target_logloss: float
    target_logloss_degradation: float
    clean_source_logloss: float
    corrupted_source_logloss: float
    source_logloss_degradation: float
    finite: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _model_nll(model: JointGateRoutingModel, cases: Sequence[PixelTransitionCase]) -> float:
    inputs, deltas = encode_pixel_cases(cases)
    return model._nll(model._features(inputs), deltas)


def _target_logloss(
    model: JointGateRoutingModel,
    graph: V2GraphVariant,
    cases: Sequence[PixelTransitionCase],
) -> float:
    inputs, deltas = encode_pixel_cases(cases)
    logits = model._logits(model._features(inputs))
    values: list[float] = []
    cardinalities = (3, 3, 3, 4, 3)
    for source_name, target_name in graph.action_cross_edges:
        source_index = LAYER_ORDER.index(source_name)
        target_index = LAYER_ORDER.index(target_name)
        selected = np.asarray(
            [case.action.components[source_index] > 0 for case in cases],
            dtype=bool,
        )
        if not np.any(selected):
            continue
        offset = LOGIT_OFFSETS[target_index]
        cardinality = cardinalities[target_index]
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
        raise ValueError("no graph-sensitive target interventions were found")
    return float(np.mean(values))


def _source_logloss(
    model: JointGateRoutingModel,
    graph: V2GraphVariant,
    cases: Sequence[PixelTransitionCase],
) -> float:
    inputs, deltas = encode_pixel_cases(cases)
    logits = model._logits(model._features(inputs))
    values: list[float] = []
    cardinalities = (3, 3, 3, 4, 3)
    for source_name, target_name in graph.source_cross_edges:
        target_index = LAYER_ORDER.index(target_name)
        selected = np.asarray(
            [case.action.components[target_index] > 0 for case in cases],
            dtype=bool,
        )
        if not np.any(selected):
            continue
        offset = LOGIT_OFFSETS[target_index]
        cardinality = cardinalities[target_index]
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
        raise ValueError("no source-sensitive target interventions were found")
    return float(np.mean(values))


def _source_ablation_nll(
    model: JointGateRoutingModel,
    cases: Sequence[PixelTransitionCase],
) -> float:
    inputs, deltas = encode_pixel_cases(cases)
    action_only = inputs.copy()
    action_only[:, :26] = 0.0
    return model._nll(model._features(action_only), deltas)


def _prediction_shift(
    model: JointGateRoutingModel,
    clean: Sequence[PixelTransitionCase],
    corrupted: Sequence[PixelTransitionCase],
) -> tuple[float, float]:
    clean_inputs, _ = encode_pixel_cases(clean)
    corrupted_inputs, _ = encode_pixel_cases(corrupted)
    clean_logits = model._logits(model._features(clean_inputs))
    corrupted_logits = model._logits(model._features(corrupted_inputs))
    cardinalities = (3, 3, 3, 4, 3)
    offsets = (0, 3, 6, 9, 13)
    kl_values: list[float] = []
    flips: list[float] = []
    for offset, cardinality in zip(offsets, cardinalities, strict=True):
        clean_prob = model._softmax(clean_logits[:, offset : offset + cardinality])
        corrupted_prob = model._softmax(corrupted_logits[:, offset : offset + cardinality])
        kl_values.extend(
            np.sum(
                clean_prob
                * (
                    np.log(np.maximum(clean_prob, np.finfo(float).tiny))
                    - np.log(np.maximum(corrupted_prob, np.finfo(float).tiny))
                ),
                axis=1,
            )
        )
        flips.extend(
            (
                np.argmax(clean_prob, axis=1)
                != np.argmax(corrupted_prob, axis=1)
            ).astype(np.float64)
        )
    return float(np.mean(kl_values)), float(np.mean(flips))


def _feature_shift(
    clean: Sequence[PixelTransitionCase],
    corrupted: Sequence[PixelTransitionCase],
) -> float:
    clean_features, _ = encode_pixel_cases(clean)
    corrupted_features, _ = encode_pixel_cases(corrupted)
    return float(np.mean(np.linalg.norm(clean_features - corrupted_features, axis=1)))


def run_pixel_robustness_development(
    *,
    worlds: int = 4,
    world_start: int = 0,
    mechanism_slot: int | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    updates: int = 500,
) -> tuple[PixelRobustnessResult, ...]:
    if worlds <= 0:
        raise ValueError("worlds must be positive")
    if world_start < 0:
        raise ValueError("world_start must be nonnegative")
    conditions = (
        ("gaussian_0.25", {"gaussian_noise": 0.25}),
        ("gaussian_0.50", {"gaussian_noise": 0.50}),
        ("dropout_0.25", {"dropout_probability": 0.25}),
        ("dropout_0.50_quantized", {"dropout_probability": 0.50, "quantization_levels": 4}),
        ("near_blank_0.99", {"dropout_probability": 0.99, "quantization_levels": 2}),
    )
    results: list[PixelRobustnessResult] = []
    for world_index in range(world_start, world_start + worlds):
        dataset = (
            build_v2_world_dataset(world_index)
            if mechanism_slot is None
            else build_balanced_v2_world_dataset(world_index, mechanism_slot)
        )
        raw = dict(dataset.partitions)
        for seed in seeds:
            train = build_observed_partitions(
                raw,
                entity_count=3,
                regime="pixel_object_observation",
                seed=10_000 + world_index,
            )
            model = JointGateRoutingModel(WorldFamily.CONTEXT_DEPENDENT, seed)
            trace = model.fit(train["train"], updates=updates)
            clean = train["test"]
            clean_nll = _model_nll(model, clean)
            source_ablation_nll = _source_ablation_nll(model, clean)
            clean_target_logloss = _target_logloss(model, dataset.graph, clean)
            clean_source_logloss = _source_logloss(model, dataset.graph, clean)
            for condition, kwargs in conditions:
                corrupted = tuple(
                    corrupt_pixel_case(case, seed=20_000 + world_index + index, **kwargs)
                    for index, case in enumerate(clean)
                )
                corrupted_nll = _model_nll(model, corrupted)
                corrupted_target_logloss = _target_logloss(model, dataset.graph, corrupted)
                corrupted_source_logloss = _source_logloss(model, dataset.graph, corrupted)
                posterior_kl, argmax_flip_rate = _prediction_shift(model, clean, corrupted)
                absolute = corrupted_nll - clean_nll
                results.append(
                    PixelRobustnessResult(
                        world_index=world_index,
                        graph_variant=dataset.graph.identifier,
                        mechanism_slot=mechanism_slot,
                        bridge_coefficient=dataset.mechanism.bridge_coefficient,
                        layer_multipliers=dataset.mechanism.layer_multipliers,
                        seed=seed,
                        condition=condition,
                        clean_nll=clean_nll,
                        corrupted_nll=corrupted_nll,
                        absolute_degradation=absolute,
                        relative_degradation=absolute / max(abs(clean_nll), 1e-12),
                        feature_shift_l2=_feature_shift(clean, corrupted),
                        posterior_kl=posterior_kl,
                        argmax_flip_rate=argmax_flip_rate,
                        source_ablation_nll=source_ablation_nll,
                        clean_target_logloss=clean_target_logloss,
                        corrupted_target_logloss=corrupted_target_logloss,
                        target_logloss_degradation=corrupted_target_logloss - clean_target_logloss,
                        clean_source_logloss=clean_source_logloss,
                        corrupted_source_logloss=corrupted_source_logloss,
                        source_logloss_degradation=corrupted_source_logloss - clean_source_logloss,
                        finite=bool(trace.finite and np.isfinite(corrupted_nll)),
                    )
                )
    return tuple(results)


def run_pixel_robustness_balanced_validation(
    *,
    worlds: int = 4,
    world_start: int = 40,
    mechanism_slots: tuple[int, ...] = (0, 1, 2, 3),
    seeds: tuple[int, ...] = (0, 1, 2),
    updates: int = 500,
) -> tuple[PixelRobustnessResult, ...]:
    """Run a fresh factorial panel with fixed graph worlds and crossed mechanisms."""
    if not mechanism_slots:
        raise ValueError("mechanism_slots must be nonempty")
    if len(set(mechanism_slots)) != len(mechanism_slots):
        raise ValueError("mechanism_slots must not contain duplicates")
    results: list[PixelRobustnessResult] = []
    for mechanism_slot in mechanism_slots:
        results.extend(
            run_pixel_robustness_development(
                worlds=worlds,
                world_start=world_start,
                mechanism_slot=mechanism_slot,
                seeds=seeds,
                updates=updates,
            )
        )
    return tuple(results)


def run_pixel_leakage_audit(*, world_index: int = 0) -> dict[str, object]:
    dataset = build_v2_world_dataset(world_index)
    observed = build_observed_partitions(
        dict(dataset.partitions),
        entity_count=3,
        regime="pixel_object_observation",
        seed=40_000 + world_index,
    )
    return pixel_feature_leakage_audit(observed["train"] + observed["test"])
