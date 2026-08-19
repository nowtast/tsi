"""Development-only learned routing pilot for P3-5A.

The pilot first fits a dense model, scores source/action blocks by the fitted
parameter contribution, and selects a pre-registered number of cross-layer
edges.  It then refits a masked model using only the inferred graph.  The
oracle graph is used only by external audit code, never by the inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .paper3_independence_contract import WorldFamily
from .paper3_multiworld import GeneratedTransitionCase, LAYER_ORDER
from .paper3_routing_controls import (
    ESTIMATED_TRAINING_MACS_PER_WORLD,
    MULTIPLY_ADDS_PER_EXAMPLE,
    TRANSITION_ACTIVE_PARAMETER_BUDGET,
    TRAINING_UPDATES,
    TUNING_CANDIDATES_PER_MODEL,
    RoutingControlManifest,
    SELF_EDGES,
    routing_control_manifests,
)
from .paper3_routing_model import (
    ACTION_FEATURE_SLICES,
    encode_cases,
    LOGIT_OFFSETS,
    SOURCE_FEATURE_SLICES,
    TrainableRoutingModel,
    RoutingTrainingTrace,
)


P3_LEARNED_ROUTING_MODEL_ID = "P3-5A-DENSE-IMPORTANCE-ROUTING-v1"
LEARNED_SOURCE_CROSS_EDGE_BUDGET = {
    WorldFamily.SEPARABLE: 0,
    WorldFamily.BRIDGE_COUPLED: 1,
    WorldFamily.CONTEXT_DEPENDENT: 2,
}
LEARNED_ACTION_CROSS_EDGE_BUDGET = {
    WorldFamily.SEPARABLE: 0,
    WorldFamily.BRIDGE_COUPLED: 1,
    WorldFamily.CONTEXT_DEPENDENT: 1,
}


def _dense_manifest(family: WorldFamily) -> RoutingControlManifest:
    for manifest in routing_control_manifests(family):
        if manifest.identifier == "dense_active_matched":
            return manifest
    raise RuntimeError("dense routing manifest is unavailable")


def _cross_candidates() -> tuple[tuple[str, str], ...]:
    return tuple(
        (source, target)
        for source in LAYER_ORDER
        for target in LAYER_ORDER
        if source != target
    )


def _layer_nll(
    model: TrainableRoutingModel,
    inputs: np.ndarray,
    deltas: np.ndarray,
    layer: int,
) -> float:
    features = model.basis.transform_inputs(inputs)
    logits = model._logits(features)
    offset = LOGIT_OFFSETS[layer]
    cardinality = 4 if layer == 3 else 3
    probabilities = model._softmax(logits[:, offset : offset + cardinality])
    rows = np.arange(len(inputs))
    return float(
        np.mean(
            -np.log(
                np.maximum(
                    probabilities[rows, deltas[:, layer]],
                    np.finfo(np.float64).tiny,
                )
            )
        )
    )


def _block_scores(
    model: TrainableRoutingModel,
    cases: Sequence[GeneratedTransitionCase],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Measure validation loss increase after ablating each input block."""

    inputs, deltas = encode_cases(cases)
    baseline = tuple(
        _layer_nll(model, inputs, deltas, layer) for layer in range(len(LAYER_ORDER))
    )
    layer_index = {name: index for index, name in enumerate(LAYER_ORDER)}
    source_scores: dict[tuple[str, str], float] = {}
    action_scores: dict[tuple[str, str], float] = {}
    for target_index, target in enumerate(LAYER_ORDER):
        for source in LAYER_ORDER:
            source_index = layer_index[source]
            block = SOURCE_FEATURE_SLICES[source_index]
            ablated = inputs.copy()
            ablated[:, block] = 0.0
            source_scores[(source, target)] = _layer_nll(
                model, ablated, deltas, target_index
            ) - baseline[target_index]
            action_block = ACTION_FEATURE_SLICES[source_index]
            ablated = inputs.copy()
            ablated[:, action_block] = 0.0
            action_scores[(source, target)] = _layer_nll(
                model, ablated, deltas, target_index
            ) - baseline[target_index]
    return source_scores, action_scores


def _top_edges(
    scores: dict[tuple[str, str], float],
    budget: int,
) -> tuple[tuple[str, str], ...]:
    if type(budget) is not int or budget < 0:
        raise ValueError("edge budget must be a nonnegative integer")
    ranked = sorted(
        _cross_candidates(),
        key=lambda edge: (-scores[edge], edge[0], edge[1]),
    )
    return tuple(ranked[:budget])


def learned_manifest_from_dense_fit(
    dense_model: TrainableRoutingModel,
    family: WorldFamily,
    selection_cases: Sequence[GeneratedTransitionCase],
) -> RoutingControlManifest:
    """Infer a sparse routing manifest from a fitted dense model."""

    if dense_model.manifest.identifier != "dense_active_matched":
        raise ValueError("learned routing must begin from the dense pilot model")
    if not selection_cases:
        raise ValueError("learned routing selection requires validation cases")
    source_scores, action_scores = _block_scores(dense_model, selection_cases)
    source_edges = (
        *SELF_EDGES,
        *_top_edges(source_scores, LEARNED_SOURCE_CROSS_EDGE_BUDGET[family]),
    )
    action_edges = (
        *SELF_EDGES,
        *_top_edges(action_scores, LEARNED_ACTION_CROSS_EDGE_BUDGET[family]),
    )
    return RoutingControlManifest(
        identifier="learned_signature_routing",
        family=family,
        source_edges=source_edges,
        action_edges=action_edges,
        transition_parameterization=(
            "data_inferred_source_and_action_masks_with_delta_softmax_heads"
        ),
        base_active_parameters=TRANSITION_ACTIVE_PARAMETER_BUDGET,
        capacity_adapter_parameters=0,
        total_active_parameters=TRANSITION_ACTIVE_PARAMETER_BUDGET,
        multiply_adds_per_example=MULTIPLY_ADDS_PER_EXAMPLE,
        estimated_training_macs_per_world=ESTIMATED_TRAINING_MACS_PER_WORLD,
        input_fields=("exact_structural_state", "full_five_component_action"),
        training_updates=TRAINING_UPDATES,
        tuning_candidates=TUNING_CANDIDATES_PER_MODEL,
        optimizer_seeds_per_world=3,
    )


@dataclass(frozen=True)
class LearnedRoutingPilotResult:
    family: str
    world_index: int
    optimizer_seed: int
    dense_model: TrainableRoutingModel
    dense_trace: RoutingTrainingTrace
    learned_trace: RoutingTrainingTrace
    source_edges: tuple[tuple[str, str], ...]
    action_edges: tuple[tuple[str, str], ...]
    source_scores: tuple[tuple[tuple[str, str], float], ...]
    action_scores: tuple[tuple[tuple[str, str], float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "world_index": self.world_index,
            "optimizer_seed": self.optimizer_seed,
            "dense_training": self.dense_trace.as_dict(),
            "learned_training": self.learned_trace.as_dict(),
            "source_edges": [list(edge) for edge in self.source_edges],
            "action_edges": [list(edge) for edge in self.action_edges],
            "source_scores": [[list(edge), score] for edge, score in self.source_scores],
            "action_scores": [[list(edge), score] for edge, score in self.action_scores],
        }


def run_learned_routing_pilot(
    cases: Sequence[GeneratedTransitionCase],
    *,
    selection_cases: Sequence[GeneratedTransitionCase] | None = None,
    family: WorldFamily,
    world_index: int,
    optimizer_seed: int,
    updates: int = TRAINING_UPDATES,
) -> tuple[LearnedRoutingPilotResult, TrainableRoutingModel]:
    """Fit dense then learned routing on one public development world."""

    if not cases:
        raise ValueError("learned routing pilot requires at least one case")
    dense = TrainableRoutingModel(_dense_manifest(family), optimizer_seed)
    dense_trace = dense.fit(cases, updates=updates)
    selected_cases = cases if selection_cases is None else selection_cases
    source_scores, action_scores = _block_scores(dense, selected_cases)
    manifest = learned_manifest_from_dense_fit(dense, family, selected_cases)
    learned = TrainableRoutingModel(manifest, optimizer_seed)
    learned_trace = learned.fit(cases, updates=updates)
    result = LearnedRoutingPilotResult(
        family=family.value,
        world_index=world_index,
        optimizer_seed=optimizer_seed,
        dense_model=dense,
        dense_trace=dense_trace,
        learned_trace=learned_trace,
        source_edges=manifest.source_edges,
        action_edges=manifest.action_edges,
        source_scores=tuple(sorted(source_scores.items())),
        action_scores=tuple(sorted(action_scores.items())),
    )
    return result, learned


def edge_f1(
    inferred: Sequence[tuple[str, str]],
    expected: Sequence[tuple[str, str]],
) -> float:
    inferred_set = set(inferred)
    expected_set = set(expected)
    if not inferred_set and not expected_set:
        return 1.0
    if not inferred_set or not expected_set:
        return 0.0
    true_positive = len(inferred_set & expected_set)
    precision = true_positive / len(inferred_set)
    recall = true_positive / len(expected_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
