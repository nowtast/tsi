"""Capacity-matched routing-control manifest for TSI P3-3A."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .paper3_independence_contract import (
    MODEL_CONTROLS,
    P3_INDEPENDENCE_CONTRACT_ID,
    WorldFamily,
)
from .paper3_multiworld import LAYER_ORDER


P3_ROUTING_CONTROL_ID = "P3-3A-ROUTING-CONTROLS-v2"
TRANSITION_ACTIVE_PARAMETER_BUDGET = 420
ACTIVE_PARAMETER_RELATIVE_TOLERANCE = 0.05
TRAINING_UPDATES = 1_000
TRAIN_EXAMPLES_PER_WORLD = 1_308
TUNING_CANDIDATES_PER_MODEL = 1
PAIRED_OPTIMIZER_SEEDS_PER_WORLD = 3
TRAINING_MINIBATCH_SIZE = 128
FULL_INPUT_WIDTH = 31
OUTPUT_LOGIT_COUNT = 16
TRAINABLE_RANDOM_FEATURE_COEFFICIENTS = (
    TRANSITION_ACTIVE_PARAMETER_BUDGET - OUTPUT_LOGIT_COUNT
)
MULTIPLY_ADDS_PER_EXAMPLE = TRAINABLE_RANDOM_FEATURE_COEFFICIENTS * (
    FULL_INPUT_WIDTH + 1
)
ESTIMATED_TRAINING_MACS_PER_WORLD = (
    TRAINABLE_RANDOM_FEATURE_COEFFICIENTS * FULL_INPUT_WIDTH * TRAIN_EXAMPLES_PER_WORLD
    + 2
    * TRAINABLE_RANDOM_FEATURE_COEFFICIENTS
    * TRAINING_MINIBATCH_SIZE
    * TRAINING_UPDATES
)


Edge = tuple[str, str]
SELF_EDGES: tuple[Edge, ...] = tuple((layer, layer) for layer in LAYER_ORDER)


def correct_source_cross_edges(family: WorldFamily) -> tuple[Edge, ...]:
    if family is WorldFamily.SEPARABLE:
        return ()
    if family is WorldFamily.BRIDGE_COUPLED:
        return (("topology", "relation"),)
    return (
        ("topology", "relation"),
        ("order", "metric"),
    )


def correct_action_cross_edges(family: WorldFamily) -> tuple[Edge, ...]:
    if family is WorldFamily.SEPARABLE:
        return ()
    return (("topology", "relation"),)


def correct_cross_edges(family: WorldFamily) -> tuple[Edge, ...]:
    """Backward-compatible alias for declared source-state dependencies."""

    return correct_source_cross_edges(family)


def wrong_source_cross_edges(family: WorldFamily) -> tuple[Edge, ...]:
    return tuple(
        (target, source) for source, target in correct_source_cross_edges(family)
    )


def wrong_action_cross_edges(family: WorldFamily) -> tuple[Edge, ...]:
    return tuple(
        (target, source) for source, target in correct_action_cross_edges(family)
    )


def wrong_cross_edges(family: WorldFamily) -> tuple[Edge, ...]:
    """Backward-compatible alias for reversed source-state dependencies."""

    return wrong_source_cross_edges(family)


def _random_cross_edges(
    family: WorldFamily,
    *,
    channel: str,
) -> tuple[Edge, ...]:
    if channel == "source":
        correct = correct_source_cross_edges(family)
        wrong = wrong_source_cross_edges(family)
    elif channel == "action":
        correct = correct_action_cross_edges(family)
        wrong = wrong_action_cross_edges(family)
    else:
        raise ValueError("routing channel must be source or action")
    required = len(correct)
    if required == 0:
        return ()
    forbidden = set(SELF_EDGES).union(correct).union(wrong)
    candidates = tuple(
        (source, target)
        for source in LAYER_ORDER
        for target in LAYER_ORDER
        if (source, target) not in forbidden
    )
    seed = sha256(
        f"{P3_ROUTING_CONTROL_ID}:{family.value}:{channel}".encode("utf-8")
    ).digest()
    ranked = sorted(
        candidates,
        key=lambda edge: sha256(seed + f"{edge[0]}:{edge[1]}".encode("utf-8")).digest(),
    )
    return tuple(ranked[:required])


@dataclass(frozen=True)
class RoutingControlManifest:
    identifier: str
    family: WorldFamily
    source_edges: tuple[Edge, ...]
    action_edges: tuple[Edge, ...]
    transition_parameterization: str
    base_active_parameters: int
    capacity_adapter_parameters: int
    total_active_parameters: int
    multiply_adds_per_example: int
    estimated_training_macs_per_world: int
    input_fields: tuple[str, ...]
    training_updates: int
    tuning_candidates: int
    optimizer_seeds_per_world: int

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "family": self.family.value,
            "source_edges": [list(edge) for edge in self.source_edges],
            "action_edges": [list(edge) for edge in self.action_edges],
            "transition_parameterization": self.transition_parameterization,
            "base_active_parameters": self.base_active_parameters,
            "capacity_adapter_parameters": (self.capacity_adapter_parameters),
            "total_active_parameters": self.total_active_parameters,
            "multiply_adds_per_example": self.multiply_adds_per_example,
            "estimated_training_macs_per_world": (
                self.estimated_training_macs_per_world
            ),
            "input_fields": list(self.input_fields),
            "training_updates": self.training_updates,
            "tuning_candidates": self.tuning_candidates,
            "optimizer_seeds_per_world": self.optimizer_seeds_per_world,
        }


def _masked_manifest(
    identifier: str,
    family: WorldFamily,
    source_edges: tuple[Edge, ...],
    action_edges: tuple[Edge, ...],
    parameterization: str,
) -> RoutingControlManifest:
    return RoutingControlManifest(
        identifier=identifier,
        family=family,
        source_edges=source_edges,
        action_edges=action_edges,
        transition_parameterization=parameterization,
        base_active_parameters=TRANSITION_ACTIVE_PARAMETER_BUDGET,
        capacity_adapter_parameters=0,
        total_active_parameters=TRANSITION_ACTIVE_PARAMETER_BUDGET,
        multiply_adds_per_example=MULTIPLY_ADDS_PER_EXAMPLE,
        estimated_training_macs_per_world=ESTIMATED_TRAINING_MACS_PER_WORLD,
        input_fields=(
            "full_exact_structural_state",
            "full_five_component_action",
        ),
        training_updates=TRAINING_UPDATES,
        tuning_candidates=TUNING_CANDIDATES_PER_MODEL,
        optimizer_seeds_per_world=PAIRED_OPTIMIZER_SEEDS_PER_WORLD,
    )


def routing_control_manifests(
    family: WorldFamily,
) -> tuple[RoutingControlManifest, ...]:
    """Build all six controls with one exact active-parameter budget."""

    correct_source = (*SELF_EDGES, *correct_source_cross_edges(family))
    correct_action = (*SELF_EDGES, *correct_action_cross_edges(family))
    random_source = (
        *SELF_EDGES,
        *_random_cross_edges(family, channel="source"),
    )
    random_action = (
        *SELF_EDGES,
        *_random_cross_edges(family, channel="action"),
    )
    wrong_source = (*SELF_EDGES, *wrong_source_cross_edges(family))
    wrong_action = (*SELF_EDGES, *wrong_action_cross_edges(family))
    all_edges = tuple(
        (source, target) for source in LAYER_ORDER for target in LAYER_ORDER
    )

    dense = _masked_manifest(
        "dense_active_matched",
        family,
        all_edges,
        all_edges,
        "all_inputs_masked_random_features_with_delta_softmax_heads",
    )
    layer_routed_dense_action = _masked_manifest(
        "layer_routed_dense_action",
        family,
        SELF_EDGES,
        all_edges,
        "self_source_mask_all_action_masks_with_delta_softmax_heads",
    )
    strict = _masked_manifest(
        "strict_factorized_action",
        family,
        SELF_EDGES,
        SELF_EDGES,
        "strict_self_source_action_masks_with_delta_softmax_heads",
    )
    signature_routed = _masked_manifest(
        "signature_routed_oracle",
        family,
        correct_source,
        correct_action,
        "declared_source_and_action_masks_with_delta_softmax_heads",
    )
    random_routed = _masked_manifest(
        "random_routed_matched_sparsity",
        family,
        random_source,
        random_action,
        "frozen_random_cross_masks_with_delta_softmax_heads",
    )
    wrong_routed = _masked_manifest(
        "permuted_or_wrong_routed",
        family,
        wrong_source,
        wrong_action,
        "reversed_declared_cross_masks_with_delta_softmax_heads",
    )
    return (
        dense,
        layer_routed_dense_action,
        strict,
        signature_routed,
        random_routed,
        wrong_routed,
    )


def routing_control_digest() -> str:
    payload = {
        "identifier": P3_ROUTING_CONTROL_ID,
        "parent_contract": P3_INDEPENDENCE_CONTRACT_ID,
        "layer_order": list(LAYER_ORDER),
        "full_input_width": FULL_INPUT_WIDTH,
        "output_logit_count": OUTPUT_LOGIT_COUNT,
        "trainable_random_feature_coefficients": (
            TRAINABLE_RANDOM_FEATURE_COEFFICIENTS
        ),
        "training_minibatch_size": TRAINING_MINIBATCH_SIZE,
        "estimated_training_macs_per_world": ESTIMATED_TRAINING_MACS_PER_WORLD,
        "active_parameter_budget": TRANSITION_ACTIVE_PARAMETER_BUDGET,
        "active_parameter_relative_tolerance": (ACTIVE_PARAMETER_RELATIVE_TOLERANCE),
        "manifests": {
            family.value: [
                manifest.as_dict() for manifest in routing_control_manifests(family)
            ]
            for family in WorldFamily
        },
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RoutingControlAudit:
    control_digest: str
    model_count_per_family: int
    active_parameter_budget: int
    max_relative_parameter_difference: float
    random_masks_equal_correct_masks: bool
    wrong_masks_equal_correct_masks: bool
    information_fields_matched: bool
    training_updates_matched: bool
    compute_budgets_matched: bool
    tuning_budgets_matched: bool
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "control_digest": self.control_digest,
            "model_count_per_family": self.model_count_per_family,
            "active_parameter_budget": self.active_parameter_budget,
            "max_relative_parameter_difference": (
                self.max_relative_parameter_difference
            ),
            "random_masks_equal_correct_masks": (self.random_masks_equal_correct_masks),
            "wrong_masks_equal_correct_masks": (self.wrong_masks_equal_correct_masks),
            "information_fields_matched": self.information_fields_matched,
            "training_updates_matched": self.training_updates_matched,
            "compute_budgets_matched": self.compute_budgets_matched,
            "tuning_budgets_matched": self.tuning_budgets_matched,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def audit_routing_controls() -> RoutingControlAudit:
    errors: list[str] = []
    expected_ids = tuple(model.identifier for model in MODEL_CONTROLS)
    all_manifests = {
        family: routing_control_manifests(family) for family in WorldFamily
    }
    max_relative_difference = 0.0
    random_equal = False
    wrong_equal = False
    information_sets: set[tuple[str, ...]] = set()
    update_counts: set[int] = set()
    compute_counts: set[tuple[int, int]] = set()
    tuning_counts: set[int] = set()

    for family, manifests in all_manifests.items():
        identifiers = tuple(manifest.identifier for manifest in manifests)
        if identifiers != expected_ids:
            errors.append(f"{family.value} control order or identifiers changed")
        counts = tuple(manifest.total_active_parameters for manifest in manifests)
        relative_difference = (
            max(counts) - min(counts)
        ) / TRANSITION_ACTIVE_PARAMETER_BUDGET
        max_relative_difference = max(
            max_relative_difference,
            relative_difference,
        )
        if relative_difference > ACTIVE_PARAMETER_RELATIVE_TOLERANCE:
            errors.append(f"{family.value} active parameter budget is unmatched")
        for manifest in manifests:
            if (
                manifest.base_active_parameters + manifest.capacity_adapter_parameters
                != manifest.total_active_parameters
            ):
                errors.append(
                    f"{family.value}/{manifest.identifier} budget does not close"
                )
            information_sets.add(manifest.input_fields)
            update_counts.add(manifest.training_updates)
            compute_counts.add(
                (
                    manifest.multiply_adds_per_example,
                    manifest.estimated_training_macs_per_world,
                )
            )
            tuning_counts.add(manifest.tuning_candidates)

        by_id = {manifest.identifier: manifest for manifest in manifests}
        correct = by_id["signature_routed_oracle"]
        random = by_id["random_routed_matched_sparsity"]
        wrong = by_id["permuted_or_wrong_routed"]
        if family is not WorldFamily.SEPARABLE:
            random_equal = random_equal or (
                random.source_edges == correct.source_edges
                and random.action_edges == correct.action_edges
            )
            wrong_equal = wrong_equal or (
                wrong.source_edges == correct.source_edges
                and wrong.action_edges == correct.action_edges
            )

    if random_equal:
        errors.append("a random routing mask equals the correct mask")
    if wrong_equal:
        errors.append("a wrong routing mask equals the correct mask")
    if len(information_sets) != 1:
        errors.append("model input information fields are unmatched")
    if len(update_counts) != 1:
        errors.append("training update counts are unmatched")
    if len(compute_counts) != 1:
        errors.append("training compute budgets are unmatched")
    if len(tuning_counts) != 1:
        errors.append("tuning candidate counts are unmatched")

    return RoutingControlAudit(
        control_digest=routing_control_digest(),
        model_count_per_family=len(MODEL_CONTROLS),
        active_parameter_budget=TRANSITION_ACTIVE_PARAMETER_BUDGET,
        max_relative_parameter_difference=max_relative_difference,
        random_masks_equal_correct_masks=random_equal,
        wrong_masks_equal_correct_masks=wrong_equal,
        information_fields_matched=len(information_sets) == 1,
        training_updates_matched=len(update_counts) == 1,
        compute_budgets_matched=len(compute_counts) == 1,
        tuning_budgets_matched=len(tuning_counts) == 1,
        errors=tuple(errors),
    )
