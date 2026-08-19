"""Frozen structural interface for the minimal TSI Paper 3 experiment.

The module does not implement a neural network.  It fixes the exact state,
transition, objective, and evaluation contracts that a later implementation
must satisfy.  Exact finite evaluators are kept separate from differentiable
training surrogates.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from math import inf, isfinite
from typing import Hashable

from .coherent import (
    CoherentStructuralState,
    CorrespondenceCosts,
    correspondence_costs,
    typed_correspondences,
)
from .dynamical import TrackedTransition


FROZEN_LAYER_ORDER = ("label", "simplicial", "metric", "relation", "order")


class AccessRegime(str, Enum):
    """How a structural state becomes available to the model or evaluator."""

    EXACT_ORACLE = "exact_oracle"
    DECODED_VALID = "decoded_valid"
    NOISY_RECOVERED = "noisy_recovered"


class ObjectiveRole(str, Enum):
    """Whether a quantity trains a model or audits its decoded output."""

    TRAINING_SURROGATE = "training_surrogate"
    EXACT_EVALUATOR = "exact_evaluator"


class ClaimStatus(str, Enum):
    """Evidence status attached to one objective or evaluation term."""

    THEOREM_BACKED = "theorem_backed"
    EMPIRICAL = "empirical"


@dataclass(frozen=True)
class ObjectiveTerm:
    """One frozen objective-ledger entry."""

    name: str
    role: ObjectiveRole
    status: ClaimStatus
    target: str
    stage2_anchor: str


@dataclass(frozen=True)
class Paper3ObjectiveWeights:
    """Nonnegative coefficients for the nine empirical training terms."""

    jepa_latent: float = 1.0
    label_surrogate: float = 1.0
    simplicial_surrogate: float = 1.0
    metric_surrogate: float = 1.0
    relation_surrogate: float = 1.0
    order_surrogate: float = 1.0
    bridge_surrogate: float = 1.0
    tracking_surrogate: float = 1.0
    validity_surrogate: float = 1.0

    def __post_init__(self) -> None:
        values = {field.name: float(getattr(self, field.name)) for field in fields(self)}
        if any(not isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("objective weights must be finite and nonnegative")
        required_positive = set(values) - {"bridge_surrogate"}
        if any(values[name] <= 0 for name in required_positive):
            raise ValueError("every core Paper 3 objective weight must be positive")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def validate_for_state(self, state: CoherentStructuralState) -> None:
        if state.signature.bridges and self.bridge_surrogate <= 0:
            raise ValueError("declared bridges require a positive bridge weight")


@dataclass(frozen=True)
class Paper3TrainingLosses:
    """One measured vector of empirical Paper 3 training surrogates."""

    jepa_latent: float
    label_surrogate: float
    simplicial_surrogate: float
    metric_surrogate: float
    relation_surrogate: float
    order_surrogate: float
    bridge_surrogate: float
    tracking_surrogate: float
    validity_surrogate: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not isfinite(value) or value < 0:
                raise ValueError("training losses must be finite and nonnegative")
            object.__setattr__(self, field.name, value)

    def weighted_total(
        self,
        weights: Paper3ObjectiveWeights,
        state: CoherentStructuralState,
    ) -> float:
        weights.validate_for_state(state)
        return sum(
            getattr(self, field.name) * getattr(weights, field.name)
            for field in fields(self)
        )


FROZEN_OBJECTIVE_WEIGHTS = Paper3ObjectiveWeights()


@dataclass(frozen=True)
class Paper3InterfaceSpec:
    """The immutable scope of the first Structural JEPA implementation."""

    identifier: str
    state_type: str
    required_layers: tuple[str, ...]
    conditionally_required: tuple[str, ...]
    excluded_components: tuple[str, ...]
    input_regime: AccessRegime
    target_regime: AccessRegime
    prediction_regime: AccessRegime
    carrier_policy: str
    transition_type: str
    objective_terms: tuple[ObjectiveTerm, ...]
    max_exact_correspondences: int = 100_000

    def __post_init__(self) -> None:
        if self.required_layers != FROZEN_LAYER_ORDER:
            raise ValueError("Paper 3 must use the frozen five-layer I0 order")
        if len({term.name for term in self.objective_terms}) != len(
            self.objective_terms
        ):
            raise ValueError("objective term names must be unique")
        if self.max_exact_correspondences <= 0:
            raise ValueError("max_exact_correspondences must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "state_type": self.state_type,
            "required_layers": list(self.required_layers),
            "conditionally_required": list(self.conditionally_required),
            "excluded_components": list(self.excluded_components),
            "input_regime": self.input_regime.value,
            "target_regime": self.target_regime.value,
            "prediction_regime": self.prediction_regime.value,
            "carrier_policy": self.carrier_policy,
            "transition_type": self.transition_type,
            "max_exact_correspondences": self.max_exact_correspondences,
            "default_objective_weights": {
                field.name: getattr(FROZEN_OBJECTIVE_WEIGHTS, field.name)
                for field in fields(FROZEN_OBJECTIVE_WEIGHTS)
            },
            "objective_terms": [
                {
                    "name": term.name,
                    "role": term.role.value,
                    "status": term.status.value,
                    "target": term.target,
                    "stage2_anchor": term.stage2_anchor,
                }
                for term in self.objective_terms
            ],
        }


FROZEN_PAPER3_INTERFACE = Paper3InterfaceSpec(
    identifier="P3-I0-FIXED-v1",
    state_type="CoherentStructuralState",
    required_layers=FROZEN_LAYER_ORDER,
    conditionally_required=("declared exact bridges",),
    excluded_components=(
        "attributes",
        "mass",
        "noisy structural recovery",
        "causal identification",
        "continuous motion witnesses",
    ),
    input_regime=AccessRegime.EXACT_ORACLE,
    target_regime=AccessRegime.EXACT_ORACLE,
    prediction_regime=AccessRegime.DECODED_VALID,
    carrier_policy="fixed typed local identifiers for training and tracking",
    transition_type="typed label-preserving partial bijection conditioned on action",
    objective_terms=(
        ObjectiveTerm(
            "jepa_latent",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "future target-encoder representation",
            "not a Stage 2 theorem",
        ),
        ObjectiveTerm(
            "label_surrogate",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "typed labels",
            "I0 label component",
        ),
        ObjectiveTerm(
            "simplicial_surrogate",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "exact simplex membership after decoding",
            "Paper 2A and I0 simplicial component",
        ),
        ObjectiveTerm(
            "metric_surrogate",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "intrinsic metric",
            "Paper 2B and I0 metric component",
        ),
        ObjectiveTerm(
            "relation_surrogate",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "typed generator relations",
            "Paper 2C and I0 relation component",
        ),
        ObjectiveTerm(
            "order_surrogate",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "finite preorder",
            "Paper 2A-X1 and I0 order component",
        ),
        ObjectiveTerm(
            "bridge_surrogate",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "every bridge declared by the frozen signature",
            "I0 bridge defects",
        ),
        ObjectiveTerm(
            "tracking_surrogate",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "oracle partial-bijection graph",
            "Paper 2D tracking metric",
        ),
        ObjectiveTerm(
            "validity_surrogate",
            ObjectiveRole.TRAINING_SURROGATE,
            ClaimStatus.EMPIRICAL,
            "decoded membership in the coherent-state domain",
            "Stage 2 well-formedness conditions",
        ),
        ObjectiveTerm(
            "fixed_carrier_exact",
            ObjectiveRole.EXACT_EVALUATOR,
            ClaimStatus.THEOREM_BACKED,
            "literal equality on the fixed typed carrier",
            "identity-aligned specialization of the I0 layers",
        ),
        ObjectiveTerm(
            "i0_quotient_metric",
            ObjectiveRole.EXACT_EVALUATOR,
            ClaimStatus.THEOREM_BACKED,
            "integrated structural isomorphism class",
            "I0 coherent common-correspondence metric",
        ),
        ObjectiveTerm(
            "tracking_exact",
            ObjectiveRole.EXACT_EVALUATOR,
            ClaimStatus.THEOREM_BACKED,
            "oracle partial-bijection graph on fixed local identifiers",
            "Paper 2D tracking graph metric",
        ),
    ),
)


def _validate_fixed_carrier_pair(
    left: CoherentStructuralState,
    right: CoherentStructuralState,
) -> None:
    if left.schema != right.schema:
        raise ValueError("fixed-carrier states must use the same schema")
    if left.signature != right.signature:
        raise ValueError("fixed-carrier states must use the same signature")
    for object_name in left.schema.objects:
        if (
            left.core.relational.carriers[object_name]
            != right.core.relational.carriers[object_name]
        ):
            raise ValueError(
                "fixed-carrier states must use identical typed local identifiers"
            )


@dataclass(frozen=True)
class StructuralTransitionExample:
    """One oracle-supervised action transition in the frozen Paper 3 regime."""

    source: CoherentStructuralState
    action: Hashable
    target: CoherentStructuralState
    tracking: TrackedTransition

    def __post_init__(self) -> None:
        _validate_fixed_carrier_pair(self.source, self.target)
        try:
            hash(self.action)
        except TypeError as error:
            raise TypeError("action must be hashable") from error
        if self.tracking.source != self.source.core:
            raise ValueError("oracle tracking source does not match the source state")
        if self.tracking.target != self.target.core:
            raise ValueError("oracle tracking target does not match the target state")


@dataclass(frozen=True)
class FixedCarrierLayerErrors:
    """Exact identity-aligned layer errors on one frozen typed carrier."""

    label: float
    simplicial: float
    metric: float
    relation: float
    order: float
    total: float

    @property
    def is_zero(self) -> bool:
        return self.total == 0.0


def fixed_carrier_exact_losses(
    predicted: CoherentStructuralState,
    target: CoherentStructuralState,
) -> FixedCarrierLayerErrors:
    """Return exact hard-output errors under the fixed identifier alignment.

    Because every signature weight is positive, the total is zero exactly when
    the two coherent states are literally equal on their fixed typed carrier.
    This evaluator is stronger than quotient equality and is not permutation
    invariant.
    """

    _validate_fixed_carrier_pair(predicted, target)
    vertex_count = len(target.core.tagged_entities)
    label_error = (
        sum(
            left != right
            for left, right in zip(
                predicted.core.tagged_labels,
                target.core.tagged_labels,
                strict=True,
            )
        )
        / vertex_count
    )
    simplicial_error = len(
        predicted.core.simplices.symmetric_difference(target.core.simplices)
    ) / (1 << vertex_count)

    predicted_index = predicted.core.distance_index
    target_index = target.core.distance_index
    raw_metric_error = max(
        abs(
            predicted.core.distances[predicted_index[left]][predicted_index[right]]
            - target.core.distances[target_index[left]][target_index[right]]
        )
        for left in target.core.tagged_entities
        for right in target.core.tagged_entities
    )
    metric_error = min(1.0, raw_metric_error / target.signature.metric_scale)

    relation_error = max(
        (
            len(
                predicted.core.relational.generators[
                    arrow.name
                ].pairs.symmetric_difference(
                    target.core.relational.generators[arrow.name].pairs
                )
            )
            / (
                len(target.core.relational.carriers[arrow.source])
                * len(target.core.relational.carriers[arrow.target])
            )
            for arrow in target.schema.arrows
        ),
        default=0.0,
    )
    order_error = len(
        predicted.order.relation.symmetric_difference(target.order.relation)
    ) / (vertex_count * vertex_count)

    signature = target.signature
    total = (
        signature.label_weight * label_error
        + signature.simplicial_weight * simplicial_error
        + signature.metric_weight * metric_error
        + signature.relation_weight * relation_error
        + signature.order_weight * order_error
    )
    return FixedCarrierLayerErrors(
        label=label_error,
        simplicial=simplicial_error,
        metric=metric_error,
        relation=relation_error,
        order=order_error,
        total=total,
    )


def optimal_correspondence_costs(
    predicted: CoherentStructuralState,
    target: CoherentStructuralState,
    *,
    max_correspondences: int = 100_000,
) -> CorrespondenceCosts:
    """Return a deterministic optimal witness for the exact I0 discrepancy."""

    best: CorrespondenceCosts | None = None
    best_key = (inf, inf, inf, inf, inf, inf)
    for correspondence in typed_correspondences(
        predicted,
        target,
        max_correspondences=max_correspondences,
    ):
        costs = correspondence_costs(correspondence, predicted, target)
        key = (
            costs.total,
            costs.label,
            costs.simplicial,
            costs.metric,
            costs.relation,
            costs.order,
        )
        if key < best_key:
            best = costs
            best_key = key
    if best is None:
        raise RuntimeError("no typed correspondence was generated")
    return best


def fixed_carrier_tracking_error(
    predicted: TrackedTransition,
    target: TrackedTransition,
) -> float:
    """Return normalized graph error without requiring equal endpoint structure."""

    if predicted.source.relational.schema != target.source.relational.schema:
        raise ValueError("tracking predictions must use the same schema")
    if predicted.target.relational.schema != target.target.relational.schema:
        raise ValueError("tracking predictions must use the same schema")

    denominator = 0
    difference = 0
    for object_name in target.source.relational.schema.objects:
        predicted_component = predicted.components[object_name]
        target_component = target.components[object_name]
        if predicted_component.source != target_component.source:
            raise ValueError("tracking sources must use identical local identifiers")
        if predicted_component.target != target_component.target:
            raise ValueError("tracking targets must use identical local identifiers")
        denominator += 2 * min(
            len(target_component.source),
            len(target_component.target),
        )
        difference += len(
            predicted_component.pairs.symmetric_difference(target_component.pairs)
        )
    if denominator == 0:
        raise ValueError("tracking carriers must be nonempty")
    return difference / denominator


@dataclass(frozen=True)
class Paper3Evaluation:
    """Separated exact and empirical diagnostics for one decoded prediction."""

    quotient: CorrespondenceCosts
    fixed_carrier: FixedCarrierLayerErrors
    tracking: float
    latent_prediction: float | None = None

    @property
    def state_isomorphic(self) -> bool:
        return self.quotient.total == 0.0

    @property
    def tracking_exact(self) -> bool:
        return self.tracking == 0.0

    @property
    def jointly_exact(self) -> bool:
        return self.state_isomorphic and self.tracking_exact

    def as_dict(self) -> dict[str, object]:
        return {
            "quotient": {
                field.name: getattr(self.quotient, field.name)
                for field in fields(self.quotient)
            },
            "fixed_carrier": {
                field.name: getattr(self.fixed_carrier, field.name)
                for field in fields(self.fixed_carrier)
            },
            "tracking": self.tracking,
            "latent_prediction": self.latent_prediction,
            "state_isomorphic": self.state_isomorphic,
            "tracking_exact": self.tracking_exact,
            "jointly_exact": self.jointly_exact,
        }


def evaluate_decoded_prediction(
    example: StructuralTransitionExample,
    predicted_target: CoherentStructuralState,
    predicted_tracking: TrackedTransition,
    *,
    latent_prediction_error: float | None = None,
    max_correspondences: int | None = None,
) -> Paper3Evaluation:
    """Evaluate one valid decoded state and its explicit persistence tracking."""

    _validate_fixed_carrier_pair(predicted_target, example.target)
    if predicted_tracking.source != example.source.core:
        raise ValueError("predicted tracking source must equal the oracle source")
    if predicted_tracking.target != predicted_target.core:
        raise ValueError("predicted tracking target must equal the decoded target")

    latent_error = None
    if latent_prediction_error is not None:
        latent_error = float(latent_prediction_error)
        if not isfinite(latent_error) or latent_error < 0:
            raise ValueError("latent_prediction_error must be finite and nonnegative")

    budget = (
        FROZEN_PAPER3_INTERFACE.max_exact_correspondences
        if max_correspondences is None
        else max_correspondences
    )
    return Paper3Evaluation(
        quotient=optimal_correspondence_costs(
            predicted_target,
            example.target,
            max_correspondences=budget,
        ),
        fixed_carrier=fixed_carrier_exact_losses(
            predicted_target,
            example.target,
        ),
        tracking=fixed_carrier_tracking_error(
            predicted_tracking,
            example.tracking,
        ),
        latent_prediction=latent_error,
    )


@dataclass(frozen=True)
class Paper3InterfaceAudit:
    """Machine-readable audit of the frozen interface's claim discipline."""

    errors: tuple[str, ...]
    training_surrogate_count: int
    exact_evaluator_count: int

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "passed" if self.passed else "failed",
            "training_surrogate_count": self.training_surrogate_count,
            "exact_evaluator_count": self.exact_evaluator_count,
            "errors": list(self.errors),
            "interface": FROZEN_PAPER3_INTERFACE.as_dict(),
        }


def audit_frozen_paper3_interface() -> Paper3InterfaceAudit:
    """Check that no empirical surrogate is mislabeled as a theorem."""

    spec = FROZEN_PAPER3_INTERFACE
    errors: list[str] = []
    if spec.input_regime is not AccessRegime.EXACT_ORACLE:
        errors.append("the minimal input regime must remain exact-oracle")
    if spec.target_regime is not AccessRegime.EXACT_ORACLE:
        errors.append("the minimal target regime must remain exact-oracle")
    if spec.prediction_regime is not AccessRegime.DECODED_VALID:
        errors.append("the minimal prediction regime must remain decoded-valid")
    if "noisy structural recovery" not in spec.excluded_components:
        errors.append("noisy recovery must remain outside the frozen core")
    if "attributes" not in spec.excluded_components or "mass" not in spec.excluded_components:
        errors.append("attribute and mass extensions must remain outside the frozen core")

    training_terms = tuple(
        term
        for term in spec.objective_terms
        if term.role is ObjectiveRole.TRAINING_SURROGATE
    )
    exact_terms = tuple(
        term
        for term in spec.objective_terms
        if term.role is ObjectiveRole.EXACT_EVALUATOR
    )
    if any(term.status is not ClaimStatus.EMPIRICAL for term in training_terms):
        errors.append("every training surrogate must retain empirical status")
    if any(term.status is not ClaimStatus.THEOREM_BACKED for term in exact_terms):
        errors.append("every exact evaluator must retain theorem-backed status")

    expected_training = {
        "jepa_latent",
        "label_surrogate",
        "simplicial_surrogate",
        "metric_surrogate",
        "relation_surrogate",
        "order_surrogate",
        "bridge_surrogate",
        "tracking_surrogate",
        "validity_surrogate",
    }
    if {term.name for term in training_terms} != expected_training:
        errors.append("the frozen training objective ledger is incomplete")
    expected_exact = {
        "fixed_carrier_exact",
        "i0_quotient_metric",
        "tracking_exact",
    }
    if {term.name for term in exact_terms} != expected_exact:
        errors.append("the frozen exact-evaluation ledger is incomplete")

    return Paper3InterfaceAudit(
        errors=tuple(errors),
        training_surrogate_count=len(training_terms),
        exact_evaluator_count=len(exact_terms),
    )
