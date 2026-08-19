from __future__ import annotations

import unittest

from tsi.coherent import CoherenceSignature, CoherentStructuralState
from tsi.dynamical import IntegratedStructuralState, PartialBijection, TrackedTransition
from tsi.order_topology import FinitePreorder
from tsi.paper3_interface import (
    AccessRegime,
    ClaimStatus,
    FROZEN_LAYER_ORDER,
    FROZEN_OBJECTIVE_WEIGHTS,
    FROZEN_PAPER3_INTERFACE,
    ObjectiveRole,
    Paper3Evaluation,
    Paper3ObjectiveWeights,
    Paper3TrainingLosses,
    StructuralTransitionExample,
    audit_frozen_paper3_interface,
    evaluate_decoded_prediction,
    fixed_carrier_exact_losses,
    fixed_carrier_tracking_error,
    optimal_correspondence_costs,
)
from tsi.relational import (
    ArrowSpec,
    FiniteRelation,
    FiniteRelationAssignment,
    FiniteRelationalSchema,
)


SCHEMA = FiniteRelationalSchema(
    objects=("entity",),
    arrows=(ArrowSpec("rel", "entity", "entity"),),
)


def make_state(
    entities: tuple[object, ...],
    *,
    labels: tuple[object, ...] | None = None,
    relation_pairs: frozenset[tuple[object, object]] = frozenset(),
    edges: frozenset[frozenset[object]] = frozenset(),
    spacing: float = 1.0,
    linear_order: bool = False,
    signature: CoherenceSignature | None = None,
) -> CoherentStructuralState:
    labels = labels or tuple("same" for _ in entities)
    relational = FiniteRelationAssignment(
        SCHEMA,
        {"entity": entities},
        {"entity": labels},
        {"rel": FiniteRelation(entities, entities, relation_pairs)},
    )
    tagged = {entity: ("entity", entity) for entity in entities}
    simplices = {
        frozenset(),
        *(frozenset((tagged[entity],)) for entity in entities),
        *(
            frozenset(tagged[entity] for entity in edge)
            for edge in edges
        ),
    }
    distances = tuple(
        tuple(
            abs(left_index - right_index) * spacing
            for right_index, _ in enumerate(entities)
        )
        for left_index, _ in enumerate(entities)
    )
    core = IntegratedStructuralState(
        relational=relational,
        simplices=frozenset(simplices),
        distances=distances,
    )
    if linear_order:
        order_relation = frozenset(
            (left, right)
            for left_index, left in enumerate(core.tagged_entities)
            for right_index, right in enumerate(core.tagged_entities)
            if left_index <= right_index
        )
    else:
        order_relation = frozenset(
            (entity, entity) for entity in core.tagged_entities
        )
    order = FinitePreorder(
        core.tagged_entities,
        order_relation,
        core.tagged_labels,
    )
    return CoherentStructuralState(
        core=core,
        order=order,
        signature=signature or CoherenceSignature(),
    )


def make_transition(
    source: CoherentStructuralState,
    target: CoherentStructuralState,
    pairs: frozenset[tuple[object, object]],
) -> TrackedTransition:
    return TrackedTransition(
        source=source.core,
        target=target.core,
        components={
            "entity": PartialBijection(
                source.core.relational.carriers["entity"],
                target.core.relational.carriers["entity"],
                pairs,
            )
        },
    )


class FrozenContractTest(unittest.TestCase):
    def test_state_regime_and_component_scope_are_frozen(self) -> None:
        spec = FROZEN_PAPER3_INTERFACE

        self.assertEqual(spec.required_layers, FROZEN_LAYER_ORDER)
        self.assertIs(spec.input_regime, AccessRegime.EXACT_ORACLE)
        self.assertIs(spec.target_regime, AccessRegime.EXACT_ORACLE)
        self.assertIs(spec.prediction_regime, AccessRegime.DECODED_VALID)
        self.assertIn("attributes", spec.excluded_components)
        self.assertIn("mass", spec.excluded_components)
        self.assertIn("noisy structural recovery", spec.excluded_components)

    def test_empirical_surrogates_are_not_mislabeled_as_theorems(self) -> None:
        training_terms = [
            term
            for term in FROZEN_PAPER3_INTERFACE.objective_terms
            if term.role is ObjectiveRole.TRAINING_SURROGATE
        ]
        exact_terms = [
            term
            for term in FROZEN_PAPER3_INTERFACE.objective_terms
            if term.role is ObjectiveRole.EXACT_EVALUATOR
        ]

        self.assertTrue(training_terms)
        self.assertTrue(exact_terms)
        self.assertTrue(
            all(term.status is ClaimStatus.EMPIRICAL for term in training_terms)
        )
        self.assertTrue(
            all(term.status is ClaimStatus.THEOREM_BACKED for term in exact_terms)
        )

    def test_static_interface_audit_passes(self) -> None:
        report = audit_frozen_paper3_interface()

        self.assertTrue(report.passed)
        self.assertEqual(report.training_surrogate_count, 9)
        self.assertEqual(report.exact_evaluator_count, 3)

    def test_training_objective_uses_only_nonnegative_empirical_terms(self) -> None:
        state = make_state((0,))
        losses = Paper3TrainingLosses(
            jepa_latent=1.0,
            label_surrogate=2.0,
            simplicial_surrogate=3.0,
            metric_surrogate=4.0,
            relation_surrogate=5.0,
            order_surrogate=6.0,
            bridge_surrogate=0.0,
            tracking_surrogate=7.0,
            validity_surrogate=8.0,
        )

        self.assertEqual(
            losses.weighted_total(FROZEN_OBJECTIVE_WEIGHTS, state),
            36.0,
        )
        with self.assertRaisesRegex(ValueError, "core Paper 3"):
            Paper3ObjectiveWeights(jepa_latent=0.0)


class FixedCarrierEvaluationTest(unittest.TestCase):
    def test_exact_layer_total_is_zero_only_for_literal_fixed_state(self) -> None:
        target = make_state((0, 1))
        same = make_state((0, 1))
        changed_order = make_state((0, 1), linear_order=True)

        self.assertTrue(fixed_carrier_exact_losses(same, target).is_zero)
        changed = fixed_carrier_exact_losses(changed_order, target)
        self.assertGreater(changed.order, 0.0)
        self.assertGreater(changed.total, 0.0)

    def test_exact_zero_does_not_absorb_a_small_metric_perturbation(self) -> None:
        target = make_state((0, 1), spacing=1.0)
        perturbed = make_state((0, 1), spacing=1.0 + 1e-12)

        fixed = fixed_carrier_exact_losses(perturbed, target)
        quotient = optimal_correspondence_costs(perturbed, target)
        self.assertGreater(fixed.total, 0.0)
        self.assertFalse(fixed.is_zero)
        self.assertGreater(quotient.total, 0.0)
        self.assertFalse(Paper3Evaluation(quotient, fixed, 0.0).state_isomorphic)

    def test_relation_error_is_resolved_from_other_layers(self) -> None:
        target = make_state(
            (0, 1),
            relation_pairs=frozenset({(0, 1)}),
        )
        prediction = make_state((0, 1))

        fixed = fixed_carrier_exact_losses(prediction, target)
        quotient = optimal_correspondence_costs(prediction, target)
        self.assertEqual(fixed.label, 0.0)
        self.assertEqual(fixed.simplicial, 0.0)
        self.assertEqual(fixed.metric, 0.0)
        self.assertGreater(fixed.relation, 0.0)
        self.assertGreater(quotient.relation, 0.0)

    def test_quotient_isomorphism_allows_a_nontrivial_renaming(self) -> None:
        left = make_state(
            (0, 1),
            labels=("red", "blue"),
            relation_pairs=frozenset({(0, 1)}),
            edges=frozenset({frozenset((0, 1))}),
        )
        right = make_state(
            ("b", "a"),
            labels=("blue", "red"),
            relation_pairs=frozenset({("a", "b")}),
            edges=frozenset({frozenset(("a", "b"))}),
        )

        self.assertEqual(optimal_correspondence_costs(left, right).total, 0.0)
        with self.assertRaisesRegex(ValueError, "local identifiers"):
            fixed_carrier_exact_losses(left, right)

    def test_example_rejects_mixed_signature_or_carrier_regimes(self) -> None:
        source = make_state((0, 1))
        tracking = TrackedTransition.identity(source.core)
        changed_signature = make_state(
            (0, 1),
            signature=CoherenceSignature(metric_scale=2.0),
        )
        changed_carrier = make_state((0, 1, 2))

        with self.assertRaisesRegex(ValueError, "same signature"):
            StructuralTransitionExample(
                source,
                "stay",
                changed_signature,
                tracking,
            )
        with self.assertRaisesRegex(ValueError, "local identifiers"):
            StructuralTransitionExample(
                source,
                "stay",
                changed_carrier,
                tracking,
            )


class TransitionEvaluationTest(unittest.TestCase):
    def test_exact_endpoints_do_not_imply_exact_tracking(self) -> None:
        state = make_state((0, 1))
        oracle_tracking = TrackedTransition.identity(state.core)
        swapped_tracking = make_transition(
            state,
            state,
            frozenset({(0, 1), (1, 0)}),
        )
        example = StructuralTransitionExample(
            source=state,
            action="stay",
            target=state,
            tracking=oracle_tracking,
        )

        report = evaluate_decoded_prediction(
            example,
            predicted_target=state,
            predicted_tracking=swapped_tracking,
            latent_prediction_error=0.0,
        )
        self.assertTrue(report.state_isomorphic)
        self.assertTrue(report.fixed_carrier.is_zero)
        self.assertFalse(report.tracking_exact)
        self.assertFalse(report.jointly_exact)
        self.assertEqual(report.tracking, 1.0)

    def test_jointly_exact_prediction_passes_every_exact_gate(self) -> None:
        source = make_state((0, 1))
        target = make_state(
            (0, 1),
            relation_pairs=frozenset({(0, 1)}),
        )
        tracking = make_transition(
            source,
            target,
            frozenset({(0, 0), (1, 1)}),
        )
        example = StructuralTransitionExample(
            source=source,
            action="connect",
            target=target,
            tracking=tracking,
        )

        report = evaluate_decoded_prediction(
            example,
            predicted_target=target,
            predicted_tracking=tracking,
            latent_prediction_error=0.25,
        )
        self.assertTrue(report.jointly_exact)
        self.assertEqual(report.quotient.total, 0.0)
        self.assertEqual(report.fixed_carrier.total, 0.0)
        self.assertEqual(report.latent_prediction, 0.25)

    def test_tracking_error_is_a_normalized_fixed_carrier_graph_error(self) -> None:
        state = make_state((0, 1, 2))
        identity = TrackedTransition.identity(state.core)
        partial = make_transition(
            state,
            state,
            frozenset({(0, 0), (1, 1)}),
        )

        error = fixed_carrier_tracking_error(partial, identity)
        self.assertGreater(error, 0.0)
        self.assertLessEqual(error, 1.0)

    def test_invalid_latent_diagnostic_is_rejected(self) -> None:
        state = make_state((0,))
        transition = TrackedTransition.identity(state.core)
        example = StructuralTransitionExample(
            source=state,
            action="stay",
            target=state,
            tracking=transition,
        )

        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    evaluate_decoded_prediction(
                        example,
                        predicted_target=state,
                        predicted_tracking=transition,
                        latent_prediction_error=value,
                    )


if __name__ == "__main__":
    unittest.main()
