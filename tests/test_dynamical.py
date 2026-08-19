from __future__ import annotations

import math
import unittest
from itertools import combinations, permutations, product

from tsi.dynamical import (
    FiniteActionHistory,
    IntegratedStructuralState,
    PartialBijection,
    TrackedTransition,
    are_integrated_isomorphic,
    build_action_history,
    causal_structural_rollout,
    collision_pairs,
    integrated_structural_discrepancy,
    intervene_actions,
    rollout_error_bound,
    tracking_composition_error_bound,
    tracking_difference,
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
    *,
    entities: tuple[object, ...] = (0, 1),
    labels: tuple[object, ...] = ("same", "same"),
    relation_pairs: frozenset[tuple[object, object]] = frozenset(),
    edge: bool = False,
    distance: float = 1.0,
) -> IntegratedStructuralState:
    relation = FiniteRelation(entities, entities, relation_pairs)
    relational = FiniteRelationAssignment(
        SCHEMA,
        {"entity": entities},
        {"entity": labels},
        {"rel": relation},
    )
    tagged = tuple(("entity", entity) for entity in entities)
    simplices = {
        frozenset(),
        *(frozenset((vertex,)) for vertex in tagged),
    }
    if edge:
        simplices.add(frozenset(tagged))
    distances = tuple(
        tuple(0.0 if left == right else float(distance) for right in entities)
        for left in entities
    )
    return IntegratedStructuralState(
        relational=relational,
        simplices=frozenset(simplices),
        distances=distances,
    )


def all_partial_bijections(
    source: tuple[object, ...],
    target: tuple[object, ...],
) -> tuple[PartialBijection, ...]:
    result = []
    for size in range(min(len(source), len(target)) + 1):
        for domain in combinations(source, size):
            for image in combinations(target, size):
                for ordered_image in permutations(image):
                    result.append(
                        PartialBijection(
                            source,
                            target,
                            frozenset(zip(domain, ordered_image, strict=True)),
                        )
                    )
    return tuple(result)


def transition(
    source: IntegratedStructuralState,
    target: IntegratedStructuralState,
    component: PartialBijection,
) -> TrackedTransition:
    return TrackedTransition(source, target, {"entity": component})


class PartialBijectionCategoryTest(unittest.TestCase):
    def test_category_laws_exhaustively_on_two_entities(self) -> None:
        carrier = (0, 1)
        arrows = all_partial_bijections(carrier, carrier)
        identity = PartialBijection.identity(carrier)

        for arrow in arrows:
            self.assertEqual(identity.compose(arrow), arrow)
            self.assertEqual(arrow.compose(identity), arrow)
            inverse = arrow.inverse()
            self.assertEqual(
                inverse.compose(arrow).pairs,
                frozenset((value, value) for value in arrow.domain),
            )
            self.assertEqual(
                arrow.compose(inverse).pairs,
                frozenset((value, value) for value in arrow.image),
            )

        for first, second, third in product(arrows, repeat=3):
            self.assertEqual(
                third.compose(second.compose(first)),
                third.compose(second).compose(first),
            )

    def test_many_to_one_pairs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "injective"):
            PartialBijection(
                (0, 1),
                (0, 1),
                frozenset({(0, 0), (1, 0)}),
            )


class IntegratedStructuralDiscrepancyTest(unittest.TestCase):
    def test_zero_exactly_detects_simultaneous_relabeling(self) -> None:
        left = make_state(
            entities=("a", "b"),
            labels=("red", "blue"),
            relation_pairs=frozenset({("a", "b")}),
            edge=True,
            distance=2.0,
        )
        right = make_state(
            entities=("v", "u"),
            labels=("blue", "red"),
            relation_pairs=frozenset({("u", "v")}),
            edge=True,
            distance=2.0,
        )
        wrong_relation = make_state(
            entities=("v", "u"),
            labels=("blue", "red"),
            relation_pairs=frozenset({("v", "u")}),
            edge=True,
            distance=2.0,
        )

        self.assertTrue(are_integrated_isomorphic(left, right))
        self.assertEqual(integrated_structural_discrepancy(left, right), 0.0)
        self.assertGreater(
            integrated_structural_discrepancy(left, wrong_relation),
            0.0,
        )

    def test_extended_metric_axioms_on_two_entity_state_family(self) -> None:
        carrier = (0, 1)
        all_pairs = tuple(product(carrier, repeat=2))
        states = []
        for mask in range(16):
            pairs = frozenset(
                pair for index, pair in enumerate(all_pairs) if mask & (1 << index)
            )
            for edge in (False, True):
                for distance in (1.0, 2.0):
                    states.append(
                        make_state(
                            relation_pairs=pairs,
                            edge=edge,
                            distance=distance,
                        )
                    )

        distances = {
            (left_index, right_index): integrated_structural_discrepancy(left, right)
            for left_index, left in enumerate(states)
            for right_index, right in enumerate(states)
        }
        for index in range(len(states)):
            self.assertEqual(distances[(index, index)], 0.0)
        for left_index in range(len(states)):
            for right_index in range(len(states)):
                self.assertAlmostEqual(
                    distances[(left_index, right_index)],
                    distances[(right_index, left_index)],
                )
        for left_index, middle_index, right_index in product(
            range(len(states)),
            repeat=3,
        ):
            self.assertLessEqual(
                distances[(left_index, right_index)],
                distances[(left_index, middle_index)]
                + distances[(middle_index, right_index)]
                + 1e-9,
            )

    def test_incompatible_label_multiplicity_gives_infinity(self) -> None:
        left = make_state(labels=("red", "blue"))
        right = make_state(labels=("red", "red"))
        self.assertTrue(
            math.isinf(integrated_structural_discrepancy(left, right))
        )


class TrackedTransitionTest(unittest.TestCase):
    def test_category_laws_and_tracking_bound_exhaustively(self) -> None:
        state = make_state(edge=True, distance=1.0)
        components = all_partial_bijections((0, 1), (0, 1))
        transitions = tuple(transition(state, state, item) for item in components)
        identity = TrackedTransition.identity(state)

        for arrow in transitions:
            self.assertEqual(identity.compose(arrow), arrow)
            self.assertEqual(arrow.compose(identity), arrow)

        for first, second, third in product(transitions, repeat=3):
            self.assertEqual(
                third.compose(second.compose(first)),
                third.compose(second).compose(first),
            )

        for before, perturbed_before, after, perturbed_after in product(
            transitions,
            repeat=4,
        ):
            actual, bound = tracking_composition_error_bound(
                before,
                perturbed_before,
                after,
                perturbed_after,
            )
            self.assertLessEqual(actual, bound)

    def test_layer_preservation_is_closed_under_composition(self) -> None:
        states = (
            make_state(
                relation_pairs=frozenset({(0, 1)}),
                edge=True,
                distance=1.0,
            ),
            make_state(
                relation_pairs=frozenset({(1, 0)}),
                edge=True,
                distance=1.0,
            ),
            make_state(
                relation_pairs=frozenset({(0, 0), (1, 1)}),
                edge=False,
                distance=2.0,
            ),
        )
        components = all_partial_bijections((0, 1), (0, 1))
        first_steps = tuple(transition(states[0], states[1], item) for item in components)
        second_steps = tuple(transition(states[1], states[2], item) for item in components)

        for first, second in product(first_steps, second_steps):
            composite = second.compose(first)
            if first.preserves_topology and second.preserves_topology:
                self.assertTrue(composite.preserves_topology)
            if first.preserves_geometry and second.preserves_geometry:
                self.assertTrue(composite.preserves_geometry)
            if first.preserves_relations and second.preserves_relations:
                self.assertTrue(composite.preserves_relations)

    def test_full_zero_defect_is_integrated_isomorphism(self) -> None:
        state = make_state(
            relation_pairs=frozenset({(0, 1), (1, 0)}),
            edge=True,
            distance=1.0,
        )
        exact = TrackedTransition.identity(state)
        self.assertTrue(exact.is_exact_conservative)
        self.assertEqual(exact.defects.turnover, 0)
        self.assertEqual(exact.defects.topological, 0)
        self.assertEqual(exact.defects.geometric, 0.0)
        self.assertEqual(exact.defects.relational, 0)

        changed = make_state(
            relation_pairs=frozenset({(0, 1), (1, 0)}),
            edge=True,
            distance=2.0,
        )
        changed_transition = transition(
            state,
            changed,
            PartialBijection.identity((0, 1)),
        )
        self.assertFalse(changed_transition.is_exact_conservative)
        self.assertGreater(changed_transition.defects.geometric, 0.0)

    def test_endpoint_state_does_not_determine_tracking(self) -> None:
        state = make_state(
            relation_pairs=frozenset({(0, 1), (1, 0)}),
            edge=True,
            distance=1.0,
        )
        identity = TrackedTransition.identity(state)
        swap = transition(
            state,
            state,
            PartialBijection(
                (0, 1),
                (0, 1),
                frozenset({(0, 1), (1, 0)}),
            ),
        )
        self.assertTrue(identity.is_exact_conservative)
        self.assertTrue(swap.is_exact_conservative)
        self.assertEqual(
            integrated_structural_discrepancy(identity.target, swap.target),
            0.0,
        )
        self.assertEqual(tracking_difference(identity, swap), 4)

    def test_empty_tracking_is_vacuously_preserving_with_positive_turnover(
        self,
    ) -> None:
        source = make_state(
            relation_pairs=frozenset({(0, 1)}),
            edge=True,
            distance=1.0,
        )
        target = make_state(
            relation_pairs=frozenset({(1, 0)}),
            edge=False,
            distance=2.0,
        )
        empty = transition(
            source,
            target,
            PartialBijection((0, 1), (0, 1), frozenset()),
        )

        self.assertTrue(empty.preserves_topology)
        self.assertTrue(empty.preserves_geometry)
        self.assertTrue(empty.preserves_relations)
        self.assertEqual(empty.defects.topological, 0)
        self.assertEqual(empty.defects.geometric, 0.0)
        self.assertEqual(empty.defects.relational, 0)
        self.assertEqual(empty.turnover, 4)
        self.assertFalse(empty.is_exact_conservative)


class ActionHistoryTest(unittest.TestCase):
    def test_unique_composites_and_local_to_global_preservation(self) -> None:
        state = make_state(
            relation_pairs=frozenset({(0, 1), (1, 0)}),
            edge=True,
            distance=1.0,
        )

        def update(
            source: IntegratedStructuralState,
            action: str,
            word: tuple[str, ...],
        ) -> TrackedTransition:
            del word
            if action == "stay":
                component = PartialBijection.identity((0, 1))
            else:
                component = PartialBijection(
                    (0, 1),
                    (0, 1),
                    frozenset({(0, 1), (1, 0)}),
                )
            return transition(source, state, component)

        history = build_action_history(state, ("stay", "swap"), 3, update)
        self.assertIsInstance(history, FiniteActionHistory)

        for word in history.states:
            for split in range(len(word) + 1):
                prefix = word[:split]
                direct = history.transition((), word)
                staged = history.transition(prefix, word).compose(
                    history.transition((), prefix)
                )
                self.assertEqual(direct, staged)
            self.assertTrue(history.transition((), word).is_exact_conservative)


class StabilityAndCausalityTest(unittest.TestCase):
    def test_rollout_error_bound_unrolls_the_recursion(self) -> None:
        self.assertAlmostEqual(
            rollout_error_bound((1.0, 2.0, 3.0), (2.0, 0.5, 3.0)),
            10.5,
        )
        self.assertAlmostEqual(
            rollout_error_bound((0.1,) * 4, (1.0,) * 4),
            0.4,
        )

    def test_intervention_preserves_the_preintervention_prefix(self) -> None:
        states_by_bit = {
            0: make_state(relation_pairs=frozenset()),
            1: make_state(relation_pairs=frozenset({(0, 0)})),
        }
        def update(
            source: IntegratedStructuralState,
            action: int,
            noise: object,
            time: int,
        ) -> TrackedTransition:
            del time
            current_bit = 0 if source == states_by_bit[0] else 1
            next_bit = current_bit ^ int(action) ^ int(noise)
            return transition(
                source,
                states_by_bit[next_bit],
                PartialBijection.identity((0, 1)),
            )

        factual_actions = (0, 1, 0)
        counterfactual_actions = intervene_actions(factual_actions, {1: 0})
        exogenous = (0, 1, 0)
        factual_states, _ = causal_structural_rollout(
            states_by_bit[0],
            factual_actions,
            exogenous,
            update,
        )
        counterfactual_states, _ = causal_structural_rollout(
            states_by_bit[0],
            counterfactual_actions,
            exogenous,
            update,
        )
        self.assertEqual(factual_states[:2], counterfactual_states[:2])
        self.assertNotEqual(factual_states[2], counterfactual_states[2])

    def test_observation_does_not_identify_intervention(self) -> None:
        observational_model_one = []
        observational_model_two = []
        intervention_model_one = []
        intervention_model_two = []
        for exogenous in (0, 1):
            action = exogenous
            observational_model_one.append((action, action))
            observational_model_two.append((action, exogenous))
            intervention_model_one.append((0, 0))
            intervention_model_two.append((0, exogenous))

        self.assertEqual(observational_model_one, observational_model_two)
        self.assertNotEqual(intervention_model_one, intervention_model_two)


class AmbientCollisionTest(unittest.TestCase):
    def test_identical_endpoints_admit_collision_and_collision_free_motion(self) -> None:
        radii = {"left": 0.25, "right": 0.25}
        start = {"left": (-1.0, 0.0), "right": (1.0, 0.0)}
        end = {"left": (1.0, 0.0), "right": (-1.0, 0.0)}
        self.assertEqual(collision_pairs(start, radii), frozenset())
        self.assertEqual(collision_pairs(end, radii), frozenset())

        straight_midpoint = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
        self.assertEqual(
            collision_pairs(straight_midpoint, radii),
            frozenset({frozenset({"left", "right"})}),
        )

        for step in range(101):
            time = step / 100
            first = (-math.cos(math.pi * time), math.sin(math.pi * time))
            second = (math.cos(math.pi * time), -math.sin(math.pi * time))
            self.assertEqual(
                collision_pairs({"left": first, "right": second}, radii),
                frozenset(),
            )

    def test_rigid_motion_preserves_collision_relation(self) -> None:
        positions = {
            "a": (0.0, 0.0),
            "b": (0.5, 0.0),
            "c": (3.0, 0.0),
        }
        radii = {"a": 0.3, "b": 0.3, "c": 0.3}
        transformed = {
            entity: (-point[1] + 4.0, point[0] - 2.0)
            for entity, point in positions.items()
        }
        self.assertEqual(
            collision_pairs(positions, radii),
            collision_pairs(transformed, radii),
        )


if __name__ == "__main__":
    unittest.main()
