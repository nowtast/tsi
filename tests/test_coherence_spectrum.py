from __future__ import annotations

import math
import unittest
from itertools import product

from tsi.coherence_spectrum import (
    CorrespondenceSpectrum,
    LayerDistortionVector,
    alignment_frustration,
    audit_pareto_triangle,
    coherent_correspondence_spectrum,
    pareto_minima,
    signature_weights,
)
from tsi.coherent import (
    CoherenceSignature,
    CoherentStructuralState,
    coherent_structural_discrepancy,
)
from tsi.dynamical import IntegratedStructuralState
from tsi.order_topology import FinitePreorder
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


def state(
    entities: tuple[object, ...],
    *,
    labels: tuple[object, ...] | None = None,
    relation_pairs: frozenset[tuple[object, object]] = frozenset(),
    spacing: float = 1.0,
    edge: bool = False,
    linear_order: bool = False,
) -> CoherentStructuralState:
    labels = labels or tuple("same" for _ in entities)
    relational = FiniteRelationAssignment(
        SCHEMA,
        {"entity": entities},
        {"entity": labels},
        {"rel": FiniteRelation(entities, entities, relation_pairs)},
    )
    tagged = tuple(("entity", entity) for entity in entities)
    simplices = {
        frozenset(),
        *(frozenset((entity,)) for entity in tagged),
    }
    if edge and len(tagged) == 2:
        simplices.add(frozenset(tagged))
    distances = tuple(
        tuple(
            abs(left_index - right_index) * spacing
            for right_index in range(len(entities))
        )
        for left_index in range(len(entities))
    )
    core = IntegratedStructuralState(
        relational=relational,
        simplices=frozenset(simplices),
        distances=distances,
    )
    if linear_order:
        order_relation = frozenset(
            (left, right)
            for left_index, left in enumerate(tagged)
            for right_index, right in enumerate(tagged)
            if left_index <= right_index
        )
    else:
        order_relation = frozenset((entity, entity) for entity in tagged)
    order = FinitePreorder(tagged, order_relation, core.tagged_labels)
    return CoherentStructuralState(core, order, CoherenceSignature())


def conflict_pair() -> tuple[CoherentStructuralState, CoherentStructuralState]:
    labels = ("red", "blue")
    left = state(
        (0, 1),
        labels=labels,
        relation_pairs=frozenset({(0, 0)}),
    )
    right = state(
        (0, 1),
        labels=labels,
        relation_pairs=frozenset({(1, 1)}),
    )
    return left, right


def audit_family() -> tuple[CoherentStructuralState, ...]:
    conflict_left, conflict_right = conflict_pair()
    return (
        state((0,)),
        state((0, 1)),
        state((0, 1), spacing=2.0),
        conflict_left,
        conflict_right,
        state((0, 1), edge=True, linear_order=True),
    )


class VectorAndFrontierTest(unittest.TestCase):
    def test_pareto_filter_removes_dominated_vectors_and_duplicates(self) -> None:
        first = LayerDistortionVector(0, 1, 0, 1, 0)
        second = LayerDistortionVector(1, 0, 0, 0, 0)
        dominated = LayerDistortionVector(1, 1, 0, 1, 0)

        self.assertEqual(
            pareto_minima((dominated, second, first, first)),
            (first, second),
        )

    def test_spectrum_records_unattained_coordinatewise_ideal(self) -> None:
        first = LayerDistortionVector(0, 1, 0, 1, 0)
        second = LayerDistortionVector(1, 0, 0, 0, 0)
        spectrum = CorrespondenceSpectrum.from_vectors((first, second))

        self.assertEqual(spectrum.pareto, (first, second))
        self.assertEqual(spectrum.ideal, LayerDistortionVector.zero())
        self.assertFalse(spectrum.ideal_is_attainable)
        self.assertTrue(spectrum.upper_contains(first + second))

    def test_invalid_vectors_spectra_and_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            LayerDistortionVector(-1, 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            LayerDistortionVector(math.inf, 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            CorrespondenceSpectrum.from_vectors(())

        vector = LayerDistortionVector.zero()
        for weights in ((1, 1), (1, 1, 1, 1, 0), (1, 1, math.nan, 1, 1)):
            with self.assertRaises(ValueError):
                vector.scalarize(weights)


class AlignmentConflictTest(unittest.TestCase):
    def test_two_point_example_has_exact_strict_frontier(self) -> None:
        left, right = conflict_pair()
        spectrum = coherent_correspondence_spectrum(left, right)
        expected = (
            LayerDistortionVector(0, 0, 0, 1, 0),
            LayerDistortionVector(1, 0, 0, 0, 0),
        )

        self.assertEqual(spectrum.pareto, expected)
        self.assertEqual(spectrum.ideal, LayerDistortionVector.zero())
        self.assertFalse(spectrum.ideal_is_attainable)
        self.assertFalse(spectrum.has_zero)

    def test_frustration_equals_the_cheaper_of_two_conflicting_layers(self) -> None:
        left, right = conflict_pair()
        spectrum = coherent_correspondence_spectrum(left, right)
        weights = (2, 3, 4, 5, 6)
        frustration = alignment_frustration(spectrum, weights)

        self.assertEqual(frustration.independent_lower_bound, 0.0)
        self.assertEqual(frustration.joint_cost, 2.0)
        self.assertEqual(frustration.gap, min(weights[0], weights[3]))
        self.assertFalse(frustration.is_zero)

    def test_one_sided_singleton_has_no_alignment_choice_or_frustration(self) -> None:
        cases = (
            (state((0,)), state(("a",))),
            (state((0,)), state(("a", "b"))),
            (state((0, 1)), state(("a",))),
        )
        for left, right in cases:
            with self.subTest(left=len(left.core.tagged_entities), right=len(right.core.tagged_entities)):
                spectrum = coherent_correspondence_spectrum(left, right)
                self.assertEqual(spectrum.correspondence_count, 1)
                self.assertTrue(spectrum.ideal_is_attainable)
                self.assertTrue(
                    alignment_frustration(
                        spectrum,
                        signature_weights(left),
                    ).is_zero
                )


class ExhaustiveSmallFamilyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.states = audit_family()
        cls.spectra = {
            (i, j): coherent_correspondence_spectrum(left, right)
            for i, left in enumerate(cls.states)
            for j, right in enumerate(cls.states)
        }

    def test_symmetry_diagonal_and_i0_scalarization_on_all_36_pairs(self) -> None:
        for i, j in product(range(len(self.states)), repeat=2):
            with self.subTest(i=i, j=j):
                forward = self.spectra[(i, j)]
                backward = self.spectra[(j, i)]
                self.assertEqual(forward.attainable, backward.attainable)
                self.assertEqual(forward.pareto, backward.pareto)
                expected = coherent_structural_discrepancy(
                    self.states[i],
                    self.states[j],
                )
                self.assertAlmostEqual(
                    forward.scalarized_value(signature_weights(self.states[i])),
                    expected,
                )
                frustration = alignment_frustration(
                    forward,
                    signature_weights(self.states[i]),
                )
                self.assertEqual(
                    frustration.is_zero,
                    forward.ideal_is_attainable,
                )
                if i == j:
                    self.assertEqual(
                        forward.pareto,
                        (LayerDistortionVector.zero(),),
                    )

    def test_pareto_minkowski_triangle_on_all_216_ordered_triples(self) -> None:
        tested_frontier_pairs = 0
        for i, j, k in product(range(len(self.states)), repeat=3):
            audit = audit_pareto_triangle(
                self.spectra[(i, j)],
                self.spectra[(j, k)],
                self.spectra[(i, k)],
            )
            tested_frontier_pairs += audit.tested_frontier_pairs
            with self.subTest(i=i, j=j, k=k):
                self.assertTrue(audit.passed, audit.violations)
        self.assertGreaterEqual(tested_frontier_pairs, 216)


if __name__ == "__main__":
    unittest.main()
