from __future__ import annotations

from itertools import product
import unittest

from tsi.labeled_topology import (
    LabeledSimplicialComplex,
    are_contiguous,
    commuting_chain_maps_hold,
    commuting_square_holds,
    contiguity_chain_homotopy_audit,
    filtration_betti_signature,
    induced_label_subcomplex,
    is_label_preserving_isomorphism,
    label_filtration,
    label_filtration_preserved,
    label_filtration_stability_audit,
    label_stratum_preserved,
)
from tsi.topological import validate_complex


def complex_from_facets(*facets):
    simplices = {frozenset()}
    for facet in facets:
        members = tuple(facet)
        for mask in range(1, 1 << len(members)):
            simplices.add(
                frozenset(
                    members[index]
                    for index in range(len(members))
                    if mask & (1 << index)
                )
            )
    return frozenset(simplices)


class LabeledFiltrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = LabeledSimplicialComplex(
            complex_from_facets((0, 1, 2)),
            {0: "red", 1: "blue", 2: "red"},
        )

    def test_label_induced_full_subcomplex_is_well_formed(self) -> None:
        red = induced_label_subcomplex(self.state, frozenset(("red",)))

        self.assertEqual(
            red,
            frozenset(
                {
                    frozenset(),
                    frozenset((0,)),
                    frozenset((2,)),
                    frozenset((0, 2)),
                }
            ),
        )
        self.assertEqual(validate_complex(red), red)

    def test_max_label_filtration_is_monotone_and_constructible(self) -> None:
        values = label_filtration(self.state, {"red": 0.0, "blue": 2.0})

        self.assertEqual(values[frozenset()], 0.0)
        self.assertEqual(values[frozenset((0, 2))], 0.0)
        self.assertEqual(values[frozenset((0, 1, 2))], 2.0)
        self.assertEqual(set(values.values()), {0.0, 2.0})
        with self.assertRaisesRegex(ValueError, "missing"):
            label_filtration(self.state, {"red": 0.0})

    def test_label_preserving_isomorphism_preserves_every_tested_stratum(self) -> None:
        renamed = LabeledSimplicialComplex(
            complex_from_facets(("a", "b", "c")),
            {"a": "red", "b": "blue", "c": "red"},
        )
        alignment = {0: "c", 1: "b", 2: "a"}

        self.assertTrue(
            is_label_preserving_isomorphism(self.state, renamed, alignment)
        )
        for labels in (
            frozenset(),
            frozenset(("red",)),
            frozenset(("blue",)),
            frozenset(("red", "blue")),
        ):
            self.assertTrue(
                label_stratum_preserved(self.state, renamed, alignment, labels)
            )
        self.assertTrue(
            label_filtration_preserved(
                self.state,
                renamed,
                alignment,
                {"red": 0.0, "blue": 2.0},
            )
        )


class AlignedTransitionTest(unittest.TestCase):
    def test_commuting_square_implies_equal_chain_composites(self) -> None:
        source = LabeledSimplicialComplex(
            complex_from_facets((0, 1)),
            {0: "x", 1: "x"},
        )
        target = LabeledSimplicialComplex(
            complex_from_facets(("a", "b", "c")),
            {"a": "x", "b": "x", "c": "x"},
        )
        aligned_source = LabeledSimplicialComplex(
            complex_from_facets(("u", "v")),
            {"u": "x", "v": "x"},
        )
        aligned_target = LabeledSimplicialComplex(
            complex_from_facets(("p", "q", "r")),
            {"p": "x", "q": "x", "r": "x"},
        )
        transition = {0: "a", 1: "b"}
        source_alignment = {0: "v", 1: "u"}
        target_alignment = {"a": "r", "b": "p", "c": "q"}
        aligned_transition = {"u": "p", "v": "r"}

        self.assertTrue(
            commuting_square_holds(
                source,
                target,
                aligned_source,
                aligned_target,
                transition,
                aligned_transition,
                source_alignment,
                target_alignment,
            )
        )
        self.assertTrue(
            commuting_chain_maps_hold(
                source,
                target,
                aligned_source,
                aligned_target,
                transition,
                aligned_transition,
                source_alignment,
                target_alignment,
            )
        )
        noncommuting = {"u": "q", "v": "r"}
        self.assertFalse(
            commuting_square_holds(
                source,
                target,
                aligned_source,
                aligned_target,
                transition,
                noncommuting,
                source_alignment,
                target_alignment,
            )
        )


class ContiguityTest(unittest.TestCase):
    def test_contiguity_has_an_explicit_prism_chain_homotopy(self) -> None:
        source = LabeledSimplicialComplex(
            complex_from_facets((0, 1)),
            {0: "x", 1: "x"},
        )
        filled = LabeledSimplicialComplex(
            complex_from_facets(("a", "b", "c")),
            {"a": "x", "b": "x", "c": "x"},
        )
        left = {0: "a", 1: "b"}
        right = {0: "b", 1: "c"}

        self.assertTrue(are_contiguous(source, filled, left, right))
        self.assertTrue(
            contiguity_chain_homotopy_audit(source, filled, left, right)
        )

    def test_edgewise_valid_maps_need_not_be_contiguous(self) -> None:
        source = LabeledSimplicialComplex(
            complex_from_facets((0, 1)),
            {0: "x", 1: "x"},
        )
        boundary = LabeledSimplicialComplex(
            complex_from_facets(("a", "b"), ("b", "c"), ("a", "c")),
            {"a": "x", "b": "x", "c": "x"},
        )
        left = {0: "a", 1: "b"}
        right = {0: "b", 1: "c"}

        self.assertFalse(are_contiguous(source, boundary, left, right))
        self.assertFalse(
            contiguity_chain_homotopy_audit(source, boundary, left, right)
        )


    def test_prism_identity_exhaustively_for_two_to_three_vertex_maps(self) -> None:
        source = LabeledSimplicialComplex(
            complex_from_facets((0, 1)),
            {0: "x", 1: "x"},
        )
        targets = (
            LabeledSimplicialComplex(
                complex_from_facets(("a", "b", "c")),
                {"a": "x", "b": "x", "c": "x"},
            ),
            LabeledSimplicialComplex(
                complex_from_facets(("a", "b"), ("b", "c"), ("a", "c")),
                {"a": "x", "b": "x", "c": "x"},
            ),
        )
        maps = tuple(
            {0: images[0], 1: images[1]}
            for images in product(("a", "b", "c"), repeat=2)
        )

        for target in targets:
            for left, right in product(maps, repeat=2):
                with self.subTest(
                    filled=frozenset(("a", "b", "c")) in target.complex,
                    left=left,
                    right=right,
                ):
                    contiguous = are_contiguous(source, target, left, right)
                    self.assertEqual(
                        contiguity_chain_homotopy_audit(
                            source,
                            target,
                            left,
                            right,
                        ),
                        contiguous,
                    )


class StabilityAndIncompletenessTest(unittest.TestCase):
    def test_common_scale_label_perturbation_obeys_h0_stability(self) -> None:
        state = LabeledSimplicialComplex(
            complex_from_facets((0, 1), (1, 2)),
            {0: "a", 1: "b", 2: "c"},
        )
        audit = label_filtration_stability_audit(
            state,
            {"a": 0.0, "b": 1.0, "c": 2.0},
            {"a": 0.1, "b": 1.2, "c": 1.9},
        )

        self.assertAlmostEqual(audit.epsilon, 0.2)
        self.assertAlmostEqual(audit.filtration_sup_distance, 0.2)
        self.assertTrue(audit.interleaving_holds)
        self.assertTrue(audit.bound_holds)

    def test_persistence_betti_signature_is_not_a_complete_invariant(self) -> None:
        edge = LabeledSimplicialComplex(
            complex_from_facets((0, 1)),
            {0: "only", 1: "only"},
        )
        triangle = LabeledSimplicialComplex(
            complex_from_facets(("a", "b", "c")),
            {"a": "only", "b": "only", "c": "only"},
        )

        self.assertEqual(
            filtration_betti_signature(
                edge,
                {"only": 0.0},
                max_dimension=2,
            ),
            filtration_betti_signature(
                triangle,
                {"only": 0.0},
                max_dimension=2,
            ),
        )
        self.assertNotEqual(len(edge.vertices), len(triangle.vertices))


if __name__ == "__main__":
    unittest.main()

