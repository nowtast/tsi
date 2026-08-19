import itertools
import unittest

from tsi.relational import (
    ArrowSpec,
    FiniteRelation,
    FiniteRelationAssignment,
    FiniteRelationalSchema,
    PathEquation,
    TypedPath,
    are_naturally_isomorphic,
    composition_error_bound,
    face_inclusion_relation,
    graph_bijection,
    is_relation_isomorphism,
    metric_threshold_relation,
    relational_discrepancy,
    threshold_profile_preserved,
)
from tsi import FiniteMetricState


def relation(
    source: tuple[str, ...],
    target: tuple[str, ...],
    pairs: set[tuple[str, str]],
) -> FiniteRelation:
    return FiniteRelation(source, target, frozenset(pairs))


def composition_schema() -> FiniteRelationalSchema:
    return FiniteRelationalSchema(
        objects=("A", "B", "C"),
        arrows=(
            ArrowSpec("f", "A", "B"),
            ArrowSpec("g", "B", "C"),
            ArrowSpec("h", "A", "C"),
        ),
        equations=(
            PathEquation(
                TypedPath("A", ("h",)),
                TypedPath("A", ("f", "g")),
            ),
        ),
    )


def composition_assignment(*, swapped_h: bool) -> FiniteRelationAssignment:
    schema = composition_schema()
    carriers = {
        "A": ("a1", "a2"),
        "B": ("b1", "b2"),
        "C": ("c1", "c2"),
    }
    return FiniteRelationAssignment(
        schema=schema,
        carriers=carriers,
        labels={name: ("entity", "entity") for name in carriers},
        generators={
            "f": relation(
                carriers["A"],
                carriers["B"],
                {("a1", "b1"), ("a2", "b2")},
            ),
            "g": relation(
                carriers["B"],
                carriers["C"],
                {("b1", "c1"), ("b2", "c2")},
            ),
            "h": relation(
                carriers["A"],
                carriers["C"],
                (
                    {("a1", "c2"), ("a2", "c1")}
                    if swapped_h
                    else {("a1", "c1"), ("a2", "c2")}
                ),
            ),
        },
    )


def loop_schema() -> FiniteRelationalSchema:
    return FiniteRelationalSchema(
        objects=("node",),
        arrows=(ArrowSpec("edge", "node", "node"),),
    )


def loop_state(
    entities: tuple[str, ...],
    pairs: set[tuple[str, str]],
) -> FiniteRelationAssignment:
    schema = loop_schema()
    return FiniteRelationAssignment(
        schema=schema,
        carriers={"node": entities},
        labels={"node": tuple("entity" for _ in entities)},
        generators={"edge": relation(entities, entities, pairs)},
    )


class FiniteRelationCategoryTest(unittest.TestCase):
    def test_associativity_and_diagonal_identities(self) -> None:
        first = relation(("x0", "x1"), ("y0", "y1"), {("x0", "y0"), ("x1", "y1")})
        second = relation(("y0", "y1"), ("z0", "z1"), {("y0", "z1"), ("y1", "z0")})
        third = relation(("z0", "z1"), ("w0",), {("z0", "w0"), ("z1", "w0")})

        self.assertEqual(
            third.compose(second.compose(first)),
            third.compose(second).compose(first),
        )
        self.assertEqual(FiniteRelation.identity(first.target).compose(first), first)
        self.assertEqual(first.compose(FiniteRelation.identity(first.source)), first)

    def test_category_axioms_exhaustively_on_two_entities(self) -> None:
        carrier = ("0", "1")
        possible_pairs = tuple(itertools.product(carrier, repeat=2))
        relations = [
            relation(
                carrier,
                carrier,
                {pair for index, pair in enumerate(possible_pairs) if mask & (1 << index)},
            )
            for mask in range(1 << len(possible_pairs))
        ]
        identity = FiniteRelation.identity(carrier)

        for current in relations:
            self.assertEqual(identity.compose(current), current)
            self.assertEqual(current.compose(identity), current)
        for first, second, third in itertools.product(relations, repeat=3):
            self.assertEqual(
                third.compose(second.compose(first)),
                third.compose(second).compose(first),
            )

    def test_isomorphisms_are_bijection_graphs(self) -> None:
        forward = FiniteRelation.graph(
            ("x0", "x1"),
            ("y0", "y1"),
            {"x0": "y1", "x1": "y0"},
        )
        inverse = forward.converse()

        self.assertEqual(dict(graph_bijection(forward) or {}), {"x0": "y1", "x1": "y0"})
        self.assertTrue(is_relation_isomorphism(forward, inverse))

        multivalued = relation(
            ("x0", "x1"),
            ("y0", "y1"),
            {("x0", "y0"), ("x0", "y1"), ("x1", "y1")},
        )
        self.assertIsNone(graph_bijection(multivalued))
        self.assertFalse(is_relation_isomorphism(multivalued, multivalued.converse()))

    def test_isomorphism_characterization_exhaustively_on_two_entities(self) -> None:
        source = ("x0", "x1")
        target = ("y0", "y1")
        forward_pairs = tuple(itertools.product(source, target))
        inverse_pairs = tuple(itertools.product(target, source))
        forward_relations = [
            relation(
                source,
                target,
                {pair for index, pair in enumerate(forward_pairs) if mask & (1 << index)},
            )
            for mask in range(1 << len(forward_pairs))
        ]
        inverse_relations = [
            relation(
                target,
                source,
                {pair for index, pair in enumerate(inverse_pairs) if mask & (1 << index)},
            )
            for mask in range(1 << len(inverse_pairs))
        ]

        for forward in forward_relations:
            mapping = graph_bijection(forward)
            for inverse in inverse_relations:
                categorical_inverse = is_relation_isomorphism(forward, inverse)
                self.assertEqual(
                    categorical_inverse,
                    mapping is not None and inverse == forward.converse(),
                )


class PresentedSchemaTest(unittest.TestCase):
    def test_free_path_evaluation_and_zero_defect_descent(self) -> None:
        assignment = composition_assignment(swapped_h=False)
        composite = assignment.path_relation(TypedPath("A", ("f", "g")))

        self.assertEqual(composite, assignment.generators["h"])
        self.assertEqual(assignment.composition_defect, 0.0)
        self.assertTrue(assignment.is_functorial)

    def test_swapped_composite_has_maximal_defect(self) -> None:
        assignment = composition_assignment(swapped_h=True)

        self.assertEqual(assignment.composition_defect, 1.0)
        self.assertFalse(assignment.is_functorial)
        with self.assertRaisesRegex(ValueError, "functorial states"):
            relational_discrepancy(assignment, assignment)


class RelationalDiscrepancyTest(unittest.TestCase):
    def test_zero_detects_natural_isomorphism_under_relabeling(self) -> None:
        left = loop_state(
            ("a", "b", "c"),
            {("a", "b"), ("b", "c")},
        )
        right = loop_state(
            ("v", "u", "w"),
            {("u", "v"), ("v", "w")},
        )

        self.assertEqual(relational_discrepancy(left, right), 0.0)
        self.assertTrue(are_naturally_isomorphic(left, right))

    def test_path_and_fork_with_equal_counts_are_not_isomorphic(self) -> None:
        path = loop_state(
            ("1", "2", "3"),
            {("1", "2"), ("2", "3")},
        )
        fork = loop_state(
            ("1", "2", "3"),
            {("1", "2"), ("1", "3")},
        )

        self.assertEqual(len(path.generators["edge"].pairs), len(fork.generators["edge"].pairs))
        self.assertGreater(relational_discrepancy(path, fork), 0.0)

    def test_extended_metric_axioms_exhaustively_on_two_entities(self) -> None:
        entities = ("0", "1")
        possible_pairs = tuple(itertools.product(entities, repeat=2))
        states = [
            loop_state(
                entities,
                {pair for index, pair in enumerate(possible_pairs) if mask & (1 << index)},
            )
            for mask in range(1 << len(possible_pairs))
        ]
        distances = {
            (i, j): relational_discrepancy(left, right)
            for i, left in enumerate(states)
            for j, right in enumerate(states)
        }

        for i in range(len(states)):
            self.assertEqual(distances[i, i], 0.0)
            for j in range(len(states)):
                self.assertEqual(distances[i, j], distances[j, i])
                for k in range(len(states)):
                    self.assertLessEqual(
                        distances[i, k],
                        distances[i, j] + distances[j, k] + 1e-12,
                    )


class CompositionBoundTest(unittest.TestCase):
    def test_bound_holds_for_all_relations_on_two_element_carriers(self) -> None:
        x = ("x0", "x1")
        y = ("y0", "y1")
        z = ("z0", "z1")
        xy_pairs = tuple(itertools.product(x, y))
        yz_pairs = tuple(itertools.product(y, z))
        first_relations = [
            relation(
                x,
                y,
                {pair for index, pair in enumerate(xy_pairs) if mask & (1 << index)},
            )
            for mask in range(1 << len(xy_pairs))
        ]
        second_relations = [
            relation(
                y,
                z,
                {pair for index, pair in enumerate(yz_pairs) if mask & (1 << index)},
            )
            for mask in range(1 << len(yz_pairs))
        ]

        for first, perturbed_first, second, perturbed_second in itertools.product(
            first_relations,
            first_relations,
            second_relations,
            second_relations,
        ):
            actual, bound = composition_error_bound(
                first,
                perturbed_first,
                second,
                perturbed_second,
            )
            self.assertLessEqual(actual, bound)


class StructuralBridgeTest(unittest.TestCase):
    def test_face_inclusion_relation_recovers_incidence(self) -> None:
        a = frozenset({"a"})
        b = frozenset({"b"})
        edge = frozenset({"a", "b"})
        inclusion = face_inclusion_relation((a, b, edge))

        self.assertEqual(
            inclusion.pairs,
            frozenset({(a, a), (b, b), (edge, edge), (a, edge), (b, edge)}),
        )
        with self.assertRaisesRegex(ValueError, "closed under"):
            face_inclusion_relation((a, edge))

    def test_critical_thresholds_exactly_detect_isometry(self) -> None:
        left = FiniteMetricState(
            entities=("a", "b", "c"),
            distances=((0, 1, 2), (1, 0, 1), (2, 1, 0)),
            labels=("x", "x", "x"),
        )
        right = FiniteMetricState(
            entities=("v", "u", "w"),
            distances=((0, 1, 1), (1, 0, 2), (1, 2, 0)),
            labels=("x", "x", "x"),
        )
        isometry = {"a": "u", "b": "v", "c": "w"}
        non_isometry = {"a": "v", "b": "u", "c": "w"}

        self.assertTrue(threshold_profile_preserved(left, right, isometry))
        self.assertFalse(threshold_profile_preserved(left, right, non_isometry))
        self.assertEqual(
            len(metric_threshold_relation(left, 1.0).pairs),
            7,
        )


if __name__ == "__main__":
    unittest.main()
