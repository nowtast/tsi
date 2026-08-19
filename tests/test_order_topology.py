from itertools import product
from math import inf
import unittest

from tsi.order_topology import (
    FinitePreorder,
    are_order_isomorphic,
    containment_prediction_error,
    is_continuous,
    is_discrete_topology,
    is_monotone,
    is_t0_topology,
    is_t1_topology,
    is_topology,
    monotonicity_defect,
    order_discrepancy,
    specialization_relation,
)


def all_preorders(elements: tuple[int, ...]) -> tuple[FinitePreorder, ...]:
    pairs = tuple(product(elements, repeat=2))
    orders: list[FinitePreorder] = []
    for mask in range(1 << len(pairs)):
        relation = frozenset(
            pair for index, pair in enumerate(pairs) if mask & (1 << index)
        )
        try:
            orders.append(FinitePreorder(elements, relation, elements))
        except ValueError:
            pass
    return tuple(orders)


def all_topologies(
    elements: tuple[int, ...],
) -> tuple[frozenset[frozenset[int]], ...]:
    subsets = tuple(
        frozenset(
            elements[index]
            for index in range(len(elements))
            if mask & (1 << index)
        )
        for mask in range(1 << len(elements))
    )
    topologies: list[frozenset[frozenset[int]]] = []
    for mask in range(1 << len(subsets)):
        candidate = frozenset(
            subset
            for index, subset in enumerate(subsets)
            if mask & (1 << index)
        )
        if is_topology(elements, candidate):
            topologies.append(candidate)
    return tuple(topologies)


class FiniteOrderTopologyTest(unittest.TestCase):
    def test_public_api_exports_finite_preorder(self) -> None:
        import tsi

        self.assertIs(tsi.FinitePreorder, FinitePreorder)
        self.assertIs(tsi.order_discrepancy, order_discrepancy)

    def test_invalid_preorders_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reflexive"):
            FinitePreorder((0, 1), frozenset({(0, 0), (0, 1)}), ("a", "b"))
        with self.assertRaisesRegex(ValueError, "transitive"):
            FinitePreorder(
                (0, 1, 2),
                frozenset(
                    {
                        (0, 0),
                        (1, 1),
                        (2, 2),
                        (0, 1),
                        (1, 2),
                    }
                ),
                ("a", "b", "c"),
            )

    def test_all_three_point_preorders_recover_from_upper_topology(self) -> None:
        orders = all_preorders((0, 1, 2))
        self.assertEqual(len(orders), 29)
        for order in orders:
            topology = order.upper_topology
            self.assertTrue(is_topology(order.elements, topology))
            self.assertEqual(
                specialization_relation(order.elements, topology),
                order.relation,
            )

    def test_all_three_point_topologies_reconstruct_as_upper_sets(self) -> None:
        topologies = all_topologies((0, 1, 2))
        self.assertEqual(len(topologies), 29)
        for topology in topologies:
            relation = specialization_relation((0, 1, 2), topology)
            reconstructed = FinitePreorder(
                (0, 1, 2),
                relation,
                (0, 1, 2),
            ).upper_topology
            self.assertEqual(reconstructed, topology)

    def test_monotone_is_equivalent_to_continuous_exhaustively(self) -> None:
        orders = all_preorders((0, 1))
        self.assertEqual(len(orders), 4)
        for source in orders:
            for target in orders:
                for values in product(target.elements, repeat=len(source.elements)):
                    mapping = dict(zip(source.elements, values, strict=True))
                    self.assertEqual(
                        is_monotone(mapping, source, target),
                        is_continuous(mapping, source, target),
                    )

    def test_separation_characterizations_exhaustively(self) -> None:
        for order in all_preorders((0, 1, 2)):
            topology = order.upper_topology
            self.assertEqual(
                is_t0_topology(order.elements, topology),
                order.is_antisymmetric,
            )
            self.assertEqual(
                is_t1_topology(order.elements, topology),
                is_discrete_topology(order.elements, topology),
            )
            self.assertEqual(
                is_t1_topology(order.elements, topology),
                order.is_equality_order,
            )

    def test_kolmogorov_quotient_is_antisymmetric(self) -> None:
        indiscrete = FinitePreorder(
            ("a", "b"),
            frozenset(product(("a", "b"), repeat=2)),
            ("left", "right"),
        )
        quotient = indiscrete.kolmogorov_quotient()
        self.assertEqual(quotient.elements, (frozenset({"a", "b"}),))
        self.assertTrue(quotient.is_antisymmetric)
        self.assertEqual(len(quotient.upper_topology), 2)

    def test_order_discrepancy_is_an_extended_metric_on_isomorphism_classes(
        self,
    ) -> None:
        orders = tuple(
            FinitePreorder(order.elements, order.relation, ("x", "x"))
            for order in all_preorders((0, 1))
        )
        for left in orders:
            self.assertEqual(order_discrepancy(left, left), 0.0)
            for right in orders:
                distance = order_discrepancy(left, right)
                self.assertEqual(distance, order_discrepancy(right, left))
                self.assertEqual(distance == 0.0, are_order_isomorphic(left, right))
                for third in orders:
                    self.assertLessEqual(
                        order_discrepancy(left, third),
                        distance + order_discrepancy(right, third),
                    )

        unmatched = FinitePreorder(
            (2, 3),
            frozenset({(2, 2), (3, 3)}),
            ("x", "y"),
        )
        self.assertEqual(order_discrepancy(orders[0], unmatched), inf)

    def test_monotonicity_defect_and_prediction_zero_criterion(self) -> None:
        chain = FinitePreorder(
            (0, 1),
            frozenset({(0, 0), (1, 1), (0, 1)}),
            ("low", "high"),
        )
        identity = {0: 0, 1: 1}
        swap = {0: 1, 1: 0}
        self.assertEqual(monotonicity_defect(identity, chain, chain), 0.0)
        self.assertEqual(monotonicity_defect(swap, chain, chain), 1 / 3)
        self.assertEqual(
            containment_prediction_error(chain, chain, chain, identity),
            0.0,
        )
        self.assertGreater(
            containment_prediction_error(chain, chain, chain, swap),
            0.0,
        )

        reversed_labels = FinitePreorder(
            ("u", "v"),
            frozenset({("u", "u"), ("v", "v"), ("v", "u")}),
            ("high", "low"),
        )
        self.assertEqual(order_discrepancy(chain, reversed_labels), 0.0)

    def test_required_counterexamples(self) -> None:
        vertices = ("a", "b", "c")
        full_simplex = frozenset(
            frozenset(
                vertices[index]
                for index in range(len(vertices))
                if mask & (1 << index)
            )
            for mask in range(1 << len(vertices))
        )
        chain = FinitePreorder(
            vertices,
            frozenset(
                {
                    ("a", "a"),
                    ("b", "b"),
                    ("c", "c"),
                    ("a", "b"),
                    ("b", "c"),
                    ("a", "c"),
                }
            ),
            vertices,
        )
        vee = FinitePreorder(
            vertices,
            frozenset(
                {
                    ("a", "a"),
                    ("b", "b"),
                    ("c", "c"),
                    ("a", "b"),
                    ("a", "c"),
                }
            ),
            vertices,
        )
        self.assertEqual(len(full_simplex), 8)
        self.assertNotEqual(chain.relation, vee.relation)
        self.assertNotEqual(chain.upper_topology, vee.upper_topology)

        indiscrete = FinitePreorder(
            ("a", "b"),
            frozenset(product(("a", "b"), repeat=2)),
            ("a", "b"),
        )
        self.assertFalse(is_t0_topology(indiscrete.elements, indiscrete.upper_topology))

        edge_adjacency = frozenset({frozenset({0, 1})})
        swap = {0: 1, 1: 0}
        transported_adjacency = frozenset(
            frozenset(swap[vertex] for vertex in edge)
            for edge in edge_adjacency
        )
        two_chain = FinitePreorder(
            (0, 1),
            frozenset({(0, 0), (1, 1), (0, 1)}),
            ("x", "x"),
        )
        self.assertEqual(transported_adjacency, edge_adjacency)
        self.assertFalse(is_monotone(swap, two_chain, two_chain))
        self.assertFalse(is_continuous(swap, two_chain, two_chain))

    def test_alignment_limit_and_weight_validation(self) -> None:
        discrete = FinitePreorder(
            (0, 1, 2),
            frozenset({(0, 0), (1, 1), (2, 2)}),
            ("x", "x", "x"),
        )
        with self.assertRaisesRegex(ValueError, "exceed"):
            order_discrepancy(discrete, discrete, max_alignments=5)
        with self.assertRaisesRegex(ValueError, "positive"):
            containment_prediction_error(
                discrete,
                discrete,
                discrete,
                {0: 0, 1: 1, 2: 2},
                relation_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
