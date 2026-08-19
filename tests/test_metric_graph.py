from __future__ import annotations

from itertools import combinations
import unittest

from tsi.geometric import FiniteMetricState
from tsi.metric_graph import (
    InteriorPoint,
    PositiveWeightedGraph,
    VertexPoint,
    all_pairs_vertex_distances,
    curvature_isomorphism_audit,
    curvature_perturbation_audit,
    complete_metric_realization_graph,
    finite_metric_geodesic_obstruction,
    induced_realization_map,
    interior_point,
    is_length_preserving_graph_isomorphism,
    is_reduced_graph,
    lazy_neighbor_measure,
    ollivier_ricci_curvature,
    realization_distance,
    shortest_realization_path,
    shortest_vertex_path,
    vertex_distance,
    wasserstein_1,
)


def weighted_graph(vertices, edge_lengths) -> PositiveWeightedGraph:
    return PositiveWeightedGraph(tuple(vertices), edge_lengths)


class WeightedGraphDomainTest(unittest.TestCase):
    def test_domain_rejects_nonpositive_disconnected_and_duplicate_edges(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            weighted_graph((0, 1), {(0, 1): 0.0})
        with self.assertRaisesRegex(ValueError, "connected"):
            weighted_graph((0, 1, 2), {(0, 1): 1.0})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            weighted_graph((0, 1), {(0, 1): 1.0, (1, 0): 1.0})

    def test_every_nontrivial_finite_metric_has_a_missing_geodesic_parameter(self) -> None:
        state = FiniteMetricState(
            entities=("a", "b", "c"),
            distances=(
                (0.0, 1.0, 2.0),
                (1.0, 0.0, 1.0),
                (2.0, 1.0, 0.0),
            ),
            labels=("x", "x", "x"),
        )

        witness = finite_metric_geodesic_obstruction(state, "a", "c")

        self.assertGreater(witness.missing_parameter, 0.0)
        self.assertLess(witness.missing_parameter, witness.endpoint_distance)
        self.assertNotIn(witness.missing_parameter, witness.radial_distances)
        realization_graph = complete_metric_realization_graph(state)
        for left_index, left in enumerate(state.entities):
            for right_index, right in enumerate(state.entities):
                self.assertEqual(
                    vertex_distance(realization_graph, left, right),
                    state.distances[left_index][right_index],
                )

        tiny = FiniteMetricState(
            entities=("u", "v"),
            distances=((0.0, 1e-12), (1e-12, 0.0)),
            labels=("x", "x"),
        )
        tiny_witness = finite_metric_geodesic_obstruction(tiny, "u", "v")
        self.assertEqual(tiny_witness.missing_parameter, 5e-13)


class MetricRealizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = weighted_graph(
            ("a", "b", "c"),
            {
                ("a", "b"): 2.0,
                ("b", "c"): 1.0,
                ("a", "c"): 5.0,
            },
        )

    def test_vertex_shortest_path_uses_the_minimum_simple_path(self) -> None:
        self.assertEqual(shortest_vertex_path(self.graph, "a", "c"), ("a", "b", "c"))
        self.assertAlmostEqual(vertex_distance(self.graph, "a", "c"), 3.0)

    def test_arbitrary_edge_points_have_an_attained_shortest_path(self) -> None:
        start = interior_point(self.graph, "a", "b", 0.5)
        end = interior_point(self.graph, "b", "c", 0.25)

        witness = shortest_realization_path(self.graph, start, end)

        self.assertAlmostEqual(witness.length, 1.75)
        self.assertEqual(witness.vertex_path, ("b",))
        self.assertAlmostEqual(realization_distance(self.graph, start, end), 1.75)

        tiny_graph = weighted_graph((0, 1), {(0, 1): 1e-12})
        midpoint = interior_point(tiny_graph, 0, 1, 5e-13)
        self.assertIsInstance(midpoint, InteriorPoint)
        self.assertEqual(
            realization_distance(tiny_graph, VertexPoint(0), midpoint),
            5e-13,
        )

    def test_same_edge_distance_can_use_a_shorter_route_around_a_cycle(self) -> None:
        graph = weighted_graph(
            (0, 1, 2),
            {(0, 1): 10.0, (0, 2): 1.0, (2, 1): 1.0},
        )
        left = interior_point(graph, 0, 1, 1.0)
        right = interior_point(graph, 0, 1, 9.0)

        witness = shortest_realization_path(graph, left, right)

        self.assertFalse(witness.direct_same_edge)
        self.assertAlmostEqual(witness.length, 4.0)

    def test_weighted_isomorphism_extends_to_an_isometry_on_sampled_points(self) -> None:
        renamed = weighted_graph(
            ("u", "v", "w"),
            {
                ("u", "v"): 2.0,
                ("v", "w"): 1.0,
                ("u", "w"): 5.0,
            },
        )
        mapping = {"a": "u", "b": "v", "c": "w"}
        points = (
            VertexPoint("a"),
            VertexPoint("c"),
            interior_point(self.graph, "a", "b", 0.4),
            interior_point(self.graph, "c", "a", 1.5),
        )

        self.assertTrue(
            is_length_preserving_graph_isomorphism(
                self.graph,
                renamed,
                mapping,
            )
        )
        almost_renamed = weighted_graph(
            ("u", "v", "w"),
            {
                ("u", "v"): 2.0 + 1e-12,
                ("v", "w"): 1.0,
                ("u", "w"): 5.0,
            },
        )
        self.assertFalse(
            is_length_preserving_graph_isomorphism(
                self.graph,
                almost_renamed,
                mapping,
            )
        )
        images = tuple(
            induced_realization_map(self.graph, renamed, mapping, point)
            for point in points
        )
        for (left, right), (image_left, image_right) in zip(
            combinations(points, 2),
            combinations(images, 2),
        ):
            self.assertAlmostEqual(
                realization_distance(self.graph, left, right),
                realization_distance(renamed, image_left, image_right),
            )

    def test_degree_two_subdivision_is_invisible_to_the_realization_metric(self) -> None:
        single_edge = weighted_graph(("a", "b"), {("a", "b"): 2.0})
        subdivided = weighted_graph(
            ("a", "m", "b"),
            {("a", "m"): 1.0, ("m", "b"): 1.0},
        )

        self.assertTrue(is_reduced_graph(single_edge))
        self.assertFalse(is_reduced_graph(subdivided))
        self.assertNotEqual(len(single_edge.vertices), len(subdivided.vertices))
        for left_coordinate in (0.0, 0.3, 1.0, 1.7, 2.0):
            for right_coordinate in (0.0, 0.6, 1.0, 1.4, 2.0):
                left_single = interior_point(
                    single_edge,
                    "a",
                    "b",
                    left_coordinate,
                )
                right_single = interior_point(
                    single_edge,
                    "a",
                    "b",
                    right_coordinate,
                )

                def subdivided_point(coordinate):
                    if coordinate <= 1.0:
                        return interior_point(
                            subdivided,
                            "a",
                            "m",
                            coordinate,
                        )
                    return interior_point(
                        subdivided,
                        "m",
                        "b",
                        coordinate - 1.0,
                    )

                self.assertAlmostEqual(
                    realization_distance(single_edge, left_single, right_single),
                    realization_distance(
                        subdivided,
                        subdivided_point(left_coordinate),
                        subdivided_point(right_coordinate),
                    ),
                )


class OllivierRicciTest(unittest.TestCase):
    def test_transport_solver_matches_path_closed_form_on_225_measure_pairs(self) -> None:
        graph = weighted_graph(
            (0, 1, 2),
            {(0, 1): 1.25, (1, 2): 0.75},
        )
        measures = tuple(
            {0: left / 4, 1: middle / 4, 2: right / 4}
            for left in range(5)
            for middle in range(5 - left)
            for right in (4 - left - middle,)
        )

        for left in measures:
            for right in measures:
                expected = (
                    1.25 * abs(left[0] - right[0])
                    + 0.75 * abs(
                        left[0] + left[1] - right[0] - right[1]
                    )
                )
                with self.subTest(left=left, right=right):
                    self.assertAlmostEqual(
                        wasserstein_1(graph, left, right),
                        expected,
                    )

    def test_transport_solver_has_expected_dirac_and_symmetry_values(self) -> None:
        graph = weighted_graph((0, 1), {(0, 1): 2.5})
        left = {0: 1.0}
        right = {1: 1.0}

        self.assertAlmostEqual(wasserstein_1(graph, left, right), 2.5)
        self.assertAlmostEqual(wasserstein_1(graph, right, left), 2.5)
        self.assertAlmostEqual(wasserstein_1(graph, left, left), 0.0)

    def test_lazy_k2_curvature_has_closed_form_value(self) -> None:
        graph = weighted_graph((0, 1), {(0, 1): 3.0})

        self.assertEqual(dict(lazy_neighbor_measure(graph, 0)), {0: 0.5, 1: 0.5})
        self.assertAlmostEqual(
            ollivier_ricci_curvature(graph, 0, 1, idleness=0.5),
            1.0,
        )
        self.assertAlmostEqual(
            ollivier_ricci_curvature(graph, 0, 1, idleness=0.8),
            0.4,
        )
        tiny = weighted_graph((0, 1), {(0, 1): 1e-12})
        self.assertAlmostEqual(
            ollivier_ricci_curvature(tiny, 0, 1, idleness=0.8),
            0.4,
        )

    def test_curvature_is_invariant_under_weighted_graph_renaming(self) -> None:
        source = weighted_graph(
            (0, 1, 2, 3),
            {(0, 1): 1.0, (1, 2): 1.5, (2, 3): 0.8, (3, 0): 1.2},
        )
        target = weighted_graph(
            ("a", "b", "c", "d"),
            {
                ("b", "d"): 1.0,
                ("d", "a"): 1.5,
                ("a", "c"): 0.8,
                ("c", "b"): 1.2,
            },
        )
        mapping = {0: "b", 1: "d", 2: "a", 3: "c"}

        audit = curvature_isomorphism_audit(
            source,
            target,
            mapping,
            idleness=0.35,
        )

        self.assertTrue(audit.holds)
        self.assertAlmostEqual(audit.maximum_error, 0.0)

    def test_fixed_graph_length_perturbation_obeys_both_explicit_bounds(self) -> None:
        source = weighted_graph(
            (0, 1, 2, 3),
            {
                (0, 1): 1.0,
                (1, 2): 1.4,
                (2, 3): 0.9,
                (3, 0): 1.3,
                (0, 2): 1.8,
            },
        )
        perturbed = weighted_graph(
            (0, 1, 2, 3),
            {
                (0, 1): 1.05,
                (1, 2): 1.32,
                (2, 3): 0.93,
                (3, 0): 1.24,
                (0, 2): 1.82,
            },
        )

        audit = curvature_perturbation_audit(
            source,
            perturbed,
            idleness=0.5,
        )

        self.assertAlmostEqual(audit.edge_length_sup_error, 0.08)
        self.assertAlmostEqual(audit.path_metric_bound, 0.24)
        self.assertTrue(audit.path_metric_bound_holds)
        self.assertTrue(audit.curvature_bound_holds)
        self.assertLessEqual(
            audit.path_metric_sup_error,
            audit.path_metric_bound + 1e-9,
        )
        self.assertLessEqual(
            audit.curvature_sup_error,
            audit.curvature_bound + 1e-9,
        )

    def test_all_pair_distances_change_by_at_most_n_minus_one_times_eta(self) -> None:
        source = weighted_graph(
            ("a", "b", "c"),
            {("a", "b"): 1.0, ("b", "c"): 1.0, ("a", "c"): 3.0},
        )
        perturbed = weighted_graph(
            ("a", "b", "c"),
            {("a", "b"): 1.1, ("b", "c"): 0.9, ("a", "c"): 2.95},
        )
        left = all_pairs_vertex_distances(source)
        right = all_pairs_vertex_distances(perturbed)

        self.assertLessEqual(
            max(abs(left[pair] - right[pair]) for pair in left),
            2 * 0.1 + 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
