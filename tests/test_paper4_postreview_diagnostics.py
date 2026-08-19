import unittest

from tsi.paper3_replication_family import PRIMITIVE_ACTIONS, successor
from tsi.paper4_postreview_diagnostics import (
    COMPOSITION_ACTIONS,
    MAGNITUDE_TWO_ACTIONS,
    _cases,
    run_cell,
)


class Paper4PostReviewDiagnosticTests(unittest.TestCase):
    def test_composition_actions_have_two_distinct_active_coordinates(self) -> None:
        self.assertEqual(len(COMPOSITION_ACTIONS), 10)
        self.assertTrue(all(sum(action) == 2 for action in COMPOSITION_ACTIONS))
        self.assertTrue(all(max(action) == 1 for action in COMPOSITION_ACTIONS))

    def test_misspecification_is_dormant_on_primitive_training_actions(self) -> None:
        graph = "metric_to_relation"
        source = (1, 1, 2, 1, 1)
        for action in PRIMITIVE_ACTIONS:
            case = next(
                item
                for item in _cases(graph, 7, (action,), misspecified=True)
                if item.source == source
            )
            self.assertEqual(case.target, successor(source, action, graph, 7))

    def test_misspecified_panel_breaks_exact_tsi_prediction(self) -> None:
        result = run_cell("metric_to_relation", 7)
        panels = result["panels"]
        self.assertEqual(
            panels["original_magnitude_two"]["correct_graph_factorized"], 1.0
        )
        self.assertEqual(
            panels["two_coordinate_composition"]["correct_graph_factorized"], 1.0
        )
        self.assertLess(
            panels["misspecified_curvature"]["correct_graph_factorized"], 1.0
        )
        self.assertEqual(len(MAGNITUDE_TWO_ACTIONS), 5)


if __name__ == "__main__":
    unittest.main()
