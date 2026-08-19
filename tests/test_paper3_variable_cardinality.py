import unittest

from tsi.paper3_variable_cardinality import (
    CARDINALITY_PANELS,
    COMBINATIONS,
    GRAPH_NAMES,
    build_variable_dataset,
    evaluate,
    factorize,
)


class VariableCardinalityTests(unittest.TestCase):
    def test_all_panels_recover_parameters_and_interventions(self) -> None:
        for panel_index, cardinalities in enumerate(CARDINALITY_PANELS):
            for graph in GRAPH_NAMES:
                for combination_index, expected in enumerate(COMBINATIONS):
                    dataset = build_variable_dataset(
                        panel_index, graph, combination_index
                    )
                    self.assertEqual(factorize(dataset), expected)
                    self.assertEqual(evaluate(dataset), 1.0)


if __name__ == "__main__":
    unittest.main()
