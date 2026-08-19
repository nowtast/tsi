import unittest

from tsi.paper3_replication_family import (
    COMBINATIONS,
    GRAPH_NAMES,
    build_replication_dataset,
)
from tsi.paper3_replication_factorized import evaluate, factorize


class ReplicationFactorizedTests(unittest.TestCase):
    def test_all_cells_identify_parameters(self) -> None:
        for graph in GRAPH_NAMES:
            for index, expected in enumerate(COMBINATIONS):
                signature = factorize(build_replication_dataset(graph, index))
                self.assertEqual(signature.multipliers, expected[0])
                self.assertEqual(signature.coefficient, expected[1])

    def test_all_cells_predict_interventions(self) -> None:
        for graph in GRAPH_NAMES:
            for index in range(len(COMBINATIONS)):
                self.assertEqual(
                    evaluate(build_replication_dataset(graph, index))["exact_accuracy"],
                    1.0,
                )


if __name__ == "__main__":
    unittest.main()
