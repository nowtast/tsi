import unittest

from tsi.paper3_replication_family import build_replication_dataset
from tsi.paper4_comparative_validation import (
    evaluate_model,
    fit_tsi_factorized,
    fit_unstructured_lookup,
    fit_vector_only,
)


class Paper4ComparativeValidationTests(unittest.TestCase):
    def test_tsi_beats_controls_on_cross_layer_intervention(self) -> None:
        dataset = build_replication_dataset("metric_to_relation", 7)
        results = [
            evaluate_model(model, dataset)
            for model in (
                fit_vector_only(dataset),
                fit_unstructured_lookup(dataset),
                fit_tsi_factorized(dataset),
            )
        ]
        by_name = {result["model"]: result for result in results}
        self.assertEqual(
            by_name["tsi_graph_discovered_factorized"]["exact_accuracy"], 1.0
        )
        self.assertEqual(
            by_name["tsi_graph_discovered_factorized"]["intervention_exact_accuracy"],
            1.0,
        )
        self.assertLess(
            by_name["vector_only_diagonal"]["intervention_exact_accuracy"], 1.0
        )
        self.assertLess(
            by_name["unstructured_lookup"]["intervention_exact_accuracy"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
