import unittest

from tsi.paper3_learned_v3_identification import (
    identify_v3_training_mechanism,
    run_v3_identification_audit,
)
from tsi.paper3_learned_v3_generator import build_v3_world_dataset


class LearnedV3IdentificationTests(unittest.TestCase):
    def test_identification_uses_v3_train_partition_only(self) -> None:
        dataset = build_v3_world_dataset(0, 132, graph_index=2)
        report = identify_v3_training_mechanism(dataset)
        self.assertTrue(report["graph_exact"], report)
        self.assertTrue(report["active_mechanism_exact"], report)
        self.assertEqual(report["mechanism_split"], "test")

    def test_small_factorial_audit_is_exact(self) -> None:
        results = run_v3_identification_audit(combination_indices=(0, 128, 132))
        self.assertEqual(len(results), 12)
        self.assertTrue(all(row["graph_exact"] and row["active_mechanism_exact"] for row in results))


if __name__ == "__main__":
    unittest.main()
