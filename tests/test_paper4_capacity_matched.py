import unittest

from tsi.paper3_replication_family import build_replication_dataset
from tsi.paper4_capacity_matched import BOOTSTRAP_SEEDS, run_capacity_matched_cell


class Paper4CapacityMatchedTests(unittest.TestCase):
    def test_capacity_matched_panel_is_finite_and_tsi_is_exact(self) -> None:
        rows = run_capacity_matched_cell(
            build_replication_dataset("metric_to_relation", 7)
        )
        self.assertEqual(len(rows), len(BOOTSTRAP_SEEDS) * 2 + 1)
        tsi_rows = [
            row for row in rows if row["model"] == "tsi_graph_discovered_factorized"
        ]
        self.assertEqual(tsi_rows[0]["exact_accuracy"], 1.0)
        self.assertTrue(all(0.0 <= row["exact_accuracy"] <= 1.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
