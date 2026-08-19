import unittest

from tsi.paper3_learned_v2_stratified_gate import (
    build_stratified_gate_report,
    build_stratified_power_freeze,
)


class LearnedV2StratifiedGateTests(unittest.TestCase):
    ARTIFACT = "experiments/paper3_learned_v2/balanced_graph_mechanism_factorial.json"

    def test_gate_reports_all_factorial_strata_and_separates_control(self) -> None:
        report = build_stratified_gate_report(self.ARTIFACT)
        self.assertEqual(report["stratum_count"], 12)
        self.assertEqual(len(report["strata"]), 16)
        self.assertFalse(report["positive_strata_passed"])
        self.assertTrue(report["negative_control_reported_separately"])
        self.assertTrue(all("source" in row for row in report["strata"] if not row["missing"]))

    def test_power_freeze_remains_unsealed_without_world_level_validation(self) -> None:
        report = build_stratified_gate_report(self.ARTIFACT)
        freeze = build_stratified_power_freeze(report)
        self.assertFalse(freeze["power_gate_passed"])
        self.assertEqual(freeze["status"], "stratified_power_freeze_not_sealed")
        self.assertIn("gate failed", freeze["power_gate_reason"])


if __name__ == "__main__":
    unittest.main()
