import json
from pathlib import Path
import unittest

from tsi.paper34_retrospective_power import estimate_retrospective_power


class Paper34RetrospectivePowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.development = json.loads(
            (
                root / "experiments/paper34_resolution_v1/development_report.json"
            ).read_text(encoding="utf-8")
        )

    def test_report_is_explicitly_retrospective(self) -> None:
        report = estimate_retrospective_power(
            self.development,
            world_counts=(120,),
            iterations=200,
            batch_size=100,
        )
        self.assertFalse(report["uses_confirmatory_results"])
        self.assertIn("not_preregistered", report["status"])
        self.assertEqual(report["development_world_count"], 24)

    def test_selected_power_is_a_probability(self) -> None:
        report = estimate_retrospective_power(
            self.development,
            world_counts=(120,),
            iterations=200,
            batch_size=100,
        )
        self.assertGreaterEqual(report["selected_conjunctive_gate_power"], 0.0)
        self.assertLessEqual(report["selected_conjunctive_gate_power"], 1.0)


if __name__ == "__main__":
    unittest.main()
