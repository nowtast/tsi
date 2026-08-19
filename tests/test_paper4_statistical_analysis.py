from pathlib import Path
import unittest

from tsi.paper4_statistical_analysis import analyze_final_audit


class Paper4StatisticalAnalysisTests(unittest.TestCase):
    def test_primary_contrast_is_positive_and_reproducible(self) -> None:
        audit = Path("experiments/paper4_final_comparative_audit.json")
        if not audit.exists():
            audit = Path("artifacts/paper4/paper4_final_comparative_audit.json")
        first = analyze_final_audit(audit)
        second = analyze_final_audit(audit)
        self.assertEqual(first, second)
        interval = first["primary_contrast_interval"]
        self.assertGreater(interval["mean"], 0.0)
        self.assertGreater(interval["ci95_low"], 0.0)
        self.assertFalse(first["all_primary_differences_positive"])
        distributions = first["intervention_cell_distributions"]
        self.assertEqual(distributions["dense_polynomial_trainable"]["exact_cell_count"], 33)
        self.assertEqual(distributions["dense_polynomial_trainable"]["zero_cell_count"], 32)
        self.assertEqual(
            distributions["wrong_routed_factorized"]["distribution"],
            {"0": 192, "0.6": 96},
        )
        self.assertEqual(
            distributions["unstructured_lookup"]["zero_cell_count"], 288
        )


if __name__ == "__main__":
    unittest.main()
