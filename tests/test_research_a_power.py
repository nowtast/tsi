import unittest

from tsi.research_a_development import run_development
from tsi.research_a_power import estimate_prospective_power


class ResearchAPowerTests(unittest.TestCase):
    def test_power_uses_development_only_and_balanced_strata(self) -> None:
        development = run_development(
            world_count=9,
            sample_sizes=(5, 10, 15, 20, 25, 30, 40, 50),
            test_case_count=20,
        )
        report = estimate_prospective_power(
            development,
            world_counts=(18, 126),
            iterations=20,
            batch_size=10,
        )
        self.assertFalse(report["uses_confirmatory_results"])
        self.assertEqual(report["bonferroni_endpoint_count"], 16)
        self.assertEqual(report["selected_world_count"], 126)


if __name__ == "__main__":
    unittest.main()
