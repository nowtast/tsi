import unittest

from tsi.research_a2_development import run_a2_development
from tsi.research_a2_power import estimate_a2_prospective_power


class ResearchA2PowerTests(unittest.TestCase):
    def test_power_uses_only_development_and_balances_both_stratum_systems(
        self,
    ) -> None:
        development = run_a2_development(
            matched_world_count=18,
            misspecification_world_count=45,
            width_sample_sizes=(10, 15, 20, 25, 30, 40),
            noise_sample_sizes=(15, 20, 30, 40, 80, 160),
            misspecification_sample_sizes=(20, 40, 80, 160, 320),
            test_case_count=12,
        )
        report = estimate_a2_prospective_power(
            development,
            world_counts=(135,),
            iterations=10,
            batch_size=5,
        )
        self.assertFalse(report["uses_confirmatory_results"])
        self.assertFalse(report["confirmatory_seed_created"])
        self.assertEqual(report["multiplicity"]["candidate_width_endpoint_count"], 36)
        self.assertEqual(report["multiplicity"]["training_noise_endpoint_count"], 48)
        self.assertEqual(
            report["selected_world_count_per_axis_or_scope_condition"], 135
        )


if __name__ == "__main__":
    unittest.main()
