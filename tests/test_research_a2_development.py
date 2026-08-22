import unittest

from tsi.research_a2_development import run_a2_development


class ResearchA2DevelopmentTests(unittest.TestCase):
    def test_small_development_run_covers_all_axes_without_confirmatory_seed(
        self,
    ) -> None:
        report = run_a2_development(
            matched_world_count=2,
            misspecification_world_count=2,
            width_sample_sizes=(10,),
            noise_sample_sizes=(15,),
            misspecification_sample_sizes=(20,),
            test_case_count=12,
        )
        self.assertEqual(report["status"], "development_only_not_confirmatory")
        self.assertFalse(report["confirmatory_seed_created"])
        self.assertFalse(report["held_out_test_used_for_fit_or_selection"])
        axes = report["axes"]
        self.assertEqual(
            set(axes), {"candidate_width", "training_noise", "misspecification"}
        )
        self.assertEqual(len(axes["candidate_width"]["records"]), 2 * 3)
        self.assertEqual(len(axes["training_noise"]["records"]), 2 * 4)
        self.assertEqual(len(axes["misspecification"]["records"]), 2 * 3)
        self.assertTrue(all(audit["passed"] for audit in report["audits"].values()))
        matched = [
            row
            for row in axes["misspecification"]["records"]
            if row["condition"] == "matched"
        ]
        self.assertTrue(all(isinstance(row["generic_exact"], bool) for row in matched))


if __name__ == "__main__":
    unittest.main()
