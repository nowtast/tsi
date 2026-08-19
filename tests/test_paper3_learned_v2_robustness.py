import unittest

from tsi.paper3_learned_v2_robustness import (
    run_pixel_leakage_audit,
    run_pixel_robustness_development,
)


class LearnedV2RobustnessTests(unittest.TestCase):
    def test_leakage_audit_has_machine_readable_warning(self) -> None:
        audit = run_pixel_leakage_audit(world_index=0)
        self.assertIn("warning", audit)
        self.assertGreater(audit["case_count"], 0)

    def test_robustness_smoke_covers_strong_corruptions(self) -> None:
        results = run_pixel_robustness_development(worlds=1, seeds=(0,), updates=3)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(result.finite for result in results))
        self.assertTrue(all(result.feature_shift_l2 >= 0.0 for result in results))


if __name__ == "__main__":
    unittest.main()
