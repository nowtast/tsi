import unittest
from pathlib import Path

from tsi.paper3_learned_v2_power import (
    build_v2_power_report,
    simulate_one_sided_power,
    variance_decomposition,
)


class LearnedV2PowerTests(unittest.TestCase):
    def test_variance_decomposition_separates_world_and_seed_variance(self) -> None:
        report = variance_decomposition(
            __import__("numpy").array([[0.0, 0.1, 0.0], [0.5, 0.4, 0.6]])
        )
        self.assertGreater(report["world_variance"], 0.0)
        self.assertGreater(report["within_world_seed_variance"], 0.0)

    def test_power_simulation_is_deterministic(self) -> None:
        first = simulate_one_sided_power(
            world_count=24,
            world_variance=0.01,
            seed_variance=0.001,
            iterations=200,
        )
        second = simulate_one_sided_power(
            world_count=24,
            world_variance=0.01,
            seed_variance=0.001,
            iterations=200,
        )
        self.assertEqual(first, second)

    def test_power_report_uses_development_and_independent_artifacts(self) -> None:
        development = Path("experiments/paper3_learned_v2/source_conditioned_robustness_development.json")
        independent = Path("experiments/paper3_learned_v2/independent_source_conditioned_validation.json")
        if not development.exists() or not independent.exists():
            self.skipTest("development artifacts are not present")
        report = build_v2_power_report(development, independent_path=independent)
        self.assertEqual(report["endpoint"], "source_logloss_degradation")
        self.assertIsNotNone(report["independent_validation_variance"])
        self.assertEqual(len(report["power_curve"]), 4)


if __name__ == "__main__":
    unittest.main()
