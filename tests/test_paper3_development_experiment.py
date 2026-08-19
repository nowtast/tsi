from __future__ import annotations

import unittest

from tsi.paper3_development_experiment import run_development_pilot


class DevelopmentExperimentTest(unittest.TestCase):
    def test_small_public_only_pilot_is_complete(self) -> None:
        report = run_development_pilot(
            worlds_per_family=1,
            optimizer_seeds=(0,),
            updates=5,
        )

        self.assertFalse(report["test_output_used"])
        self.assertEqual(report["run_count"], 7)
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(
            {run["family"] for run in report["runs"]},
            {"separable", "bridge_coupled"},
        )
        self.assertTrue(
            all("bridge_consistent_shift" in run["metrics"] for run in report["runs"])
        )


if __name__ == "__main__":
    unittest.main()
