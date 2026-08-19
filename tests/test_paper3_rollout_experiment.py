from __future__ import annotations

import unittest

from tsi.paper3_rollout_experiment import run_rollout_development_pilot


class Paper3RolloutExperimentTest(unittest.TestCase):
    def test_small_development_run_is_complete_and_codebook_free(self) -> None:
        result = run_rollout_development_pilot(
            world_count=1,
            optimizer_seeds=(0,),
            updates=2,
            control_ids=("signature_routed_oracle",),
        )

        self.assertFalse(result["test_output_used"])
        self.assertEqual(result["run_count"], 1)
        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(
            result["constructive_metric_cache"]["global_target_state_candidates"],
            0,
        )
        self.assertEqual(
            result["runs"][0]["metrics"]["recursive_bound_violation_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
