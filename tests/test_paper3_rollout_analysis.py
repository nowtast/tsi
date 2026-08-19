from __future__ import annotations

import unittest

from tsi.paper3_independence_contract import MODEL_CONTROLS
from tsi.paper3_rollout_analysis import analyze_rollout_confirmatory
from tsi.paper3_rollout_contract import OPTIMIZER_SEEDS
from tsi.paper3_rollout_experiment import P3_ROLLOUT_SEALED_RAW_ID
from tsi.paper3_rollout_power import _analysis_plan


def synthetic_sealed_result(world_count: int) -> dict[str, object]:
    errors = {
        "signature_routed_oracle": 0.0,
        "dense_active_matched": 0.0,
        "random_routed_matched_sparsity": 0.20,
        "permuted_or_wrong_routed": 0.20,
        "layer_routed_dense_action": 0.16,
        "strict_factorized_action": 0.18,
    }
    runs = []
    for world in range(world_count):
        variation = (world % 5) * 1.0e-4
        for model in MODEL_CONTROLS:
            for seed in OPTIMIZER_SEEDS:
                error = errors[model.identifier]
                if error:
                    error += variation
                metrics = {
                    "teacher_forced_i0_auc": error / 3.0,
                    "open_loop_i0_auc": error,
                    "exposure_gap_i0_auc": (
                        0.0
                        if model.identifier == "signature_routed_oracle"
                        else error * 2 / 3
                    ),
                    "terminal_open_loop_i0_error": error,
                    "terminal_open_loop_fixed_error": error,
                    "terminal_open_loop_tracking_error": 0.0,
                    "self_conditioned_local_law_violation_rate": (
                        0.0 if model.identifier == "signature_routed_oracle" else error
                    ),
                    "state_coherence_bridge_violation_rate": 0.0,
                    "terminal_trajectory_survival_rate": (1.0 if error == 0.0 else 0.0),
                    "mean_first_structural_failure_time": (
                        33.0 if error == 0.0 else 2.0
                    ),
                    "recursive_bound_violation_count": 0,
                    "horizon_summary": {
                        "32": {
                            "teacher_forced_mean_i0_error": error / 3.0,
                            "open_loop_mean_i0_error": error,
                            "open_loop_state_exact_rate": (
                                1.0 if error == 0.0 else 0.0
                            ),
                            "trajectory_survival_rate": (1.0 if error == 0.0 else 0.0),
                        }
                    },
                }
                runs.append(
                    {
                        "status": "completed",
                        "world_index": world,
                        "model": model.identifier,
                        "optimizer_seed": seed,
                        "metrics": metrics,
                    }
                )
    return {
        "identifier": P3_ROLLOUT_SEALED_RAW_ID,
        "test_output_used": True,
        "analysis_plan_digest": _analysis_plan(
            "a" * 64,
            world_count,
        )["analysis_plan_digest"],
        "world_count": world_count,
        "trajectory_count_per_world": 32,
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "run_count": len(runs),
        "failure_count": 0,
        "constructive_metric_cache": {
            "global_target_state_candidates": 0,
        },
        "report_digest": "b" * 64,
        "runs": runs,
    }


class Paper3RolloutAnalysisTest(unittest.TestCase):
    def test_conjunctive_analysis_passes_only_with_all_effects(self) -> None:
        world_count = 62
        plan = _analysis_plan("a" * 64, world_count)
        raw = synthetic_sealed_result(world_count)

        analysis = analyze_rollout_confirmatory(raw, plan)

        self.assertTrue(analysis["passed"])
        self.assertTrue(all(analysis["decision_requirements"].values()))
        self.assertEqual(len(analysis["student_t_holm"]), 8)
        self.assertTrue(analysis["routing_point_effects"]["both_meet_0.05_sesoi"])

    def test_failed_recursive_bound_blocks_gate(self) -> None:
        world_count = 62
        plan = _analysis_plan("a" * 64, world_count)
        raw = synthetic_sealed_result(world_count)
        raw["runs"][0]["metrics"]["recursive_bound_violation_count"] = 1

        analysis = analyze_rollout_confirmatory(raw, plan)

        self.assertFalse(analysis["passed"])
        self.assertFalse(
            analysis["decision_requirements"]["recursive_rollout_bounds_passed"]
        )


if __name__ == "__main__":
    unittest.main()
