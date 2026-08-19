from __future__ import annotations

import unittest

import numpy as np

from tsi.paper3_independence_contract import MODEL_CONTROLS
from tsi.paper3_rollout_contract import (
    DEVELOPMENT_WORLDS,
    OPTIMIZER_SEEDS,
    SUCCESS_EFFECT_NAMES,
)
from tsi.paper3_rollout_experiment import P3_ROLLOUT_DEVELOPMENT_ID
from tsi.paper3_rollout_power import (
    build_rollout_power_report,
    planning_covariance,
    seed_level_success_effects,
    simulate_holm_power,
)


def synthetic_pilot() -> dict[str, object]:
    runs = []
    errors = {
        "signature_routed_oracle": 0.0,
        "dense_active_matched": 0.0,
        "random_routed_matched_sparsity": 0.20,
        "permuted_or_wrong_routed": 0.20,
        "layer_routed_dense_action": 0.16,
        "strict_factorized_action": 0.18,
    }
    for world in range(DEVELOPMENT_WORLDS):
        for model in MODEL_CONTROLS:
            for seed in OPTIMIZER_SEEDS:
                error = errors[model.identifier] + world * 1.0e-4
                metrics = {
                    "open_loop_i0_auc": error,
                    "terminal_open_loop_i0_error": error,
                    "exposure_gap_i0_auc": (
                        0.0 if model.identifier == "signature_routed_oracle" else 0.02
                    ),
                    "terminal_open_loop_tracking_error": 0.0,
                    "self_conditioned_local_law_violation_rate": 0.0,
                    "recursive_bound_violation_count": 0,
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
        "identifier": P3_ROLLOUT_DEVELOPMENT_ID,
        "test_output_used": False,
        "world_count": DEVELOPMENT_WORLDS,
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "failure_count": 0,
        "report_digest": "a" * 64,
        "runs": runs,
    }


class Paper3RolloutPowerTest(unittest.TestCase):
    def test_seed_effects_use_worlds_and_nested_seeds(self) -> None:
        effects, detail = seed_level_success_effects(
            synthetic_pilot(),
            expected_world_count=DEVELOPMENT_WORLDS,
        )

        self.assertEqual(
            effects.shape,
            (
                DEVELOPMENT_WORLDS,
                len(OPTIMIZER_SEEDS),
                len(SUCCESS_EFFECT_NAMES),
            ),
        )
        self.assertEqual(len(detail), DEVELOPMENT_WORLDS)
        self.assertAlmostEqual(effects[0, 0, -2], 0.20)
        self.assertAlmostEqual(effects[0, 0, -1], 0.20)

    def test_power_simulation_is_deterministic(self) -> None:
        covariance = np.eye(len(SUCCESS_EFFECT_NAMES)) * 0.01
        first = simulate_holm_power(covariance, iterations=200)
        second = simulate_holm_power(covariance, iterations=200)

        self.assertEqual(first, second)
        self.assertIsNotNone(first["selected_worlds"])

    def test_power_report_freezes_analysis_plan(self) -> None:
        effects, _detail = seed_level_success_effects(
            synthetic_pilot(),
            expected_world_count=DEVELOPMENT_WORLDS,
        )
        _observed, planning, covariance = planning_covariance(np.mean(effects, axis=1))
        self.assertTrue(np.all(planning >= 0.10))
        self.assertEqual(
            covariance.shape,
            (len(SUCCESS_EFFECT_NAMES), len(SUCCESS_EFFECT_NAMES)),
        )

        report = build_rollout_power_report(
            synthetic_pilot(),
            iterations=500,
        )
        self.assertTrue(report["passed"])
        self.assertFalse(report["test_output_used"])
        self.assertGreaterEqual(report["planned_test_worlds"], 50)
        self.assertEqual(
            len(report["analysis_plan"]["analysis_plan_digest"]),
            64,
        )


if __name__ == "__main__":
    unittest.main()
