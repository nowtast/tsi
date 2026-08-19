from __future__ import annotations

import unittest

import numpy as np

from tsi.paper3_analysis_plan import analysis_plan_digest
from tsi.paper3_confirmatory_analysis import (
    analyze_confirmatory_experiment,
    holm_adjusted_pvalues,
    student_t_survival,
)
from tsi.paper3_confirmatory_experiment import P3_CONFIRMATORY_EXPERIMENT_ID
from tsi.paper3_routing_controls import routing_control_manifests
from tsi.paper3_independence_contract import WorldFamily


class ConfirmatoryAnalysisTest(unittest.TestCase):
    def test_student_t_survival_known_reference_points(self) -> None:
        self.assertAlmostEqual(student_t_survival(0.0, 49), 0.5, places=12)
        self.assertAlmostEqual(
            student_t_survival(1.6765508926, 49),
            0.05,
            places=7,
        )

    def test_holm_adjustment_is_step_down_monotone(self) -> None:
        adjusted = holm_adjusted_pvalues(
            np.asarray([0.01, 0.03, 0.2], dtype=np.float64)
        )

        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.2])

    def test_conjunctive_analysis_passes_clear_synthetic_effect(self) -> None:
        identifiers = tuple(
            manifest.identifier
            for manifest in routing_control_manifests(WorldFamily.BRIDGE_COUPLED)
        )
        errors = {
            "dense_active_matched": 0.0,
            "layer_routed_dense_action": 0.2,
            "strict_factorized_action": 0.2,
            "signature_routed_oracle": 0.0,
            "random_routed_matched_sparsity": 0.2,
            "permuted_or_wrong_routed": 0.2,
        }
        runs = []
        for world in range(50):
            for model in identifiers:
                for seed in range(3):
                    runs.append(
                        {
                            "status": "completed",
                            "family": "bridge_coupled",
                            "world_index": world,
                            "model": model,
                            "optimizer_seed": seed,
                            "metrics": {
                                "bridge_consistent_shift": {
                                    "mean_normalized_i0_quotient_error": (
                                        errors[model]
                                    ),
                                    "fixed_joint_exact_rate": 1.0,
                                    "bridge_violation_rate": 0.0,
                                    "tracking_exact_rate": 1.0,
                                }
                            },
                        }
                    )
        raw = {
            "identifier": P3_CONFIRMATORY_EXPERIMENT_ID,
            "analysis_plan_digest": analysis_plan_digest(),
            "world_count": 50,
            "run_count": 900,
            "failure_count": 0,
            "constructive_metric_cache": {"global_target_state_candidates": 0},
            "report_digest": "a" * 64,
            "runs": runs,
        }

        report = analyze_confirmatory_experiment(raw)

        self.assertTrue(report["passed"])
        self.assertTrue(all(report["decision_requirements"].values()))


if __name__ == "__main__":
    unittest.main()
