import unittest
from unittest.mock import patch

from tsi.paper3_validity_analysis import (
    _canonical_digest,
    analyze_validity_confirmatory,
    holm_adjusted_pvalues,
)
from tsi.paper3_validity_contract import (
    OPTIMIZER_SEEDS,
    PRIMARY_EFFECT_NAMES,
    PRIMARY_PREDICTIVE_MODELS,
    UNITS_PER_WORLD,
)
from tsi.paper3_validity_experiment import P3_VALIDITY_SEALED_RAW_ID
from tsi.paper3_validity_power import P3_VALIDITY_ANALYSIS_PLAN_ID
from tsi.paper3_validity_predictor import MODEL_IDS


class Paper3ValidityAnalysisTests(unittest.TestCase):
    def test_holm_adjustment_preserves_original_effect_order(self) -> None:
        adjusted = holm_adjusted_pvalues([0.03, 0.01])
        self.assertEqual(adjusted.tolist(), [0.03, 0.02])

    def test_sealed_shaped_positive_fixture_passes_all_decision_gates(self) -> None:
        world_count = 2
        frozen_digest = "frozen-predictor"
        plan_payload = {
            "identifier": P3_VALIDITY_ANALYSIS_PLAN_ID,
            "planned_test_worlds": world_count,
            "primary_success_effects": list(PRIMARY_EFFECT_NAMES),
            "sealed_predictor_refitting": False,
            "frozen_predictor_digest": frozen_digest,
        }
        plan = {
            **plan_payload,
            "analysis_plan_digest": _canonical_digest(plan_payload),
        }
        unit_records = [
            {
                "outcome_uses_tsi_metric": False,
                "probe_task_domains_separated": True,
            }
            for _ in range(UNITS_PER_WORLD)
        ]
        raw = {
            "identifier": P3_VALIDITY_SEALED_RAW_ID,
            "analysis_plan_digest": plan["analysis_plan_digest"],
            "frozen_predictor_digest": frozen_digest,
            "report_digest": "raw",
            "world_count": world_count,
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "run_count": world_count * len(OPTIMIZER_SEEDS) * len(MODEL_IDS),
            "failure_count": 0,
            "unit_count_per_world": UNITS_PER_WORLD,
            "constructive_metric_cache": {
                "global_target_state_candidates": 0,
            },
            "runs": [
                {
                    "status": "completed",
                    "unit_records": unit_records,
                }
                for _ in range(
                    world_count * len(OPTIMIZER_SEEDS) * len(MODEL_IDS)
                )
            ],
        }
        scored = []
        for world in range(world_count):
            for seed in OPTIMIZER_SEEDS:
                for model in MODEL_IDS:
                    for unit in range(UNITS_PER_WORLD):
                        is_primary = model in PRIMARY_PREDICTIVE_MODELS
                        event = int(is_primary and unit % 2 == 0)
                        scored.append(
                            {
                                "world_index": world,
                                "optimizer_seed": seed,
                                "model": model,
                                "unit_index": unit,
                                "any_task_failure": event,
                                "first_failure_time": 3 if event else 9,
                                "binary_baseline_brier": 0.25,
                                "binary_tsi_brier": 0.24,
                                "binary_brier_improvement": 0.01,
                                "first_failure_baseline_integrated_brier": 0.20,
                                "first_failure_tsi_integrated_brier": 0.19,
                                "first_failure_integrated_brier_improvement": 0.01,
                                "layer_aware_binary_brier_improvement": 0.005,
                                "layer_aware_integrated_brier_improvement": 0.005,
                            }
                        )
        world_seed = [
            {
                "world_index": world,
                "optimizer_seed": seed,
                "binary_brier_improvement": 0.01,
                "first_failure_integrated_brier_improvement": 0.01,
            }
            for world in range(world_count)
            for seed in OPTIMIZER_SEEDS
        ]
        with (
            patch(
                "tsi.paper3_validity_analysis.validate_frozen_predictors",
                return_value={"frozen_predictor_digest": frozen_digest},
            ),
            patch(
                "tsi.paper3_validity_analysis.score_validity_result",
                return_value=(scored, world_seed),
            ),
        ):
            analysis = analyze_validity_confirmatory(raw, plan, {})
        self.assertTrue(analysis["passed"])
        self.assertTrue(all(analysis["decision_requirements"].values()))
        self.assertEqual(analysis["sealed_primary_event_rate"], 0.5)
        self.assertTrue(analysis["oracle_endpoint_diagnostics_used"])
        self.assertFalse(analysis["temporal_prognostic_validity_supported"])


if __name__ == "__main__":
    unittest.main()
