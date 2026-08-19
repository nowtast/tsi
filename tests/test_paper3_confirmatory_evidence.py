from __future__ import annotations

import unittest

from tsi.paper3_confirmatory_evidence import build_evidence_report
from tsi.paper3_independence_contract import MODEL_CONTROLS, WorldFamily
from tsi.paper3_multiworld import _ranked_active_parameters


class ConfirmatoryEvidenceTest(unittest.TestCase):
    def test_all_conjunctive_requirements_promote_only_to_level_three(self) -> None:
        candidates = _ranked_active_parameters(WorldFamily.BRIDGE_COUPLED)[36:86]
        manifest = {
            "world_count": 50,
            "manifest_digest": "a" * 64,
            "worlds": [
                {
                    "active_parameter_signature": [
                        list(multipliers),
                        bridge,
                    ]
                }
                for multipliers, bridge, _context in candidates
            ],
        }
        runs = [
            {
                "model": model.identifier,
                "world_index": world,
                "optimizer_seed": seed,
            }
            for world in range(50)
            for model in MODEL_CONTROLS
            for seed in range(3)
        ]
        raw = {
            "world_count": 50,
            "optimizer_seeds": [0, 1, 2],
            "run_count": 900,
            "failure_count": 0,
            "constructive_metric_cache": {"global_target_state_candidates": 0},
            "runs": runs,
            "report_digest": "b" * 64,
        }
        analysis = {
            "passed": True,
            "hierarchical_world_seed_analysis": {
                "model": ("world_random_intercept_seed_nested_variance_decomposition")
            },
            "world_cluster_bootstrap": {
                "cluster_unit": "world_with_all_nested_optimizer_seeds"
            },
            "report_digest": "c" * 64,
        }
        access = {"passed": True}

        report = build_evidence_report(manifest, raw, analysis, access)

        self.assertTrue(report["level_3_attained"])
        self.assertEqual(report["evidence_level_after"], 3)
        self.assertTrue(report["publication_blocked"])
        self.assertTrue(all(report["requirements"].values()))


if __name__ == "__main__":
    unittest.main()
