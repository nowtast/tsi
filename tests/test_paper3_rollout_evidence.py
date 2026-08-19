from __future__ import annotations

import unittest

from tsi.paper3_rollout_analysis import P3_ROLLOUT_CONFIRMATORY_ANALYSIS_ID
from tsi.paper3_rollout_evidence import build_rollout_evidence_report
from tsi.paper3_rollout_experiment import P3_ROLLOUT_SEALED_RAW_ID


class Paper3RolloutEvidenceTest(unittest.TestCase):
    def test_rollout_satisfies_one_level_four_requirement_only(self) -> None:
        p3b = {
            "level_3_attained": True,
            "evidence_level_after": 3,
            "requirements": {"requirement": True},
            "report_digest": "a" * 64,
        }
        raw = {
            "identifier": P3_ROLLOUT_SEALED_RAW_ID,
            "test_output_used": True,
            "failure_count": 0,
            "report_digest": "b" * 64,
        }
        analysis = {
            "identifier": P3_ROLLOUT_CONFIRMATORY_ANALYSIS_ID,
            "passed": True,
            "maximum_horizon": 32,
            "report_digest": "c" * 64,
        }
        access = {
            "passed": True,
            "seed_reveals": 1,
            "result_evaluations": 1,
            "audit_digest": "d" * 64,
        }

        report = build_rollout_evidence_report(
            p3b,
            raw,
            analysis,
            access,
        )

        self.assertEqual(report["evidence_level_after"], 3)
        self.assertFalse(report["level_4_attained"])
        self.assertTrue(report["publication_blocked"])
        self.assertTrue(
            report["level_4_requirements"]["open_loop_multihorizon_rollout"]
        )
        self.assertEqual(
            sum(report["level_4_requirements"].values()),
            1,
        )


if __name__ == "__main__":
    unittest.main()
