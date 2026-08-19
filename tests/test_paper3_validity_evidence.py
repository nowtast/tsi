from hashlib import sha256
import json
import unittest

from tsi.paper3_rollout_evidence import P3_ROLLOUT_EVIDENCE_ID
from tsi.paper3_validity_analysis import P3_VALIDITY_CONFIRMATORY_ANALYSIS_ID
from tsi.paper3_validity_evidence import build_validity_evidence_report
from tsi.paper3_validity_experiment import P3_VALIDITY_SEALED_RAW_ID


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Paper3ValidityEvidenceTests(unittest.TestCase):
    def _p3a(self) -> dict[str, object]:
        payload = {
            "identifier": P3_ROLLOUT_EVIDENCE_ID,
            "level_3_retained": True,
            "evidence_level_after": 3,
            "level_4_requirements": {
                "open_loop_multihorizon_rollout": True,
                "downstream_predictive_validity": False,
                "learned_routing_or_structure": False,
                "noisy_perception": False,
                "variable_cardinality": False,
                "public_benchmark": False,
                "cross_family_replication": False,
                "artifact_reproducibility": False,
            },
        }
        return {**payload, "report_digest": _digest(payload)}

    def test_pass_adds_only_downstream_requirement(self) -> None:
        report = build_validity_evidence_report(
            self._p3a(),
            {
                "identifier": P3_VALIDITY_SEALED_RAW_ID,
                "test_output_used": True,
                "failure_count": 0,
                "report_digest": "a",
            },
            {
                "identifier": P3_VALIDITY_CONFIRMATORY_ANALYSIS_ID,
                "passed": True,
                "report_digest": "b",
            },
            {
                "passed": True,
                "seed_reveals": 1,
                "result_evaluations": 1,
                "audit_digest": "c",
            },
        )
        self.assertEqual(report["satisfied_requirement_count"], 2)
        self.assertEqual(report["evidence_level_after"], 3)
        self.assertFalse(report["level_4_attained"])
        self.assertTrue(report["publication_blocked"])
        self.assertEqual(
            report["newly_satisfied_requirements"],
            ["downstream_predictive_validity"],
        )
        self.assertFalse(report["temporal_prognostic_validity_supported"])

    def test_failed_analysis_does_not_add_requirement(self) -> None:
        report = build_validity_evidence_report(
            self._p3a(),
            {
                "identifier": P3_VALIDITY_SEALED_RAW_ID,
                "test_output_used": True,
                "failure_count": 0,
            },
            {
                "identifier": P3_VALIDITY_CONFIRMATORY_ANALYSIS_ID,
                "passed": False,
            },
            {
                "passed": True,
                "seed_reveals": 1,
                "result_evaluations": 1,
            },
        )
        self.assertEqual(report["satisfied_requirement_count"], 1)


if __name__ == "__main__":
    unittest.main()
