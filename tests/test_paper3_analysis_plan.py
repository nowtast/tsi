from __future__ import annotations

import unittest

from tsi.paper3_analysis_plan import (
    PRIMARY_CONTRASTS,
    PRIMARY_CONTROLS,
    analysis_plan_digest,
    analysis_plan_payload,
    audit_analysis_plan,
)


class AnalysisPlanTest(unittest.TestCase):
    def test_three_primary_controls_are_frozen(self) -> None:
        self.assertEqual(
            tuple(contrast.control for contrast in PRIMARY_CONTRASTS),
            PRIMARY_CONTROLS,
        )

    def test_test_world_count_is_frozen_after_power_pilot(self) -> None:
        payload = analysis_plan_payload()

        self.assertEqual(
            payload["primary_ood_slice"],
            "bridge_consistent_shift",
        )
        self.assertEqual(
            payload["excluded_from_primary_endpoint"],
            ["bridge_violating_control"],
        )
        self.assertEqual(
            payload["test_world_count_status"],
            "frozen_after_p3_3a_power_gate",
        )
        self.assertEqual(payload["planned_test_worlds"], 50)

    def test_digest_is_deterministic(self) -> None:
        self.assertEqual(analysis_plan_digest(), analysis_plan_digest())

    def test_analysis_plan_audit_passes_with_frozen_sample_size(self) -> None:
        audit = audit_analysis_plan()

        self.assertTrue(audit.passed)
        self.assertEqual(audit.primary_contrast_count, 3)
        self.assertTrue(audit.test_world_count_frozen)


if __name__ == "__main__":
    unittest.main()
