from __future__ import annotations

import unittest

from tsi.paper3_rollout_contract import (
    SUCCESS_EFFECT_NAMES,
    analytic_world_floor,
    audit_rollout_contract,
    holm_normal_criticals,
    rollout_contract_digest,
)


class Paper3RolloutContractTest(unittest.TestCase):
    def test_contract_is_complete_and_deterministic(self) -> None:
        audit = audit_rollout_contract()

        self.assertTrue(audit["passed"])
        self.assertEqual(len(SUCCESS_EFFECT_NAMES), 8)
        self.assertEqual(len(rollout_contract_digest()), 64)
        self.assertEqual(rollout_contract_digest(), rollout_contract_digest())

    def test_power_constants_are_conservative(self) -> None:
        criticals = holm_normal_criticals()

        self.assertEqual(len(criticals), 8)
        self.assertGreater(criticals[0], criticals[-1])
        self.assertGreaterEqual(analytic_world_floor(0.10), 50)
        with self.assertRaisesRegex(ValueError, "underpowered"):
            analytic_world_floor(1.0)


if __name__ == "__main__":
    unittest.main()
