import unittest

from tsi.paper3_learned_v2_sealed_gate import (
    audit_confirmatory_freeze,
    build_zero_access_ledger,
)


class LearnedV2SealedGateTests(unittest.TestCase):
    def test_zero_access_ledger_has_no_sealed_access(self) -> None:
        ledger = build_zero_access_ledger(analysis_inputs=())
        self.assertEqual(ledger["sealed_test_access_count"], 0)
        self.assertEqual(ledger["status"], "zero_access_confirmed")

    def test_confirmatory_freeze_blocks_unresolved_performance(self) -> None:
        audit = audit_confirmatory_freeze(performance_gate_passed=False)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "preseal_blocked")
        self.assertIn("source-conditioned performance gate is unresolved", audit["errors"])


if __name__ == "__main__":
    unittest.main()
