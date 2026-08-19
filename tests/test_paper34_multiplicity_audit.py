import json
from pathlib import Path
import unittest

from tsi.paper34_multiplicity_audit import audit_multiplicity
from tsi.paper34_resolution_contract import (
    PRIMARY_INFERENTIAL_EFFECT_COUNT,
    PRIMARY_INFERENTIAL_QUANTITIES,
)


class Paper34MultiplicityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (
                root
                / "experiments/paper34_resolution_v1/confirmatory/"
                "confirmatory_analysis.json"
            ).read_text(encoding="utf-8")
        )
        cls.audit = audit_multiplicity(payload["analysis"])

    def test_frozen_divisor_now_has_explicit_postfreeze_labels(self) -> None:
        self.assertEqual(
            len(PRIMARY_INFERENTIAL_QUANTITIES),
            PRIMARY_INFERENTIAL_EFFECT_COUNT,
        )
        self.assertFalse(self.audit["frozen_contract_named_members"])

    def test_reported_relationships_are_verified_worldwise(self) -> None:
        bookkeeping = self.audit["bookkeeping"]
        self.assertTrue(bookkeeping["exact_duplicate_verified_worldwise"])
        self.assertTrue(bookkeeping["sign_flip_verified_worldwise"])
        self.assertTrue(bookkeeping["deterministic_zero_verified_worldwise"])
        self.assertEqual(
            bookkeeping["informationally_distinct_stochastic_quantity_count"], 7
        )

    def test_divisor_ten_sensitivity_preserves_all_gates(self) -> None:
        self.assertTrue(self.audit["all_sensitivity_gates_passed"])
        intervals = self.audit["sensitivity_effect_intervals"]
        self.assertGreater(
            intervals["large_generic_graph_nll"]["simultaneous_lower"], 0.0
        )
        self.assertGreater(
            intervals["criterion_brier"]["simultaneous_lower"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
