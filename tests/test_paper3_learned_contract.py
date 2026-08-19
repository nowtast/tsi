import unittest

from tsi.paper3_learned_contract import (
    GRAPH_EDGE_F1_MINIMUM,
    HELD_OUT_ENTITY_COUNTS,
    NONNEGOTIABLE_POLICIES,
    PRIMARY_ENDPOINTS,
    REGIMES,
    TRAIN_ENTITY_COUNT,
    audit_learned_contract,
    holm_normal_criticals,
    learned_contract_digest,
)


class LearnedContractTests(unittest.TestCase):
    def test_contract_audit_passes_and_has_separate_regimes(self) -> None:
        audit = audit_learned_contract()
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["regime_count"], 5)
        self.assertEqual(audit["endpoint_count"], 4)
        self.assertEqual(len(set(REGIMES)), len(REGIMES))

    def test_oracle_is_not_a_primary_learned_endpoint(self) -> None:
        self.assertNotIn("oracle", PRIMARY_ENDPOINTS[1])
        self.assertIn("Oracle and learned results", NONNEGOTIABLE_POLICIES[0])

    def test_cardinality_is_held_out(self) -> None:
        self.assertNotIn(TRAIN_ENTITY_COUNT, HELD_OUT_ENTITY_COUNTS)
        self.assertEqual(set(HELD_OUT_ENTITY_COUNTS), {2, 4})

    def test_thresholds_and_holm_critical_values_are_valid(self) -> None:
        self.assertGreaterEqual(GRAPH_EDGE_F1_MINIMUM, 0.90)
        criticals = holm_normal_criticals()
        self.assertEqual(len(criticals), 4)
        self.assertEqual(tuple(sorted(criticals, reverse=True)), criticals)

    def test_contract_digest_is_deterministic(self) -> None:
        self.assertEqual(learned_contract_digest(), learned_contract_digest())
        self.assertEqual(len(learned_contract_digest()), 64)


if __name__ == "__main__":
    unittest.main()
