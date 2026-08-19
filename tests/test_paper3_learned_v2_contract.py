import unittest

from tsi.paper3_learned_v2_contract import (
    ABLATION_FACTORS,
    ABLATION_ROWS,
    GRAPH_VARIANTS,
    PRIMARY_ENDPOINTS,
    SPLITS,
    audit_learned_v2_contract,
    learned_v2_contract_digest,
)


class LearnedV2ContractTests(unittest.TestCase):
    def test_v2_contract_audit_passes(self) -> None:
        audit = audit_learned_v2_contract()
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["split_count"], 4)
        self.assertGreaterEqual(audit["graph_variant_count"], 3)

    def test_selection_and_evaluation_are_distinct_splits(self) -> None:
        self.assertEqual(SPLITS[:2], ("train", "routing_selection"))
        self.assertNotEqual(SPLITS[1], SPLITS[2])
        self.assertNotEqual(SPLITS[2], SPLITS[3])

    def test_ablation_matrix_has_v1_and_one_row_per_factor(self) -> None:
        self.assertEqual(len(ABLATION_ROWS), len(ABLATION_FACTORS) + 1)
        self.assertTrue(ABLATION_ROWS[0].startswith("v1_"))

    def test_primary_utility_endpoints_are_not_only_i0(self) -> None:
        self.assertIn("held_out_intervention_target_logloss", PRIMARY_ENDPOINTS)
        self.assertIn("held_out_intervention_regret", PRIMARY_ENDPOINTS)

    def test_graph_variants_are_not_single_family(self) -> None:
        self.assertGreaterEqual(len(set(GRAPH_VARIANTS)), 3)

    def test_contract_digest_is_stable_length(self) -> None:
        self.assertEqual(len(learned_v2_contract_digest()), 64)


if __name__ == "__main__":
    unittest.main()
