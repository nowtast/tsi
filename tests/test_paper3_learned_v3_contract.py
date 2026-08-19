import unittest

from tsi.paper3_learned_v3_contract import (
    TEST_COMBINATION_INDICES,
    TRAIN_COMBINATION_INDICES,
    VALIDATION_COMBINATION_INDICES,
    audit_learned_v3_contract,
    learned_v3_contract_digest,
)


class LearnedV3ContractTests(unittest.TestCase):
    def test_contract_and_combination_splits_are_valid(self) -> None:
        audit = audit_learned_v3_contract()
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(len(TRAIN_COMBINATION_INDICES), 96)
        self.assertEqual(len(VALIDATION_COMBINATION_INDICES), 48)
        self.assertEqual(len(TEST_COMBINATION_INDICES), 48)
        self.assertEqual(
            len(TRAIN_COMBINATION_INDICES | VALIDATION_COMBINATION_INDICES | TEST_COMBINATION_INDICES),
            192,
        )
        self.assertEqual(len(learned_v3_contract_digest()), 64)

    def test_combination_splits_are_pairwise_disjoint(self) -> None:
        self.assertFalse(TRAIN_COMBINATION_INDICES & VALIDATION_COMBINATION_INDICES)
        self.assertFalse(TRAIN_COMBINATION_INDICES & TEST_COMBINATION_INDICES)
        self.assertFalse(VALIDATION_COMBINATION_INDICES & TEST_COMBINATION_INDICES)


if __name__ == "__main__":
    unittest.main()
