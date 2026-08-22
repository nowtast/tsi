import unittest

from tsi.research_a2_contract import (
    NOISE_ENDPOINTS,
    SCOPE_ENDPOINTS,
    WIDTH_ENDPOINTS,
    audit_contract,
    contract_digest,
    contract_payload,
)


class ResearchA2ContractTests(unittest.TestCase):
    def test_numeric_draft_is_complete_but_unfrozen_and_unseeded(self) -> None:
        audit = audit_contract()
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["confirmatory_seed_created"])
        self.assertEqual(len(contract_digest()), 64)
        self.assertEqual(contract_payload()["status"], audit["status"])

    def test_all_three_multiplicity_families_are_explicit(self) -> None:
        self.assertEqual(len(WIDTH_ENDPOINTS), 36)
        self.assertEqual(len(NOISE_ENDPOINTS), 48)
        self.assertEqual(len(SCOPE_ENDPOINTS), 6)


if __name__ == "__main__":
    unittest.main()
