import unittest

from tsi.research_a_contract import (
    PRIMARY_ENDPOINTS,
    REVIEW_SAFEGUARDS,
    WORLD_COUNT,
    audit_contract,
)


class ResearchAContractTests(unittest.TestCase):
    def test_contract_names_every_primary_endpoint(self) -> None:
        self.assertTrue(audit_contract()["passed"])
        self.assertEqual(len(PRIMARY_ENDPOINTS), 16)
        self.assertEqual(len(set(PRIMARY_ENDPOINTS)), 16)

    def test_worlds_and_review_safeguards_are_balanced(self) -> None:
        self.assertEqual(WORLD_COUNT % 9, 0)
        self.assertEqual(len(REVIEW_SAFEGUARDS), 10)


if __name__ == "__main__":
    unittest.main()
