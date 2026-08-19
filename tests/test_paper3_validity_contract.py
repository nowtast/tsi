import unittest

from tsi.paper3_validity_contract import (
    MINIMUM_TEST_WORLDS,
    PRIMARY_EFFECT_NAMES,
    analytic_world_floor,
    audit_validity_contract,
    holm_normal_criticals,
    validity_contract_digest,
)


class Paper3ValidityContractTests(unittest.TestCase):
    def test_contract_audit_passes(self) -> None:
        audit = audit_validity_contract()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["errors"], [])
        self.assertEqual(audit["effect_count"], 2)

    def test_contract_digest_is_stable_sha256(self) -> None:
        digest = validity_contract_digest()
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, validity_contract_digest())

    def test_holm_criticals_cover_both_effects(self) -> None:
        criticals = holm_normal_criticals()
        self.assertEqual(len(criticals), len(PRIMARY_EFFECT_NAMES))
        self.assertGreater(criticals[0], criticals[1])

    def test_analytic_floor_respects_minimum(self) -> None:
        self.assertEqual(analytic_world_floor(0.0), MINIMUM_TEST_WORLDS)


if __name__ == "__main__":
    unittest.main()
