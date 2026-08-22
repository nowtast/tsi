import unittest

from tsi.research_a2_contract import (
    NOISE_ADVANTAGE_PROBABILITIES,
    NOISE_BOUNDARY_STRESS_PROBABILITY,
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

    def test_noise_gate_and_seed_source_are_nontrivial_and_prespecified(self) -> None:
        payload = contract_payload()
        self.assertEqual(NOISE_ADVANTAGE_PROBABILITIES, (0.08, 0.3, 0.6))
        self.assertEqual(NOISE_BOUNDARY_STRESS_PROBABILITY, 0.8)
        self.assertEqual(
            payload["noise"]["advantage_required_at_each_probability"],
            [0.08, 0.3, 0.6],
        )
        self.assertFalse(
            payload["seed_selection"]["author_generated_or_selected_seed_allowed"]
        )


if __name__ == "__main__":
    unittest.main()
