from __future__ import annotations

import unittest

from tsi.paper3_evidence_contract import (
    ALL_REQUIREMENTS,
    EVIDENCE_CONTRACT_ID,
    EVIDENCE_PHASES,
    LEVEL_3_REQUIREMENTS,
    LEVEL_4_REQUIREMENTS,
    LEVEL_5_REQUIREMENTS,
    NONNEGOTIABLE_POLICIES,
    EvidenceLevel,
    attained_evidence_level,
    audit_paper3_evidence_contract,
    evidence_contract_digest,
    missing_requirements,
)


class EvidencePromotionTest(unittest.TestCase):
    def test_empty_future_ledger_remains_level_two(self) -> None:
        self.assertIs(
            attained_evidence_level(()),
            EvidenceLevel.DEVELOPMENT_DIAGNOSTIC,
        )

    def test_level_three_requires_every_confirmatory_requirement(self) -> None:
        keys = tuple(requirement.key for requirement in LEVEL_3_REQUIREMENTS)
        self.assertIs(
            attained_evidence_level(keys),
            EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        )
        for omitted in keys:
            reduced = tuple(key for key in keys if key != omitted)
            self.assertIs(
                attained_evidence_level(reduced),
                EvidenceLevel.DEVELOPMENT_DIAGNOSTIC,
            )

    def test_level_four_is_conjunctive_across_level_three_and_four(self) -> None:
        level_three = tuple(requirement.key for requirement in LEVEL_3_REQUIREMENTS)
        level_four = tuple(requirement.key for requirement in LEVEL_4_REQUIREMENTS)
        self.assertIs(
            attained_evidence_level((*level_three, *level_four)),
            EvidenceLevel.MULTI_REGIME_VALIDATION,
        )
        self.assertIs(
            attained_evidence_level(level_four),
            EvidenceLevel.DEVELOPMENT_DIAGNOSTIC,
        )
        for omitted in level_four:
            reduced = tuple(key for key in level_four if key != omitted)
            self.assertIs(
                attained_evidence_level((*level_three, *reduced)),
                EvidenceLevel.CONFIRMATORY_STRUCTURAL,
            )

    def test_level_five_requires_external_replication(self) -> None:
        through_level_four = tuple(
            requirement.key
            for requirement in ALL_REQUIREMENTS
            if requirement.minimum_level <= EvidenceLevel.MULTI_REGIME_VALIDATION
        )
        level_five = tuple(requirement.key for requirement in LEVEL_5_REQUIREMENTS)
        self.assertIs(
            attained_evidence_level((*through_level_four, *level_five)),
            EvidenceLevel.INDEPENDENT_REPLICATION,
        )


class EvidenceContractAuditTest(unittest.TestCase):
    def test_static_contract_passes_and_blocks_publication(self) -> None:
        audit = audit_paper3_evidence_contract()

        self.assertTrue(audit.passed)
        self.assertEqual(audit.identifier, EVIDENCE_CONTRACT_ID)
        self.assertIs(
            audit.current_level,
            EvidenceLevel.DEVELOPMENT_DIAGNOSTIC,
        )
        self.assertIs(
            audit.target_level,
            EvidenceLevel.MULTI_REGIME_VALIDATION,
        )
        self.assertTrue(audit.publication_blocked)
        self.assertEqual(audit.next_phase, "P3-3A-INDEPENDENCE-v1")
        self.assertEqual(len(audit.missing_for_target), 16)

    def test_phase_order_and_ceilings_are_frozen(self) -> None:
        self.assertEqual(EVIDENCE_PHASES[0].identifier, EVIDENCE_CONTRACT_ID)
        self.assertEqual(
            EVIDENCE_PHASES[1].identifier,
            "P3-3A-INDEPENDENCE-v1",
        )
        self.assertEqual(
            EVIDENCE_PHASES[-2].evidence_ceiling,
            EvidenceLevel.MULTI_REGIME_VALIDATION,
        )
        self.assertEqual(
            EVIDENCE_PHASES[-1].evidence_ceiling,
            EvidenceLevel.INDEPENDENT_REPLICATION,
        )

    def test_primary_oracle_shortcuts_are_explicitly_forbidden(self) -> None:
        policy_text = " ".join(NONNEGOTIABLE_POLICIES).lower()
        self.assertIn("target-state codebook", policy_text)
        self.assertIn("oracle upper bounds", policy_text)
        self.assertIn("seeds are nested", policy_text)
        self.assertIn("paper 4", policy_text)

    def test_digest_is_stable_and_semantic(self) -> None:
        first = evidence_contract_digest()
        second = evidence_contract_digest()

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)

    def test_missing_requirements_respects_target_level(self) -> None:
        level_three_keys = tuple(
            requirement.key for requirement in LEVEL_3_REQUIREMENTS
        )

        self.assertEqual(
            missing_requirements(
                level_three_keys,
                EvidenceLevel.CONFIRMATORY_STRUCTURAL,
            ),
            (),
        )
        self.assertEqual(
            len(
                missing_requirements(
                    level_three_keys,
                    EvidenceLevel.MULTI_REGIME_VALIDATION,
                )
            ),
            len(LEVEL_4_REQUIREMENTS),
        )


if __name__ == "__main__":
    unittest.main()
