from __future__ import annotations

import unittest

from tsi.paper3_independence_contract import (
    DECODER_CONSTRUCTIONS,
    MODEL_CONTROLS,
    OOD_SLICES,
    P3_INDEPENDENCE_CONTRACT_ID,
    PRIMARY_CONTROLS,
    RELATION_GENERATORS,
    REQUIRED_ARTIFACTS,
    DecoderRegime,
    WorldFamily,
    audit_p3_3a_independence_contract,
    independence_contract_digest,
    planned_test_world_count,
)


class IndependenceStaticContractTest(unittest.TestCase):
    def test_static_contract_passes_but_gate_remains_blocked(self) -> None:
        audit = audit_p3_3a_independence_contract()

        self.assertEqual(audit.identifier, P3_INDEPENDENCE_CONTRACT_ID)
        self.assertTrue(audit.static_contract_passed)
        self.assertFalse(audit.gate_passed)
        self.assertEqual(audit.evidence_level_before, 2)
        self.assertEqual(audit.evidence_level_after, 2)
        self.assertEqual(audit.test_seed_reveals, 0)
        self.assertEqual(audit.test_result_evaluations, 0)
        self.assertEqual(audit.artifact_blockers, REQUIRED_ARTIFACTS)

    def test_all_artifacts_can_pass_only_with_zero_test_access(self) -> None:
        completed = {artifact: True for artifact in REQUIRED_ARTIFACTS}

        self.assertTrue(audit_p3_3a_independence_contract(completed).gate_passed)
        self.assertFalse(
            audit_p3_3a_independence_contract(
                completed,
                test_seed_reveals=1,
            ).gate_passed
        )
        self.assertFalse(
            audit_p3_3a_independence_contract(
                completed,
                test_result_evaluations=1,
            ).gate_passed
        )

    def test_unknown_artifacts_are_contract_errors(self) -> None:
        audit = audit_p3_3a_independence_contract({"not_registered": True})

        self.assertFalse(audit.static_contract_passed)
        self.assertIn("unknown artifact", audit.static_errors[0])

    def test_model_control_matrix_is_complete_and_unique(self) -> None:
        identifiers = tuple(model.identifier for model in MODEL_CONTROLS)

        self.assertEqual(len(identifiers), 6)
        self.assertEqual(len(set(identifiers)), 6)
        self.assertIn("signature_routed_oracle", identifiers)
        self.assertTrue(set(PRIMARY_CONTROLS).issubset(identifiers))

    def test_generator_has_three_families_and_six_ood_slices(self) -> None:
        self.assertEqual(
            {family.value for family in WorldFamily},
            {"separable", "bridge_coupled", "context_dependent"},
        )
        self.assertEqual(len(OOD_SLICES), 6)
        self.assertEqual(len(set(OOD_SLICES)), 6)

    def test_relation_contract_contains_independent_information(self) -> None:
        self.assertEqual(
            RELATION_GENERATORS["adjacent"],
            "simplicial_1_skeleton_bridge_bound",
        )
        self.assertEqual(
            RELATION_GENERATORS["influences"],
            "independent_directed_relation",
        )

    def test_constructive_decoder_covers_every_output_component(self) -> None:
        self.assertEqual(
            set(DECODER_CONSTRUCTIONS),
            {"label", "topology", "metric", "relation", "order", "tracking"},
        )
        self.assertEqual(
            DecoderRegime.CONSTRUCTIVE_VALID_PRIMARY.value,
            "constructive_valid_primary",
        )

    def test_digest_is_deterministic(self) -> None:
        first = independence_contract_digest()
        second = independence_contract_digest()

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)


class FrozenPowerRuleTest(unittest.TestCase):
    def test_world_count_has_a_conservative_floor(self) -> None:
        self.assertEqual(planned_test_world_count(0.0), 47)
        self.assertEqual(planned_test_world_count(0.08), 47)

    def test_world_count_uses_development_variance_only(self) -> None:
        self.assertEqual(planned_test_world_count(0.10), 73)

    def test_underpowered_maximum_blocks_test_reveal(self) -> None:
        with self.assertRaisesRegex(ValueError, "underpowered"):
            planned_test_world_count(0.20)

    def test_negative_standard_deviation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            planned_test_world_count(-0.01)


if __name__ == "__main__":
    unittest.main()
