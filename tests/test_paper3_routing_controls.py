from __future__ import annotations

import unittest

from tsi.paper3_independence_contract import WorldFamily
from tsi.paper3_routing_controls import (
    TRANSITION_ACTIVE_PARAMETER_BUDGET,
    audit_routing_controls,
    correct_action_cross_edges,
    correct_cross_edges,
    routing_control_digest,
    routing_control_manifests,
    wrong_cross_edges,
)


class RoutingControlTest(unittest.TestCase):
    def test_every_family_has_six_capacity_matched_models(self) -> None:
        for family in WorldFamily:
            manifests = routing_control_manifests(family)

            self.assertEqual(len(manifests), 6)
            self.assertEqual(
                {manifest.total_active_parameters for manifest in manifests},
                {TRANSITION_ACTIVE_PARAMETER_BUDGET},
            )

    def test_correct_dependencies_match_world_family(self) -> None:
        self.assertEqual(correct_cross_edges(WorldFamily.SEPARABLE), ())
        self.assertEqual(
            correct_cross_edges(WorldFamily.BRIDGE_COUPLED),
            (("topology", "relation"),),
        )
        self.assertEqual(
            correct_cross_edges(WorldFamily.CONTEXT_DEPENDENT),
            (
                ("topology", "relation"),
                ("order", "metric"),
            ),
        )
        self.assertEqual(
            correct_action_cross_edges(WorldFamily.CONTEXT_DEPENDENT),
            (("topology", "relation"),),
        )

    def test_wrong_dependencies_reverse_declared_edges(self) -> None:
        self.assertEqual(
            wrong_cross_edges(WorldFamily.BRIDGE_COUPLED),
            (("relation", "topology"),),
        )
        self.assertEqual(
            wrong_cross_edges(WorldFamily.CONTEXT_DEPENDENT),
            (
                ("relation", "topology"),
                ("metric", "order"),
            ),
        )

    def test_random_and_wrong_masks_differ_from_correct(self) -> None:
        for family in (
            WorldFamily.BRIDGE_COUPLED,
            WorldFamily.CONTEXT_DEPENDENT,
        ):
            by_id = {
                manifest.identifier: manifest
                for manifest in routing_control_manifests(family)
            }
            correct = by_id["signature_routed_oracle"]
            random = by_id["random_routed_matched_sparsity"]
            wrong = by_id["permuted_or_wrong_routed"]

            self.assertNotEqual(random.source_edges, correct.source_edges)
            self.assertNotEqual(wrong.source_edges, correct.source_edges)

    def test_digest_is_deterministic(self) -> None:
        self.assertEqual(routing_control_digest(), routing_control_digest())

    def test_routing_machine_audit_passes(self) -> None:
        audit = audit_routing_controls()

        self.assertTrue(audit.passed)
        self.assertEqual(audit.max_relative_parameter_difference, 0.0)
        self.assertTrue(audit.information_fields_matched)
        self.assertTrue(audit.training_updates_matched)
        self.assertTrue(audit.compute_budgets_matched)
        self.assertTrue(audit.tuning_budgets_matched)


if __name__ == "__main__":
    unittest.main()
