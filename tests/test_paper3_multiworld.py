from __future__ import annotations

import unittest

from tsi.coherent import bridge_defects
from tsi.paper3_independence_contract import BenchmarkSplit, WorldFamily
from tsi.paper3_multiworld import (
    BRIDGE_PROBE_ACTION,
    PRIMITIVE_ACTIONS,
    MultiworldStateCode,
    all_multiworld_state_codes,
    build_multiworld_state,
    build_world_dataset,
    build_world_mechanism,
    development_validation_world_manifest,
    multiworld_generator_digest,
    successor_code,
    violating_bridge_successor_code,
)
from tsi.paper3_multiworld_audit import (
    REQUIRED_OOD_SLICES,
    audit_multiworld_generator,
)


class MultiworldStateTest(unittest.TestCase):
    def test_state_code_family_has_324_unique_members(self) -> None:
        codes = all_multiworld_state_codes()

        self.assertEqual(len(codes), 324)
        self.assertEqual(len(set(codes)), 324)

    def test_every_state_is_coherent(self) -> None:
        for code in all_multiworld_state_codes():
            state = build_multiworld_state(code)
            self.assertTrue(
                all(
                    value == 0.0
                    for value in bridge_defects(
                        state.core,
                        state.order,
                        state.signature,
                    ).values()
                )
            )

    def test_influence_relation_is_not_a_topology_duplicate(self) -> None:
        topology_fixed_a = build_multiworld_state(MultiworldStateCode(0, 0, 0, 0, 0))
        topology_fixed_b = build_multiworld_state(MultiworldStateCode(0, 0, 0, 1, 0))
        relation_fixed_a = build_multiworld_state(MultiworldStateCode(0, 0, 0, 2, 0))
        relation_fixed_b = build_multiworld_state(MultiworldStateCode(0, 1, 0, 2, 0))

        self.assertEqual(
            topology_fixed_a.core.simplices,
            topology_fixed_b.core.simplices,
        )
        self.assertNotEqual(
            topology_fixed_a.core.relational.generators["influences"],
            topology_fixed_b.core.relational.generators["influences"],
        )
        self.assertEqual(
            relation_fixed_a.core.relational.generators["influences"],
            relation_fixed_b.core.relational.generators["influences"],
        )
        self.assertNotEqual(
            relation_fixed_a.core.simplices,
            relation_fixed_b.core.simplices,
        )


class MultiworldMechanismTest(unittest.TestCase):
    def test_manifest_has_only_public_development_validation_worlds(self) -> None:
        manifest = development_validation_world_manifest()

        self.assertEqual(len(manifest), 108)
        self.assertEqual(len({world.identifier for world in manifest}), 108)
        self.assertEqual(
            sum(world.cohort is BenchmarkSplit.DEVELOPMENT for world in manifest),
            72,
        )
        self.assertEqual(
            sum(world.cohort is BenchmarkSplit.VALIDATION for world in manifest),
            36,
        )
        self.assertFalse(
            any(world.cohort is BenchmarkSplit.SEALED_TEST for world in manifest)
        )
        for family in WorldFamily:
            active_signatures = {
                world.active_parameter_signature
                for world in manifest
                if world.family is family
            }
            self.assertEqual(len(active_signatures), 36)

    def test_world_mechanism_derivation_is_deterministic_and_distinct(self) -> None:
        first = build_world_mechanism(
            WorldFamily.BRIDGE_COUPLED,
            BenchmarkSplit.DEVELOPMENT,
            0,
        )
        replay = build_world_mechanism(
            WorldFamily.BRIDGE_COUPLED,
            BenchmarkSplit.DEVELOPMENT,
            0,
        )
        second = build_world_mechanism(
            WorldFamily.BRIDGE_COUPLED,
            BenchmarkSplit.DEVELOPMENT,
            1,
        )

        self.assertEqual(first, replay)
        self.assertNotEqual(first.mechanism_digest, second.mechanism_digest)

    def test_declared_dependency_families_have_witnesses(self) -> None:
        topology_action = next(
            action for action in PRIMITIVE_ACTIONS if action.name == "topology_step"
        )
        metric_action = next(
            action for action in PRIMITIVE_ACTIONS if action.name == "metric_step"
        )
        topology_zero = MultiworldStateCode(0, 0, 0, 0, 0)
        topology_two = MultiworldStateCode(0, 2, 0, 0, 0)
        order_zero = MultiworldStateCode(0, 0, 0, 0, 0)
        order_one = MultiworldStateCode(0, 0, 0, 0, 1)
        separable = build_world_mechanism(
            WorldFamily.SEPARABLE,
            BenchmarkSplit.DEVELOPMENT,
            0,
        )
        bridge = build_world_mechanism(
            WorldFamily.BRIDGE_COUPLED,
            BenchmarkSplit.DEVELOPMENT,
            0,
        )
        context = build_world_mechanism(
            WorldFamily.CONTEXT_DEPENDENT,
            BenchmarkSplit.DEVELOPMENT,
            0,
        )

        self.assertEqual(
            successor_code(
                topology_zero,
                topology_action,
                separable,
            ).influence_mode,
            successor_code(
                topology_two,
                topology_action,
                separable,
            ).influence_mode,
        )
        self.assertNotEqual(
            successor_code(
                topology_zero,
                topology_action,
                bridge,
            ).influence_mode,
            successor_code(
                topology_two,
                topology_action,
                bridge,
            ).influence_mode,
        )
        self.assertNotEqual(
            successor_code(
                order_zero,
                metric_action,
                context,
            ).metric_mode,
            successor_code(
                order_one,
                metric_action,
                context,
            ).metric_mode,
        )

    def test_world_dataset_has_disjoint_inputs_and_all_ood_slices(self) -> None:
        for family in WorldFamily:
            mechanism = build_world_mechanism(
                family,
                BenchmarkSplit.DEVELOPMENT,
                0,
            )
            dataset = build_world_dataset(mechanism)
            train = {case.input_key for case in dataset.partitions["train"]}
            validation = {case.input_key for case in dataset.partitions["validation"]}
            ood = {case.input_key for case in dataset.partitions["ood"]}

            self.assertFalse(train.intersection(validation))
            self.assertFalse(train.intersection(ood))
            self.assertEqual(
                set(dataset.ood_by_slice),
                set(REQUIRED_OOD_SLICES),
            )

    def test_bridge_negative_control_changes_only_declared_target_code(self) -> None:
        source = MultiworldStateCode(0, 0, 0, 0, 0)
        for family in WorldFamily:
            mechanism = build_world_mechanism(
                family,
                BenchmarkSplit.DEVELOPMENT,
                0,
            )
            correct = successor_code(source, BRIDGE_PROBE_ACTION, mechanism)
            violating = violating_bridge_successor_code(
                source,
                BRIDGE_PROBE_ACTION,
                mechanism,
            )

            self.assertNotEqual(correct.influence_mode, violating.influence_mode)
            self.assertEqual(correct.label_phase, violating.label_phase)
            self.assertEqual(correct.topology_mode, violating.topology_mode)
            self.assertEqual(correct.metric_mode, violating.metric_mode)
            self.assertEqual(correct.order_mode, violating.order_mode)

    def test_generator_digest_is_deterministic(self) -> None:
        self.assertEqual(
            multiworld_generator_digest(),
            multiworld_generator_digest(),
        )

    def test_generator_machine_audit_passes(self) -> None:
        audit = audit_multiworld_generator()

        self.assertTrue(audit.passed)
        self.assertEqual(audit.state_count, 324)
        self.assertEqual(audit.manifest_world_count, 108)
        self.assertEqual(set(audit.active_signature_counts.values()), {36})
        self.assertEqual(audit.independent_relation_witnesses, 2)
        self.assertEqual(audit.sealed_test_worlds_materialized, 0)


if __name__ == "__main__":
    unittest.main()
