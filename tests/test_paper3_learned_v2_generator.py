import unittest

from tsi.paper3_learned_v2_generator import (
    audit_v2_dataset,
    build_balanced_v2_world_dataset,
    build_v2_world_dataset,
    graph_variant_for_world,
)


class LearnedV2GeneratorTests(unittest.TestCase):
    def test_four_way_split_is_disjoint_and_has_interventions(self) -> None:
        dataset = build_v2_world_dataset(0)
        audit = audit_v2_dataset(dataset)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(set(dataset.partitions), {"train", "routing_selection", "downstream_evaluation", "test"})
        self.assertTrue(any(case.intervention for case in dataset.partitions["test"]))

    def test_graph_variant_changes_across_worlds(self) -> None:
        self.assertNotEqual(graph_variant_for_world(0).identifier, graph_variant_for_world(1).identifier)
        self.assertNotEqual(graph_variant_for_world(1).identifier, graph_variant_for_world(2).identifier)

    def test_balanced_generator_crosses_graph_and_mechanism_slots(self) -> None:
        first = build_balanced_v2_world_dataset(0, 0)
        second = build_balanced_v2_world_dataset(1, 0)
        third = build_balanced_v2_world_dataset(0, 1)
        self.assertEqual(first.mechanism.active_parameter_signature, second.mechanism.active_parameter_signature)
        self.assertNotEqual(first.graph.identifier, second.graph.identifier)
        self.assertNotEqual(first.mechanism.active_parameter_signature, third.mechanism.active_parameter_signature)


    def test_balanced_validation_rejects_duplicate_mechanism_slots(self) -> None:
        from tsi.paper3_learned_v2_robustness import run_pixel_robustness_balanced_validation

        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            run_pixel_robustness_balanced_validation(mechanism_slots=(0, 0))

    def test_generator_is_deterministic(self) -> None:
        first = build_v2_world_dataset(3)
        second = build_v2_world_dataset(3)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.graph, second.graph)


if __name__ == "__main__":
    unittest.main()
