import unittest

from tsi.paper3_learned_v3_generator import audit_v3_dataset, build_v3_world_dataset
from tsi.paper3_learned_v3_contract import TEST_COMBINATION_INDICES, mechanism_split_for_combination


class LearnedV3GeneratorTests(unittest.TestCase):
    def test_five_way_split_is_disjoint_and_has_interventions(self) -> None:
        dataset = build_v3_world_dataset(0, 150, graph_index=0)
        audit = audit_v3_dataset(dataset)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(
            set(dataset.partitions),
            {"train", "routing_selection", "calibration", "downstream_evaluation", "test"},
        )
        self.assertTrue(any(case.intervention for case in dataset.partitions["test"]))
        self.assertEqual(mechanism_split_for_combination(next(iter(TEST_COMBINATION_INDICES))), "test")

    def test_world_index_does_not_change_mechanism_combination(self) -> None:
        first = build_v3_world_dataset(0, 7, graph_index=0)
        second = build_v3_world_dataset(1, 7, graph_index=1)
        self.assertEqual(first.mechanism.active_parameter_signature, second.mechanism.active_parameter_signature)
        self.assertNotEqual(first.graph.identifier, second.graph.identifier)


if __name__ == "__main__":
    unittest.main()
