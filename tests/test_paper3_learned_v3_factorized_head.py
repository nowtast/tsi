import unittest

from tsi.paper3_learned_v3_factorized_head import (
    GraphConditionedFactorizedHead,
    factorize_training_signature,
)
from tsi.paper3_learned_v3_generator import build_v3_world_dataset


class LearnedV3FactorizedHeadTests(unittest.TestCase):
    def test_active_signature_is_factorized_exactly(self) -> None:
        dataset = build_v3_world_dataset(0, 0, graph_index=0)
        signature = factorize_training_signature(
            dataset.partitions["train"], dataset.graph.identifier
        )
        self.assertEqual(
            signature.layer_multipliers, dataset.mechanism.layer_multipliers
        )
        self.assertEqual(
            signature.bridge_coefficient, dataset.mechanism.bridge_coefficient
        )

    def test_factorized_head_handles_held_out_case(self) -> None:
        train = tuple(
            build_v3_world_dataset(index * 4 + graph, index, graph_index=graph)
            for index in (0, 8, 12)
            for graph in range(4)
        )
        evaluation = build_v3_world_dataset(11 * 4 + 2, 11, graph_index=2)
        head = GraphConditionedFactorizedHead.fit(train)
        result = head.evaluate(evaluation)
        self.assertEqual(result["exact_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
