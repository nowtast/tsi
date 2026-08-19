import unittest

from tsi.paper3_learned_v3_generator import build_v3_world_dataset
from tsi.paper3_learned_v3_pooled_model import PooledMechanismConditionedModel, encode_v3_cases


class LearnedV3PooledModelTests(unittest.TestCase):
    def test_encoding_includes_signature_and_five_layer_targets(self) -> None:
        dataset = build_v3_world_dataset(0, 0, graph_index=0)
        from tsi.paper3_learned_v2_mechanism import identify_observable_mechanism

        signature = identify_observable_mechanism(dataset.partitions["train"])
        inputs, deltas = encode_v3_cases(dataset.partitions["train"][:8], signature)
        self.assertEqual(inputs.shape, (8, 42))
        self.assertEqual(deltas.shape, (8, 5))
        self.assertGreater(inputs[:, 31:].sum(), 0.0)

    def test_pooled_model_fits_and_evaluates(self) -> None:
        train = tuple(
            build_v3_world_dataset(index, combination, graph_index=index % 4)
            for index, combination in enumerate((0, 4, 8, 12))
        )
        model = PooledMechanismConditionedModel(seed=0)
        trace = model.fit(train, updates=20)
        result = model.evaluate(train[0])
        self.assertTrue(trace.finite, trace)
        self.assertTrue(result["finite"], result)
        self.assertEqual(result["case_count"], len(train[0].partitions["test"]))


if __name__ == "__main__":
    unittest.main()
