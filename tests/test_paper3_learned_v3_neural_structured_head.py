import unittest

from tsi.paper3_learned_v3_generator import build_v3_world_dataset
from tsi.paper3_learned_v3_neural_structured_head import (
    NeuralStructuredTransitionHead,
    encode_structured_cases,
)
from tsi.paper3_learned_v2_mechanism import identify_observable_mechanism


class LearnedV3NeuralStructuredHeadTests(unittest.TestCase):
    def test_structured_encoding_shape(self) -> None:
        d = build_v3_world_dataset(0, 0, graph_index=0)
        s = identify_observable_mechanism(d.partitions["train"])
        x, y = encode_structured_cases(d.partitions["train"][:4], s)
        self.assertEqual(x.shape, (4, 77))
        self.assertEqual(y.shape, (4, 5))

    def test_head_fits(self) -> None:
        ds = tuple(
            build_v3_world_dataset(i, c, graph_index=i % 4)
            for i, c in enumerate((0, 8, 12))
        )
        h = NeuralStructuredTransitionHead(seed=0)
        trace = h.fit(ds, updates=20)
        result = h.evaluate(ds[0])
        self.assertTrue(trace.finite)
        self.assertTrue(result["finite"])


if __name__ == "__main__":
    unittest.main()
