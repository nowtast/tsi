import unittest

from tsi.paper3_learned_v3_generator import build_v3_world_dataset
from tsi.paper3_learned_v3_structured_head import StructuredParameterizedTransitionHead


class LearnedV3StructuredHeadTests(unittest.TestCase):
    def test_explicit_head_fits_training_panel(self) -> None:
        datasets = tuple(build_v3_world_dataset(index, index, graph_index=index % 4) for index in (0, 8, 12))
        head = StructuredParameterizedTransitionHead.fit(datasets)
        self.assertEqual(head.trace.training_exact_accuracy, 1.0)
        self.assertEqual(head.trace.training_world_count, 3)

    def test_explicit_head_generalizes_held_out_combination(self) -> None:
        train = tuple(build_v3_world_dataset(index, index, graph_index=index % 4) for index in (0, 8, 12))
        held_out = build_v3_world_dataset(19, 19, graph_index=3)
        result = StructuredParameterizedTransitionHead.fit(train).evaluate(held_out)
        self.assertEqual(result["exact_accuracy"], 1.0, result)


if __name__ == "__main__":
    unittest.main()
