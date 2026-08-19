import unittest

from tsi.paper3_independence_contract import BenchmarkSplit, WorldFamily
from tsi.paper3_learned_routing import (
    LEARNED_ACTION_CROSS_EDGE_BUDGET,
    LEARNED_SOURCE_CROSS_EDGE_BUDGET,
    edge_f1,
    run_learned_routing_pilot,
)
from tsi.paper3_multiworld import build_world_dataset, build_world_mechanism


class LearnedRoutingPilotTests(unittest.TestCase):
    def test_pilot_returns_a_budgeted_learned_manifest(self) -> None:
        family = WorldFamily.CONTEXT_DEPENDENT
        dataset = build_world_dataset(
            build_world_mechanism(family, BenchmarkSplit.DEVELOPMENT, 0)
        )
        result, model = run_learned_routing_pilot(
            dataset.partitions["train"],
            family=family,
            world_index=0,
            optimizer_seed=0,
            updates=10,
        )
        self.assertTrue(result.dense_trace.finite)
        self.assertTrue(result.learned_trace.finite)
        self.assertEqual(model.manifest.identifier, "learned_signature_routing")
        self.assertEqual(
            len(result.source_edges),
            5 + LEARNED_SOURCE_CROSS_EDGE_BUDGET[family],
        )
        self.assertEqual(
            len(result.action_edges),
            5 + LEARNED_ACTION_CROSS_EDGE_BUDGET[family],
        )

    def test_edge_f1_is_defined_on_empty_and_nonempty_graphs(self) -> None:
        self.assertEqual(edge_f1((), ()), 1.0)
        self.assertEqual(edge_f1((("a", "b"),), (("a", "c"),)), 0.0)
        self.assertEqual(edge_f1((("a", "b"),), (("a", "b"),)), 1.0)


if __name__ == "__main__":
    unittest.main()
