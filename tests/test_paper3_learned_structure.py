import unittest

from tsi.paper3_learned_structure import discover_replication_graph, discover_v3_graph
from tsi.paper3_learned_v3_generator import build_v3_world_dataset
from tsi.paper3_replication_family import build_replication_dataset


class LearnedStructureTests(unittest.TestCase):
    def test_v3_discovers_graph_without_graph_input(self) -> None:
        for graph in range(4):
            result = discover_v3_graph(
                build_v3_world_dataset(graph, 7, graph_index=graph)
            )
            self.assertTrue(result["graph_exact"])
            self.assertIn("candidate_training_accuracies", result)

    def test_replication_discovers_graph_without_graph_input(self) -> None:
        for graph in ("topology_to_metric", "metric_to_relation", "relation_to_order"):
            result = discover_replication_graph(build_replication_dataset(graph, 7))
            self.assertTrue(result["graph_exact"])
            self.assertIn("candidate_training_accuracies", result)


if __name__ == "__main__":
    unittest.main()
