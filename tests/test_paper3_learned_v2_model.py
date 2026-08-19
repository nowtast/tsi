import unittest

import numpy as np

from tsi.paper3_independence_contract import WorldFamily
from tsi.paper3_learned_v2_generator import build_v2_world_dataset
from tsi.paper3_learned_v2_model import JointGateRoutingModel


class LearnedV2ModelTests(unittest.TestCase):
    def test_joint_gate_fit_is_finite_and_predicts_without_codebook(self) -> None:
        dataset = build_v2_world_dataset(0)
        model = JointGateRoutingModel(WorldFamily.CONTEXT_DEPENDENT, optimizer_seed=0)
        trace = model.fit(dataset.partitions["train"], updates=10)
        self.assertTrue(trace.finite, trace)
        self.assertEqual(len(model.predict_codes(dataset.partitions["test"][:8])), 8)
        source_edges, action_edges = model.inferred_edges()
        self.assertIsInstance(source_edges, tuple)
        self.assertIsInstance(action_edges, tuple)

    def test_gate_values_are_bounded_and_parameter_count_includes_gates(self) -> None:
        model = JointGateRoutingModel(WorldFamily.CONTEXT_DEPENDENT, optimizer_seed=1)
        self.assertEqual(model.gate_values().shape, (5, 10))
        self.assertTrue(((model.gate_values() > 0.0) & (model.gate_values() < 1.0)).all())
        self.assertGreater(model.parameter_count, 420)

    def test_threshold_selection_uses_routing_split_and_freezes_support(self) -> None:
        dataset = build_v2_world_dataset(0)
        model = JointGateRoutingModel(WorldFamily.CONTEXT_DEPENDENT, optimizer_seed=2)
        selection = model.select_threshold(
            dataset.partitions["train"],
            dataset.partitions["routing_selection"],
            thresholds=(0.10, 0.30, 0.50),
            updates=3,
            refit_updates=3,
        )
        self.assertIn(selection.selected_threshold, (0.10, 0.30, 0.50))
        selected_values = selection.selected_model.gate_values()
        self.assertTrue(np.all((selected_values < 1e-8) | (selected_values > 1.0 - 1e-8)))
        self.assertEqual(len(selection.candidate_scores), 3)


if __name__ == "__main__":
    unittest.main()
