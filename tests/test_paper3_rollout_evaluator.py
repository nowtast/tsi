from __future__ import annotations

import unittest

from tsi.paper3_development_experiment import ConstructiveMetricCache
from tsi.paper3_independence_contract import BenchmarkSplit, WorldFamily
from tsi.paper3_multiworld import build_world_dataset, build_world_mechanism
from tsi.paper3_rollout_evaluator import (
    audit_fixed_metric_and_lipschitz,
    evaluate_rollout_model,
)
from tsi.paper3_rollout_generator import development_rollout_trajectories
from tsi.paper3_routing_controls import routing_control_manifests
from tsi.paper3_routing_model import TrainableRoutingModel


class Paper3RolloutEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mechanism = build_world_mechanism(
            WorldFamily.CONTEXT_DEPENDENT,
            BenchmarkSplit.DEVELOPMENT,
            0,
        )

    def test_fixed_metric_and_exact_lipschitz_audit(self) -> None:
        audit = audit_fixed_metric_and_lipschitz(self.mechanism)

        self.assertTrue(audit["passed"])
        self.assertLessEqual(audit["maximum_triangle_excess"], 1.0e-12)
        self.assertEqual(len(audit["exact_lipschitz_constants"]), 6)

    def test_signature_model_has_valid_recursive_rollout_bound(self) -> None:
        manifest = next(
            item
            for item in routing_control_manifests(WorldFamily.CONTEXT_DEPENDENT)
            if item.identifier == "signature_routed_oracle"
        )
        model = TrainableRoutingModel(manifest, 0)
        dataset = build_world_dataset(self.mechanism)
        model.fit(dataset.partitions["train"])

        metrics = evaluate_rollout_model(
            ConstructiveMetricCache(),
            model,
            self.mechanism,
            development_rollout_trajectories(0)[:2],
        )

        self.assertEqual(metrics["recursive_bound_violation_count"], 0)
        self.assertLessEqual(
            metrics["maximum_recursive_bound_excess"],
            1.0e-12,
        )
        self.assertEqual(metrics["global_target_state_candidates"], 0)
        self.assertEqual(metrics["trajectory_count"], 2)


if __name__ == "__main__":
    unittest.main()
