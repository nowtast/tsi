from __future__ import annotations

import unittest

import numpy as np

from tsi.paper3_independence_contract import BenchmarkSplit, WorldFamily
from tsi.paper3_multiworld import build_world_dataset, build_world_mechanism
from tsi.paper3_routing_controls import (
    TRANSITION_ACTIVE_PARAMETER_BUDGET,
    routing_control_manifests,
)
from tsi.paper3_routing_model import (
    ACTION_FEATURE_SLICES,
    SOURCE_FEATURE_SLICES,
    MaskedRandomFeatureBasis,
    TrainableRoutingModel,
    routing_input_masks,
    routing_model_digest,
)


class RoutingModelTest(unittest.TestCase):
    def test_every_model_has_exact_trainable_budget(self) -> None:
        for family in WorldFamily:
            for manifest in routing_control_manifests(family):
                model = TrainableRoutingModel(manifest, optimizer_seed=0)
                self.assertEqual(
                    model.parameter_count,
                    TRANSITION_ACTIVE_PARAMETER_BUDGET,
                )

    def test_context_source_and_action_paths_are_distinct(self) -> None:
        manifest = {
            item.identifier: item
            for item in routing_control_manifests(WorldFamily.CONTEXT_DEPENDENT)
        }["signature_routed_oracle"]
        masks = routing_input_masks(manifest)
        metric_target = 2
        self.assertTrue(np.all(masks[metric_target, SOURCE_FEATURE_SLICES[4]] == 1.0))
        self.assertTrue(np.all(masks[metric_target, ACTION_FEATURE_SLICES[2]] == 1.0))
        self.assertTrue(np.all(masks[metric_target, ACTION_FEATURE_SLICES[4]] == 0.0))

    def test_separable_equivalent_masks_share_paired_features(self) -> None:
        manifests = {
            item.identifier: item
            for item in routing_control_manifests(WorldFamily.SEPARABLE)
        }
        strict = MaskedRandomFeatureBasis(
            manifests["strict_factorized_action"],
            optimizer_seed=2,
        )
        signature = MaskedRandomFeatureBasis(
            manifests["signature_routed_oracle"],
            optimizer_seed=2,
        )
        np.testing.assert_array_equal(strict.weights, signature.weights)
        np.testing.assert_array_equal(strict.biases, signature.biases)

    def test_training_decreases_loss_and_predicts_valid_codes(self) -> None:
        mechanism = build_world_mechanism(
            WorldFamily.BRIDGE_COUPLED,
            BenchmarkSplit.DEVELOPMENT,
            0,
        )
        dataset = build_world_dataset(mechanism)
        manifest = {
            item.identifier: item
            for item in routing_control_manifests(mechanism.family)
        }["signature_routed_oracle"]
        model = TrainableRoutingModel(manifest, optimizer_seed=0)
        features, deltas = model.basis.transform_cases(dataset.partitions["train"])
        trace = model.fit_precomputed(features, deltas, updates=20)

        self.assertTrue(trace.finite)
        self.assertLess(trace.final_nll, trace.initial_nll)
        validation_features = model.basis.transform_cases(
            dataset.partitions["validation"]
        )[0]
        predictions = model.predict_codes_precomputed(
            dataset.partitions["validation"],
            validation_features,
        )
        self.assertEqual(len(predictions), len(dataset.partitions["validation"]))

    def test_digest_is_deterministic(self) -> None:
        self.assertEqual(routing_model_digest(), routing_model_digest())


if __name__ == "__main__":
    unittest.main()
