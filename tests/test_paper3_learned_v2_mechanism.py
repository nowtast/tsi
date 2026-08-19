import unittest

from tsi.paper3_learned_v2_identifiability import pixel_carrier_collision_audit
from dataclasses import replace

from tsi.paper3_learned_v2_generator import build_balanced_v2_world_dataset
from tsi.paper3_learned_v2_mechanism import (
    identify_observable_mechanism,
    evaluate_identified_signature,
    evaluate_pixel_identified_signature,
    MechanismConditionedStructuredHead,
    DenoisedMechanismConditionedStructuredHead,
)


class LearnedV2MechanismConditioningTests(unittest.TestCase):
    def test_full_pixel_carrier_is_collision_free(self) -> None:
        audit = pixel_carrier_collision_audit()
        self.assertEqual(audit["state_count"], 324)
        self.assertEqual(audit["unique_clean_images"], 324)
        self.assertEqual(audit["ambiguous_image_classes"], 0)

    def test_identification_uses_training_transitions_not_graph_label(self) -> None:
        dataset = build_balanced_v2_world_dataset(40, 0)
        training = tuple(replace(case, graph_variant="wrong_direction_negative_control") for case in dataset.partitions["train"])
        signature = identify_observable_mechanism(training)
        self.assertEqual(signature.graph_variant, "bridge_topology_to_relation")
        self.assertGreaterEqual(signature.candidate_count, 1)

    def test_pixel_conditioned_predictor_uses_raster_source_not_source_code(self) -> None:
        dataset = build_balanced_v2_world_dataset(40, 0)
        from tsi.paper3_learned_v2_observation import build_observed_partitions

        observed = build_observed_partitions(
            dict(dataset.partitions), entity_count=3, regime="pixel_object_observation", seed=10040
        )
        report = evaluate_pixel_identified_signature(observed["train"], observed["test"])
        self.assertTrue(report["all_exact"], report)

    def test_soft_denoised_head_returns_valid_delta_distribution(self) -> None:
        dataset = build_balanced_v2_world_dataset(40, 0)
        from tsi.paper3_learned_v2_observation import build_observed_partitions

        observed = build_observed_partitions(
            dict(dataset.partitions), entity_count=3, regime="pixel_object_observation", seed=10040
        )
        head = DenoisedMechanismConditionedStructuredHead.fit(observed["train"])
        probabilities = head.delta_probabilities(observed["test"][0])
        self.assertTrue(all(abs(sum(row) - 1.0) < 1.0e-9 for row in probabilities))

    def test_structured_head_exposes_signature_and_clean_logloss(self) -> None:
        dataset = build_balanced_v2_world_dataset(40, 0)
        from tsi.paper3_learned_v2_observation import build_observed_partitions

        observed = build_observed_partitions(
            dict(dataset.partitions), entity_count=3, regime="pixel_object_observation", seed=10040
        )
        head = MechanismConditionedStructuredHead.fit(observed["train"])
        self.assertEqual(head.signature.graph_variant, "bridge_topology_to_relation")
        self.assertLess(head.mean_logloss(observed["test"]), 1.0e-4)

    def test_conditioned_predictor_handles_unseen_mechanism_actions(self) -> None:
        dataset = build_balanced_v2_world_dataset(42, 2)
        report = evaluate_identified_signature(dataset.partitions["train"], dataset.partitions["test"])
        self.assertTrue(report["all_exact"], report)
        self.assertEqual(report["exact_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
