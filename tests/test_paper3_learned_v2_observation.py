import unittest

import numpy as np

from tsi.paper3_independence_contract import WorldFamily
from tsi.paper3_learned_v2_generator import build_v2_world_dataset
from tsi.paper3_learned_v2_model import JointGateRoutingModel
from tsi.paper3_learned_v2_observation import (
    build_observed_partitions,
    encode_observed_cases,
    observed_case_from_v2_case,
    encode_pixel_cases,
    corrupt_pixel_case,
    pixel_feature_leakage_audit,
    pixel_case_from_v2_case,
)


class LearnedV2ObservationTests(unittest.TestCase):
    def test_encoder_is_fixed_width_and_permutation_invariant(self) -> None:
        dataset = build_v2_world_dataset(0)
        case = observed_case_from_v2_case(dataset.partitions["train"][0], entity_count=3)
        permuted = type(case).from_v2_case(
            dataset.partitions["train"][0],
            case.observation.permute((2, 0, 1)),
        )
        first, _ = encode_observed_cases((case,))
        second, _ = encode_observed_cases((permuted,))
        self.assertEqual(first.shape, (1, 31))
        np.testing.assert_allclose(first, second)

    def test_noise_and_cardinality_are_in_actual_input_path(self) -> None:
        dataset = build_v2_world_dataset(0)
        partitions = dict(dataset.partitions)
        noisy_two = build_observed_partitions(
            partitions,
            entity_count=2,
            regime="noisy_recovered_structure",
            noise=0.10,
            seed=7,
        )
        noisy_four = build_observed_partitions(
            partitions,
            entity_count=4,
            regime="noisy_recovered_structure",
            noise=0.10,
            seed=7,
        )
        two, _ = encode_observed_cases(noisy_two["test"][:4])
        four, _ = encode_observed_cases(noisy_four["test"][:4])
        self.assertFalse(np.allclose(two, four))
        self.assertTrue(np.isfinite(two).all())
        self.assertTrue(np.isfinite(four).all())

    def test_strong_pixel_corruption_changes_encoded_input(self) -> None:
        dataset = build_v2_world_dataset(0)
        case = pixel_case_from_v2_case(
            dataset.partitions["train"][0],
            entity_count=3,
        )
        corrupted = corrupt_pixel_case(
            case,
            gaussian_noise=0.50,
            dropout_probability=0.50,
            quantization_levels=4,
            seed=9,
        )
        clean, _ = encode_pixel_cases((case,))
        noisy, _ = encode_pixel_cases((corrupted,))
        self.assertGreater(float(np.linalg.norm(clean - noisy)), 0.0)

    def test_pixel_leakage_audit_is_explicitly_diagnostic(self) -> None:
        dataset = build_v2_world_dataset(0)
        cases = tuple(
            pixel_case_from_v2_case(case, entity_count=3)
            for case in dataset.partitions["train"][:8]
        )
        audit = pixel_feature_leakage_audit(cases)
        self.assertIn("unique_feature_fraction", audit)
        self.assertIn("warning", audit)
        self.assertEqual(audit["feature_width"], 31)

    def test_pixel_encoder_uses_raster_input_and_model_path(self) -> None:
        dataset = build_v2_world_dataset(0)
        case = pixel_case_from_v2_case(
            dataset.partitions["train"][0],
            entity_count=3,
            noise=0.10,
            seed=3,
        )
        features, _ = encode_pixel_cases((case,))
        self.assertEqual(features.shape, (1, 31))
        self.assertTrue(np.isfinite(features).all())

    def test_joint_gate_model_fits_observed_train_and_predicts_held_out_count(self) -> None:
        dataset = build_v2_world_dataset(0)
        train = build_observed_partitions(
            dict(dataset.partitions),
            entity_count=3,
            regime="noisy_recovered_structure",
            noise=0.10,
            seed=11,
        )
        held_out = build_observed_partitions(
            dict(dataset.partitions),
            entity_count=2,
            regime="noisy_recovered_structure",
            noise=0.10,
            seed=12,
        )
        model = JointGateRoutingModel(WorldFamily.CONTEXT_DEPENDENT, optimizer_seed=0)
        trace = model.fit(train["train"], updates=5)
        self.assertTrue(trace.finite)
        self.assertEqual(len(model.predict_codes(held_out["test"][:6])), 6)


if __name__ == "__main__":
    unittest.main()
