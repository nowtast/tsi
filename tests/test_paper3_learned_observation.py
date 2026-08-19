import unittest

import numpy as np

from tsi.paper3_learned_observation import (
    observation_from_arrays,
)


class LearnedObservationTests(unittest.TestCase):
    def _observation(self):
        entities = np.asarray(
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [2.0, 1.0, 1.0, 0.0]]
        )
        pairs = np.zeros((3, 3, 2), dtype=np.float64)
        for source in range(3):
            for target in range(3):
                pairs[source, target] = (source + 0.1, target + 0.2)
        return observation_from_arrays("exact_state_learned_routing", entities, pairs)

    def test_presentation_permutation_preserves_entity_and_pair_semantics(self) -> None:
        observation = self._observation()
        permuted = observation.permute((2, 0, 1))
        np.testing.assert_array_equal(
            permuted.entity_features,
            observation.entity_features[[2, 0, 1]],
        )
        expected_pairs = observation.pair_features[[2, 0, 1]][:, [2, 0, 1]]
        np.testing.assert_array_equal(permuted.pair_features, expected_pairs)
        self.assertEqual(observation.entity_count, permuted.entity_count)

    def test_model_facing_arrays_do_not_expose_sample_local_keys(self) -> None:
        observation = self._observation()
        self.assertEqual(observation.entity_features.shape, (3, 4))
        self.assertEqual(observation.pair_features.shape, (3, 3, 2))
        self.assertFalse(hasattr(observation.entity_features, "key"))

    def test_noise_is_deterministic_and_preserves_structure(self) -> None:
        observation = self._observation()
        first = observation.with_gaussian_noise(0.10, seed=7)
        second = observation.with_gaussian_noise(0.10, seed=7)
        np.testing.assert_array_equal(first.entity_features, second.entity_features)
        np.testing.assert_array_equal(first.pair_features, second.pair_features)
        self.assertEqual(first.entity_count, observation.entity_count)

    def test_held_out_cardinalities_are_supported(self) -> None:
        for count in (2, 4):
            entities = np.zeros((count, 4), dtype=np.float64)
            pairs = np.zeros((count, count, 2), dtype=np.float64)
            observation = observation_from_arrays("held_out_entity_count_learned_routing", entities, pairs)
            self.assertEqual(observation.entity_count, count)

    def test_invalid_permutation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._observation().permute((0, 0, 1))


if __name__ == "__main__":
    unittest.main()
