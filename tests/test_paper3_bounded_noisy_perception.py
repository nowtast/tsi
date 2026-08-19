import unittest

from tsi.paper3_bounded_noisy_perception import (
    evaluate_bounded_noise,
    majority_decode,
    noisy_views,
)
from tsi.paper3_replication_family import build_replication_dataset


class BoundedNoisyPerceptionTests(unittest.TestCase):
    def test_majority_recovers_one_corrupted_view(self) -> None:
        case = build_replication_dataset("metric_to_relation", 7).partitions["train"][0]
        decoded = majority_decode(noisy_views(case))
        self.assertEqual(decoded.source, case.source)
        self.assertEqual(decoded.target, case.target)

    def test_bounded_noise_preserves_exact_factorization(self) -> None:
        result = evaluate_bounded_noise(
            build_replication_dataset("metric_to_relation", 7)
        )
        self.assertEqual(result["exact_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
