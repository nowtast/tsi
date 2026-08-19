import unittest

from tsi.paper34_noise_sensitivity import (
    build_noise_sensitivity,
    run_noise_world,
)


class Paper34NoiseSensitivityTests(unittest.TestCase):
    def test_noise_world_is_reproducible(self) -> None:
        first = run_noise_world(0, 0.04, 0.06)
        second = run_noise_world(0, 0.04, 0.06)
        self.assertEqual(first, second)

    def test_small_grid_summary_is_descriptive(self) -> None:
        rows = []
        for train_noise in (0.04, 0.08, 0.16):
            for ood_noise in (0.06, 0.12, 0.24):
                rows.extend(
                    run_noise_world(index, train_noise, ood_noise)
                    for index in range(2)
                )
        report = build_noise_sensitivity(rows)
        self.assertEqual(len(report["cells"]), 9)
        self.assertIn("not_confirmatory", report["status"])


if __name__ == "__main__":
    unittest.main()
