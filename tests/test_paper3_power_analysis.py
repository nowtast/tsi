from __future__ import annotations

import unittest

import numpy as np

from tsi.paper3_power_analysis import (
    planning_covariance,
    simulate_holm_power,
)


class PowerAnalysisTest(unittest.TestCase):
    def test_planning_sd_has_frozen_floor(self) -> None:
        effects = np.asarray(
            [
                [0.05, 0.18, 0.19],
                [0.05, 0.20, 0.21],
                [0.05, 0.22, 0.23],
            ],
            dtype=np.float64,
        )
        observed, planning, covariance = planning_covariance(effects)

        self.assertEqual(observed.shape, (3,))
        self.assertTrue(np.all(planning >= 0.10))
        self.assertEqual(covariance.shape, (3, 3))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))

    def test_holm_power_simulation_is_deterministic(self) -> None:
        covariance = np.eye(3, dtype=np.float64) * 0.01
        first = simulate_holm_power(
            covariance,
            iterations=500,
            minimum_worlds=36,
            maximum_worlds=40,
        )
        second = simulate_holm_power(
            covariance,
            iterations=500,
            minimum_worlds=36,
            maximum_worlds=40,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["power_curve"]), 5)


if __name__ == "__main__":
    unittest.main()
