from __future__ import annotations

import unittest

from tsi.geometric import FiniteMetricState
from tsi.geometric_validation import (
    ambient_alignment_error,
    apply_rigid_motion,
    coupling_distortion,
    is_orthogonal,
    is_special_orthogonal,
    pairwise_distances,
    signed_area_2d,
    zero_coupling_bijection,
)


class AmbientGeometryAuditTest(unittest.TestCase):
    def test_reflection_preserves_intrinsic_metric_but_reverses_orientation(self) -> None:
        source = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0)}
        reflection = ((-1.0, 0.0), (0.0, 1.0))
        target = apply_rigid_motion(source, reflection, (0.0, 0.0))

        self.assertTrue(is_orthogonal(reflection))
        self.assertFalse(is_special_orthogonal(reflection))
        self.assertEqual(pairwise_distances(source), pairwise_distances(target))
        self.assertAlmostEqual(
            signed_area_2d(source[1], source[2], source[3]),
            -signed_area_2d(target[1], target[2], target[3]),
        )
        self.assertEqual(
            ambient_alignment_error(source, target, reflection, (0.0, 0.0)),
            0.0,
        )

    def test_proper_rigid_motion_has_zero_alignment_error_and_preserves_area(self) -> None:
        source = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0)}
        quarter_turn = ((0.0, -1.0), (1.0, 0.0))
        target = apply_rigid_motion(source, quarter_turn, (2.0, -3.0))

        self.assertTrue(is_special_orthogonal(quarter_turn))
        self.assertEqual(
            ambient_alignment_error(source, target, quarter_turn, (2.0, -3.0)),
            0.0,
        )
        self.assertAlmostEqual(
            signed_area_2d(source[1], source[2], source[3]),
            signed_area_2d(target[1], target[2], target[3]),
        )


class MetricMeasureAuditTest(unittest.TestCase):
    def test_zero_graph_coupling_recovers_measure_preserving_isometry(self) -> None:
        left = FiniteMetricState(
            ("a", "b"),
            ((0.0, 2.0), (2.0, 0.0)),
            ("red", "blue"),
        )
        right = FiniteMetricState(
            ("v", "u"),
            ((0.0, 2.0), (2.0, 0.0)),
            ("blue", "red"),
        )
        coupling = ((0.0, 0.25), (0.75, 0.0))

        self.assertEqual(
            coupling_distortion(
                left,
                right,
                (0.25, 0.75),
                (0.75, 0.25),
                coupling,
            ),
            0.0,
        )
        self.assertEqual(
            zero_coupling_bijection(
                left,
                right,
                (0.25, 0.75),
                (0.75, 0.25),
                coupling,
            ),
            {"a": "u", "b": "v"},
        )

    def test_full_support_is_necessary_for_zero_exactness(self) -> None:
        left = FiniteMetricState(("a",), ((0.0,),), ("same",))
        right = FiniteMetricState(
            ("u", "ignored"),
            ((0.0, 7.0), (7.0, 0.0)),
            ("same", "same"),
        )
        coupling = ((1.0, 0.0),)

        self.assertEqual(
            coupling_distortion(
                left,
                right,
                (1.0,),
                (1.0, 0.0),
                coupling,
                full_support=False,
            ),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "full support"):
            coupling_distortion(
                left,
                right,
                (1.0,),
                (1.0, 0.0),
                coupling,
                full_support=True,
            )


if __name__ == "__main__":
    unittest.main()
