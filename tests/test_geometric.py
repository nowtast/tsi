import math
import unittest

from tsi import (
    FiniteMetricState,
    find_label_preserving_isometry,
    geometric_discrepancy,
)


def two_point_state(name: str, distance: float) -> FiniteMetricState:
    return FiniteMetricState(
        entities=(f"{name}0", f"{name}1"),
        distances=((0.0, distance), (distance, 0.0)),
        labels=("left", "right"),
    )


class FiniteMetricStateTest(unittest.TestCase):
    def test_rejects_triangle_inequality_violation(self) -> None:
        with self.assertRaisesRegex(ValueError, "triangle inequality"):
            FiniteMetricState(
                entities=("a", "b", "c"),
                distances=((0, 1, 3), (1, 0, 1), (3, 1, 0)),
                labels=("x", "x", "x"),
            )

    def test_diameter(self) -> None:
        self.assertEqual(two_point_state("x", 3.5).diameter, 3.5)


class GeometricDiscrepancyTest(unittest.TestCase):
    def test_zero_exactly_detects_label_preserving_isometry(self) -> None:
        left = FiniteMetricState(
            entities=("a", "b", "c"),
            distances=((0, 1, 2), (1, 0, 1), (2, 1, 0)),
            labels=("end", "middle", "end"),
        )
        right = FiniteMetricState(
            entities=("v", "u", "w"),
            distances=((0, 1, 1), (1, 0, 2), (1, 2, 0)),
            labels=("middle", "end", "end"),
        )

        self.assertEqual(geometric_discrepancy(left, right), 0.0)
        self.assertIsNotNone(find_label_preserving_isometry(left, right))

    def test_incompatible_labels_give_infinity(self) -> None:
        left = FiniteMetricState(("a",), ((0,),), ("red",))
        right = FiniteMetricState(("b",), ((0,),), ("blue",))

        self.assertTrue(math.isinf(geometric_discrepancy(left, right)))

    def test_triangle_inequality_and_diameter_bound(self) -> None:
        first = two_point_state("a", 1.0)
        second = two_point_state("b", 2.0)
        third = two_point_state("c", 4.0)

        d12 = geometric_discrepancy(first, second)
        d23 = geometric_discrepancy(second, third)
        d13 = geometric_discrepancy(first, third)

        self.assertEqual(d12, 1.0)
        self.assertEqual(d23, 2.0)
        self.assertEqual(d13, 3.0)
        self.assertLessEqual(d13, d12 + d23)
        self.assertLessEqual(abs(first.diameter - third.diameter), d13)

    def test_equal_diameter_does_not_imply_zero_discrepancy(self) -> None:
        line = FiniteMetricState(
            entities=("a", "b", "c"),
            distances=((0, 1, 2), (1, 0, 1), (2, 1, 0)),
            labels=("x", "x", "x"),
        )
        equilateral = FiniteMetricState(
            entities=("u", "v", "w"),
            distances=((0, 2, 2), (2, 0, 2), (2, 2, 0)),
            labels=("x", "x", "x"),
        )

        self.assertEqual(line.diameter, equilateral.diameter)
        self.assertGreater(geometric_discrepancy(line, equilateral), 0.0)


if __name__ == "__main__":
    unittest.main()
