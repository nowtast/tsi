from __future__ import annotations

import unittest

from tsi.topological import (
    betti_numbers,
    bottleneck_distance,
    filtration_interleaving_audit,
    sublevel_complex,
    validate_filtration,
    zero_dimensional_barcode,
)


def graph_complex(
    vertices: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    *,
    fill: bool = False,
):
    simplices = {
        frozenset(),
        *(frozenset((vertex,)) for vertex in vertices),
        *(frozenset(edge) for edge in edges),
    }
    if fill:
        simplices.add(frozenset(vertices))
    return frozenset(simplices)


class FiniteHomologyTest(unittest.TestCase):
    def test_triangle_boundary_and_filling_have_expected_betti_numbers(self) -> None:
        boundary = graph_complex((0, 1, 2), ((0, 1), (1, 2), (0, 2)))
        filled = graph_complex(
            (0, 1, 2),
            ((0, 1), (1, 2), (0, 2)),
            fill=True,
        )

        self.assertEqual(betti_numbers(boundary, max_dimension=2), (1, 1, 0))
        self.assertEqual(betti_numbers(filled, max_dimension=2), (1, 0, 0))

    def test_equal_betti_vectors_do_not_identify_a_complex(self) -> None:
        triangle = graph_complex((0, 1, 2), ((0, 1), (1, 2), (0, 2)))
        square = graph_complex(
            (0, 1, 2, 3),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
        )

        self.assertEqual(
            betti_numbers(triangle, max_dimension=1),
            betti_numbers(square, max_dimension=1),
        )
        self.assertNotEqual(len(triangle), len(square))


class FinitePersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.complex = graph_complex(
            (0, 1, 2),
            ((0, 1), (1, 2), (0, 2)),
            fill=True,
        )
        self.left = {
            simplex: (
                -1.0
                if not simplex
                else 0.0
                if len(simplex) == 1
                else 1.0
                if len(simplex) == 2
                else 2.0
            )
            for simplex in self.complex
        }
        self.right = {
            simplex: value + (0.2 if simplex else 0.0)
            for simplex, value in self.left.items()
        }

    def test_sublevel_complex_and_monotonicity_validation(self) -> None:
        sublevel = sublevel_complex(self.complex, self.left, 0.5)
        self.assertEqual(
            sublevel,
            frozenset(
                {
                    frozenset(),
                    frozenset((0,)),
                    frozenset((1,)),
                    frozenset((2,)),
                }
            ),
        )
        invalid = dict(self.left)
        invalid[frozenset((0, 1))] = -2.0
        with self.assertRaisesRegex(ValueError, "monotone"):
            validate_filtration(self.complex, invalid)

    def test_h0_algebraic_stability_on_a_common_finite_complex(self) -> None:
        delta, inclusions_hold = filtration_interleaving_audit(
            self.complex,
            self.left,
            self.right,
        )
        left_barcode = zero_dimensional_barcode(self.complex, self.left)
        right_barcode = zero_dimensional_barcode(self.complex, self.right)
        distance = bottleneck_distance(left_barcode, right_barcode)

        self.assertTrue(inclusions_hold)
        self.assertAlmostEqual(delta, 0.2)
        self.assertLessEqual(distance, delta + 1e-9)


if __name__ == "__main__":
    unittest.main()
