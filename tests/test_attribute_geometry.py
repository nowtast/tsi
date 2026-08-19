from __future__ import annotations

import itertools
import unittest

from tsi.attribute_geometry import (
    FiniteAttributedMetricMeasureState,
    FiniteAttributedMetricState,
    all_correspondences,
    attribute_aware_discrepancy,
    attribute_correspondence_audit,
    combined_power_error_bound,
    compose_correspondences,
    coupling_perturbation_audit,
    empirical_mass,
    find_attribute_preserving_isometry,
    finite_support_tv_radius,
    fused_coupling_audit,
    fused_sampling_bound,
    optimal_attribute_correspondence,
    total_variation,
    validate_attribute_metric,
    zero_fused_coupling_isometry,
)


def discrete_attribute_distance(left: int, right: int) -> float:
    return float(left != right)


def line_attribute_distance(left: int, right: int) -> float:
    return float(abs(left - right))


def state(
    name: str,
    *,
    distance: float = 1.0,
    labels: tuple[str, str] = ("a", "b"),
    attributes: tuple[int, int] = (0, 1),
) -> FiniteAttributedMetricState:
    return FiniteAttributedMetricState(
        (f"{name}0", f"{name}1"),
        ((0.0, distance), (distance, 0.0)),
        labels,
        attributes,
    )


class AttributeCorrespondenceTest(unittest.TestCase):
    def test_attribute_metric_validation_rejects_a_pseudometric(self) -> None:
        sample = state("x")

        with self.assertRaisesRegex(ValueError, "separate"):
            validate_attribute_metric((sample,), lambda _left, _right: 0.0)

    def test_minimum_is_attained_and_reports_components(self) -> None:
        left = state("x")
        right = state("y", labels=("b", "a"), attributes=(1, 0))

        audit = optimal_attribute_correspondence(
            left,
            right,
            line_attribute_distance,
        )

        self.assertEqual(audit.discrepancy, 0.0)
        self.assertEqual(audit.metric_distortion, 0.0)
        self.assertEqual(audit.label_distortion, 0.0)
        self.assertEqual(audit.attribute_distortion, 0.0)
        self.assertEqual(audit.correspondence, frozenset({(0, 1), (1, 0)}))

    def test_zero_exactness_matches_attribute_preserving_isometry_exhaustively(self) -> None:
        fixtures = [
            state(
                f"s{index}",
                distance=distance,
                labels=labels,
                attributes=attributes,
            )
            for index, (distance, labels, attributes) in enumerate(
                itertools.product(
                    (1.0, 2.0),
                    (("a", "a"), ("a", "b"), ("b", "a")),
                    ((0, 0), (0, 1), (1, 0)),
                )
            )
        ]
        checked = 0
        for left in fixtures:
            for right in fixtures:
                discrepancy = attribute_aware_discrepancy(
                    left,
                    right,
                    discrete_attribute_distance,
                )
                isometry = find_attribute_preserving_isometry(
                    left,
                    right,
                    discrete_attribute_distance,
                )
                self.assertEqual(discrepancy == 0.0, isometry is not None)
                checked += 1
        self.assertEqual(checked, 324)

    def test_unequal_carriers_cannot_have_zero_discrepancy(self) -> None:
        left = FiniteAttributedMetricState(("x",), ((0.0,),), ("a",), (0,))
        right = state("y", labels=("a", "a"), attributes=(0, 0))

        audit = optimal_attribute_correspondence(
            left,
            right,
            discrete_attribute_distance,
        )

        self.assertGreater(audit.metric_distortion, 0.0)
        self.assertGreater(audit.discrepancy, 0.0)
        self.assertIsNone(
            find_attribute_preserving_isometry(
                left,
                right,
                discrete_attribute_distance,
            )
        )

    def test_composition_witnesses_triangle_inequality_on_512_triples(self) -> None:
        fixtures = [
            state(
                f"s{index}",
                distance=distance,
                labels=labels,
                attributes=attributes,
            )
            for index, (distance, labels, attributes) in enumerate(
                itertools.product(
                    (1.0, 2.0),
                    (("a", "b"), ("b", "a")),
                    ((0, 1), (1, 0)),
                )
            )
        ]
        checked = 0
        for left, middle, right in itertools.product(fixtures, repeat=3):
            first = optimal_attribute_correspondence(
                left,
                middle,
                discrete_attribute_distance,
            )
            second = optimal_attribute_correspondence(
                middle,
                right,
                discrete_attribute_distance,
            )
            composite = compose_correspondences(
                first.correspondence,
                second.correspondence,
            )
            composite_audit = attribute_correspondence_audit(
                left,
                right,
                composite,
                discrete_attribute_distance,
            )
            direct = attribute_aware_discrepancy(
                left,
                right,
                discrete_attribute_distance,
            )
            self.assertLessEqual(
                direct,
                first.discrepancy + second.discrepancy + 1e-9,
            )
            self.assertLessEqual(
                composite_audit.discrepancy,
                first.discrepancy + second.discrepancy + 1e-9,
            )
            checked += 1
        self.assertEqual(checked, 512)

    def test_correspondence_enumerator_covers_both_carriers(self) -> None:
        correspondences = tuple(all_correspondences(state("x"), state("y")))

        self.assertEqual(len(correspondences), 7)
        for correspondence in correspondences:
            self.assertEqual({i for i, _ in correspondence}, {0, 1})
            self.assertEqual({j for _, j in correspondence}, {0, 1})


class FusedMetricMeasureTest(unittest.TestCase):
    def test_zero_coupling_recovers_measure_preserving_attributed_isometry(self) -> None:
        left = FiniteAttributedMetricMeasureState(state("x"), (0.25, 0.75))
        right = FiniteAttributedMetricMeasureState(
            state("y", labels=("b", "a"), attributes=(1, 0)),
            (0.75, 0.25),
        )
        coupling = ((0.0, 0.25), (0.75, 0.0))

        audit = fused_coupling_audit(
            left,
            right,
            coupling,
            line_attribute_distance,
            p=2.0,
        )

        self.assertEqual(audit.power_value, 0.0)
        self.assertEqual(
            zero_fused_coupling_isometry(
                left,
                right,
                coupling,
                line_attribute_distance,
                p=2.0,
            ),
            {"x0": "y1", "x1": "y0"},
        )

    def test_attribute_or_label_mismatch_has_positive_fused_cost(self) -> None:
        left = FiniteAttributedMetricMeasureState(state("x"), (0.5, 0.5))
        right = FiniteAttributedMetricMeasureState(
            state("y", labels=("a", "wrong"), attributes=(0, 2)),
            (0.5, 0.5),
        )

        audit = fused_coupling_audit(
            left,
            right,
            ((0.5, 0.0), (0.0, 0.5)),
            line_attribute_distance,
        )

        self.assertEqual(audit.structural_power, 0.0)
        self.assertGreater(audit.label_power, 0.0)
        self.assertGreater(audit.attribute_power, 0.0)
        self.assertIsNone(
            zero_fused_coupling_isometry(
                left,
                right,
                ((0.5, 0.0), (0.0, 0.5)),
                line_attribute_distance,
            )
        )

    def test_full_support_is_necessary_for_zero_exactness(self) -> None:
        left_state = FiniteAttributedMetricState(("x",), ((0.0,),), ("a",), (0,))
        right_state = state("y", labels=("a", "a"), attributes=(0, 0))
        left = FiniteAttributedMetricMeasureState(left_state, (1.0,))
        right = FiniteAttributedMetricMeasureState(right_state, (1.0, 0.0))
        coupling = ((1.0, 0.0),)

        self.assertEqual(
            fused_coupling_audit(
                left,
                right,
                coupling,
                discrete_attribute_distance,
            ).power_value,
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "full support"):
            zero_fused_coupling_isometry(
                left,
                right,
                coupling,
                discrete_attribute_distance,
            )

    def test_coupling_perturbation_witness_satisfies_both_bounds(self) -> None:
        left = FiniteAttributedMetricMeasureState(state("x"), (0.6, 0.4))
        right = FiniteAttributedMetricMeasureState(
            state("y", labels=("b", "a"), attributes=(1, 0)),
            (0.3, 0.7),
        )
        audit = coupling_perturbation_audit(
            left,
            right,
            ((0.0, 0.6), (0.3, 0.1)),
            (0.45, 0.55),
            (0.5, 0.5),
            line_attribute_distance,
            p=2.0,
        )

        self.assertLessEqual(
            audit.coupling_tv,
            audit.source_tv + audit.target_tv + 1e-9,
        )
        self.assertLessEqual(
            audit.objective_difference,
            audit.objective_bound + 1e-9,
        )


class StatisticalEstimatorTest(unittest.TestCase):
    def test_empirical_mass_retains_unobserved_declared_cells(self) -> None:
        sample_state = state("x")

        self.assertEqual(
            empirical_mass(sample_state, ("x0", "x0", "x0")),
            (1.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            empirical_mass(sample_state, ("missing",))

    def test_two_sample_bound_has_the_stated_hoeffding_radii(self) -> None:
        left = state("x")
        right = state("y")
        bound = fused_sampling_bound(
            left,
            right,
            line_attribute_distance,
            source_sample_size=200,
            target_sample_size=300,
            failure_probability=0.05,
        )

        self.assertAlmostEqual(
            bound.source_tv_radius,
            finite_support_tv_radius(2, 200, 0.025),
        )
        self.assertAlmostEqual(
            bound.target_tv_radius,
            finite_support_tv_radius(2, 300, 0.025),
        )
        self.assertEqual(bound.confidence, 0.95)
        self.assertAlmostEqual(
            bound.statistical_power_error,
            bound.coupling_lipschitz_constant
            * (bound.source_tv_radius + bound.target_tv_radius),
        )

    def test_deterministic_empirical_sequence_converges_in_total_variation(self) -> None:
        population = (0.3, 0.7)
        errors = [
            total_variation(
                empirical_mass(
                    state("x"),
                    tuple(["x0"] * (3 * scale) + ["x1"] * (7 * scale)),
                ),
                population,
            )
            for scale in (1, 2, 5, 10)
        ]

        self.assertEqual(errors, [0.0, 0.0, 0.0, 0.0])

    def test_statistical_and_optimization_errors_remain_separate(self) -> None:
        self.assertEqual(combined_power_error_bound(0.12, 0.03), 0.15)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            combined_power_error_bound(0.1, -0.01)

    def test_missing_labels_are_not_identifiable_from_unlabeled_observations(self) -> None:
        red = FiniteAttributedMetricState(("x",), ((0.0,),), ("red",), (0,))
        blue = FiniteAttributedMetricState(("y",), ((0.0,),), ("blue",), (0,))

        self.assertEqual(
            empirical_mass(red, ("x", "x", "x")),
            empirical_mass(blue, ("y", "y", "y")),
        )
        self.assertGreater(
            attribute_aware_discrepancy(
                red,
                blue,
                discrete_attribute_distance,
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

