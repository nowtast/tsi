import unittest

from tsi.research_a_sample_complexity import (
    GENERIC_ADAPTIVE_COMPARISON_COUNT,
    STRUCTURED_CLASS_COUNT,
    audit_research_a_constants,
    generic_greedy_failure_bound,
    generic_population_improvement_margin,
    generic_sufficient_sample_size,
    qary_mode_gap,
    structured_erm_failure_bound,
    structured_erm_sufficient_sample_size,
    typed_recovery_failure_bound,
    typed_sufficient_sample_size,
)


class ResearchASampleComplexityTests(unittest.TestCase):
    def test_modulo_seven_mode_gap(self) -> None:
        self.assertAlmostEqual(qary_mode_gap(0.08), 1.0 - 7.0 * 0.08 / 6.0)
        self.assertGreater(qary_mode_gap(0.08), 0.0)
        with self.assertRaises(ValueError):
            qary_mode_gap(6.0 / 7.0)

    def test_population_margin_scales_with_mode_gap(self) -> None:
        self.assertAlmostEqual(
            generic_population_improvement_margin(0.08),
            4.0 * qary_mode_gap(0.08) / 35.0,
        )

    def test_failure_envelopes_decrease_with_sample_size(self) -> None:
        self.assertGreater(
            typed_recovery_failure_bound(100, 0.08),
            typed_recovery_failure_bound(1000, 0.08),
        )
        self.assertGreater(
            generic_greedy_failure_bound(1000, 0.08),
            generic_greedy_failure_bound(10000, 0.08),
        )

    def test_closed_form_thresholds_satisfy_requested_delta(self) -> None:
        delta = 0.05
        typed_n = typed_sufficient_sample_size(0.08, delta)
        generic_n = generic_sufficient_sample_size(0.08, delta)
        self.assertLessEqual(typed_recovery_failure_bound(typed_n, 0.08), delta)
        self.assertLessEqual(generic_greedy_failure_bound(generic_n, 0.08), delta)
        self.assertLess(typed_n, generic_n)

    def test_structured_erm_threshold_satisfies_requested_delta(self) -> None:
        delta = 0.05
        sample_size = structured_erm_sufficient_sample_size(0.08, delta)
        self.assertLessEqual(
            structured_erm_failure_bound(sample_size, 0.08), delta
        )
        self.assertLess(
            sample_size, generic_sufficient_sample_size(0.08, delta)
        )

    def test_adaptive_comparison_count_is_explicit(self) -> None:
        self.assertEqual(GENERIC_ADAPTIVE_COMPARISON_COUNT, 41910)
        self.assertEqual(STRUCTURED_CLASS_COUNT, 9 * 6**7)

    def test_finite_population_constants_are_exhaustively_audited(self) -> None:
        audit = audit_research_a_constants()
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(
            audit["minimum_scaled_head_disagreement"], 5.0 / 7.0
        )
        self.assertAlmostEqual(
            audit["maximum_false_useful_agreement"], 2.0 / 7.0
        )


if __name__ == "__main__":
    unittest.main()
