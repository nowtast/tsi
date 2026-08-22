import unittest

from tsi.research_a2_analysis import _efficiency_axis, analyze_a2_axes
from tsi.research_a2_contract import NOISE_ENDPOINTS
from tsi.research_a2_development import run_a2_development


class ResearchA2AnalysisTests(unittest.TestCase):
    def test_analysis_consumes_all_declared_axes_and_keeps_scope_separate(self) -> None:
        report = run_a2_development(
            matched_world_count=2,
            misspecification_world_count=2,
            test_case_count=12,
        )
        axes = {name: payload["records"] for name, payload in report["axes"].items()}
        analysis = analyze_a2_axes(axes)
        self.assertEqual(analysis["candidate_width"]["bonferroni_endpoint_count"], 36)
        self.assertEqual(analysis["training_noise"]["bonferroni_endpoint_count"], 48)
        self.assertEqual(
            analysis["misspecification_scope"]["bonferroni_endpoint_count"], 6
        )
        self.assertTrue(analysis["scope_cannot_change_efficiency_gate"])

    def test_noise_gate_requires_each_level_through_point_six(self) -> None:
        probabilities = (0.08, 0.3, 0.6, 0.8)
        sizes = (15,)

        def records(failing_probability: float | None) -> list[dict[str, object]]:
            rows = []
            for probability in probabilities:
                passes = probability != failing_probability and probability != 0.8
                for world in range(8):
                    rows.append(
                        {
                            "train_noise_probability": probability,
                            "sample_size": 15,
                            "generic_minus_typed_nll": 0.2 if passes else 0.0,
                            "typed_minus_generic_exact": 1.0 if passes else 0.0,
                            "typed_exact": passes,
                            "generic_exact": False,
                            "world_index": world,
                        }
                    )
            return rows

        boundary_only_failure = _efficiency_axis(
            records(None),
            group_key="train_noise_probability",
            group_values=probabilities,
            sample_sizes=sizes,
            endpoint_count=len(NOISE_ENDPOINTS),
            required_advantage_groups=(0.08, 0.3, 0.6),
        )
        self.assertTrue(boundary_only_failure["gate_passed"])
        self.assertEqual(boundary_only_failure["descriptive_only_groups"], [0.8])

        required_failure = _efficiency_axis(
            records(0.6),
            group_key="train_noise_probability",
            group_values=probabilities,
            sample_sizes=sizes,
            endpoint_count=len(NOISE_ENDPOINTS),
            required_advantage_groups=(0.08, 0.3, 0.6),
        )
        self.assertFalse(required_failure["gate_passed"])


if __name__ == "__main__":
    unittest.main()
