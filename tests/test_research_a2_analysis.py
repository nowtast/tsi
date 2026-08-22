import unittest

from tsi.research_a2_analysis import analyze_a2_axes
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


if __name__ == "__main__":
    unittest.main()
