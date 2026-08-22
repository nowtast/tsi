import unittest

from tsi.research_a_analysis import analyze_confirmatory_rows
from tsi.research_a_contract import PRIMARY_SAMPLE_SIZES


class ResearchAAnalysisTests(unittest.TestCase):
    def test_transition_band_requires_advantage_then_equivalence(self) -> None:
        rows = []
        for world in range(20):
            estimates = []
            for size in PRIMARY_SAMPLE_SIZES:
                advantage = size <= 20
                estimates.append(
                    {
                        "sample_size": size,
                        "generic_minus_typed_nll": 0.2 if advantage else 0.0,
                        "typed_minus_generic_exact": 1.0 if advantage else 0.0,
                        "typed_minus_isomorphic_nll": 0.0,
                        "typed_exact": True,
                        "generic_exact": not advantage,
                    }
                )
            rows.append({"world_index": world, "estimates": estimates})
        analysis = analyze_confirmatory_rows(rows)
        self.assertTrue(analysis["a1_supported"])
        self.assertEqual(
            analysis["transition_band"],
            {"last_joint_advantage_n": 20, "first_later_equivalence_n": 25},
        )


if __name__ == "__main__":
    unittest.main()
