import unittest

from tsi.research_a_development import run_development


class ResearchADevelopmentTests(unittest.TestCase):
    def test_small_pilot_preserves_information_and_notation_invariants(self) -> None:
        report = run_development(
            world_count=9,
            sample_sizes=(50,),
            test_case_count=30,
        )
        self.assertEqual(report["status"], "development_only_not_confirmatory")
        self.assertFalse(report["test_used_for_fit_or_selection"])
        self.assertEqual(len(report["rows"]), 9)
        summary = report["summaries"][0]
        self.assertEqual(summary["maximum_absolute_notation_nll_difference"], 0.0)
        observed_pairs = {
            tuple(row["families"]) for row in report["rows"]
        }
        self.assertEqual(len(observed_pairs), 9)


if __name__ == "__main__":
    unittest.main()
