import unittest

from tsi.research_a_confirmatory import run_cohort


class ResearchAConfirmatoryTests(unittest.TestCase):
    def test_small_cohort_audits_balance_and_notation(self) -> None:
        rows, portable, audit = run_cohort(
            bytes(range(32)),
            world_count=9,
            sample_sizes=(5,),
            test_case_count=20,
        )
        self.assertEqual(len(rows), 9)
        self.assertEqual(portable["world_count"], 9)
        self.assertTrue(audit["family_pairs_balanced"])
        self.assertTrue(audit["notation_invariant_passed"])


if __name__ == "__main__":
    unittest.main()
