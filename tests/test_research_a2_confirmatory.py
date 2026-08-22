import unittest

from tsi.research_a2_confirmatory import run_a2_cohort


class ResearchA2ConfirmatoryTests(unittest.TestCase):
    def test_small_seed_driven_cohort_balances_and_covers_all_axes(self) -> None:
        records, portable, audit = run_a2_cohort(
            bytes(range(32)),
            world_count=45,
            width_sample_sizes=(10,),
            noise_sample_sizes=(15,),
            noise_probabilities=(0.08,),
            scope_sample_size=20,
            test_case_count=12,
        )
        self.assertEqual(len(records["candidate_width"]), 45 * 3)
        self.assertEqual(len(records["training_noise"]), 45)
        self.assertEqual(len(records["misspecification"]), 45 * 3)
        self.assertEqual(
            portable["status"], "confirmatory_portable_replay_with_answers"
        )
        self.assertTrue(audit["candidate_width"]["family_pairs_balanced"])
        self.assertTrue(audit["scope_typed_misspecified"]["family_pairs_balanced"])
        self.assertTrue(audit["paired_scope_nonfamily_parameters_equal"])

    def test_seed_must_be_exactly_32_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "32 bytes"):
            run_a2_cohort(b"short")


if __name__ == "__main__":
    unittest.main()
