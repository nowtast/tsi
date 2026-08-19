import unittest

from tsi.paper4_misspecified_resolution import run_stress_world, summarize_stress


class Paper4MisspecifiedResolutionTests(unittest.TestCase):
    def test_synergy_is_unseen_and_breaks_exactness(self) -> None:
        row = run_stress_world(5, 88421)
        self.assertTrue(row["all_primitive_training_synergies_zero"])
        self.assertTrue(row["graph_exact"])
        self.assertTrue(row["head_exact"])
        self.assertLess(row["learned_center_accuracy"], 1.0)
        self.assertGreater(row["graph_nll_effect"], 0.0)

    def test_stress_summary_has_four_gates(self) -> None:
        rows = tuple(run_stress_world(index, 5000 + index) for index in range(3))
        self.assertEqual(len(summarize_stress(rows)["gates"]), 4)


if __name__ == "__main__":
    unittest.main()
