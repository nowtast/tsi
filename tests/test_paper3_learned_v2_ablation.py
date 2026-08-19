import unittest

from tsi.paper3_learned_v2_ablation import CONTROL_ORDER, run_v2_development_ablation


class LearnedV2AblationTests(unittest.TestCase):
    def test_smoke_runs_all_controls_on_disjoint_evaluation_splits(self) -> None:
        results = run_v2_development_ablation(worlds=1, seeds=(0,), updates=3)
        self.assertEqual(tuple(result.control for result in results), CONTROL_ORDER)
        self.assertTrue(all(result.finite for result in results))
        self.assertTrue(all(result.test_nll >= 0.0 for result in results))


if __name__ == "__main__":
    unittest.main()
