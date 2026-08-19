import unittest

from tsi.paper3_learned_development import run_learned_development_pilot


class LearnedDevelopmentPilotTests(unittest.TestCase):
    def test_one_world_pilot_has_complete_seed_panel(self) -> None:
        report = run_learned_development_pilot(
            worlds=1,
            optimizer_seeds=(0,),
            updates=10,
        )
        self.assertEqual(report["run_count"], 1)
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(report["world_count_with_complete_seed_panel"], 1)
        self.assertFalse(report["oracle_graph_used_for_inference"])
        self.assertTrue(report["oracle_graph_used_for_external_audit"])


if __name__ == "__main__":
    unittest.main()
