import unittest

import numpy as np

from tsi.paper34_resolution_benchmark import (
    GRAPH_MANIFEST,
    deterministic_successor,
    generate_cases,
    learn_factorized,
    run_world,
    world_spec,
)
from tsi.paper34_resolution_analysis import summarize_cohort
from tsi.paper34_resolution_contract import audit_contract


class Paper34ResolutionTests(unittest.TestCase):
    def test_contract_and_graph_manifest(self) -> None:
        self.assertTrue(audit_contract()["passed"])
        self.assertEqual(len(GRAPH_MANIFEST), 30)

    def test_composition_cases_activate_two_coordinates(self) -> None:
        rng = np.random.default_rng(17)
        spec = world_spec(4, rng)
        cases = generate_cases(spec, 30, rng, composition=True, noise_probability=0.0)
        self.assertTrue(all(sum(value != 0 for value in case.action) == 2 for case in cases))
        self.assertEqual({case.composition_stratum for case in cases}, {
            "both_true_mechanisms", "one_true_mechanism", "distractor_composition"
        })

    def test_train_only_search_recovers_noise_free_world(self) -> None:
        rng = np.random.default_rng(23)
        spec = world_spec(7, rng)
        train = generate_cases(spec, 700, rng, composition=False, noise_probability=0.0)
        selection = generate_cases(spec, 350, rng, composition=False, noise_probability=0.0)
        learned = learn_factorized(train, selection)
        self.assertEqual(learned.graph, spec.graph)
        self.assertEqual(learned.families, spec.families)

    def test_stochastic_world_runs_with_complete_factorial(self) -> None:
        result = run_world(2, 9917)
        metrics = result["metrics"]
        self.assertEqual(metrics["correct_graph_correct_head"]["active_parameter_count"], 7)
        self.assertEqual(metrics["correct_graph_generic_7"]["active_parameter_count"], 7)
        self.assertEqual(metrics["correct_graph_generic_55"]["active_parameter_count"], 55)
        self.assertEqual(len(result["rollouts"]), 160)

    def test_analysis_reports_all_resolution_gates(self) -> None:
        rows = [run_world(index, 1200 + index) for index in range(3)]
        report = summarize_cohort(rows)
        self.assertEqual(len(report["gates"]), 9)
        self.assertEqual(report["world_count"], 3)


if __name__ == "__main__":
    unittest.main()
