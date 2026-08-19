import unittest

import numpy as np

from tsi.paper3_development_experiment import ConstructiveMetricCache
from tsi.paper3_validity_evaluator import evaluate_validity_units
from tsi.paper3_validity_generator import (
    development_validity_units,
    development_validity_worlds,
)


class _OracleBasis:
    def transform_cases(self, cases):
        return np.zeros((len(cases), 1), dtype=np.float64), np.zeros(
            (len(cases), 5),
            dtype=np.int64,
        )


class _OracleModel:
    def __init__(self) -> None:
        self.basis = _OracleBasis()

    def predict_codes_precomputed(self, cases, _features):
        return tuple(case.target_code for case in cases)


class Paper3ValidityEvaluatorTests(unittest.TestCase):
    def test_oracle_predictions_have_zero_probe_error_and_no_task_failure(self) -> None:
        world = development_validity_worlds()[0]
        units = development_validity_units(world)[:2]
        records = evaluate_validity_units(
            ConstructiveMetricCache(),
            _OracleModel(),
            world,
            units,
        )
        self.assertEqual(len(records), 2)
        for record in records:
            generic = record["generic_predictors"]
            tsi = record["tsi_predictors"]
            outcomes = record["outcomes"]
            self.assertEqual(generic["probe_teacher_one_step_mse"], 0.0)
            self.assertEqual(generic["probe_open_loop_latent_mse"], 0.0)
            self.assertEqual(generic["probe_terminal_exactness"], 1.0)
            self.assertEqual(tsi["probe_i0_correspondence_auc"], 0.0)
            self.assertEqual(tsi["probe_fixed_total_auc"], 0.0)
            self.assertEqual(outcomes["any_task_failure"], 0)
            self.assertEqual(outcomes["first_failure_time"], 9)
            self.assertFalse(record["outcome_uses_tsi_metric"])


if __name__ == "__main__":
    unittest.main()
