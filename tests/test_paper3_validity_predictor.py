import copy
import unittest

import numpy as np

from tsi.paper3_validity_contract import (
    PRIMARY_PREDICTIVE_MODELS,
    TASKS_PER_UNIT,
)
from tsi.paper3_validity_predictor import (
    LAYER_AWARE_FEATURE_NAMES,
    LAYER_AWARE_TSI_FEATURE_NAMES,
    SCALAR_GENERIC_FEATURE_NAMES,
    TSI_FEATURE_NAMES,
    _fit_model_set,
    fit_ridge_logistic,
    predict_ridge_logistic,
    score_frozen_predictors,
)


class Paper3ValidityPredictorTests(unittest.TestCase):
    def test_ridge_logistic_is_deterministic_and_digest_protected(self) -> None:
        x = np.linspace(-2.0, 2.0, 100)[:, np.newaxis]
        y = (x[:, 0] > 0.0).astype(np.float64)
        model = fit_ridge_logistic(x, y, ("x",))
        self.assertTrue(model["converged"])
        predictions = predict_ridge_logistic(model, x, ("x",))
        self.assertLess(float(predictions[0]), 0.5)
        self.assertGreater(float(predictions[-1]), 0.5)
        self.assertTrue(
            np.array_equal(
                predictions,
                predict_ridge_logistic(model, x, ("x",)),
            )
        )
        tampered = copy.deepcopy(model)
        tampered["coefficients"][0] += 0.1
        with self.assertRaises(ValueError):
            predict_ridge_logistic(tampered, x, ("x",))

    def test_fitting_rejects_single_class_targets(self) -> None:
        with self.assertRaises(ValueError):
            fit_ridge_logistic(
                np.ones((4, 1), dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                ("x",),
            )

    def test_discrete_hazard_scores_are_finite_and_monotone(self) -> None:
        rows = []
        for index in range(80):
            event = int(index % 4 in (2, 3))
            scalar_panel = []
            tsi_panel = []
            layer_panel = []
            layer_tsi_panel = []
            for time in range(TASKS_PER_UNIT):
                scalar = np.zeros(
                    len(SCALAR_GENERIC_FEATURE_NAMES),
                    dtype=np.float64,
                )
                scalar[0] = (index % 10) / 10.0
                scalar[-1] = time / TASKS_PER_UNIT
                tsi = np.zeros(len(TSI_FEATURE_NAMES), dtype=np.float64)
                tsi[: len(scalar)] = scalar
                tsi[-1] = float(event)
                layer = np.zeros(
                    len(LAYER_AWARE_FEATURE_NAMES),
                    dtype=np.float64,
                )
                layer[: len(scalar)] = scalar
                layer_tsi = np.zeros(
                    len(LAYER_AWARE_TSI_FEATURE_NAMES),
                    dtype=np.float64,
                )
                layer_tsi[: len(layer)] = layer
                layer_tsi[-1] = float(event)
                scalar_panel.append(scalar.tolist())
                tsi_panel.append(tsi.tolist())
                layer_panel.append(layer.tolist())
                layer_tsi_panel.append(layer_tsi.tolist())
            failures = [0] * TASKS_PER_UNIT
            if event:
                failures[2] = 1
            rows.append(
                {
                    "world_index": index // 8,
                    "model": PRIMARY_PREDICTIVE_MODELS[
                        index % len(PRIMARY_PREDICTIVE_MODELS)
                    ],
                    "optimizer_seed": index % 3,
                    "unit_index": index,
                    "is_primary_predictive_model": True,
                    "scalar_task_features": scalar_panel,
                    "tsi_task_features": tsi_panel,
                    "layer_task_features": layer_panel,
                    "layer_tsi_task_features": layer_tsi_panel,
                    "task_failures": failures,
                    "any_task_failure": event,
                    "first_failure_time": 3 if event else TASKS_PER_UNIT + 1,
                }
            )
        models = _fit_model_set(rows)
        scored = score_frozen_predictors(rows, models)
        self.assertEqual(
            set(models),
            {
                "hazard_baseline",
                "hazard_tsi",
                "hazard_layer_aware_baseline",
                "hazard_layer_aware_tsi",
            },
        )
        self.assertEqual(len(scored), len(rows))
        for row in scored:
            baseline = row["baseline_cumulative_failure_probability"]
            tsi = row["tsi_cumulative_failure_probability"]
            self.assertTrue(all(np.isfinite(baseline)))
            self.assertTrue(all(np.isfinite(tsi)))
            self.assertTrue(
                all(left <= right for left, right in zip(baseline, baseline[1:]))
            )
            self.assertTrue(
                all(left <= right for left, right in zip(tsi, tsi[1:]))
            )


if __name__ == "__main__":
    unittest.main()
