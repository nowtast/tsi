import json
from pathlib import Path

import numpy as np

from tsi.paper3_multiworld import LAYER_ORDER
from tsi.paper3_validity_contract import (
    PRIMARY_PREDICTIVE_MODELS,
    TASKS_PER_UNIT,
)
from tsi.paper3_validity_predictor import (
    _hazard_probabilities,
    _hazard_training_data,
    fit_ridge_logistic,
)


FIXED_LAYER_ORDER = ("label", "simplicial", "metric", "relation", "order")
TASK_TO_FIXED = {
    "label": "label",
    "topology": "simplicial",
    "metric": "metric",
    "relation": "relation",
    "order": "order",
}
BASE_NAMES = (
    "training_final_nll",
    "probe_teacher_one_step_mse",
    "probe_open_loop_latent_mse",
    "probe_terminal_exactness",
    "stratum_unseen_structural_mode",
    *(
        f"model_is_{model}"
        for model in PRIMARY_PREDICTIVE_MODELS[1:]
    ),
    "local_normalized_plan_horizon",
    *(f"local_goal_weight_{layer}" for layer in LAYER_ORDER),
    "local_predicted_utility_gap",
)
LAYER_ADDITIONS = (
    *(f"probe_raw_layer_mismatch_{layer}" for layer in LAYER_ORDER),
    "probe_goal_aligned_raw_mismatch",
)
TSI_ADDITIONS = (
    "probe_i0_correspondence_auc",
    "probe_fixed_total_auc",
    *(f"probe_fixed_{layer}" for layer in FIXED_LAYER_ORDER),
    "probe_goal_aligned_fixed_auc",
)
LAYER_NAMES = (*BASE_NAMES, *LAYER_ADDITIONS)
TSI_NAMES = (*BASE_NAMES, *TSI_ADDITIONS)
LAYER_TSI_NAMES = (*LAYER_NAMES, *TSI_ADDITIONS)


def rows_from_raw(raw):
    rows = []
    for run in raw["runs"]:
        if run["status"] != "completed":
            continue
        model = run["model"]
        if model not in PRIMARY_PREDICTIVE_MODELS:
            continue
        training_nll = float(run["training"]["final_nll"])
        for record in run["unit_records"]:
            generic = record["generic_predictors"]
            tsi = record["tsi_predictors"]
            raw_layers = generic["probe_open_loop_layer_mismatch_rate"]
            fixed_layers = tsi["probe_fixed_layer_auc"]
            base_prefix = [
                training_nll,
                float(generic["probe_teacher_one_step_mse"]),
                float(generic["probe_open_loop_latent_mse"]),
                float(generic["probe_terminal_exactness"]),
                float(record["stratum"] == "unseen_structural_mode"),
                *[
                    float(model == candidate)
                    for candidate in PRIMARY_PREDICTIVE_MODELS[1:]
                ],
            ]
            base_panel = []
            layer_panel = []
            tsi_panel = []
            layer_tsi_panel = []
            failures = []
            for task in record["task_records"]:
                local = task["generic_task_predictors"]
                weights = local["goal_layer_weights"]
                base = [
                    *base_prefix,
                    float(local["normalized_plan_horizon"]),
                    *[float(weights[layer]) for layer in LAYER_ORDER],
                    float(local["predicted_utility_gap"]),
                ]
                aligned_raw = sum(
                    float(weights[layer]) * float(raw_layers[layer])
                    for layer in LAYER_ORDER
                )
                layer = [
                    *base,
                    *[float(raw_layers[name]) for name in LAYER_ORDER],
                    aligned_raw,
                ]
                aligned_fixed = sum(
                    float(weights[layer])
                    * float(fixed_layers[TASK_TO_FIXED[layer]])
                    for layer in LAYER_ORDER
                )
                tsi_values = [
                    float(tsi["probe_i0_correspondence_auc"]),
                    float(tsi["probe_fixed_total_auc"]),
                    *[float(fixed_layers[name]) for name in FIXED_LAYER_ORDER],
                    aligned_fixed,
                ]
                base_panel.append(base)
                layer_panel.append(layer)
                tsi_panel.append([*base, *tsi_values])
                layer_tsi_panel.append([*layer, *tsi_values])
                failures.append(int(task["task_failure"]))
            rows.append(
                {
                    "world_index": int(run["world_index"]),
                    "optimizer_seed": int(run["optimizer_seed"]),
                    "model": model,
                    "base": base_panel,
                    "layer": layer_panel,
                    "tsi": tsi_panel,
                    "layer_tsi": layer_tsi_panel,
                    "task_failures": failures,
                    "first_failure_time": int(
                        record["outcomes"]["first_failure_time"]
                    ),
                }
            )
    return rows


def fit(rows, field, names):
    x, y, expanded_names = _hazard_training_data(rows, field, names)
    return fit_ridge_logistic(x, y, expanded_names)


def score(rows, baseline_model, tsi_model, baseline_field, baseline_names,
          tsi_field, tsi_names):
    baseline_hazards = _hazard_probabilities(
        baseline_model, rows, baseline_field, baseline_names
    )
    tsi_hazards = _hazard_probabilities(
        tsi_model, rows, tsi_field, tsi_names
    )
    baseline_failure = 1.0 - np.cumprod(1.0 - baseline_hazards, axis=1)
    tsi_failure = 1.0 - np.cumprod(1.0 - tsi_hazards, axis=1)
    effects = []
    for index, row in enumerate(rows):
        first = row["first_failure_time"]
        observed = np.asarray(
            [float(first <= time) for time in range(1, TASKS_PER_UNIT + 1)]
        )
        base_brier = (observed - baseline_failure[index]) ** 2
        tsi_brier = (observed - tsi_failure[index]) ** 2
        effects.append(
            (
                base_brier[-1] - tsi_brier[-1],
                np.mean(base_brier - tsi_brier),
            )
        )
    return np.asarray(effects)


def lowo(rows, baseline_field, baseline_names, tsi_field, tsi_names):
    output = []
    for world in sorted({row["world_index"] for row in rows}):
        training = [row for row in rows if row["world_index"] != world]
        testing = [row for row in rows if row["world_index"] == world]
        effects = score(
            testing,
            fit(training, baseline_field, baseline_names),
            fit(training, tsi_field, tsi_names),
            baseline_field,
            baseline_names,
            tsi_field,
            tsi_names,
        )
        for seed in sorted({row["optimizer_seed"] for row in testing}):
            indices = [
                index
                for index, row in enumerate(testing)
                if row["optimizer_seed"] == seed
            ]
            output.append(np.mean(effects[indices], axis=0))
    return np.asarray(output)


raw = json.loads(
    Path(
        "experiments/paper3_validity_v2/development_validity_results.json"
    ).read_text()
)
rows = rows_from_raw(raw)
scalar = lowo(rows, "base", BASE_NAMES, "tsi", TSI_NAMES)
layer = lowo(
    rows,
    "layer",
    LAYER_NAMES,
    "layer_tsi",
    LAYER_TSI_NAMES,
)
print(
    json.dumps(
        {
            "row_count": len(rows),
            "scalar_generic_vs_prefix_tsi_mean": np.mean(
                scalar, axis=0
            ).tolist(),
            "scalar_generic_vs_prefix_tsi_world_seed_sd": np.std(
                scalar, axis=0, ddof=1
            ).tolist(),
            "layer_generic_vs_prefix_tsi_mean": np.mean(
                layer, axis=0
            ).tolist(),
            "layer_generic_vs_prefix_tsi_world_seed_sd": np.std(
                layer, axis=0, ddof=1
            ).tolist(),
        },
        indent=2,
        sort_keys=True,
    )
)
