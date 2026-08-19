import json
from pathlib import Path

import numpy as np

from tsi.paper3_multiworld import LAYER_ORDER
from tsi.paper3_validity_contract import PRIMARY_PREDICTIVE_MODELS
from tsi.paper3_validity_predictor import (
    fit_ridge_logistic,
    predict_ridge_logistic,
)


HORIZON = 4
FIXED_LAYER_ORDER = ("label", "simplicial", "metric", "relation", "order")
TASK_TO_FIXED = {
    "label": "label",
    "topology": "simplicial",
    "metric": "metric",
    "relation": "relation",
    "order": "order",
}
CALIBRATION_SCALARS = (
    "candidate_latent_mse_mean",
    "candidate_latent_mse_max",
    "candidate_latent_mse_selected_minus_alternative",
    "candidate_endpoint_exact_rate",
    "candidate_endpoint_exact_selected_minus_alternative",
    "predicted_utility_gap",
)
CALIBRATION_TSI_SCALARS = (
    "candidate_i0_mean",
    "candidate_i0_max",
    "candidate_i0_selected_minus_alternative",
    "candidate_fixed_total_mean",
    "candidate_fixed_total_max",
    "candidate_fixed_total_selected_minus_alternative",
)
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
    *(f"calibration_{name}" for name in CALIBRATION_SCALARS),
    "calibration_failure_rate",
    "outcome_normalized_plan_horizon",
    *(f"outcome_goal_weight_{layer}" for layer in LAYER_ORDER),
    "outcome_predicted_utility_gap",
)
LAYER_ADDITIONS = (
    *(f"probe_raw_layer_{layer}" for layer in LAYER_ORDER),
    *(f"calibration_raw_layer_{layer}" for layer in LAYER_ORDER),
    *(
        f"calibration_raw_contrast_{layer}"
        for layer in LAYER_ORDER
    ),
    "outcome_goal_aligned_calibration_raw",
    "outcome_goal_aligned_calibration_raw_contrast",
)
TSI_ADDITIONS = (
    "probe_i0",
    "probe_fixed_total",
    *(f"probe_fixed_{layer}" for layer in FIXED_LAYER_ORDER),
    *(f"calibration_{name}" for name in CALIBRATION_TSI_SCALARS),
    *(f"calibration_fixed_{layer}" for layer in FIXED_LAYER_ORDER),
    *(
        f"calibration_fixed_contrast_{layer}"
        for layer in FIXED_LAYER_ORDER
    ),
    "outcome_goal_aligned_calibration_fixed",
    "outcome_goal_aligned_calibration_fixed_contrast",
)
LAYER_NAMES = (*BASE_NAMES, *LAYER_ADDITIONS)
TSI_NAMES = (*BASE_NAMES, *TSI_ADDITIONS)
LAYER_TSI_NAMES = (*LAYER_NAMES, *TSI_ADDITIONS)
TIME_NAMES = tuple(f"failure_round_{time}" for time in range(2, HORIZON + 1))


def mean(tasks, section, name):
    return float(np.mean([task[section][name] for task in tasks]))


def vector_mean(tasks, section, name, layer):
    return float(
        np.mean([task[section][name][layer] for task in tasks])
    )


def rows_from_raw(raw):
    rows = []
    for run in raw["runs"]:
        if (
            run["status"] != "completed"
            or run["model"] not in PRIMARY_PREDICTIVE_MODELS
        ):
            continue
        model = run["model"]
        for record in run["unit_records"]:
            generic = record["generic_predictors"]
            tsi = record["tsi_predictors"]
            calibration = record["task_records"][:4]
            outcome = record["task_records"][4:]
            calibration_raw = {
                layer: vector_mean(
                    calibration,
                    "generic_task_predictors",
                    "candidate_layer_mismatch_rate",
                    layer,
                )
                for layer in LAYER_ORDER
            }
            calibration_raw_contrast = {
                layer: vector_mean(
                    calibration,
                    "generic_task_predictors",
                    "candidate_layer_mismatch_selected_minus_alternative",
                    layer,
                )
                for layer in LAYER_ORDER
            }
            calibration_fixed = {
                layer: vector_mean(
                    calibration,
                    "tsi_task_predictors",
                    "candidate_fixed_layer_mean",
                    layer,
                )
                for layer in FIXED_LAYER_ORDER
            }
            calibration_fixed_contrast = {
                layer: vector_mean(
                    calibration,
                    "tsi_task_predictors",
                    "candidate_fixed_layer_selected_minus_alternative",
                    layer,
                )
                for layer in FIXED_LAYER_ORDER
            }
            base_prefix = [
                float(run["training"]["final_nll"]),
                float(generic["probe_teacher_one_step_mse"]),
                float(generic["probe_open_loop_latent_mse"]),
                float(generic["probe_terminal_exactness"]),
                float(record["stratum"] == "unseen_structural_mode"),
                *[
                    float(model == candidate)
                    for candidate in PRIMARY_PREDICTIVE_MODELS[1:]
                ],
                *[
                    mean(calibration, "generic_task_predictors", name)
                    for name in CALIBRATION_SCALARS
                ],
                float(
                    np.mean([task["task_failure"] for task in calibration])
                ),
            ]
            layer_prefix = [
                *[
                    float(
                        generic["probe_open_loop_layer_mismatch_rate"][layer]
                    )
                    for layer in LAYER_ORDER
                ],
                *[calibration_raw[layer] for layer in LAYER_ORDER],
                *[
                    calibration_raw_contrast[layer]
                    for layer in LAYER_ORDER
                ],
            ]
            tsi_prefix = [
                float(tsi["probe_i0_correspondence_auc"]),
                float(tsi["probe_fixed_total_auc"]),
                *[
                    float(tsi["probe_fixed_layer_auc"][layer])
                    for layer in FIXED_LAYER_ORDER
                ],
                *[
                    mean(calibration, "tsi_task_predictors", name)
                    for name in CALIBRATION_TSI_SCALARS
                ],
                *[
                    calibration_fixed[layer]
                    for layer in FIXED_LAYER_ORDER
                ],
                *[
                    calibration_fixed_contrast[layer]
                    for layer in FIXED_LAYER_ORDER
                ],
            ]
            base_panel = []
            layer_panel = []
            tsi_panel = []
            layer_tsi_panel = []
            failures = []
            for task in outcome:
                local = task["generic_task_predictors"]
                weights = local["goal_layer_weights"]
                base = [
                    *base_prefix,
                    float(local["normalized_plan_horizon"]),
                    *[float(weights[layer]) for layer in LAYER_ORDER],
                    float(local["predicted_utility_gap"]),
                ]
                aligned_raw = sum(
                    float(weights[layer]) * calibration_raw[layer]
                    for layer in LAYER_ORDER
                )
                aligned_raw_contrast = sum(
                    float(weights[layer]) * calibration_raw_contrast[layer]
                    for layer in LAYER_ORDER
                )
                layer = [
                    *base,
                    *layer_prefix,
                    aligned_raw,
                    aligned_raw_contrast,
                ]
                aligned_fixed = sum(
                    float(weights[layer])
                    * calibration_fixed[TASK_TO_FIXED[layer]]
                    for layer in LAYER_ORDER
                )
                aligned_fixed_contrast = sum(
                    float(weights[layer])
                    * calibration_fixed_contrast[TASK_TO_FIXED[layer]]
                    for layer in LAYER_ORDER
                )
                tsi_values = [
                    *tsi_prefix,
                    aligned_fixed,
                    aligned_fixed_contrast,
                ]
                base_panel.append(base)
                layer_panel.append(layer)
                tsi_panel.append([*base, *tsi_values])
                layer_tsi_panel.append([*layer, *tsi_values])
                failures.append(int(task["task_failure"]))
            first = next(
                (
                    index + 1
                    for index, failure in enumerate(failures)
                    if failure
                ),
                HORIZON + 1,
            )
            rows.append(
                {
                    "world_index": int(run["world_index"]),
                    "optimizer_seed": int(run["optimizer_seed"]),
                    "base": base_panel,
                    "layer": layer_panel,
                    "tsi": tsi_panel,
                    "layer_tsi": layer_tsi_panel,
                    "task_failures": failures,
                    "first_failure_time": first,
                }
            )
    return rows


def hazard_data(rows, field, names):
    features = []
    targets = []
    for row in rows:
        last = min(row["first_failure_time"], HORIZON)
        for time in range(1, last + 1):
            features.append(
                [
                    *row[field][time - 1],
                    *[
                        float(time == candidate)
                        for candidate in range(2, HORIZON + 1)
                    ],
                ]
            )
            targets.append(row["task_failures"][time - 1])
    return np.asarray(features), np.asarray(targets), (*names, *TIME_NAMES)


def fit(rows, field, names):
    x, y, expanded_names = hazard_data(rows, field, names)
    return fit_ridge_logistic(x, y, expanded_names)


def hazards(model, rows, field, names):
    expanded = []
    for row in rows:
        for time in range(1, HORIZON + 1):
            expanded.append(
                [
                    *row[field][time - 1],
                    *[
                        float(time == candidate)
                        for candidate in range(2, HORIZON + 1)
                    ],
                ]
            )
    values = predict_ridge_logistic(
        model,
        np.asarray(expanded),
        (*names, *TIME_NAMES),
    )
    return values.reshape(len(rows), HORIZON)


def score(rows, baseline_model, tsi_model, baseline_field, baseline_names,
          tsi_field, tsi_names):
    baseline_failure = 1.0 - np.cumprod(
        1.0 - hazards(baseline_model, rows, baseline_field, baseline_names),
        axis=1,
    )
    tsi_failure = 1.0 - np.cumprod(
        1.0 - hazards(tsi_model, rows, tsi_field, tsi_names),
        axis=1,
    )
    effects = []
    for index, row in enumerate(rows):
        first = row["first_failure_time"]
        observed = np.asarray(
            [float(first <= time) for time in range(1, HORIZON + 1)]
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
            "outcome_event_rate": float(
                np.mean(
                    [row["first_failure_time"] <= HORIZON for row in rows]
                )
            ),
            "scalar_generic_vs_split_tsi_mean": np.mean(
                scalar, axis=0
            ).tolist(),
            "scalar_generic_vs_split_tsi_world_seed_sd": np.std(
                scalar, axis=0, ddof=1
            ).tolist(),
            "layer_generic_vs_split_tsi_mean": np.mean(
                layer, axis=0
            ).tolist(),
            "layer_generic_vs_split_tsi_world_seed_sd": np.std(
                layer, axis=0, ddof=1
            ).tolist(),
        },
        indent=2,
        sort_keys=True,
    )
)
