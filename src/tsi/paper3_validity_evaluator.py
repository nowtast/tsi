"""Separated diagnostic predictors and downstream task outcomes for P3-4B."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .paper3_development_experiment import (
    ConstructiveMetricCache,
    LAYER_ERROR_FIELDS,
)
from .paper3_multiworld import (
    GeneratedTransitionCase,
    LAYER_ORDER,
    MultiworldStateCode,
    StructuredAction,
    WorldMechanism,
    successor_code,
)
from .paper3_routing_model import TrainableRoutingModel
from .paper3_validity_contract import PROBE_HORIZON, TASKS_PER_UNIT
from .paper3_validity_generator import (
    DownstreamTaskSpec,
    ValidityUnitSpec,
    goal_utility,
)


P3_VALIDITY_EVALUATOR_ID = "P3-4B-VALIDITY-EVALUATOR-v2"
FIXED_TO_TASK_LAYER = {
    "label": "label",
    "simplicial": "topology",
    "metric": "metric",
    "relation": "relation",
    "order": "order",
}
TASK_TO_FIXED_LAYER = {value: key for key, value in FIXED_TO_TASK_LAYER.items()}


def _transition_case(
    source: MultiworldStateCode,
    action: StructuredAction,
    target: MultiworldStateCode,
) -> GeneratedTransitionCase:
    return GeneratedTransitionCase(
        partition="ood",
        ood_slice="unseen_recombination",
        source_code=source,
        action=action,
        target_code=target,
        follows_declared_mechanism=True,
    )


def _predict_next(
    model: TrainableRoutingModel,
    source: MultiworldStateCode,
    action: StructuredAction,
    mechanism: WorldMechanism,
) -> MultiworldStateCode:
    local_target = successor_code(source, action, mechanism)
    case = _transition_case(source, action, local_target)
    features = model.basis.transform_cases((case,))[0]
    return model.predict_codes_precomputed((case,), features)[0]


def _raw_one_hot_mse(
    predicted: MultiworldStateCode,
    target: MultiworldStateCode,
) -> float:
    mismatch_count = sum(
        left != right
        for left, right in zip(
            predicted.as_tuple(),
            target.as_tuple(),
            strict=True,
        )
    )
    # The concatenated one-hot state has width 16 and contributes two squared
    # coordinate errors for every mismatched categorical layer.
    return mismatch_count / 8.0


def _layer_mismatches(
    predicted: MultiworldStateCode,
    target: MultiworldStateCode,
) -> dict[str, float]:
    return {
        layer: float(left != right)
        for layer, left, right in zip(
            LAYER_ORDER,
            predicted.as_tuple(),
            target.as_tuple(),
            strict=True,
        )
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty validity sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _selected_minus_alternative(
    values: Sequence[float],
    selected: int,
) -> float:
    if len(values) != 2 or selected not in (0, 1):
        raise ValueError("validity plan contrast requires two candidates")
    return float(values[selected] - values[1 - selected])


def _predict_plan_endpoint(
    model: TrainableRoutingModel,
    source: MultiworldStateCode,
    plan: Sequence[StructuredAction],
    mechanism: WorldMechanism,
) -> MultiworldStateCode:
    current = source
    for action in plan:
        current = _predict_next(model, current, action, mechanism)
    return current


def _evaluate_task(
    cache: ConstructiveMetricCache,
    model: TrainableRoutingModel,
    mechanism: WorldMechanism,
    task: DownstreamTaskSpec,
) -> dict[str, object]:
    predicted_endpoints = tuple(
        _predict_plan_endpoint(
            model,
            task.start_code,
            plan,
            mechanism,
        )
        for plan in task.candidate_plans
    )
    oracle_endpoints: list[MultiworldStateCode] = []
    for plan in task.candidate_plans:
        current = task.start_code
        for action in plan:
            current = successor_code(current, action, mechanism)
        oracle_endpoints.append(current)
    recomputed_oracle_utilities = tuple(
        goal_utility(
            endpoint,
            task.goal_layers,
            task.goal_queries,
            task.goal_values,
        )
        for endpoint in oracle_endpoints
    )
    if recomputed_oracle_utilities != task.oracle_utilities:
        raise RuntimeError("task oracle utilities are not reproducible")
    predicted_utilities = tuple(
        goal_utility(
            endpoint,
            task.goal_layers,
            task.goal_queries,
            task.goal_values,
        )
        for endpoint in predicted_endpoints
    )
    if predicted_utilities[0] == predicted_utilities[1]:
        chosen = task.prediction_tie_break_index
    else:
        chosen = int(predicted_utilities[1] > predicted_utilities[0])
    oracle_best_utility = max(task.oracle_utilities)
    realized_utility = task.oracle_utilities[chosen]
    regret = oracle_best_utility - realized_utility

    latent_mse: list[float] = []
    endpoint_exact: list[float] = []
    layer_mismatch = {layer: [] for layer in LAYER_ORDER}
    i0_errors: list[float] = []
    fixed_total: list[float] = []
    fixed_layers = {layer: [] for layer in LAYER_ERROR_FIELDS}
    aligned_raw: list[float] = []
    aligned_fixed: list[float] = []
    for predicted, oracle in zip(
        predicted_endpoints,
        oracle_endpoints,
        strict=True,
    ):
        latent_mse.append(_raw_one_hot_mse(predicted, oracle))
        endpoint_exact.append(float(predicted == oracle))
        mismatches = _layer_mismatches(predicted, oracle)
        for layer, value in mismatches.items():
            layer_mismatch[layer].append(value)
        quotient, fixed = cache.pair(predicted, oracle)
        i0_errors.append(quotient)
        fixed_total.append(float(fixed.total))
        for layer in LAYER_ERROR_FIELDS:
            fixed_layers[layer].append(float(getattr(fixed, layer)))
        raw_values = tuple(mismatches[layer] for layer in LAYER_ORDER)
        fixed_values = tuple(
            float(getattr(fixed, TASK_TO_FIXED_LAYER[layer]))
            for layer in LAYER_ORDER
        )
        aligned_raw.append(
            (
                2.0 * raw_values[task.goal_layers[0]]
                + raw_values[task.goal_layers[1]]
            )
            / 3.0
        )
        aligned_fixed.append(
            (
                2.0 * fixed_values[task.goal_layers[0]]
                + fixed_values[task.goal_layers[1]]
            )
            / 3.0
        )
    goal_weights = np.zeros(len(LAYER_ORDER), dtype=np.float64)
    goal_weights[task.goal_layers[0]] = 2.0 / 3.0
    goal_weights[task.goal_layers[1]] = 1.0 / 3.0
    return {
        "task_index": task.task_index,
        "task_digest": task.task_digest,
        "plan_horizon": task.plan_horizon,
        "predicted_endpoints": [
            list(endpoint.as_tuple()) for endpoint in predicted_endpoints
        ],
        "oracle_endpoints": [
            list(endpoint.as_tuple()) for endpoint in oracle_endpoints
        ],
        "predicted_utilities": list(predicted_utilities),
        "chosen_plan_index": chosen,
        "oracle_best_index": task.oracle_best_index,
        "oracle_utility_gap": abs(
            task.oracle_utilities[1] - task.oracle_utilities[0]
        ),
        "realized_regret": regret,
        "task_failure": int(regret > 0),
        "generic_task_predictors": {
            "candidate_latent_mse_mean": _mean(latent_mse),
            "candidate_latent_mse_max": float(max(latent_mse)),
            "candidate_latent_mse_selected_minus_alternative": (
                _selected_minus_alternative(latent_mse, chosen)
            ),
            "candidate_endpoint_exact_rate": _mean(endpoint_exact),
            "candidate_endpoint_exact_selected_minus_alternative": (
                _selected_minus_alternative(endpoint_exact, chosen)
            ),
            "candidate_layer_mismatch_rate": {
                layer: _mean(values) for layer, values in layer_mismatch.items()
            },
            "candidate_layer_mismatch_selected_minus_alternative": {
                layer: _selected_minus_alternative(values, chosen)
                for layer, values in layer_mismatch.items()
            },
            "goal_aligned_raw_mismatch": _mean(aligned_raw),
            "goal_aligned_raw_selected_minus_alternative": (
                _selected_minus_alternative(aligned_raw, chosen)
            ),
            "goal_layer_weights": {
                layer: float(goal_weights[index])
                for index, layer in enumerate(LAYER_ORDER)
            },
            "predicted_utility_gap": float(
                abs(predicted_utilities[1] - predicted_utilities[0])
            ),
            "oracle_utility_gap": float(
                abs(task.oracle_utilities[1] - task.oracle_utilities[0])
            ),
            "normalized_plan_horizon": task.plan_horizon / 8.0,
        },
        "tsi_task_predictors": {
            "candidate_i0_mean": _mean(i0_errors),
            "candidate_i0_max": float(max(i0_errors)),
            "candidate_i0_selected_minus_alternative": (
                _selected_minus_alternative(i0_errors, chosen)
            ),
            "candidate_fixed_total_mean": _mean(fixed_total),
            "candidate_fixed_total_max": float(max(fixed_total)),
            "candidate_fixed_total_selected_minus_alternative": (
                _selected_minus_alternative(fixed_total, chosen)
            ),
            "candidate_fixed_layer_mean": {
                layer: _mean(values) for layer, values in fixed_layers.items()
            },
            "candidate_fixed_layer_selected_minus_alternative": {
                layer: _selected_minus_alternative(values, chosen)
                for layer, values in fixed_layers.items()
            },
            "goal_aligned_fixed_discrepancy": _mean(aligned_fixed),
            "goal_aligned_fixed_selected_minus_alternative": (
                _selected_minus_alternative(aligned_fixed, chosen)
            ),
        },
    }


def evaluate_validity_units(
    cache: ConstructiveMetricCache,
    model: TrainableRoutingModel,
    mechanism: WorldMechanism,
    units: Sequence[ValidityUnitSpec],
) -> tuple[dict[str, object], ...]:
    """Evaluate diagnostic predictors before independent downstream tasks."""

    unit_tuple = tuple(units)
    if not unit_tuple:
        raise ValueError("validity evaluation needs at least one unit")
    records: list[dict[str, object]] = []
    for unit in unit_tuple:
        if len(unit.probe_actions) != PROBE_HORIZON:
            raise ValueError("validity probe horizon changed")
        if len(unit.tasks) != TASKS_PER_UNIT:
            raise ValueError("validity task battery changed")

        oracle_current = unit.probe_initial_code
        open_current = unit.probe_initial_code
        teacher_mse: list[float] = []
        open_mse: list[float] = []
        open_i0: list[float] = []
        open_fixed_total: list[float] = []
        layer_mismatch = {layer: [] for layer in LAYER_ORDER}
        fixed_layers = {layer: [] for layer in LAYER_ERROR_FIELDS}
        open_prediction = open_current
        oracle_next = oracle_current
        for action in unit.probe_actions:
            oracle_next = successor_code(oracle_current, action, mechanism)
            teacher_prediction = _predict_next(
                model,
                oracle_current,
                action,
                mechanism,
            )
            open_prediction = _predict_next(
                model,
                open_current,
                action,
                mechanism,
            )
            teacher_mse.append(_raw_one_hot_mse(teacher_prediction, oracle_next))
            open_mse.append(_raw_one_hot_mse(open_prediction, oracle_next))
            mismatches = _layer_mismatches(open_prediction, oracle_next)
            for layer, value in mismatches.items():
                layer_mismatch[layer].append(value)
            quotient, fixed = cache.pair(open_prediction, oracle_next)
            open_i0.append(quotient)
            open_fixed_total.append(float(fixed.total))
            for layer in LAYER_ERROR_FIELDS:
                fixed_layers[layer].append(float(getattr(fixed, layer)))
            oracle_current = oracle_next
            open_current = open_prediction

        task_records = tuple(
            _evaluate_task(cache, model, mechanism, task) for task in unit.tasks
        )
        task_failures = tuple(int(task["task_failure"]) for task in task_records)
        first_failure = next(
            (
                index + 1
                for index, failed in enumerate(task_failures)
                if failed
            ),
            TASKS_PER_UNIT + 1,
        )
        goal_weight_counts = np.zeros(len(LAYER_ORDER), dtype=np.float64)
        for task in unit.tasks:
            goal_weight_counts[task.goal_layers[0]] += 2.0
            goal_weight_counts[task.goal_layers[1]] += 1.0
        goal_weights = goal_weight_counts / float(3 * TASKS_PER_UNIT)
        mismatch_means = {
            layer: _mean(values) for layer, values in layer_mismatch.items()
        }
        fixed_means = {
            layer: _mean(values) for layer, values in fixed_layers.items()
        }
        generic_aligned = sum(
            goal_weights[index] * mismatch_means[layer]
            for index, layer in enumerate(LAYER_ORDER)
        )
        tsi_aligned = sum(
            goal_weights[index]
            * fixed_means[TASK_TO_FIXED_LAYER[layer]]
            for index, layer in enumerate(LAYER_ORDER)
        )
        records.append(
            {
                "identifier": P3_VALIDITY_EVALUATOR_ID,
                "world_index": unit.world_index,
                "unit_index": unit.unit_index,
                "unit_digest": unit.unit_digest,
                "stratum": unit.stratum,
                "generic_predictors": {
                    "probe_teacher_one_step_mse": _mean(teacher_mse),
                    "probe_open_loop_latent_mse": _mean(open_mse),
                    "probe_terminal_exactness": float(
                        open_prediction == oracle_next
                    ),
                    "probe_open_loop_layer_mismatch_rate": mismatch_means,
                    "generic_task_aligned_mismatch": float(generic_aligned),
                    "task_goal_layer_weights": {
                        layer: float(goal_weights[index])
                        for index, layer in enumerate(LAYER_ORDER)
                    },
                    "oracle_mean_reward_gap": _mean(
                        [
                            float(task["oracle_utility_gap"])
                            for task in task_records
                        ]
                    ),
                    "mean_plan_horizon": _mean(
                        [float(task.plan_horizon) for task in unit.tasks]
                    ),
                },
                "tsi_predictors": {
                    "probe_i0_correspondence_auc": _mean(open_i0),
                    "probe_fixed_total_auc": _mean(open_fixed_total),
                    "probe_fixed_layer_auc": fixed_means,
                    "tsi_task_aligned_fixed_auc": float(tsi_aligned),
                },
                "outcomes": {
                    "any_task_failure": int(any(task_failures)),
                    "task_failure_count": int(sum(task_failures)),
                    "first_failure_time": first_failure,
                    "terminal_task_survival": int(first_failure > TASKS_PER_UNIT),
                    "mean_realized_regret": _mean(
                        [
                            float(task["realized_regret"])
                            for task in task_records
                        ]
                    ),
                    "task_failures": list(task_failures),
                },
                "task_records": list(task_records),
                "probe_task_domains_separated": True,
                "outcome_uses_tsi_metric": False,
            }
        )
    return tuple(records)
