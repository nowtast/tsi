"""Teacher-forced/open-loop evaluation and exact finite rollout bounds."""

from __future__ import annotations

from functools import lru_cache
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .paper3_development_experiment import (
    ConstructiveMetricCache,
    LAYER_ERROR_FIELDS,
)
from .paper3_interface import fixed_carrier_exact_losses
from .paper3_multiworld import (
    GeneratedTransitionCase,
    MultiworldStateCode,
    PRIMITIVE_ACTIONS,
    StructuredAction,
    WorldMechanism,
    all_multiworld_state_codes,
    successor_code,
)
from .paper3_rollout_contract import MAX_HORIZON, REPORT_HORIZONS
from .paper3_rollout_generator import RolloutTrajectorySpec
from .paper3_routing_model import TrainableRoutingModel


P3_ROLLOUT_EVALUATOR_ID = "P3-4A-ROLLOUT-EVALUATOR-v1"
BOUND_TOLERANCE = 1.0e-12


def _transition_cases(
    sources: Sequence[MultiworldStateCode],
    actions: Sequence[StructuredAction],
    targets: Sequence[MultiworldStateCode],
) -> tuple[GeneratedTransitionCase, ...]:
    if not (len(sources) == len(actions) == len(targets)):
        raise ValueError("rollout transition vectors must have equal lengths")
    return tuple(
        GeneratedTransitionCase(
            partition="ood",
            ood_slice="unseen_recombination",
            source_code=source,
            action=action,
            target_code=target,
            follows_declared_mechanism=True,
        )
        for source, action, target in zip(
            sources,
            actions,
            targets,
            strict=True,
        )
    )


@lru_cache(maxsize=1)
def fixed_state_distance_matrix() -> tuple[
    tuple[MultiworldStateCode, ...],
    np.ndarray,
]:
    """Return the exact fixed-carrier metric on all 324 frozen states."""

    codes = all_multiworld_state_codes()
    cache = ConstructiveMetricCache()
    matrix = np.zeros((len(codes), len(codes)), dtype=np.float64)
    for left in range(len(codes)):
        left_state = cache.state(codes[left])
        for right in range(left + 1, len(codes)):
            distance = fixed_carrier_exact_losses(
                left_state,
                cache.state(codes[right]),
            ).total
            matrix[left, right] = distance
            matrix[right, left] = distance
    matrix.setflags(write=False)
    return codes, matrix


def exact_transition_lipschitz_constants(
    mechanism: WorldMechanism,
) -> dict[str, float]:
    """Compute the sharp finite fixed-metric Lipschitz constant per action."""

    codes, distances = fixed_state_distance_matrix()
    index = {code: position for position, code in enumerate(codes)}
    positive = distances > 0.0
    constants: dict[str, float] = {}
    for action in PRIMITIVE_ACTIONS:
        successors = np.asarray(
            [index[successor_code(code, action, mechanism)] for code in codes],
            dtype=np.int64,
        )
        output_distances = distances[np.ix_(successors, successors)]
        ratios = np.divide(
            output_distances,
            distances,
            out=np.zeros_like(output_distances),
            where=positive,
        )
        constants[action.name] = float(np.max(ratios))
    return constants


def audit_fixed_metric_and_lipschitz(
    mechanism: WorldMechanism,
) -> dict[str, object]:
    codes, distances = fixed_state_distance_matrix()
    errors: list[str] = []
    if distances.shape != (324, 324):
        errors.append("fixed-state distance matrix has the wrong shape")
    if not np.allclose(distances, distances.T, atol=0.0, rtol=0.0):
        errors.append("fixed-state distance matrix is not symmetric")
    if not np.all(np.diag(distances) == 0.0):
        errors.append("fixed-state metric has nonzero diagonal")
    off_diagonal = ~np.eye(len(codes), dtype=bool)
    if not np.all(distances[off_diagonal] > 0.0):
        errors.append("distinct state codes are not separated")
    maximum_triangle_excess = 0.0
    for middle in range(len(codes)):
        excess = distances - (
            distances[:, middle, np.newaxis] + distances[middle, np.newaxis, :]
        )
        maximum_triangle_excess = max(
            maximum_triangle_excess,
            float(np.max(excess)),
        )
    if maximum_triangle_excess > BOUND_TOLERANCE:
        errors.append("fixed-state distance violates the triangle inequality")
    constants = exact_transition_lipschitz_constants(mechanism)
    if any(value < 0.0 or not isfinite(value) for value in constants.values()):
        errors.append("a transition Lipschitz constant is invalid")
    return {
        "identifier": P3_ROLLOUT_EVALUATOR_ID,
        "state_count": len(codes),
        "positive_off_diagonal_count": int(np.sum(distances > 0.0)),
        "maximum_triangle_excess": maximum_triangle_excess,
        "exact_lipschitz_constants": constants,
        "errors": errors,
        "passed": not errors,
    }


def _cyclic_tracking_error(predicted_shift: int, oracle_shift: int) -> float:
    return 0.0 if predicted_shift % 3 == oracle_shift % 3 else 1.0


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        raise ValueError("cannot average an empty rollout array")
    return float(np.mean(values))


def evaluate_rollout_model(
    cache: ConstructiveMetricCache,
    model: TrainableRoutingModel,
    mechanism: WorldMechanism,
    trajectories: Sequence[RolloutTrajectorySpec],
    *,
    lipschitz_constants: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Evaluate one fitted model without pooling teacher-forced/open-loop paths."""

    specs = tuple(trajectories)
    if len(specs) == 0:
        raise ValueError("rollout evaluation needs at least one trajectory")
    if any(len(spec.actions) != MAX_HORIZON for spec in specs):
        raise ValueError("rollout trajectory horizon changed")
    constants = (
        exact_transition_lipschitz_constants(mechanism)
        if lipschitz_constants is None
        else {name: float(value) for name, value in lipschitz_constants.items()}
    )
    expected_actions = {action.name for action in PRIMITIVE_ACTIONS}
    if set(constants) != expected_actions:
        raise ValueError("Lipschitz constants do not cover the primitive actions")

    count = len(specs)
    shape = (count, MAX_HORIZON)
    teacher_i0 = np.zeros(shape, dtype=np.float64)
    open_i0 = np.zeros(shape, dtype=np.float64)
    teacher_fixed = np.zeros(shape, dtype=np.float64)
    open_fixed = np.zeros(shape, dtype=np.float64)
    open_layers = {
        name: np.zeros(shape, dtype=np.float64) for name in LAYER_ERROR_FIELDS
    }
    open_tracking = np.zeros(shape, dtype=np.float64)
    teacher_tracking = np.zeros(shape, dtype=np.float64)
    local_law_error = np.zeros(shape, dtype=np.float64)
    local_law_violation = np.zeros(shape, dtype=np.float64)
    bridge_violation = np.zeros(shape, dtype=np.float64)
    recursive_bounds = np.zeros(shape, dtype=np.float64)
    bound_excess = np.zeros(shape, dtype=np.float64)
    first_failure = np.full(count, MAX_HORIZON + 1, dtype=np.int64)

    oracle_current = [spec.initial_code for spec in specs]
    open_current = [spec.initial_code for spec in specs]
    oracle_tracking_shift = np.zeros(count, dtype=np.int64)
    predicted_tracking_shift = np.zeros(count, dtype=np.int64)
    previous_bounds = np.zeros(count, dtype=np.float64)

    for time in range(MAX_HORIZON):
        actions = [spec.actions[time] for spec in specs]
        oracle_next = [
            successor_code(source, action, mechanism)
            for source, action in zip(oracle_current, actions, strict=True)
        ]
        teacher_cases = _transition_cases(
            oracle_current,
            actions,
            oracle_next,
        )
        teacher_features = model.basis.transform_cases(teacher_cases)[0]
        teacher_predictions = model.predict_codes_precomputed(
            teacher_cases,
            teacher_features,
        )

        local_oracle_next = [
            successor_code(source, action, mechanism)
            for source, action in zip(open_current, actions, strict=True)
        ]
        open_cases = _transition_cases(
            open_current,
            actions,
            local_oracle_next,
        )
        open_features = model.basis.transform_cases(open_cases)[0]
        open_predictions = model.predict_codes_precomputed(
            open_cases,
            open_features,
        )

        for trajectory in range(count):
            teacher_quotient, teacher_layer = cache.pair(
                teacher_predictions[trajectory],
                oracle_next[trajectory],
            )
            open_quotient, open_layer = cache.pair(
                open_predictions[trajectory],
                oracle_next[trajectory],
            )
            local_layer = fixed_carrier_exact_losses(
                cache.state(open_predictions[trajectory]),
                cache.state(local_oracle_next[trajectory]),
            )
            teacher_i0[trajectory, time] = teacher_quotient
            open_i0[trajectory, time] = open_quotient
            teacher_fixed[trajectory, time] = teacher_layer.total
            open_fixed[trajectory, time] = open_layer.total
            for name in LAYER_ERROR_FIELDS:
                open_layers[name][trajectory, time] = float(getattr(open_layer, name))
            local_law_error[trajectory, time] = local_layer.total
            local_law_violation[trajectory, time] = float(
                open_predictions[trajectory] != local_oracle_next[trajectory]
            )
            bridge_violation[trajectory, time] = float(
                cache.bridge_violation(open_predictions[trajectory])
            )

            oracle_delta = (
                oracle_next[trajectory].label_phase
                - oracle_current[trajectory].label_phase
            ) % 3
            predicted_delta = (
                open_predictions[trajectory].label_phase
                - open_current[trajectory].label_phase
            ) % 3
            teacher_delta = (
                teacher_predictions[trajectory].label_phase
                - oracle_current[trajectory].label_phase
            ) % 3
            oracle_tracking_shift[trajectory] = (
                oracle_tracking_shift[trajectory] + oracle_delta
            ) % 3
            predicted_tracking_shift[trajectory] = (
                predicted_tracking_shift[trajectory] + predicted_delta
            ) % 3
            open_tracking[trajectory, time] = _cyclic_tracking_error(
                int(predicted_tracking_shift[trajectory]),
                int(oracle_tracking_shift[trajectory]),
            )
            teacher_tracking[trajectory, time] = _cyclic_tracking_error(
                int(teacher_delta),
                int(oracle_delta),
            )

            action_lipschitz = constants[actions[trajectory].name]
            recursive_bound = (
                local_layer.total + action_lipschitz * previous_bounds[trajectory]
            )
            recursive_bounds[trajectory, time] = recursive_bound
            bound_excess[trajectory, time] = open_layer.total - recursive_bound
            previous_bounds[trajectory] = recursive_bound
            if (
                first_failure[trajectory] == MAX_HORIZON + 1
                and open_predictions[trajectory] != oracle_next[trajectory]
            ):
                first_failure[trajectory] = time + 1

        oracle_current = oracle_next
        open_current = list(open_predictions)

    horizon_summary: dict[str, object] = {}
    for horizon in REPORT_HORIZONS:
        index = horizon - 1
        horizon_summary[str(horizon)] = {
            "teacher_forced_mean_i0_error": _mean(teacher_i0[:, index]),
            "open_loop_mean_i0_error": _mean(open_i0[:, index]),
            "teacher_forced_mean_fixed_error": _mean(teacher_fixed[:, index]),
            "open_loop_mean_fixed_error": _mean(open_fixed[:, index]),
            "open_loop_fixed_layer_error_vector": {
                name: _mean(values[:, index]) for name, values in open_layers.items()
            },
            "open_loop_tracking_error": _mean(open_tracking[:, index]),
            "teacher_forced_tracking_error": _mean(teacher_tracking[:, index]),
            "open_loop_state_exact_rate": _mean(
                (open_fixed[:, index] == 0.0).astype(np.float64)
            ),
            "trajectory_survival_rate": _mean(
                (first_failure > horizon).astype(np.float64)
            ),
            "state_coherence_bridge_violation_rate": _mean(bridge_violation[:, index]),
            "self_conditioned_local_law_violation_rate": _mean(
                local_law_violation[:, index]
            ),
        }

    trajectory_metrics = [
        {
            "trajectory_index": spec.trajectory_index,
            "stratum": spec.stratum,
            "trajectory_digest": spec.trajectory_digest,
            "teacher_forced_i0_auc": _mean(teacher_i0[index]),
            "open_loop_i0_auc": _mean(open_i0[index]),
            "terminal_open_loop_i0_error": float(open_i0[index, -1]),
            "terminal_open_loop_fixed_error": float(open_fixed[index, -1]),
            "terminal_tracking_error": float(open_tracking[index, -1]),
            "first_structural_failure_time": int(first_failure[index]),
        }
        for index, spec in enumerate(specs)
    ]
    maximum_bound_excess = float(np.max(bound_excess))
    return {
        "identifier": P3_ROLLOUT_EVALUATOR_ID,
        "trajectory_count": count,
        "maximum_horizon": MAX_HORIZON,
        "prediction_count": 2 * count * MAX_HORIZON,
        "teacher_forced_i0_auc": _mean(teacher_i0),
        "open_loop_i0_auc": _mean(open_i0),
        "exposure_gap_i0_auc": _mean(open_i0) - _mean(teacher_i0),
        "terminal_open_loop_i0_error": _mean(open_i0[:, -1]),
        "terminal_open_loop_fixed_error": _mean(open_fixed[:, -1]),
        "terminal_open_loop_tracking_error": _mean(open_tracking[:, -1]),
        "self_conditioned_local_law_error_auc": _mean(local_law_error),
        "self_conditioned_local_law_violation_rate": _mean(local_law_violation),
        "state_coherence_bridge_violation_rate": _mean(bridge_violation),
        "mean_first_structural_failure_time": _mean(first_failure.astype(np.float64)),
        "terminal_trajectory_survival_rate": _mean(
            (first_failure > MAX_HORIZON).astype(np.float64)
        ),
        "open_loop_fixed_layer_auc": {
            name: _mean(values) for name, values in open_layers.items()
        },
        "exact_lipschitz_constants": dict(constants),
        "maximum_recursive_bound_excess": maximum_bound_excess,
        "recursive_bound_violation_count": int(np.sum(bound_excess > BOUND_TOLERANCE)),
        "maximum_recursive_bound": float(np.max(recursive_bounds)),
        "horizon_summary": horizon_summary,
        "trajectory_metrics": trajectory_metrics,
        "global_target_state_candidates": 0,
    }
