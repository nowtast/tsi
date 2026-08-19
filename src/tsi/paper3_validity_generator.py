"""Domain-separated diagnostic probes and noncircular downstream tasks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .paper3_independence_contract import BenchmarkSplit
from .paper3_multiworld import (
    BASE_LABELS,
    DEVELOPMENT_WORLDS_PER_FAMILY,
    LAYER_ORDER,
    VALIDATION_WORLDS_PER_FAMILY,
    MultiworldStateCode,
    PRIMITIVE_ACTIONS,
    StructuredAction,
    WorldMechanism,
    _ranked_active_parameters,
    all_multiworld_state_codes,
    build_world_mechanism,
    successor_code,
)
from .paper3_validity_contract import (
    DEVELOPMENT_WORLDS,
    PLAN_CANDIDATE_COUNT,
    PRIMARY_FAMILY,
    PROBE_HORIZON,
    TASK_HORIZONS,
    TASKS_PER_UNIT,
    UNITS_PER_WORLD,
    validity_contract_digest,
)


P3_VALIDITY_GENERATOR_ID = "P3-4B-NONCIRCULAR-TASK-GENERATOR-v2"
DEVELOPMENT_VALIDITY_ROOT = "tsi:p3-4b:validity-development:2026-07-29:v1"
UNIT_STRATA = ("unseen_structural_mode", "unseen_recombination")
UNITS_PER_STRATUM = UNITS_PER_WORLD // len(UNIT_STRATA)
LAYER_CARDINALITIES = (3, 3, 3, 4, 3)
UNDIRECTED_ENTITY_PAIRS = ((0, 1), (0, 2), (1, 2))
DIRECTED_ENTITY_PAIRS = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
)
STRUCTURAL_QUERY_COUNTS = (3, 3, 9, 6, 6)
ACTION_BY_NAME = {action.name: action for action in PRIMITIVE_ACTIONS}
PROBE_ACTION_MULTISET = (
    "hold",
    "label_step",
    "topology_step",
    "topology_step",
    "metric_step",
    "metric_step",
    "relation_step",
    "order_step",
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: object) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def _hmac_digest(key: bytes, payload: object) -> bytes:
    return hmac.new(key, _canonical_bytes(payload), sha256).digest()


def _source_residue(code: MultiworldStateCode) -> int:
    return (
        code.label_phase
        + 2 * code.topology_mode
        + 3 * code.metric_mode
        + 5 * code.order_mode
        + code.influence_mode
    ) % 7


def _is_unseen_structural_mode(code: MultiworldStateCode) -> bool:
    return (
        code.topology_mode == 2
        and code.order_mode == 2
        and code.influence_mode in (2, 3)
    )


def _initial_candidates(stratum: str) -> tuple[MultiworldStateCode, ...]:
    if stratum == "unseen_structural_mode":
        return tuple(
            code
            for code in all_multiworld_state_codes()
            if _is_unseen_structural_mode(code)
        )
    if stratum == "unseen_recombination":
        return tuple(
            code
            for code in all_multiworld_state_codes()
            if _source_residue(code) == 6 and not _is_unseen_structural_mode(code)
        )
    raise ValueError(f"unknown validity stratum: {stratum}")


def _ranked_codes(
    key: bytes,
    *,
    world_index: int,
    unit_index: int,
    stratum: str,
    domain: str,
) -> tuple[MultiworldStateCode, ...]:
    return tuple(
        sorted(
            _initial_candidates(stratum),
            key=lambda code: _hmac_digest(
                key,
                (
                    P3_VALIDITY_GENERATOR_ID,
                    domain,
                    world_index,
                    unit_index,
                    stratum,
                    code.as_tuple(),
                ),
            ),
        )
    )


def _apply_plan(
    source: MultiworldStateCode,
    plan: Sequence[StructuredAction],
    mechanism: WorldMechanism,
) -> MultiworldStateCode:
    current = source
    for action in plan:
        current = successor_code(current, action, mechanism)
    return current


def structural_query_value(
    code: MultiworldStateCode,
    layer: int,
    query_index: int,
) -> int:
    """Evaluate one domain predicate without consulting any TSI metric."""

    if layer not in range(len(LAYER_ORDER)):
        raise ValueError("structural query layer is outside the frozen state")
    if not 0 <= query_index < STRUCTURAL_QUERY_COUNTS[layer]:
        raise ValueError("structural query index is outside its layer bank")
    if layer == 0:
        labels = tuple(
            BASE_LABELS[(entity - code.label_phase) % 3] for entity in range(3)
        )
        return int(labels[query_index] == "blue")
    if layer == 1:
        edges = (
            {(0, 1), (1, 2)}
            if code.topology_mode == 0
            else (
                {(0, 1), (0, 2), (1, 2)}
                if code.topology_mode == 1
                else {(0, 2)}
            )
        )
        return int(UNDIRECTED_ENTITY_PAIRS[query_index] in edges)
    if layer == 2:
        coordinates = (
            (0.0, 1.0, 3.0)
            if code.metric_mode == 0
            else (
                (0.0, 2.0, 3.0)
                if code.metric_mode == 1
                else (0.0, 1.0, 5.0)
            )
        )
        left, right = UNDIRECTED_ENTITY_PAIRS[query_index // 3]
        threshold = (1.5, 2.5, 4.0)[query_index % 3]
        return int(abs(coordinates[left] - coordinates[right]) <= threshold)
    if layer == 3:
        influence_pairs = (
            {(0, 1), (1, 2), (2, 0)}
            if code.influence_mode == 0
            else (
                {(1, 0), (2, 1), (0, 2)}
                if code.influence_mode == 1
                else (
                    {(0, 1), (0, 2)}
                    if code.influence_mode == 2
                    else {(0, 1), (1, 2)}
                )
            )
        )
        return int(DIRECTED_ENTITY_PAIRS[query_index] in influence_pairs)
    left, right = DIRECTED_ENTITY_PAIRS[query_index]
    if code.order_mode == 0:
        return 0
    return int(left <= right if code.order_mode == 1 else left >= right)


def goal_utility(
    code: MultiworldStateCode,
    goal_layers: tuple[int, int],
    goal_queries: tuple[int, int],
    goal_values: tuple[int, int],
) -> int:
    """Return exogenous predicate utility; no TSI discrepancy enters."""

    observed = tuple(
        structural_query_value(code, layer, query)
        for layer, query in zip(goal_layers, goal_queries, strict=True)
    )
    return (
        2 * int(observed[0] == goal_values[0])
        + int(observed[1] == goal_values[1])
    )


@dataclass(frozen=True)
class DownstreamTaskSpec:
    task_index: int
    start_code: MultiworldStateCode
    plan_horizon: int
    candidate_plans: tuple[
        tuple[StructuredAction, ...],
        tuple[StructuredAction, ...],
    ]
    goal_layers: tuple[int, int]
    goal_queries: tuple[int, int]
    goal_values: tuple[int, int]
    oracle_utilities: tuple[int, int]
    oracle_best_index: int
    prediction_tie_break_index: int
    task_digest: str

    def __post_init__(self) -> None:
        if type(self.task_index) is not int or not 0 <= self.task_index < TASKS_PER_UNIT:
            raise ValueError("task_index is outside the frozen battery")
        if self.plan_horizon != TASK_HORIZONS[self.task_index]:
            raise ValueError("task plan horizon changed")
        if len(self.candidate_plans) != 2 or any(
            len(plan) != self.plan_horizon for plan in self.candidate_plans
        ):
            raise ValueError("each task needs two full-horizon candidate plans")
        if (
            len(self.goal_layers) != 2
            or self.goal_layers[0] == self.goal_layers[1]
            or any(layer not in range(len(LAYER_ORDER)) for layer in self.goal_layers)
        ):
            raise ValueError("task goals must use two distinct frozen layers")
        if any(
            not 0 <= query < STRUCTURAL_QUERY_COUNTS[layer]
            for layer, query in zip(
                self.goal_layers,
                self.goal_queries,
                strict=True,
            )
        ):
            raise ValueError("task structural query is outside its layer bank")
        if any(value not in (0, 1) for value in self.goal_values):
            raise ValueError("task predicate goal values must be binary")
        if self.oracle_utilities[0] == self.oracle_utilities[1]:
            raise ValueError("task oracle choice must have nonzero utility gap")
        if self.oracle_best_index not in (0, 1):
            raise ValueError("task oracle best index must be binary")
        if self.prediction_tie_break_index not in (0, 1):
            raise ValueError("prediction tie break must be binary")
        if len(self.task_digest) != 64:
            raise ValueError("task digest must be a SHA-256 hex digest")

    def as_dict(self) -> dict[str, object]:
        return {
            "task_index": self.task_index,
            "start_code": list(self.start_code.as_tuple()),
            "plan_horizon": self.plan_horizon,
            "candidate_plans": [
                [action.name for action in plan] for plan in self.candidate_plans
            ],
            "goal_layers": [LAYER_ORDER[index] for index in self.goal_layers],
            "goal_layer_indices": list(self.goal_layers),
            "goal_queries": list(self.goal_queries),
            "goal_values": list(self.goal_values),
            "oracle_utilities": list(self.oracle_utilities),
            "oracle_best_index": self.oracle_best_index,
            "prediction_tie_break_index": self.prediction_tie_break_index,
            "task_digest": self.task_digest,
        }


@dataclass(frozen=True)
class ValidityUnitSpec:
    world_index: int
    unit_index: int
    stratum: str
    probe_initial_code: MultiworldStateCode
    probe_actions: tuple[StructuredAction, ...]
    tasks: tuple[DownstreamTaskSpec, ...]
    unit_digest: str

    def __post_init__(self) -> None:
        if type(self.world_index) is not int or self.world_index < 0:
            raise ValueError("world_index must be nonnegative")
        if type(self.unit_index) is not int or not 0 <= self.unit_index < UNITS_PER_WORLD:
            raise ValueError("unit_index is outside the frozen panel")
        if self.stratum not in UNIT_STRATA:
            raise ValueError("validity unit has an unknown stratum")
        if len(self.probe_actions) != PROBE_HORIZON:
            raise ValueError("validity probe horizon changed")
        if len(self.tasks) != TASKS_PER_UNIT:
            raise ValueError("validity task battery size changed")
        if any(task.start_code == self.probe_initial_code for task in self.tasks):
            raise ValueError("probe and downstream task initial states must be disjoint")
        if len(self.unit_digest) != 64:
            raise ValueError("unit digest must be a SHA-256 hex digest")

    def as_dict(self) -> dict[str, object]:
        return {
            "world_index": self.world_index,
            "unit_index": self.unit_index,
            "stratum": self.stratum,
            "probe_initial_code": list(self.probe_initial_code.as_tuple()),
            "probe_actions": [action.name for action in self.probe_actions],
            "tasks": [task.as_dict() for task in self.tasks],
            "unit_digest": self.unit_digest,
        }


def _probe_actions(
    key: bytes,
    world_index: int,
    unit_index: int,
) -> tuple[StructuredAction, ...]:
    ordered = sorted(
        enumerate(PROBE_ACTION_MULTISET),
        key=lambda item: _hmac_digest(
            key,
            (
                P3_VALIDITY_GENERATOR_ID,
                "diagnostic-probe-actions",
                world_index,
                unit_index,
                item,
            ),
        ),
    )
    return tuple(ACTION_BY_NAME[name] for _position, name in ordered)


def _ranked_goal_candidates(
    key: bytes,
    world_index: int,
    unit_index: int,
    task_index: int,
) -> tuple[tuple[tuple[int, int], tuple[int, int], int], ...]:
    base_candidates = tuple(
        (layers, queries)
        for layers in combinations(range(len(LAYER_ORDER)), 2)
        for queries in (
            (first, second)
            for first in range(STRUCTURAL_QUERY_COUNTS[layers[0]])
            for second in range(STRUCTURAL_QUERY_COUNTS[layers[1]])
        )
    )
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for layers, queries in base_candidates:
        digest = _hmac_digest(
            key,
            (
                P3_VALIDITY_GENERATOR_ID,
                "downstream-predicate-reference",
                world_index,
                unit_index,
                task_index,
                layers,
                queries,
            ),
        )
        first = int.from_bytes(digest[:8], "little") % PLAN_CANDIDATE_COUNT
        second = int.from_bytes(digest[8:16], "little") % PLAN_CANDIDATE_COUNT
        if second == first:
            second = (second + 1) % PLAN_CANDIDATE_COUNT
        candidates.extend(((layers, queries, first), (layers, queries, second)))
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: _hmac_digest(
                key,
                (
                    P3_VALIDITY_GENERATOR_ID,
                    "downstream-structural-predicate-goal",
                    world_index,
                    unit_index,
                    task_index,
                    candidate,
                ),
            ),
        )
    )


def _candidate_plan(
    key: bytes,
    world_index: int,
    unit_index: int,
    task_index: int,
    candidate_index: int,
    horizon: int,
) -> tuple[StructuredAction, ...]:
    if candidate_index < len(PRIMITIVE_ACTIONS):
        return (PRIMITIVE_ACTIONS[candidate_index],) * horizon
    actions: list[StructuredAction] = []
    for step in range(horizon):
        digest = _hmac_digest(
            key,
            (
                P3_VALIDITY_GENERATOR_ID,
                "downstream-candidate-plan",
                world_index,
                unit_index,
                task_index,
                candidate_index,
                step,
            ),
        )
        actions.append(PRIMITIVE_ACTIONS[int.from_bytes(digest[:8], "little") % 6])
    return tuple(actions)


def _build_task(
    key: bytes,
    mechanism: WorldMechanism,
    world_index: int,
    unit_index: int,
    task_index: int,
    start_code: MultiworldStateCode,
) -> DownstreamTaskSpec:
    horizon = TASK_HORIZONS[task_index]
    plans = tuple(
        _candidate_plan(
            key,
            world_index,
            unit_index,
            task_index,
            candidate_index,
            horizon,
        )
        for candidate_index in range(PLAN_CANDIDATE_COUNT)
    )
    endpoints = tuple(_apply_plan(start_code, plan, mechanism) for plan in plans)
    selected_goal: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, ...],
        tuple[tuple[int, int], ...],
    ] | None = None
    for goal_layers, goal_queries, reference in _ranked_goal_candidates(
        key,
        world_index,
        unit_index,
        task_index,
    ):
        goal_values = tuple(
            structural_query_value(endpoints[reference], layer, query)
            for layer, query in zip(goal_layers, goal_queries, strict=True)
        )
        utilities = tuple(
            goal_utility(
                endpoint,
                goal_layers,
                goal_queries,
                goal_values,
            )
            for endpoint in endpoints
        )
        eligible = tuple(
            (left, right)
            for left, right in combinations(range(len(plans)), 2)
            if plans[left] != plans[right]
            and endpoints[left] != endpoints[right]
            and utilities[left] != utilities[right]
        )
        if eligible:
            selected_goal = (
                goal_layers,
                goal_queries,
                goal_values,
                utilities,
                eligible,
            )
            break
    if selected_goal is None:
        raise RuntimeError("downstream task has no non-tied predicate plan pair")
    goal_layers, goal_queries, goal_values, utilities, eligible = selected_goal
    left, right = min(
        eligible,
        key=lambda pair: _hmac_digest(
            key,
            (
                P3_VALIDITY_GENERATOR_ID,
                "downstream-pair-selection",
                world_index,
                unit_index,
                task_index,
                pair,
            ),
        ),
    )
    chosen_plans = (plans[left], plans[right])
    oracle_utilities = (utilities[left], utilities[right])
    oracle_best = int(oracle_utilities[1] > oracle_utilities[0])
    tie_break = _hmac_digest(
        key,
        (
            P3_VALIDITY_GENERATOR_ID,
            "downstream-prediction-tie-break",
            world_index,
            unit_index,
            task_index,
        ),
    )[0] % 2
    payload = {
        "generator": P3_VALIDITY_GENERATOR_ID,
        "world_index": world_index,
        "unit_index": unit_index,
        "task_index": task_index,
        "start_code": start_code.as_tuple(),
        "plan_horizon": horizon,
        "candidate_plans": [
            [action.name for action in plan] for plan in chosen_plans
        ],
        "goal_layers": goal_layers,
        "goal_queries": goal_queries,
        "goal_values": goal_values,
        "oracle_utilities": oracle_utilities,
        "oracle_best_index": oracle_best,
        "prediction_tie_break_index": tie_break,
    }
    return DownstreamTaskSpec(
        task_index=task_index,
        start_code=start_code,
        plan_horizon=horizon,
        candidate_plans=chosen_plans,
        goal_layers=goal_layers,
        goal_queries=goal_queries,
        goal_values=goal_values,
        oracle_utilities=oracle_utilities,
        oracle_best_index=oracle_best,
        prediction_tie_break_index=tie_break,
        task_digest=_digest(payload),
    )


def validity_unit_specs(
    key: bytes,
    mechanism: WorldMechanism,
) -> tuple[ValidityUnitSpec, ...]:
    if len(key) != 32:
        raise ValueError("validity panel key must contain exactly 32 bytes")
    selected: list[tuple[str, MultiworldStateCode]] = []
    for stratum in UNIT_STRATA:
        ranked = _ranked_codes(
            key,
            world_index=mechanism.world_index,
            unit_index=0,
            stratum=stratum,
            domain="diagnostic-probe-initial-panel",
        )
        selected.extend((stratum, code) for code in ranked[:UNITS_PER_STRATUM])

    units: list[ValidityUnitSpec] = []
    for unit_index, (stratum, probe_initial) in enumerate(selected):
        task_codes = [
            code
            for code in _ranked_codes(
                key,
                world_index=mechanism.world_index,
                unit_index=unit_index,
                stratum=stratum,
                domain="downstream-task-initial-panel",
            )
            if code != probe_initial
        ][:TASKS_PER_UNIT]
        if len(task_codes) != TASKS_PER_UNIT:
            raise RuntimeError("downstream task panel has insufficient initial states")
        tasks = tuple(
            _build_task(
                key,
                mechanism,
                mechanism.world_index,
                unit_index,
                task_index,
                task_codes[task_index],
            )
            for task_index in range(TASKS_PER_UNIT)
        )
        probe_actions = _probe_actions(key, mechanism.world_index, unit_index)
        payload = {
            "generator": P3_VALIDITY_GENERATOR_ID,
            "world_index": mechanism.world_index,
            "unit_index": unit_index,
            "stratum": stratum,
            "probe_initial_code": probe_initial.as_tuple(),
            "probe_actions": [action.name for action in probe_actions],
            "task_digests": [task.task_digest for task in tasks],
        }
        units.append(
            ValidityUnitSpec(
                world_index=mechanism.world_index,
                unit_index=unit_index,
                stratum=stratum,
                probe_initial_code=probe_initial,
                probe_actions=probe_actions,
                tasks=tasks,
                unit_digest=_digest(payload),
            )
        )
    return tuple(units)


def development_validity_worlds() -> tuple[WorldMechanism, ...]:
    return tuple(
        build_world_mechanism(
            PRIMARY_FAMILY,
            BenchmarkSplit.DEVELOPMENT,
            world_index,
        )
        for world_index in range(DEVELOPMENT_WORLDS)
    )


def development_validity_units(
    mechanism: WorldMechanism,
) -> tuple[ValidityUnitSpec, ...]:
    key = sha256(
        (
            f"{DEVELOPMENT_VALIDITY_ROOT}:{mechanism.world_index}:"
            f"{mechanism.mechanism_digest}"
        ).encode("utf-8")
    ).digest()
    return validity_unit_specs(key, mechanism)


def _normalize_signature(value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("context mechanism signature must have three components")
    multipliers, bridge, context = value
    if not isinstance(multipliers, (list, tuple)):
        raise ValueError("context mechanism multipliers are malformed")
    return tuple(int(item) for item in multipliers), int(bridge), int(context)


def normalized_exclusions(
    values: Iterable[object],
) -> frozenset[tuple[object, ...]]:
    return frozenset(_normalize_signature(value) for value in values)


def sealed_validity_worlds(
    secret: bytes,
    commitment: str,
    world_count: int,
    *,
    excluded_active_signatures: Iterable[object],
) -> tuple[WorldMechanism, ...]:
    key = bytes(secret)
    if len(key) != 32 or sha256(key).hexdigest() != commitment:
        raise ValueError("validity secret does not match its commitment")
    if type(world_count) is not int or world_count <= 0:
        raise ValueError("world_count must be a positive integer")
    excluded = normalized_exclusions(excluded_active_signatures)
    public_count = DEVELOPMENT_WORLDS_PER_FAMILY + VALIDATION_WORLDS_PER_FAMILY
    candidates = tuple(
        candidate
        for candidate in _ranked_active_parameters(PRIMARY_FAMILY)[public_count:]
        if _normalize_signature(candidate) not in excluded
    )
    if world_count > len(candidates):
        raise ValueError("sealed validity world count exceeds fresh support")
    ranked = sorted(
        candidates,
        key=lambda candidate: _hmac_digest(
            key,
            (
                P3_VALIDITY_GENERATOR_ID,
                "sealed-validity-mechanism",
                PRIMARY_FAMILY.value,
                candidate,
            ),
        ),
    )
    worlds: list[WorldMechanism] = []
    for world_index, (multipliers, bridge, context) in enumerate(ranked[:world_count]):
        payload = {
            "generator": P3_VALIDITY_GENERATOR_ID,
            "family": PRIMARY_FAMILY.value,
            "world_index": world_index,
            "multipliers": multipliers,
            "bridge": bridge,
            "context": context,
            "commitment": commitment,
        }
        worlds.append(
            WorldMechanism(
                family=PRIMARY_FAMILY,
                cohort=BenchmarkSplit.SEALED_TEST,
                world_index=world_index,
                layer_multipliers=multipliers,
                bridge_coefficient=bridge,
                context_coefficient=context,
                root_commitment=commitment,
                mechanism_digest=_digest(payload),
            )
        )
    return tuple(worlds)


def sealed_validity_units(
    secret: bytes,
    commitment: str,
    mechanism: WorldMechanism,
) -> tuple[ValidityUnitSpec, ...]:
    key = bytes(secret)
    if len(key) != 32 or sha256(key).hexdigest() != commitment:
        raise ValueError("validity secret does not match its commitment")
    world_key = _hmac_digest(
        key,
        (
            P3_VALIDITY_GENERATOR_ID,
            "sealed-validity-panel",
            mechanism.world_index,
            mechanism.mechanism_digest,
        ),
    )
    return validity_unit_specs(world_key, mechanism)


def validity_manifest(
    worlds: Sequence[WorldMechanism],
    units: Mapping[int, Sequence[ValidityUnitSpec]],
) -> dict[str, object]:
    world_tuple = tuple(worlds)
    if set(units) != {world.world_index for world in world_tuple}:
        raise ValueError("each validity world needs exactly one unit panel")
    payload = {
        "identifier": P3_VALIDITY_GENERATOR_ID,
        "contract_digest": validity_contract_digest(),
        "family": PRIMARY_FAMILY.value,
        "world_count": len(world_tuple),
        "worlds": [world.as_dict() for world in world_tuple],
        "units": {
            str(world.world_index): [
                unit.as_dict() for unit in units[world.world_index]
            ]
            for world in world_tuple
        },
        "outcome_definition_inputs": (
            "predicted_plan_choice, oracle_candidate_utilities"
        ),
        "tsi_metric_used_in_task_generation": False,
    }
    return {**payload, "manifest_digest": _digest(payload)}


def development_validity_manifest() -> dict[str, object]:
    worlds = development_validity_worlds()
    units = {
        world.world_index: development_validity_units(world) for world in worlds
    }
    return validity_manifest(worlds, units)


def audit_validity_generator() -> dict[str, object]:
    errors: list[str] = []
    worlds = development_validity_worlds()
    manifest = development_validity_manifest()
    if len(worlds) != DEVELOPMENT_WORLDS:
        errors.append("development validity world count changed")
    if len({world.active_parameter_signature for world in worlds}) != len(worlds):
        errors.append("development validity mechanisms are not unique")
    task_count = 0
    for world in worlds:
        units = development_validity_units(world)
        repeated = development_validity_units(world)
        if units != repeated:
            errors.append("development validity panel is not deterministic")
        if len(units) != UNITS_PER_WORLD:
            errors.append(f"world {world.world_index} unit count changed")
        strata = {stratum: 0 for stratum in UNIT_STRATA}
        for unit in units:
            strata[unit.stratum] += 1
            if sorted(action.name for action in unit.probe_actions) != sorted(
                PROBE_ACTION_MULTISET
            ):
                errors.append("diagnostic probe action multiset changed")
            task_count += len(unit.tasks)
            for task in unit.tasks:
                endpoints = tuple(
                    _apply_plan(task.start_code, plan, world)
                    for plan in task.candidate_plans
                )
                utilities = tuple(
                    goal_utility(
                        endpoint,
                        task.goal_layers,
                        task.goal_queries,
                        task.goal_values,
                    )
                    for endpoint in endpoints
                )
                if utilities != task.oracle_utilities:
                    errors.append("task oracle utility is not reproducible")
                if utilities[task.oracle_best_index] != max(utilities):
                    errors.append("task oracle best index is incorrect")
                if utilities[0] == utilities[1]:
                    errors.append("task oracle utility gap vanished")
        if any(value != UNITS_PER_STRATUM for value in strata.values()):
            errors.append(f"world {world.world_index} unit strata are imbalanced")
    return {
        "identifier": P3_VALIDITY_GENERATOR_ID,
        "development_manifest_digest": manifest["manifest_digest"],
        "development_world_count": len(worlds),
        "development_unit_count": len(worlds) * UNITS_PER_WORLD,
        "development_task_count": task_count,
        "probe_task_hmac_domains_disjoint": True,
        "task_generation_imports_tsi_metric": False,
        "outcome_definition_uses_tsi_metric": False,
        "sealed_worlds_materialized": 0,
        "errors": errors,
        "passed": not errors,
    }
