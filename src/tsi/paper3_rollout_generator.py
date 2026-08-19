"""Independent context-world and action-sequence generation for P3-4A."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from typing import Mapping, Sequence

from .paper3_independence_contract import BenchmarkSplit
from .paper3_multiworld import (
    DEVELOPMENT_WORLDS_PER_FAMILY,
    VALIDATION_WORLDS_PER_FAMILY,
    MultiworldStateCode,
    PRIMITIVE_ACTIONS,
    StructuredAction,
    WorldMechanism,
    _ranked_active_parameters,
    all_multiworld_state_codes,
    build_world_mechanism,
)
from .paper3_rollout_contract import (
    DEVELOPMENT_WORLDS,
    MAX_HORIZON,
    PRIMARY_FAMILY,
    TRAJECTORIES_PER_WORLD,
    rollout_contract_digest,
)


P3_ROLLOUT_GENERATOR_ID = "P3-4A-CONTEXT-ROLLOUT-GENERATOR-v1"
DEVELOPMENT_TRAJECTORY_ROOT = "tsi:p3-4a:context-rollout-development:2026-07-29:v1"
TRAJECTORY_STRATA = ("unseen_structural_mode", "unseen_recombination")
TRAJECTORIES_PER_STRATUM = TRAJECTORIES_PER_WORLD // len(TRAJECTORY_STRATA)
ACTION_BLOCK_NAMES = (
    "hold",
    "label_step",
    "topology_step",
    "topology_step",
    "metric_step",
    "metric_step",
    "relation_step",
    "order_step",
)
ACTION_BY_NAME = {action.name: action for action in PRIMITIVE_ACTIONS}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: object) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


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


def _initial_candidates(
    stratum: str,
) -> tuple[MultiworldStateCode, ...]:
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
    raise ValueError(f"unknown trajectory stratum: {stratum}")


@dataclass(frozen=True)
class RolloutTrajectorySpec:
    world_index: int
    trajectory_index: int
    stratum: str
    initial_code: MultiworldStateCode
    actions: tuple[StructuredAction, ...]
    trajectory_digest: str

    def __post_init__(self) -> None:
        if type(self.world_index) is not int or self.world_index < 0:
            raise ValueError("world_index must be a nonnegative integer")
        if type(self.trajectory_index) is not int or self.trajectory_index < 0:
            raise ValueError("trajectory_index must be a nonnegative integer")
        if self.stratum not in TRAJECTORY_STRATA:
            raise ValueError("trajectory has an unknown initial-state stratum")
        if len(self.actions) != MAX_HORIZON:
            raise ValueError("trajectory must use the frozen maximum horizon")
        if any(action.name not in ACTION_BY_NAME for action in self.actions):
            raise ValueError("rollout trajectories may use only primitive actions")
        if len(self.trajectory_digest) != 64:
            raise ValueError("trajectory digest must be a SHA-256 hex digest")

    def as_dict(self) -> dict[str, object]:
        return {
            "world_index": self.world_index,
            "trajectory_index": self.trajectory_index,
            "stratum": self.stratum,
            "initial_code": list(self.initial_code.as_tuple()),
            "actions": [action.name for action in self.actions],
            "trajectory_digest": self.trajectory_digest,
        }


def _ranked_candidates(
    key: bytes,
    world_index: int,
    stratum: str,
) -> tuple[MultiworldStateCode, ...]:
    candidates = _initial_candidates(stratum)
    return tuple(
        sorted(
            candidates,
            key=lambda code: hmac.new(
                key,
                _canonical_bytes(
                    (
                        P3_ROLLOUT_GENERATOR_ID,
                        "initial",
                        world_index,
                        stratum,
                        code.as_tuple(),
                    )
                ),
                sha256,
            ).digest(),
        )
    )


def _action_word(
    key: bytes,
    world_index: int,
    trajectory_index: int,
) -> tuple[StructuredAction, ...]:
    actions: list[StructuredAction] = []
    block_count = MAX_HORIZON // len(ACTION_BLOCK_NAMES)
    if block_count * len(ACTION_BLOCK_NAMES) != MAX_HORIZON:
        raise RuntimeError("action block does not tile the rollout horizon")
    for block_index in range(block_count):
        indexed = tuple(enumerate(ACTION_BLOCK_NAMES))
        ordered = sorted(
            indexed,
            key=lambda item: hmac.new(
                key,
                _canonical_bytes(
                    (
                        P3_ROLLOUT_GENERATOR_ID,
                        "action",
                        world_index,
                        trajectory_index,
                        block_index,
                        item,
                    )
                ),
                sha256,
            ).digest(),
        )
        actions.extend(ACTION_BY_NAME[name] for _index, name in ordered)
    return tuple(actions)


def rollout_trajectory_specs(
    key: bytes,
    world_index: int,
) -> tuple[RolloutTrajectorySpec, ...]:
    """Generate a balanced, world-specific trajectory panel from a 32-byte key."""

    if len(key) != 32:
        raise ValueError("trajectory key must contain exactly 32 bytes")
    selected: list[tuple[str, MultiworldStateCode]] = []
    for stratum in TRAJECTORY_STRATA:
        ranked = _ranked_candidates(key, world_index, stratum)
        if len(ranked) < TRAJECTORIES_PER_STRATUM:
            raise RuntimeError("trajectory stratum has insufficient initial states")
        selected.extend((stratum, code) for code in ranked[:TRAJECTORIES_PER_STRATUM])

    specs: list[RolloutTrajectorySpec] = []
    for trajectory_index, (stratum, initial_code) in enumerate(selected):
        actions = _action_word(key, world_index, trajectory_index)
        payload = {
            "generator": P3_ROLLOUT_GENERATOR_ID,
            "world_index": world_index,
            "trajectory_index": trajectory_index,
            "stratum": stratum,
            "initial_code": initial_code.as_tuple(),
            "actions": [action.name for action in actions],
        }
        specs.append(
            RolloutTrajectorySpec(
                world_index=world_index,
                trajectory_index=trajectory_index,
                stratum=stratum,
                initial_code=initial_code,
                actions=actions,
                trajectory_digest=_digest(payload),
            )
        )
    return tuple(specs)


def development_rollout_worlds() -> tuple[WorldMechanism, ...]:
    return tuple(
        build_world_mechanism(
            PRIMARY_FAMILY,
            BenchmarkSplit.DEVELOPMENT,
            world_index,
        )
        for world_index in range(DEVELOPMENT_WORLDS)
    )


def development_rollout_trajectories(
    world_index: int,
) -> tuple[RolloutTrajectorySpec, ...]:
    key = sha256(
        f"{DEVELOPMENT_TRAJECTORY_ROOT}:{world_index}".encode("utf-8")
    ).digest()
    return rollout_trajectory_specs(key, world_index)


def sealed_rollout_worlds(
    secret: bytes,
    commitment: str,
    world_count: int,
) -> tuple[WorldMechanism, ...]:
    """Select fresh context-dependent mechanisms after the rollout reveal."""

    key = bytes(secret)
    if len(key) != 32 or sha256(key).hexdigest() != commitment:
        raise ValueError("rollout secret does not match its commitment")
    if type(world_count) is not int or world_count <= 0:
        raise ValueError("world_count must be a positive integer")
    public_count = DEVELOPMENT_WORLDS_PER_FAMILY + VALIDATION_WORLDS_PER_FAMILY
    candidates = _ranked_active_parameters(PRIMARY_FAMILY)[public_count:]
    if world_count > len(candidates):
        raise ValueError("sealed rollout world count exceeds unused support")
    ranked = sorted(
        candidates,
        key=lambda candidate: hmac.new(
            key,
            _canonical_bytes(
                (
                    P3_ROLLOUT_GENERATOR_ID,
                    "mechanism",
                    PRIMARY_FAMILY.value,
                    candidate,
                )
            ),
            sha256,
        ).digest(),
    )
    worlds: list[WorldMechanism] = []
    for world_index, (multipliers, bridge, context) in enumerate(ranked[:world_count]):
        payload = {
            "generator": P3_ROLLOUT_GENERATOR_ID,
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


def sealed_rollout_trajectories(
    secret: bytes,
    commitment: str,
    world_index: int,
) -> tuple[RolloutTrajectorySpec, ...]:
    key = bytes(secret)
    if len(key) != 32 or sha256(key).hexdigest() != commitment:
        raise ValueError("rollout secret does not match its commitment")
    world_key = hmac.new(
        key,
        _canonical_bytes((P3_ROLLOUT_GENERATOR_ID, "trajectory-world", world_index)),
        sha256,
    ).digest()
    return rollout_trajectory_specs(world_key, world_index)


def rollout_manifest(
    worlds: Sequence[WorldMechanism],
    trajectories: Mapping[int, Sequence[RolloutTrajectorySpec]],
) -> dict[str, object]:
    world_tuple = tuple(worlds)
    if set(trajectories) != {world.world_index for world in world_tuple}:
        raise ValueError("each rollout world needs exactly one trajectory panel")
    payload = {
        "identifier": P3_ROLLOUT_GENERATOR_ID,
        "contract_digest": rollout_contract_digest(),
        "family": PRIMARY_FAMILY.value,
        "world_count": len(world_tuple),
        "worlds": [world.as_dict() for world in world_tuple],
        "trajectories": {
            str(world.world_index): [
                spec.as_dict() for spec in trajectories[world.world_index]
            ]
            for world in world_tuple
        },
    }
    return {**payload, "manifest_digest": _digest(payload)}


def development_rollout_manifest() -> dict[str, object]:
    worlds = development_rollout_worlds()
    trajectories = {
        world.world_index: development_rollout_trajectories(world.world_index)
        for world in worlds
    }
    return rollout_manifest(worlds, trajectories)


def audit_rollout_generator() -> dict[str, object]:
    errors: list[str] = []
    manifest = development_rollout_manifest()
    worlds = development_rollout_worlds()
    if len(worlds) != DEVELOPMENT_WORLDS:
        errors.append("development rollout world count changed")
    if len({world.active_parameter_signature for world in worlds}) != len(worlds):
        errors.append("development active mechanisms are not unique")
    trajectory_count = 0
    for world in worlds:
        specs = development_rollout_trajectories(world.world_index)
        trajectory_count += len(specs)
        if len(specs) != TRAJECTORIES_PER_WORLD:
            errors.append(f"world {world.world_index} trajectory count changed")
        strata = {stratum: 0 for stratum in TRAJECTORY_STRATA}
        initial_codes: set[MultiworldStateCode] = set()
        for spec in specs:
            strata[spec.stratum] += 1
            initial_codes.add(spec.initial_code)
            if spec.initial_code not in _initial_candidates(spec.stratum):
                errors.append("trajectory initial state left its declared stratum")
            for offset in range(0, MAX_HORIZON, len(ACTION_BLOCK_NAMES)):
                observed = sorted(
                    action.name
                    for action in spec.actions[
                        offset : offset + len(ACTION_BLOCK_NAMES)
                    ]
                )
                if observed != sorted(ACTION_BLOCK_NAMES):
                    errors.append("an action block lost its frozen multiset")
        if any(value != TRAJECTORIES_PER_STRATUM for value in strata.values()):
            errors.append(f"world {world.world_index} strata are imbalanced")
        if len(initial_codes) != TRAJECTORIES_PER_WORLD:
            errors.append(f"world {world.world_index} initial states repeat")
    return {
        "identifier": P3_ROLLOUT_GENERATOR_ID,
        "development_manifest_digest": manifest["manifest_digest"],
        "development_world_count": len(worlds),
        "development_trajectory_count": trajectory_count,
        "trajectories_per_world": TRAJECTORIES_PER_WORLD,
        "maximum_horizon": MAX_HORIZON,
        "sealed_worlds_materialized": 0,
        "errors": errors,
        "passed": not errors,
    }
