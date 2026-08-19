"""Seeded sealed-test world construction, unavailable to P3-3A audits."""

from __future__ import annotations

from hashlib import sha256
import hmac
import json
from typing import Sequence

from .paper3_analysis_plan import PLANNED_TEST_WORLDS
from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_multiworld import (
    DEVELOPMENT_WORLDS_PER_FAMILY,
    VALIDATION_WORLDS_PER_FAMILY,
    WorldMechanism,
    _ranked_active_parameters,
)


P3_SEALED_WORLD_GENERATOR_ID = "P3-3B-SEALED-WORLDS-v1"


def _canonical(candidate: object) -> bytes:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sealed_world_mechanisms(
    secret: bytes,
    commitment: str,
    *,
    family: WorldFamily = WorldFamily.BRIDGE_COUPLED,
    world_count: int = PLANNED_TEST_WORLDS,
) -> tuple[WorldMechanism, ...]:
    """Select unseen active mechanisms by a committed keyed ranking."""

    key = bytes(secret)
    if len(key) != 32:
        raise ValueError("sealed world secret must contain exactly 32 bytes")
    if sha256(key).hexdigest() != commitment:
        raise ValueError("sealed world secret does not match the commitment")
    if type(world_count) is not int or world_count <= 0:
        raise ValueError("world_count must be a positive integer")
    public_count = DEVELOPMENT_WORLDS_PER_FAMILY + VALIDATION_WORLDS_PER_FAMILY
    candidates = _ranked_active_parameters(family)[public_count:]
    if world_count > len(candidates):
        raise ValueError("sealed world count exceeds unused mechanism supply")
    ranked = sorted(
        candidates,
        key=lambda candidate: hmac.new(
            key,
            (
                f"{P3_SEALED_WORLD_GENERATOR_ID}:{family.value}:".encode("utf-8")
                + _canonical(candidate)
            ),
            sha256,
        ).digest(),
    )
    mechanisms: list[WorldMechanism] = []
    for world_index, candidate in enumerate(ranked[:world_count]):
        multipliers, bridge, context = candidate
        payload = {
            "generator": P3_SEALED_WORLD_GENERATOR_ID,
            "family": family.value,
            "world_index": world_index,
            "multipliers": multipliers,
            "bridge": bridge,
            "context": context,
            "commitment": commitment,
        }
        mechanisms.append(
            WorldMechanism(
                family=family,
                cohort=BenchmarkSplit.SEALED_TEST,
                world_index=world_index,
                layer_multipliers=multipliers,
                bridge_coefficient=bridge,
                context_coefficient=context,
                root_commitment=commitment,
                mechanism_digest=sha256(_canonical(payload)).hexdigest(),
            )
        )
    return tuple(mechanisms)


def sealed_world_manifest_digest(
    mechanisms: Sequence[WorldMechanism],
) -> str:
    payload = {
        "identifier": P3_SEALED_WORLD_GENERATOR_ID,
        "world_count": len(mechanisms),
        "mechanisms": [mechanism.as_dict() for mechanism in mechanisms],
    }
    return sha256(_canonical(payload)).hexdigest()
