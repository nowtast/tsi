"""Verify Paper 3/4 world seeds and exported cases from the revealed root seed."""

from __future__ import annotations

from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np

from .paper34_resolution_benchmark import generate_cases, world_spec
from .paper34_resolution_contract import (
    OOD_CASES_PER_WORLD,
    OOD_NOISE_PROBABILITY,
    SELECTION_CASES_PER_WORLD,
    TRAIN_CASES_PER_WORLD,
    TRAIN_NOISE_PROBABILITY,
)


def derive_world_seed(root_seed: bytes, world_index: int) -> int:
    if world_index < 0:
        raise ValueError("world index must be nonnegative")
    digest = sha256(root_seed + world_index.to_bytes(4, "little")).digest()
    return int.from_bytes(digest[:8], "little")


def _portable_case(case: object) -> list[object]:
    return [
        list(case.source),  # type: ignore[attr-defined]
        list(case.action),  # type: ignore[attr-defined]
        list(case.observed),  # type: ignore[attr-defined]
    ]


def audit_world_derivation(
    portable: Mapping[str, object],
    ledger: Mapping[str, object],
    *,
    maximum_worlds: int | None = None,
) -> dict[str, object]:
    root_seed = bytes.fromhex(str(ledger["root_seed_hex_revealed_after_execution"]))
    commitment = sha256(root_seed).hexdigest()
    expected_commitment = str(portable["root_seed_commitment"])
    if commitment != expected_commitment:
        raise ValueError("revealed root seed does not match portable commitment")

    worlds = portable["worlds"]
    if not isinstance(worlds, Sequence):
        raise TypeError("portable worlds must be a sequence")
    selected = worlds if maximum_worlds is None else worlds[:maximum_worlds]
    failures: list[dict[str, object]] = []
    checks = {
        "seed_matches": 0,
        "graph_matches": 0,
        "families_match": 0,
        "train_cases_match": 0,
        "selection_cases_match": 0,
        "test_cases_match": 0,
    }

    for expected_index, world in enumerate(selected):
        if not isinstance(world, Mapping):
            raise TypeError("portable world must be a mapping")
        world_index = int(world["world_index"])
        if world_index != expected_index:
            failures.append(
                {
                    "world_index": world_index,
                    "reason": "world index is not contiguous",
                    "expected_index": expected_index,
                }
            )
            continue
        seed = derive_world_seed(root_seed, world_index)
        expected_row = world["expected_row"]
        if not isinstance(expected_row, Mapping):
            raise TypeError("expected row must be a mapping")
        seed_matches = seed == int(expected_row["seed"])
        checks["seed_matches"] += int(seed_matches)

        rng = np.random.default_rng(seed)
        spec = world_spec(world_index, rng)
        train = generate_cases(
            spec,
            TRAIN_CASES_PER_WORLD,
            rng,
            composition=False,
            noise_probability=TRAIN_NOISE_PROBABILITY,
        )
        selection = generate_cases(
            spec,
            SELECTION_CASES_PER_WORLD,
            rng,
            composition=False,
            noise_probability=TRAIN_NOISE_PROBABILITY,
        )
        test = generate_cases(
            spec,
            OOD_CASES_PER_WORLD,
            rng,
            composition=True,
            noise_probability=OOD_NOISE_PROBABILITY,
        )

        graph = [spec.graph[0], list(spec.graph[1])]
        graph_matches = graph == world["graph"]
        families_matches = list(spec.families) == world["families"]
        train_matches = [_portable_case(case) for case in train] == world["train"]
        selection_matches = (
            [_portable_case(case) for case in selection] == world["selection"]
        )
        test_matches = [_portable_case(case) for case in test] == world["test"]
        checks["graph_matches"] += int(graph_matches)
        checks["families_match"] += int(families_matches)
        checks["train_cases_match"] += int(train_matches)
        checks["selection_cases_match"] += int(selection_matches)
        checks["test_cases_match"] += int(test_matches)

        world_checks = {
            "seed": seed_matches,
            "graph": graph_matches,
            "families": families_matches,
            "train": train_matches,
            "selection": selection_matches,
            "test": test_matches,
        }
        if not all(world_checks.values()):
            failures.append({"world_index": world_index, "checks": world_checks})

    world_count = len(selected)
    return {
        "status": "paper34_world_seed_and_export_derivation_audit",
        "root_seed_commitment_verified": True,
        "world_seed_derivation": "sha256(root_seed || uint32_le(world_index))[:8]",
        "generator": "frozen Python benchmark implementation",
        "world_count": world_count,
        "checks": checks,
        "failures": failures,
        "passed": not failures
        and all(value == world_count for value in checks.values()),
    }
