"""Paired development and sealed experiments for the P3-4A rollout gate."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .paper3_development_experiment import ConstructiveMetricCache, _input_keys
from .paper3_independence_contract import BenchmarkSplit
from .paper3_multiworld import (
    WorldMechanism,
    build_world_dataset,
    multiworld_generator_digest,
)
from .paper3_rollout_contract import (
    DEVELOPMENT_WORLDS,
    OPTIMIZER_SEEDS,
    PRIMARY_FAMILY,
    TRAJECTORIES_PER_WORLD,
    rollout_contract_digest,
)
from .paper3_rollout_evaluator import (
    evaluate_rollout_model,
    exact_transition_lipschitz_constants,
)
from .paper3_rollout_generator import (
    RolloutTrajectorySpec,
    development_rollout_trajectories,
    development_rollout_worlds,
    rollout_manifest,
)
from .paper3_routing_controls import (
    TRAINING_UPDATES,
    routing_control_digest,
    routing_control_manifests,
)
from .paper3_routing_model import (
    MaskedRandomFeatureBasis,
    TrainableRoutingModel,
    encode_cases,
    routing_model_digest,
)


P3_ROLLOUT_DEVELOPMENT_ID = "P3-4A-ROLLOUT-DEVELOPMENT-v1"
P3_ROLLOUT_SEALED_RAW_ID = "P3-4A-ROLLOUT-SEALED-RAW-v1"


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_rollout_experiment(
    worlds: Sequence[WorldMechanism],
    trajectories: Mapping[int, Sequence[RolloutTrajectorySpec]],
    *,
    identifier: str,
    cohort: BenchmarkSplit,
    test_output_used: bool,
    analysis_plan_digest: str | None,
    optimizer_seeds: Sequence[int] = OPTIMIZER_SEEDS,
    updates: int = TRAINING_UPDATES,
    control_ids: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    world_tuple = tuple(worlds)
    if not world_tuple:
        raise ValueError("rollout experiment requires at least one world")
    if any(
        world.family is not PRIMARY_FAMILY or world.cohort is not cohort
        for world in world_tuple
    ):
        raise ValueError("rollout worlds do not match the requested family/cohort")
    if len({world.active_parameter_signature for world in world_tuple}) != len(
        world_tuple
    ):
        raise ValueError("rollout active mechanisms must be unique")
    if set(trajectories) != {world.world_index for world in world_tuple}:
        raise ValueError("rollout trajectory panels do not match world indices")
    if any(
        len(tuple(trajectories[world.world_index])) != TRAJECTORIES_PER_WORLD
        for world in world_tuple
    ):
        raise ValueError("each world needs the frozen trajectory count")
    seeds = tuple(optimizer_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("optimizer seeds must be nonempty and unique")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("optimizer seeds must be nonnegative integers")
    if type(updates) is not int or updates <= 0:
        raise ValueError("training updates must be positive")

    datasets = {world.world_index: build_world_dataset(world) for world in world_tuple}
    template = datasets[world_tuple[0].world_index]
    expected_train_inputs = _input_keys(template.partitions["train"])
    for dataset in datasets.values():
        if _input_keys(dataset.partitions["train"]) != expected_train_inputs:
            raise RuntimeError("training input support changed across rollout worlds")

    manifests = routing_control_manifests(PRIMARY_FAMILY)
    if control_ids is not None:
        requested = tuple(control_ids)
        manifests = tuple(
            manifest for manifest in manifests if manifest.identifier in requested
        )
        if tuple(manifest.identifier for manifest in manifests) != requested:
            raise ValueError("requested rollout control order is invalid")

    lipschitz = {
        world.world_index: exact_transition_lipschitz_constants(world)
        for world in world_tuple
    }
    cache = ConstructiveMetricCache()
    runs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for manifest in manifests:
        for seed in seeds:
            basis = MaskedRandomFeatureBasis(manifest, seed)
            train_features = basis.transform_cases(template.partitions["train"])[0]
            for world in world_tuple:
                dataset = datasets[world.world_index]
                model = TrainableRoutingModel(manifest, seed)
                model.basis = basis
                run_key = {
                    "family": world.family.value,
                    "cohort": world.cohort.value,
                    "world_index": world.world_index,
                    "world_identifier": world.identifier,
                    "mechanism_digest": world.mechanism_digest,
                    "model": manifest.identifier,
                    "optimizer_seed": seed,
                }
                try:
                    train_deltas = encode_cases(dataset.partitions["train"])[1]
                    trace = model.fit_precomputed(
                        train_features,
                        train_deltas,
                        updates=updates,
                    )
                    metrics = evaluate_rollout_model(
                        cache,
                        model,
                        world,
                        trajectories[world.world_index],
                        lipschitz_constants=lipschitz[world.world_index],
                    )
                    status = "completed" if trace.finite else "failed"
                    runs.append(
                        {
                            **run_key,
                            "status": status,
                            "parameter_count": model.parameter_count,
                            "parameter_digest": model.parameter_digest(),
                            "training": trace.as_dict(),
                            "metrics": metrics,
                        }
                    )
                    if not trace.finite:
                        failures.append(
                            {**run_key, "reason": "nonfinite training trace"}
                        )
                except Exception as error:
                    failures.append({**run_key, "reason": repr(error)})
                    runs.append(
                        {
                            **run_key,
                            "status": "failed",
                            "failure": repr(error),
                        }
                    )
            if progress is not None:
                progress(
                    f"{cohort.value}/{manifest.identifier}/seed-{seed}: "
                    f"{len(world_tuple)} worlds complete"
                )

    manifest = rollout_manifest(world_tuple, trajectories)
    payload: dict[str, object] = {
        "identifier": identifier,
        "cohort": cohort.value,
        "test_output_used": test_output_used,
        "analysis_plan_digest": analysis_plan_digest,
        "contract_digest": rollout_contract_digest(),
        "world_count": len(world_tuple),
        "trajectory_count_per_world": TRAJECTORIES_PER_WORLD,
        "optimizer_seeds": list(seeds),
        "control_ids": [manifest.identifier for manifest in manifests],
        "updates_per_run": updates,
        "rollout_manifest_digest": manifest["manifest_digest"],
        "parent_multiworld_generator_digest": multiworld_generator_digest(),
        "routing_control_digest": routing_control_digest(),
        "routing_model_digest": routing_model_digest(),
        "run_count": len(runs),
        "failure_count": len(failures),
        "failures": failures,
        "constructive_metric_cache": {
            "decoded_state_count": cache.state_count,
            "evaluated_pair_count": cache.pair_count,
            "global_target_state_candidates": 0,
        },
        "runs": runs,
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def run_rollout_development_pilot(
    *,
    world_count: int = DEVELOPMENT_WORLDS,
    optimizer_seeds: Sequence[int] = OPTIMIZER_SEEDS,
    updates: int = TRAINING_UPDATES,
    control_ids: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if type(world_count) is not int or not 1 <= world_count <= DEVELOPMENT_WORLDS:
        raise ValueError("development world_count is outside public support")
    worlds = development_rollout_worlds()[:world_count]
    trajectories = {
        world.world_index: development_rollout_trajectories(world.world_index)
        for world in worlds
    }
    return _run_rollout_experiment(
        worlds,
        trajectories,
        identifier=P3_ROLLOUT_DEVELOPMENT_ID,
        cohort=BenchmarkSplit.DEVELOPMENT,
        test_output_used=False,
        analysis_plan_digest=None,
        optimizer_seeds=optimizer_seeds,
        updates=updates,
        control_ids=control_ids,
        progress=progress,
    )


def run_rollout_sealed_experiment(
    worlds: Sequence[WorldMechanism],
    trajectories: Mapping[int, Sequence[RolloutTrajectorySpec]],
    *,
    analysis_plan_digest: str,
    optimizer_seeds: Sequence[int] = OPTIMIZER_SEEDS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    return _run_rollout_experiment(
        worlds,
        trajectories,
        identifier=P3_ROLLOUT_SEALED_RAW_ID,
        cohort=BenchmarkSplit.SEALED_TEST,
        test_output_used=True,
        analysis_plan_digest=analysis_plan_digest,
        optimizer_seeds=optimizer_seeds,
        updates=TRAINING_UPDATES,
        control_ids=None,
        progress=progress,
    )


def write_rollout_experiment(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
