"""One-shot sealed-world model fitting and codebook-free evaluation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .paper3_analysis_plan import PLANNED_TEST_WORLDS, analysis_plan_digest
from .paper3_development_experiment import (
    OPTIMIZER_SEEDS,
    ConstructiveMetricCache,
    _evaluate_partitions,
    _input_keys,
)
from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_multiworld import (
    WorldMechanism,
    build_world_dataset,
    multiworld_generator_digest,
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
from .paper3_sealed_worlds import sealed_world_manifest_digest


P3_CONFIRMATORY_EXPERIMENT_ID = "P3-3B-SEALED-TEST-RAW-v1"


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_confirmatory_experiment(
    mechanisms: Sequence[WorldMechanism],
    commitment: str,
    *,
    p3a_gate_digest: str,
    frozen_artifact_digests: Mapping[str, str],
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Fit all six frozen controls on exactly 50 sealed bridge worlds."""

    worlds = tuple(mechanisms)
    if len(worlds) != PLANNED_TEST_WORLDS:
        raise ValueError("sealed experiment requires the frozen 50 worlds")
    if any(
        mechanism.cohort is not BenchmarkSplit.SEALED_TEST
        or mechanism.family is not WorldFamily.BRIDGE_COUPLED
        or mechanism.root_commitment != commitment
        for mechanism in worlds
    ):
        raise ValueError("sealed mechanisms do not match the frozen cohort")
    if len({world.active_parameter_signature for world in worlds}) != len(worlds):
        raise ValueError("sealed active mechanism signatures must be unique")

    datasets = tuple(build_world_dataset(mechanism) for mechanism in worlds)
    template = datasets[0]
    template_cases = {
        "validation": template.partitions["validation"],
        **dict(template.ood_by_slice),
    }
    expected_inputs = {
        name: _input_keys(cases) for name, cases in template_cases.items()
    }
    for dataset in datasets[1:]:
        observed = {
            "validation": dataset.partitions["validation"],
            **dict(dataset.ood_by_slice),
        }
        if {name: _input_keys(cases) for name, cases in observed.items()} != (
            expected_inputs
        ):
            raise RuntimeError("sealed world input support changed across mechanisms")

    cache = ConstructiveMetricCache()
    runs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    manifests = routing_control_manifests(WorldFamily.BRIDGE_COUPLED)
    for manifest in manifests:
        for seed in OPTIMIZER_SEEDS:
            basis = MaskedRandomFeatureBasis(manifest, seed)
            train_features = basis.transform_cases(template.partitions["train"])[0]
            evaluation_features = {
                name: basis.transform_cases(cases)[0]
                for name, cases in template_cases.items()
            }
            for dataset in datasets:
                mechanism = dataset.mechanism
                model = TrainableRoutingModel(manifest, seed)
                model.basis = basis
                run_key = {
                    "family": mechanism.family.value,
                    "world_index": mechanism.world_index,
                    "world_identifier": mechanism.identifier,
                    "mechanism_digest": mechanism.mechanism_digest,
                    "model": manifest.identifier,
                    "optimizer_seed": seed,
                }
                try:
                    train_deltas = encode_cases(dataset.partitions["train"])[1]
                    trace = model.fit_precomputed(
                        train_features,
                        train_deltas,
                        updates=TRAINING_UPDATES,
                    )
                    current_cases = {
                        "validation": dataset.partitions["validation"],
                        **dict(dataset.ood_by_slice),
                    }
                    metrics = _evaluate_partitions(
                        cache,
                        model,
                        current_cases,
                        evaluation_features,
                    )
                    runs.append(
                        {
                            **run_key,
                            "status": "completed" if trace.finite else "failed",
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
                    failure = {**run_key, "reason": repr(error)}
                    failures.append(failure)
                    runs.append(
                        {
                            **run_key,
                            "status": "failed",
                            "failure": repr(error),
                        }
                    )
            if progress is not None:
                progress(
                    f"{manifest.identifier}/seed-{seed}: "
                    f"{len(worlds)} sealed worlds complete"
                )

    payload: dict[str, object] = {
        "identifier": P3_CONFIRMATORY_EXPERIMENT_ID,
        "cohort": BenchmarkSplit.SEALED_TEST.value,
        "test_output_used": True,
        "p3a_gate_digest": p3a_gate_digest,
        "frozen_artifact_digests": dict(frozen_artifact_digests),
        "commitment": commitment,
        "world_count": len(worlds),
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "updates_per_run": TRAINING_UPDATES,
        "generator_digest": multiworld_generator_digest(),
        "sealed_world_manifest_digest": sealed_world_manifest_digest(worlds),
        "routing_control_digest": routing_control_digest(),
        "routing_model_digest": routing_model_digest(),
        "analysis_plan_digest": analysis_plan_digest(),
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


def write_confirmatory_experiment(
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
