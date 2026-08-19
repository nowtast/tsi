"""Paired multi-world development pilot for the P3-3A evidence gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .coherent import CoherentStructuralState, bridge_defects
from .paper3_analysis_plan import analysis_plan_digest
from .paper3_constructive_decoder import (
    ConstructiveStructuralDecoder,
    build_multiworld_feature_layout,
)
from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_interface import (
    FixedCarrierLayerErrors,
    fixed_carrier_exact_losses,
    optimal_correspondence_costs,
)
from .paper3_multiworld import (
    DEVELOPMENT_WORLDS_PER_FAMILY,
    GeneratedTransitionCase,
    MultiworldStateCode,
    build_multiworld_state,
    build_world_dataset,
    build_world_mechanism,
    multiworld_generator_digest,
)
from .paper3_routing_controls import (
    PAIRED_OPTIMIZER_SEEDS_PER_WORLD,
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


P3_DEVELOPMENT_EXPERIMENT_ID = "P3-3A-DEVELOPMENT-PILOT-v1"
DEVELOPMENT_FAMILIES = (
    WorldFamily.SEPARABLE,
    WorldFamily.BRIDGE_COUPLED,
)
OPTIMIZER_SEEDS = tuple(range(PAIRED_OPTIMIZER_SEEDS_PER_WORLD))
DEVELOPMENT_CONTROL_IDS = {
    WorldFamily.SEPARABLE: ("strict_factorized_action",),
    WorldFamily.BRIDGE_COUPLED: (
        "dense_active_matched",
        "layer_routed_dense_action",
        "strict_factorized_action",
        "signature_routed_oracle",
        "random_routed_matched_sparsity",
        "permuted_or_wrong_routed",
    ),
}
LAYER_ERROR_FIELDS = ("label", "simplicial", "metric", "relation", "order")


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


@dataclass(frozen=True)
class SliceMetrics:
    example_count: int
    mean_normalized_i0_quotient_error: float
    fixed_joint_exact_rate: float
    fixed_layer_error_vector: Mapping[str, float]
    bridge_violation_rate: float
    tracking_exact_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "example_count": self.example_count,
            "mean_normalized_i0_quotient_error": (
                self.mean_normalized_i0_quotient_error
            ),
            "fixed_joint_exact_rate": self.fixed_joint_exact_rate,
            "fixed_layer_error_vector": dict(self.fixed_layer_error_vector),
            "bridge_violation_rate": self.bridge_violation_rate,
            "tracking_exact_rate": self.tracking_exact_rate,
        }


class ConstructiveMetricCache:
    """Cache codebook-free decoded states and exact I0 pair evaluations."""

    def __init__(self) -> None:
        self.layout = build_multiworld_feature_layout()
        self.decoder = ConstructiveStructuralDecoder(self.layout)
        self._states: dict[MultiworldStateCode, CoherentStructuralState] = {}
        self._pairs: dict[
            tuple[MultiworldStateCode, MultiworldStateCode],
            tuple[float, FixedCarrierLayerErrors],
        ] = {}
        self._bridge_violations: dict[MultiworldStateCode, bool] = {}

    def state(self, code: MultiworldStateCode) -> CoherentStructuralState:
        if code not in self._states:
            local_state = build_multiworld_state(code)
            self._states[code] = self.decoder.decode_state(
                self.layout.encode(local_state)
            )
        return self._states[code]

    def pair(
        self,
        predicted: MultiworldStateCode,
        target: MultiworldStateCode,
    ) -> tuple[float, FixedCarrierLayerErrors]:
        key = (predicted, target)
        if key not in self._pairs:
            predicted_state = self.state(predicted)
            target_state = self.state(target)
            quotient = optimal_correspondence_costs(
                predicted_state,
                target_state,
            )
            fixed = fixed_carrier_exact_losses(predicted_state, target_state)
            self._pairs[key] = (quotient.total, fixed)
        return self._pairs[key]

    def bridge_violation(self, code: MultiworldStateCode) -> bool:
        if code not in self._bridge_violations:
            state = self.state(code)
            defects = bridge_defects(state.core, state.order, state.signature)
            self._bridge_violations[code] = any(
                value != 0.0 for value in defects.values()
            )
        return self._bridge_violations[code]

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def pair_count(self) -> int:
        return len(self._pairs)


def evaluate_predictions(
    cache: ConstructiveMetricCache,
    cases: Sequence[GeneratedTransitionCase],
    predictions: Sequence[MultiworldStateCode],
) -> SliceMetrics:
    if len(cases) != len(predictions):
        raise ValueError("prediction count must match the case count")
    quotient_errors: list[float] = []
    fixed_joint: list[float] = []
    layer_errors: dict[str, list[float]] = {name: [] for name in LAYER_ERROR_FIELDS}
    bridge_violations: list[float] = []
    tracking_exact: list[float] = []
    for case, prediction in zip(cases, predictions, strict=True):
        quotient, fixed = cache.pair(prediction, case.target_code)
        tracking_is_exact = prediction.label_phase == case.target_code.label_phase
        quotient_errors.append(quotient)
        fixed_joint.append(1.0 if fixed.is_zero and tracking_is_exact else 0.0)
        for name in LAYER_ERROR_FIELDS:
            layer_errors[name].append(float(getattr(fixed, name)))
        bridge_violations.append(1.0 if cache.bridge_violation(prediction) else 0.0)
        tracking_exact.append(1.0 if tracking_is_exact else 0.0)
    return SliceMetrics(
        example_count=len(cases),
        mean_normalized_i0_quotient_error=_mean(quotient_errors),
        fixed_joint_exact_rate=_mean(fixed_joint),
        fixed_layer_error_vector={
            name: _mean(values) for name, values in layer_errors.items()
        },
        bridge_violation_rate=_mean(bridge_violations),
        tracking_exact_rate=_mean(tracking_exact),
    )


def _input_keys(
    cases: Sequence[GeneratedTransitionCase],
) -> tuple[tuple[MultiworldStateCode, tuple[int, ...]], ...]:
    return tuple(case.input_key for case in cases)


def _evaluate_partitions(
    cache: ConstructiveMetricCache,
    model: TrainableRoutingModel,
    dataset_cases: Mapping[str, Sequence[GeneratedTransitionCase]],
    features: Mapping[str, np.ndarray],
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, cases in dataset_cases.items():
        predictions = model.predict_codes_precomputed(cases, features[name])
        metrics[name] = evaluate_predictions(cache, cases, predictions).as_dict()
    return metrics


def run_development_pilot(
    *,
    worlds_per_family: int = DEVELOPMENT_WORLDS_PER_FAMILY,
    optimizer_seeds: Sequence[int] = OPTIMIZER_SEEDS,
    updates: int = TRAINING_UPDATES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run only public development worlds; sealed-test material is unreachable."""

    if type(worlds_per_family) is not int or not (
        1 <= worlds_per_family <= DEVELOPMENT_WORLDS_PER_FAMILY
    ):
        raise ValueError("worlds_per_family must be in the public development range")
    seeds = tuple(optimizer_seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("optimizer seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("optimizer seeds must be unique")
    if type(updates) is not int or updates <= 0:
        raise ValueError("updates must be a positive integer")

    cache = ConstructiveMetricCache()
    runs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for family in DEVELOPMENT_FAMILIES:
        datasets = tuple(
            build_world_dataset(
                build_world_mechanism(
                    family,
                    BenchmarkSplit.DEVELOPMENT,
                    world_index,
                )
            )
            for world_index in range(worlds_per_family)
        )
        template = datasets[0]
        template_cases: dict[str, Sequence[GeneratedTransitionCase]] = {
            "validation": template.partitions["validation"],
            **dict(template.ood_by_slice),
        }
        expected_inputs = {
            name: _input_keys(cases) for name, cases in template_cases.items()
        }
        for dataset in datasets[1:]:
            observed_cases = {
                "validation": dataset.partitions["validation"],
                **dict(dataset.ood_by_slice),
            }
            if {
                name: _input_keys(cases) for name, cases in observed_cases.items()
            } != expected_inputs:
                raise RuntimeError("world input support changed across mechanisms")

        allowed_controls = frozenset(DEVELOPMENT_CONTROL_IDS[family])
        manifests = tuple(
            manifest
            for manifest in routing_control_manifests(family)
            if manifest.identifier in allowed_controls
        )
        if frozenset(manifest.identifier for manifest in manifests) != (
            allowed_controls
        ):
            raise RuntimeError("development control selection is incomplete")
        for manifest in manifests:
            for seed in seeds:
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
                        "family": family.value,
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
                            updates=updates,
                        )
                        current_cases: dict[str, Sequence[GeneratedTransitionCase]] = {
                            "validation": dataset.partitions["validation"],
                            **dict(dataset.ood_by_slice),
                        }
                        metrics = _evaluate_partitions(
                            cache,
                            model,
                            current_cases,
                            evaluation_features,
                        )
                        run = {
                            **run_key,
                            "status": "completed" if trace.finite else "failed",
                            "parameter_count": model.parameter_count,
                            "parameter_digest": model.parameter_digest(),
                            "training": trace.as_dict(),
                            "metrics": metrics,
                        }
                        runs.append(run)
                        if not trace.finite:
                            failures.append(
                                {
                                    **run_key,
                                    "reason": "nonfinite training trace",
                                }
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
                        f"{family.value}/{manifest.identifier}/seed-{seed}: "
                        f"{worlds_per_family} worlds complete"
                    )

    payload: dict[str, object] = {
        "identifier": P3_DEVELOPMENT_EXPERIMENT_ID,
        "cohort": BenchmarkSplit.DEVELOPMENT.value,
        "test_output_used": False,
        "worlds_per_family": worlds_per_family,
        "families": [family.value for family in DEVELOPMENT_FAMILIES],
        "controls_by_family": {
            family.value: list(DEVELOPMENT_CONTROL_IDS[family])
            for family in DEVELOPMENT_FAMILIES
        },
        "optimizer_seeds": list(seeds),
        "updates_per_run": updates,
        "generator_digest": multiworld_generator_digest(),
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


def write_development_pilot(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
