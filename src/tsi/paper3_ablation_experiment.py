"""Exact evaluation, paired statistics, and audit for Paper 3 gate ``P3-2``."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from math import exp, isfinite, sqrt
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .coherent import bridge_defects
from .paper3_interface import (
    FROZEN_PAPER3_INTERFACE,
    FixedCarrierLayerErrors,
    fixed_carrier_exact_losses,
    fixed_carrier_tracking_error,
    optimal_correspondence_costs,
)
from .paper3_objective_ablation import (
    INTERACTION_RESIDUE_TO_SPLIT,
    OBJECTIVE_MASKS,
    P3_ABLATION_BENCHMARK_ID,
    NumericSplit,
    ObjectiveCondition,
    P3AblationBenchmark,
    P3AblationDataset,
    P3AblationSpec,
    TrainableStructuralJEPA,
    build_p3_ablation_dataset,
    decode_ablation_predictions,
    interaction_residue,
)
from .paper3_oracle_benchmark import (
    SPLIT_NAMES,
    SyntheticStateCode,
)


T_CRITICAL_95_DF4 = 2.7764451051977987


@dataclass(frozen=True)
class CachedStateEvaluation:
    """Exact state-only quantities reusable across seeds and conditions."""

    quotient_total: float
    state_isomorphic: bool
    fixed: FixedCarrierLayerErrors


class ExactStatePairCache:
    """Memoize exhaustive I0 correspondence searches by finite state code."""

    def __init__(self, benchmark: P3AblationBenchmark) -> None:
        self.benchmark = benchmark
        self._cache: dict[
            tuple[SyntheticStateCode, SyntheticStateCode],
            CachedStateEvaluation,
        ] = {}

    def evaluate(
        self,
        predicted_code: SyntheticStateCode,
        target_code: SyntheticStateCode,
    ) -> CachedStateEvaluation:
        key = (predicted_code, target_code)
        if key not in self._cache:
            predicted = self.benchmark.states[predicted_code]
            target = self.benchmark.states[target_code]
            quotient = optimal_correspondence_costs(predicted, target)
            fixed = fixed_carrier_exact_losses(predicted, target)
            self._cache[key] = CachedStateEvaluation(
                quotient_total=quotient.total,
                state_isomorphic=quotient.total == 0.0,
                fixed=fixed,
            )
        return self._cache[key]

    @property
    def pair_count(self) -> int:
        return len(self._cache)


@dataclass(frozen=True)
class EmbeddingDiagnostics:
    """Noncollapse diagnostics for learned context representations."""

    minimum_dimension_variance: float
    mean_dimension_variance: float
    covariance_rank: int
    effective_rank: float
    collision_count: int

    def as_dict(self) -> dict[str, int | float]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def embedding_diagnostics(
    model: TrainableStructuralJEPA,
    *,
    state_features: np.ndarray | None = None,
) -> EmbeddingDiagnostics:
    """Measure variance, covariance spectrum, and finite code collisions."""

    features = (
        model.dataset.candidate_features
        if state_features is None
        else np.asarray(state_features, dtype=np.float64)
    )
    if (
        features.ndim != 2
        or features.shape[1] != model.dataset.slices.dimension
        or features.shape[0] < 2
    ):
        raise ValueError(
            "embedding diagnostics require at least two full state-feature rows"
        )
    inputs = model.dataset.standardize(features)
    embeddings = model.context_embeddings(inputs)
    variances = np.var(embeddings, axis=0)
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, embeddings.shape[0] - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    largest = float(np.max(eigenvalues))
    threshold = max(1.0e-10, largest * 1.0e-8)
    positive = eigenvalues[eigenvalues > threshold]
    covariance_rank = int(positive.size)
    if positive.size:
        probabilities = positive / np.sum(positive)
        effective_rank = float(
            exp(-float(np.sum(probabilities * np.log(probabilities))))
        )
    else:
        effective_rank = 0.0

    differences = embeddings[:, None, :] - embeddings[None, :, :]
    squared = np.sum(differences**2, axis=2)
    upper = np.triu_indices(embeddings.shape[0], k=1)
    collision_count = int(np.sum(squared[upper] <= 1.0e-16))
    return EmbeddingDiagnostics(
        minimum_dimension_variance=float(np.min(variances)),
        mean_dimension_variance=float(np.mean(variances)),
        covariance_rank=covariance_rank,
        effective_rank=effective_rank,
        collision_count=collision_count,
    )


@dataclass(frozen=True)
class AblationSeedResult:
    """All primary and diagnostic metrics for one condition/seed run."""

    condition: str
    seed: int
    parameter_count: int
    update_count: int
    example_count: int
    fixed_exact_rate: float
    state_isomorphic_rate: float
    tracking_exact_rate: float
    fixed_joint_exact_rate: float
    mean_quotient_distance: float
    mean_fixed_total: float
    mean_label_error: float
    mean_simplicial_error: float
    mean_metric_error: float
    mean_relation_error: float
    mean_order_error: float
    mean_tracking_error: float
    mean_latent_error: float
    mean_soft_bridge_defect: float
    mean_soft_validity_defect: float
    mean_projection_correction: float
    post_projection_bridge_violation_rate: float
    embedding: EmbeddingDiagnostics
    initial_training_total: float
    final_training_total: float
    final_training_losses: Mapping[str, float]
    model_fingerprint: str

    def scalar_metrics(self) -> dict[str, float]:
        excluded = {
            "condition",
            "seed",
            "parameter_count",
            "update_count",
            "example_count",
            "embedding",
            "final_training_losses",
            "model_fingerprint",
        }
        result = {
            item.name: float(getattr(self, item.name))
            for item in fields(self)
            if item.name not in excluded
        }
        result.update(
            {
                f"embedding_{name}": float(value)
                for name, value in self.embedding.as_dict().items()
            }
        )
        result.update(
            {
                f"final_loss_{name}": float(value)
                for name, value in self.final_training_losses.items()
            }
        )
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
            "update_count": self.update_count,
            "example_count": self.example_count,
            **self.scalar_metrics(),
            "embedding": self.embedding.as_dict(),
            "final_training_losses": dict(self.final_training_losses),
            "model_fingerprint": self.model_fingerprint,
        }


def _model_fingerprint(model: TrainableStructuralJEPA) -> str:
    digest = sha256()
    for name in sorted(model.parameters):
        digest.update(name.encode("ascii"))
        digest.update(model.parameters[name].tobytes())
    digest.update(model.target_weight.tobytes())
    digest.update(model.target_bias.tobytes())
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def evaluate_ablation_model(
    model: TrainableStructuralJEPA,
    cache: ExactStatePairCache,
    *,
    split_name: str = "test",
    embedding_state_features: np.ndarray | None = None,
) -> AblationSeedResult:
    """Evaluate one fitted model with exact and pre-projection diagnostics."""

    if model.final_snapshot is None or model.initial_snapshot is None:
        raise RuntimeError("the model must be fitted before evaluation")
    split = model.dataset.splits[split_name]
    predictions = decode_ablation_predictions(model, split_name)
    forward = model.forward_split(split)
    target_latent = model.target_embedding(split.target_inputs)
    latent_errors = np.sqrt(
        np.mean((forward.predicted_latent - target_latent) ** 2, axis=1)
    )

    fixed_exact: list[float] = []
    state_isomorphic: list[float] = []
    tracking_exact: list[float] = []
    fixed_joint: list[float] = []
    quotient_errors: list[float] = []
    fixed_totals: list[float] = []
    label_errors: list[float] = []
    simplicial_errors: list[float] = []
    metric_errors: list[float] = []
    relation_errors: list[float] = []
    order_errors: list[float] = []
    tracking_errors: list[float] = []
    bridge_violations: list[float] = []
    for case, prediction in zip(split.cases, predictions, strict=True):
        state_result = cache.evaluate(
            prediction.target_code,
            case.target_code,
        )
        tracking_error = fixed_carrier_tracking_error(
            prediction.tracking,
            case.example.tracking,
        )
        exact_state = state_result.fixed.is_zero
        fixed_exact.append(1.0 if exact_state else 0.0)
        state_isomorphic.append(1.0 if state_result.state_isomorphic else 0.0)
        tracking_exact.append(1.0 if tracking_error == 0.0 else 0.0)
        fixed_joint.append(1.0 if exact_state and tracking_error == 0.0 else 0.0)
        quotient_errors.append(state_result.quotient_total)
        fixed_totals.append(state_result.fixed.total)
        label_errors.append(state_result.fixed.label)
        simplicial_errors.append(state_result.fixed.simplicial)
        metric_errors.append(state_result.fixed.metric)
        relation_errors.append(state_result.fixed.relation)
        order_errors.append(state_result.fixed.order)
        tracking_errors.append(tracking_error)
        defects = bridge_defects(
            prediction.target.core,
            prediction.target.order,
            prediction.target.signature,
        )
        bridge_violations.append(
            1.0 if any(value != 0.0 for value in defects.values()) else 0.0
        )

    return AblationSeedResult(
        condition=model.condition.value,
        seed=model.seed,
        parameter_count=model.parameter_count,
        update_count=model.update_count,
        example_count=len(split.cases),
        fixed_exact_rate=_mean(fixed_exact),
        state_isomorphic_rate=_mean(state_isomorphic),
        tracking_exact_rate=_mean(tracking_exact),
        fixed_joint_exact_rate=_mean(fixed_joint),
        mean_quotient_distance=_mean(quotient_errors),
        mean_fixed_total=_mean(fixed_totals),
        mean_label_error=_mean(label_errors),
        mean_simplicial_error=_mean(simplicial_errors),
        mean_metric_error=_mean(metric_errors),
        mean_relation_error=_mean(relation_errors),
        mean_order_error=_mean(order_errors),
        mean_tracking_error=_mean(tracking_errors),
        mean_latent_error=float(np.mean(latent_errors)),
        mean_soft_bridge_defect=_mean(
            [prediction.soft_bridge_defect for prediction in predictions]
        ),
        mean_soft_validity_defect=_mean(
            [prediction.soft_validity_defect for prediction in predictions]
        ),
        mean_projection_correction=_mean(
            [prediction.projection_correction for prediction in predictions]
        ),
        post_projection_bridge_violation_rate=_mean(bridge_violations),
        embedding=embedding_diagnostics(
            model,
            state_features=embedding_state_features,
        ),
        initial_training_total=model.initial_snapshot.total,
        final_training_total=model.final_snapshot.total,
        final_training_losses=model.final_snapshot.losses,
        model_fingerprint=_model_fingerprint(model),
    )


@dataclass(frozen=True)
class SummaryStatistic:
    """Mean, sample spread, and exploratory two-sided 95% t interval."""

    mean: float
    sample_standard_deviation: float
    confidence_95_low: float
    confidence_95_high: float
    values: tuple[float, ...]

    @classmethod
    def from_values(
        cls,
        values: Sequence[float],
    ) -> SummaryStatistic:
        normalized = tuple(float(value) for value in values)
        if not normalized or any(not isfinite(value) for value in normalized):
            raise ValueError("summary values must be nonempty and finite")
        mean = _mean(normalized)
        if len(normalized) == 1:
            standard_deviation = 0.0
            half_width = 0.0
        else:
            standard_deviation = float(np.std(normalized, ddof=1))
            critical = T_CRITICAL_95_DF4 if len(normalized) == 5 else 1.96
            half_width = critical * standard_deviation / sqrt(len(normalized))
        return cls(
            mean=mean,
            sample_standard_deviation=standard_deviation,
            confidence_95_low=mean - half_width,
            confidence_95_high=mean + half_width,
            values=normalized,
        )

    @property
    def excludes_zero(self) -> bool:
        return self.confidence_95_low > 0.0 or self.confidence_95_high < 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "sample_standard_deviation": self.sample_standard_deviation,
            "confidence_95": [
                self.confidence_95_low,
                self.confidence_95_high,
            ],
            "values": list(self.values),
            "excludes_zero": self.excludes_zero,
        }


@dataclass(frozen=True)
class ConditionSummary:
    """Across-seed summary for one matched objective condition."""

    condition: str
    metrics: Mapping[str, SummaryStatistic]

    def as_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "metrics": {
                name: statistic.as_dict() for name, statistic in self.metrics.items()
            },
        }


def summarize_condition(
    condition: ObjectiveCondition,
    runs: Sequence[AblationSeedResult],
) -> ConditionSummary:
    """Aggregate identically ordered seed runs for one condition."""

    if not runs or any(run.condition != condition.value for run in runs):
        raise ValueError("condition summary received mismatched runs")
    names = tuple(runs[0].scalar_metrics())
    if any(tuple(run.scalar_metrics()) != names for run in runs[1:]):
        raise ValueError("seed runs expose inconsistent metric ledgers")
    return ConditionSummary(
        condition=condition.value,
        metrics=MappingProxyType(
            {
                name: SummaryStatistic.from_values(
                    [run.scalar_metrics()[name] for run in runs]
                )
                for name in names
            }
        ),
    )


_RELEVANT_ERROR_METRIC = MappingProxyType(
    {
        ObjectiveCondition.JEPA_ONLY: "mean_fixed_total",
        ObjectiveCondition.NO_TOPOLOGY: "mean_simplicial_error",
        ObjectiveCondition.NO_METRIC: "mean_metric_error",
        ObjectiveCondition.NO_RELATION: "mean_relation_error",
        ObjectiveCondition.NO_ORDER: "mean_order_error",
        ObjectiveCondition.NO_BRIDGE: "mean_soft_bridge_defect",
        ObjectiveCondition.NO_TRACKING: "mean_tracking_error",
    }
)


def paired_comparisons(
    runs: Mapping[ObjectiveCondition, tuple[AblationSeedResult, ...]],
) -> Mapping[str, Mapping[str, SummaryStatistic]]:
    """Return paired full-minus-ablation effects with consistent directions."""

    full_runs = runs[ObjectiveCondition.FULL]
    comparisons: dict[str, Mapping[str, SummaryStatistic]] = {}
    for condition in ObjectiveCondition:
        if condition is ObjectiveCondition.FULL:
            continue
        condition_runs = runs[condition]
        if tuple(run.seed for run in condition_runs) != tuple(
            run.seed for run in full_runs
        ):
            raise ValueError("paired comparisons require identical seed order")
        relevant = _RELEVANT_ERROR_METRIC[condition]
        effects = {
            "fixed_joint_exact_gain": SummaryStatistic.from_values(
                [
                    full.fixed_joint_exact_rate - ablated.fixed_joint_exact_rate
                    for full, ablated in zip(
                        full_runs,
                        condition_runs,
                        strict=True,
                    )
                ]
            ),
            "quotient_distance_reduction": SummaryStatistic.from_values(
                [
                    ablated.mean_quotient_distance - full.mean_quotient_distance
                    for full, ablated in zip(
                        full_runs,
                        condition_runs,
                        strict=True,
                    )
                ]
            ),
            "tracking_error_reduction": SummaryStatistic.from_values(
                [
                    ablated.mean_tracking_error - full.mean_tracking_error
                    for full, ablated in zip(
                        full_runs,
                        condition_runs,
                        strict=True,
                    )
                ]
            ),
            "relevant_error_reduction": SummaryStatistic.from_values(
                [
                    ablated.scalar_metrics()[relevant] - full.scalar_metrics()[relevant]
                    for full, ablated in zip(
                        full_runs,
                        condition_runs,
                        strict=True,
                    )
                ]
            ),
        }
        comparisons[condition.value] = MappingProxyType(effects)
    return MappingProxyType(comparisons)


@dataclass(frozen=True)
class GradientAudit:
    """Finite-difference check of the hand-derived reverse-mode gradients."""

    checked_coordinates: int
    maximum_absolute_error: float
    maximum_scaled_error: float
    passed: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _slice_numeric_split(split: NumericSplit, count: int) -> NumericSplit:
    return NumericSplit(
        cases=split.cases[:count],
        source_inputs=split.source_inputs[:count],
        target_inputs=split.target_inputs[:count],
        actions=split.actions[:count],
        target_features=split.target_features[:count],
        tracking_targets=split.tracking_targets[:count],
    )


def audit_manual_gradients(
    dataset: P3AblationDataset,
    *,
    seed: int = 31_415,
    epsilon: float = 1.0e-6,
    coordinates_per_parameter: int = 2,
) -> GradientAudit:
    """Compare every parameter family's analytic gradient to central differences."""

    if epsilon <= 0.0 or not isfinite(epsilon):
        raise ValueError("gradient-audit epsilon must be positive and finite")
    if coordinates_per_parameter <= 0:
        raise ValueError("coordinates_per_parameter must be positive")
    spec = P3AblationSpec(
        seeds=(seed,),
        training_steps=1,
        gradient_clip=1.0e9,
    )
    model = TrainableStructuralJEPA(
        dataset,
        ObjectiveCondition.FULL,
        seed,
        spec,
    )
    split = _slice_numeric_split(dataset.splits["train"], 7)
    _, gradients, _ = model._losses_and_gradients(split, gradients=True)
    assert gradients is not None
    rng = np.random.default_rng(seed + 1)
    absolute_errors: list[float] = []
    scaled_errors: list[float] = []
    for name, parameter in model.parameters.items():
        flat_indices = rng.choice(
            parameter.size,
            size=min(coordinates_per_parameter, parameter.size),
            replace=False,
        )
        for flat_index in np.atleast_1d(flat_indices):
            index = np.unravel_index(int(flat_index), parameter.shape)
            original = float(parameter[index])
            parameter[index] = original + epsilon
            plus = model._losses_and_gradients(split, gradients=False)[0].total
            parameter[index] = original - epsilon
            minus = model._losses_and_gradients(split, gradients=False)[0].total
            parameter[index] = original
            numerical = (plus - minus) / (2.0 * epsilon)
            analytic = float(gradients[name][index])
            absolute = abs(numerical - analytic)
            scaled = absolute / (1.0 + abs(numerical) + abs(analytic))
            absolute_errors.append(absolute)
            scaled_errors.append(scaled)
    maximum_absolute = max(absolute_errors, default=float("inf"))
    maximum_scaled = max(scaled_errors, default=float("inf"))
    return GradientAudit(
        checked_coordinates=len(absolute_errors),
        maximum_absolute_error=maximum_absolute,
        maximum_scaled_error=maximum_scaled,
        passed=maximum_absolute <= 2.0e-5 and maximum_scaled <= 2.0e-5,
    )


def _experiment_digest(
    benchmark: P3AblationBenchmark,
    spec: P3AblationSpec,
) -> str:
    payload = {
        "benchmark_digest": benchmark.digest,
        "spec": spec.as_dict(),
        "conditions": {
            condition.value: OBJECTIVE_MASKS[condition].as_dict()
            for condition in ObjectiveCondition
        },
    }
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class P3ObjectiveAblationReport:
    """Machine-readable P3-2 result, paired statistics, and audit."""

    interface_id: str
    benchmark_id: str
    benchmark_digest: str
    experiment_digest: str
    spec: P3AblationSpec
    split_source_counts: Mapping[str, int]
    split_transition_counts: Mapping[str, int]
    parameter_count: int
    runs: Mapping[str, tuple[AblationSeedResult, ...]]
    summaries: Mapping[str, ConditionSummary]
    paired: Mapping[str, Mapping[str, SummaryStatistic]]
    gradient_audit: GradientAudit
    deterministic_replay_passed: bool
    exact_state_pair_cache_size: int
    audit_errors: tuple[str, ...]
    gate: str = "P3-2"
    claim_status: str = "empirical"

    @property
    def passed(self) -> bool:
        return not self.audit_errors

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "claim_status": self.claim_status,
            "interface_id": self.interface_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_digest": self.benchmark_digest,
            "experiment_digest": self.experiment_digest,
            "spec": self.spec.as_dict(),
            "split_source_counts": dict(self.split_source_counts),
            "split_transition_counts": dict(self.split_transition_counts),
            "parameter_count": self.parameter_count,
            "objective_masks": {
                condition.value: OBJECTIVE_MASKS[condition].as_dict()
                for condition in ObjectiveCondition
            },
            "runs": {
                condition: [run.as_dict() for run in condition_runs]
                for condition, condition_runs in self.runs.items()
            },
            "summaries": {
                condition: summary.as_dict()
                for condition, summary in self.summaries.items()
            },
            "paired_comparisons": {
                condition: {
                    name: statistic.as_dict() for name, statistic in effects.items()
                }
                for condition, effects in self.paired.items()
            },
            "gradient_audit": self.gradient_audit.as_dict(),
            "deterministic_replay_passed": self.deterministic_replay_passed,
            "exact_state_pair_cache_size": self.exact_state_pair_cache_size,
            "audit_errors": list(self.audit_errors),
        }


def _all_finite(run: AblationSeedResult) -> bool:
    return all(isfinite(value) for value in run.scalar_metrics().values())


def _audit_report_inputs(
    dataset: P3AblationDataset,
    spec: P3AblationSpec,
    runs: Mapping[ObjectiveCondition, tuple[AblationSeedResult, ...]],
    gradient_audit: GradientAudit,
    deterministic_replay_passed: bool,
) -> tuple[str, ...]:
    errors: list[str] = []
    benchmark = dataset.benchmark
    if tuple(OBJECTIVE_MASKS) != tuple(ObjectiveCondition):
        errors.append("objective-mask ledger does not match the frozen order")
    if len(spec.seeds) < 5:
        errors.append("P3-2 requires at least five paired seeds")
    if benchmark.state_count != 81 or benchmark.transition_count != 324:
        errors.append("P3-2 benchmark must contain 81 states and 324 transitions")
    source_sets = {split: benchmark.source_codes(split) for split in SPLIT_NAMES}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if source_sets[left] & source_sets[right]:
                errors.append(f"source leakage between {left} and {right}")
    for split in SPLIT_NAMES:
        expected_residue = {
            residue
            for residue, name in INTERACTION_RESIDUE_TO_SPLIT.items()
            if name == split
        }
        residues = {interaction_residue(code) for code in source_sets[split]}
        if residues != expected_residue:
            errors.append(f"{split} violates the interaction-residue split")
        if len(source_sets[split]) != 27:
            errors.append(f"{split} must contain exactly 27 source states")
    parameter_counts = {
        run.parameter_count
        for condition_runs in runs.values()
        for run in condition_runs
    }
    if len(parameter_counts) != 1:
        errors.append("objective conditions do not have matched parameter counts")
    for condition in ObjectiveCondition:
        condition_runs = runs.get(condition, ())
        if tuple(run.seed for run in condition_runs) != spec.seeds:
            errors.append(f"{condition.value} does not use the paired seed ledger")
        for run in condition_runs:
            if run.update_count != spec.training_steps:
                errors.append(f"{condition.value} has an unmatched update count")
                break
            if not _all_finite(run):
                errors.append(f"{condition.value} produced a nonfinite metric")
                break
            if run.post_projection_bridge_violation_rate != 0.0:
                errors.append(f"{condition.value} produced an invalid hard decode")
                break
    if not gradient_audit.passed:
        errors.append("manual reverse-mode gradients failed finite differences")
    if not deterministic_replay_passed:
        errors.append("identical seed/config replay is not deterministic")

    if len(spec.seeds) >= 5:
        full = runs[ObjectiveCondition.FULL]
        jepa_only = runs[ObjectiveCondition.JEPA_ONLY]
        no_tracking = runs[ObjectiveCondition.NO_TRACKING]
        if _mean([run.fixed_joint_exact_rate for run in full]) <= _mean(
            [run.fixed_joint_exact_rate for run in jepa_only]
        ):
            errors.append("full objective fails the JEPA-only positive control")
        if _mean([run.mean_tracking_error for run in full]) >= _mean(
            [run.mean_tracking_error for run in no_tracking]
        ):
            errors.append("tracking objective fails its positive control")
    return tuple(dict.fromkeys(errors))


def run_p3_objective_ablation(
    spec: P3AblationSpec | None = None,
) -> P3ObjectiveAblationReport:
    """Train all matched conditions, evaluate exact outputs, and audit P3-2."""

    resolved = spec or P3AblationSpec()
    dataset = build_p3_ablation_dataset()
    cache = ExactStatePairCache(dataset.benchmark)
    run_results: dict[
        ObjectiveCondition,
        tuple[AblationSeedResult, ...],
    ] = {}
    for condition in ObjectiveCondition:
        condition_results: list[AblationSeedResult] = []
        for seed in resolved.seeds:
            model = TrainableStructuralJEPA(
                dataset,
                condition,
                seed,
                resolved,
            ).fit()
            condition_results.append(evaluate_ablation_model(model, cache))
        run_results[condition] = tuple(condition_results)

    replay_model = TrainableStructuralJEPA(
        dataset,
        ObjectiveCondition.FULL,
        resolved.seeds[0],
        resolved,
    ).fit()
    deterministic_replay = (
        _model_fingerprint(replay_model)
        == run_results[ObjectiveCondition.FULL][0].model_fingerprint
    )
    gradient_audit = audit_manual_gradients(dataset)
    summaries = {
        condition: summarize_condition(condition, run_results[condition])
        for condition in ObjectiveCondition
    }
    paired = paired_comparisons(run_results)
    errors = _audit_report_inputs(
        dataset,
        resolved,
        run_results,
        gradient_audit,
        deterministic_replay,
    )
    parameter_count = run_results[ObjectiveCondition.FULL][0].parameter_count
    return P3ObjectiveAblationReport(
        interface_id=FROZEN_PAPER3_INTERFACE.identifier,
        benchmark_id=P3_ABLATION_BENCHMARK_ID,
        benchmark_digest=dataset.benchmark.digest,
        experiment_digest=_experiment_digest(dataset.benchmark, resolved),
        spec=resolved,
        split_source_counts={
            split: len(dataset.benchmark.source_codes(split)) for split in SPLIT_NAMES
        },
        split_transition_counts={
            split: len(dataset.benchmark.splits[split]) for split in SPLIT_NAMES
        },
        parameter_count=parameter_count,
        runs=MappingProxyType(
            {
                condition.value: run_results[condition]
                for condition in ObjectiveCondition
            }
        ),
        summaries=MappingProxyType(
            {condition.value: summaries[condition] for condition in ObjectiveCondition}
        ),
        paired=MappingProxyType(
            {
                condition: MappingProxyType(dict(effects))
                for condition, effects in paired.items()
            }
        ),
        gradient_audit=gradient_audit,
        deterministic_replay_passed=deterministic_replay,
        exact_state_pair_cache_size=cache.pair_count,
        audit_errors=errors,
    )
