"""Validation-only experiment and audit for Paper 3 gate ``P3-2R``."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .paper3_ablation_experiment import (
    AblationSeedResult,
    ExactStatePairCache,
    SummaryStatistic,
    evaluate_ablation_model,
)
from .paper3_objective_ablation import (
    P3AblationDataset,
    P3AblationSpec,
    NumericSplit,
    build_p3_ablation_dataset,
)
from .paper3_representation_repair import (
    DEFAULT_REPAIR_SEEDS,
    P3_REPAIR_BENCHMARK_ID,
    REPAIR_ALLOWED_EVALUATION_SPLITS,
    RepairStructuralJEPA,
    RepairVariant,
)


@dataclass(frozen=True)
class RepairReadinessThresholds:
    """Pre-registered P3-3 readiness thresholds."""

    train_fixed_joint_minimum: float = 0.95
    validation_fixed_joint_minimum: float = 0.25
    validation_quotient_maximum: float = 0.20
    validation_tracking_exact_minimum: float = 0.50

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("readiness thresholds must lie in [0, 1]")
            object.__setattr__(self, item.name, value)

    def as_dict(self) -> dict[str, float]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class RepairSeedResult:
    """Train and validation outcomes for one architecture and seed."""

    variant: str
    seed: int
    total_parameter_count: int
    active_parameter_count: int
    inactive_parameter_count: int
    mask_invariant_passed: bool
    train: AblationSeedResult
    validation: AblationSeedResult

    def scalar_metrics(self) -> dict[str, float]:
        return {
            "train_fixed_joint_exact_rate": self.train.fixed_joint_exact_rate,
            "train_quotient_distance": self.train.mean_quotient_distance,
            "train_tracking_exact_rate": self.train.tracking_exact_rate,
            "validation_fixed_joint_exact_rate": (
                self.validation.fixed_joint_exact_rate
            ),
            "validation_quotient_distance": self.validation.mean_quotient_distance,
            "validation_fixed_total": self.validation.mean_fixed_total,
            "validation_label_error": self.validation.mean_label_error,
            "validation_simplicial_error": self.validation.mean_simplicial_error,
            "validation_metric_error": self.validation.mean_metric_error,
            "validation_relation_error": self.validation.mean_relation_error,
            "validation_order_error": self.validation.mean_order_error,
            "validation_tracking_error": self.validation.mean_tracking_error,
            "validation_tracking_exact_rate": self.validation.tracking_exact_rate,
            "validation_soft_bridge_defect": (self.validation.mean_soft_bridge_defect),
            "validation_soft_validity_defect": (
                self.validation.mean_soft_validity_defect
            ),
            "validation_projection_correction": (
                self.validation.mean_projection_correction
            ),
            "validation_embedding_minimum_variance": (
                self.validation.embedding.minimum_dimension_variance
            ),
            "validation_embedding_effective_rank": (
                self.validation.embedding.effective_rank
            ),
            "validation_embedding_collision_count": float(
                self.validation.embedding.collision_count
            ),
            "validation_post_projection_bridge_violation_rate": (
                self.validation.post_projection_bridge_violation_rate
            ),
            "initial_training_total": self.train.initial_training_total,
            "final_training_total": self.train.final_training_total,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "seed": self.seed,
            "total_parameter_count": self.total_parameter_count,
            "active_parameter_count": self.active_parameter_count,
            "inactive_parameter_count": self.inactive_parameter_count,
            "mask_invariant_passed": self.mask_invariant_passed,
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
        }


@dataclass(frozen=True)
class RepairVariantSummary:
    """Five-seed summary for one cumulative architecture intervention."""

    variant: str
    metrics: Mapping[str, SummaryStatistic]

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "metrics": {
                name: statistic.as_dict() for name, statistic in self.metrics.items()
            },
        }


def summarize_repair_variant(
    variant: RepairVariant,
    runs: Sequence[RepairSeedResult],
) -> RepairVariantSummary:
    """Aggregate seed-level metrics without treating transitions as independent."""

    if not runs or any(run.variant != variant.value for run in runs):
        raise ValueError("repair summary received mismatched runs")
    names = tuple(runs[0].scalar_metrics())
    if any(tuple(run.scalar_metrics()) != names for run in runs[1:]):
        raise ValueError("repair runs expose inconsistent metric ledgers")
    return RepairVariantSummary(
        variant=variant.value,
        metrics=MappingProxyType(
            {
                name: SummaryStatistic.from_values(
                    [run.scalar_metrics()[name] for run in runs]
                )
                for name in names
            }
        ),
    )


def paired_repair_effects(
    candidate: Sequence[RepairSeedResult],
    reference: Sequence[RepairSeedResult],
) -> Mapping[str, SummaryStatistic]:
    """Return paired candidate improvements relative to a reference variant."""

    if tuple(run.seed for run in candidate) != tuple(run.seed for run in reference):
        raise ValueError("repair comparisons require identical seed order")
    return MappingProxyType(
        {
            "validation_fixed_joint_gain": SummaryStatistic.from_values(
                [
                    current.validation.fixed_joint_exact_rate
                    - prior.validation.fixed_joint_exact_rate
                    for current, prior in zip(candidate, reference, strict=True)
                ]
            ),
            "validation_quotient_reduction": SummaryStatistic.from_values(
                [
                    prior.validation.mean_quotient_distance
                    - current.validation.mean_quotient_distance
                    for current, prior in zip(candidate, reference, strict=True)
                ]
            ),
            "validation_tracking_exact_gain": SummaryStatistic.from_values(
                [
                    current.validation.tracking_exact_rate
                    - prior.validation.tracking_exact_rate
                    for current, prior in zip(candidate, reference, strict=True)
                ]
            ),
            "validation_tracking_error_reduction": SummaryStatistic.from_values(
                [
                    prior.validation.mean_tracking_error
                    - current.validation.mean_tracking_error
                    for current, prior in zip(candidate, reference, strict=True)
                ]
            ),
        }
    )


@dataclass(frozen=True)
class ConstrainedGradientAudit:
    """Central-difference audit on the active constrained parameter space."""

    variant: str
    checked_coordinates: int
    maximum_absolute_error: float
    maximum_scaled_error: float
    inactive_gradient_maximum: float
    mask_invariant_passed: bool
    passed: bool

    def as_dict(self) -> dict[str, str | int | float | bool]:
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


def audit_repair_gradients(
    dataset: P3AblationDataset,
    variant: RepairVariant,
    *,
    seed: int = 27_182,
    epsilon: float = 1.0e-6,
    coordinates_per_parameter: int = 2,
) -> ConstrainedGradientAudit:
    """Audit active derivatives and zero tangent gradients of one repair model."""

    if variant is RepairVariant.REFERENCE:
        raise ValueError("the reference gradient is audited by P3-2")
    if epsilon <= 0.0 or not isfinite(epsilon):
        raise ValueError("gradient-audit epsilon must be positive and finite")
    if coordinates_per_parameter <= 0:
        raise ValueError("coordinates_per_parameter must be positive")
    spec = P3AblationSpec(
        seeds=(seed,),
        training_steps=1,
        gradient_clip=1.0e9,
    )
    model = RepairStructuralJEPA(dataset, variant, seed, spec)
    split = _slice_numeric_split(dataset.splits["train"], 7)
    _, gradients, _ = model._losses_and_gradients(split, gradients=True)
    assert gradients is not None
    rng = np.random.default_rng(seed + tuple(RepairVariant).index(variant))
    absolute_errors: list[float] = []
    scaled_errors: list[float] = []
    inactive_maximum = 0.0
    for name, parameter in model.parameters.items():
        mask = model.parameter_masks[name]
        inactive = gradients[name][mask == 0.0]
        if inactive.size:
            inactive_maximum = max(
                inactive_maximum,
                float(np.max(np.abs(inactive))),
            )
        active_flat = np.flatnonzero(mask.reshape(-1) != 0.0)
        selected = rng.choice(
            active_flat,
            size=min(coordinates_per_parameter, active_flat.size),
            replace=False,
        )
        for flat_index in np.atleast_1d(selected):
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
    mask_invariant = model.mask_invariant_holds()
    return ConstrainedGradientAudit(
        variant=variant.value,
        checked_coordinates=len(absolute_errors),
        maximum_absolute_error=maximum_absolute,
        maximum_scaled_error=maximum_scaled,
        inactive_gradient_maximum=inactive_maximum,
        mask_invariant_passed=mask_invariant,
        passed=(
            maximum_absolute <= 2.0e-5
            and maximum_scaled <= 2.0e-5
            and inactive_maximum == 0.0
            and mask_invariant
        ),
    )


def evaluate_repair_model(
    model: RepairStructuralJEPA,
    cache: ExactStatePairCache,
    split_name: str,
) -> AblationSeedResult:
    """Evaluate only a split allowed by the P3-2R data-use contract."""

    if split_name not in REPAIR_ALLOWED_EVALUATION_SPLITS:
        raise ValueError("P3-2R forbids test-transition evaluation")
    allowed_codes = sorted(
        set().union(
            *(
                model.dataset.benchmark.source_codes(split)
                for split in REPAIR_ALLOWED_EVALUATION_SPLITS
            )
        )
    )
    allowed_features = np.stack(
        [
            model.dataset.benchmark.layout.encode(model.dataset.benchmark.states[code])
            for code in allowed_codes
        ]
    )
    return evaluate_ablation_model(
        model,
        cache,
        split_name=split_name,
        embedding_state_features=allowed_features,
    )


def _model_fingerprint(model: RepairStructuralJEPA) -> str:
    digest = sha256()
    digest.update(model.variant.value.encode("ascii"))
    for name in sorted(model.parameters):
        digest.update(name.encode("ascii"))
        digest.update(model.parameters[name].tobytes())
    digest.update(model.target_weight.tobytes())
    digest.update(model.target_bias.tobytes())
    return digest.hexdigest()


def _all_finite(run: RepairSeedResult) -> bool:
    return all(isfinite(value) for value in run.scalar_metrics().values())


def _meets_numeric_thresholds(
    summary: RepairVariantSummary,
    thresholds: RepairReadinessThresholds,
) -> bool:
    metrics = summary.metrics
    return (
        metrics["train_fixed_joint_exact_rate"].mean
        >= thresholds.train_fixed_joint_minimum
        and metrics["validation_fixed_joint_exact_rate"].mean
        >= thresholds.validation_fixed_joint_minimum
        and metrics["validation_quotient_distance"].mean
        <= thresholds.validation_quotient_maximum
        and metrics["validation_tracking_exact_rate"].mean
        >= thresholds.validation_tracking_exact_minimum
    )


def _consistent_improvement(
    candidate: Sequence[RepairSeedResult],
    prior: Sequence[RepairSeedResult],
) -> bool:
    fixed_not_worse = all(
        current.validation.fixed_joint_exact_rate
        >= previous.validation.fixed_joint_exact_rate
        for current, previous in zip(candidate, prior, strict=True)
    )
    quotient_not_worse = all(
        current.validation.mean_quotient_distance
        <= previous.validation.mean_quotient_distance
        for current, previous in zip(candidate, prior, strict=True)
    )
    tracking_not_worse = all(
        current.validation.tracking_exact_rate
        >= previous.validation.tracking_exact_rate
        for current, previous in zip(candidate, prior, strict=True)
    )
    strict_primary = any(
        current.validation.fixed_joint_exact_rate
        > previous.validation.fixed_joint_exact_rate
        or current.validation.mean_quotient_distance
        < previous.validation.mean_quotient_distance
        for current, previous in zip(candidate, prior, strict=True)
    )
    return (
        fixed_not_worse and quotient_not_worse and tracking_not_worse and strict_primary
    )


def _variant_is_ready(
    variant: RepairVariant,
    runs: Mapping[RepairVariant, tuple[RepairSeedResult, ...]],
    summaries: Mapping[RepairVariant, RepairVariantSummary],
    thresholds: RepairReadinessThresholds,
) -> bool:
    variant_runs = runs[variant]
    if not _meets_numeric_thresholds(summaries[variant], thresholds):
        return False
    if any(
        not run.mask_invariant_passed
        or run.validation.embedding.collision_count != 0
        or run.validation.post_projection_bridge_violation_rate != 0.0
        or not _all_finite(run)
        for run in variant_runs
    ):
        return False
    if variant is RepairVariant.REFERENCE:
        return True
    prior = tuple(RepairVariant)[tuple(RepairVariant).index(variant) - 1]
    return _consistent_improvement(variant_runs, runs[prior])


def _experiment_digest(
    dataset: P3AblationDataset,
    spec: P3AblationSpec,
    thresholds: RepairReadinessThresholds,
) -> str:
    active_parameter_counts: dict[str, int] = {}
    parameter_mask_digests: dict[str, str] = {}
    for variant in RepairVariant:
        probe = RepairStructuralJEPA(
            dataset,
            variant,
            spec.seeds[0],
            P3AblationSpec(
                seeds=(spec.seeds[0],),
                training_steps=1,
                learning_rate=spec.learning_rate,
                minimum_learning_rate_fraction=spec.minimum_learning_rate_fraction,
                latent_dimension=spec.latent_dimension,
                ema_momentum=spec.ema_momentum,
                adam_beta1=spec.adam_beta1,
                adam_beta2=spec.adam_beta2,
                adam_epsilon=spec.adam_epsilon,
                gradient_clip=spec.gradient_clip,
            ),
        )
        active_parameter_counts[variant.value] = probe.active_parameter_count
        mask_digest = sha256()
        for name in sorted(probe.parameter_masks):
            mask_digest.update(name.encode("ascii"))
            mask_digest.update(probe.parameter_masks[name].tobytes())
        parameter_mask_digests[variant.value] = mask_digest.hexdigest()
    embedding_codes = sorted(
        set().union(
            *(
                dataset.benchmark.source_codes(split)
                for split in REPAIR_ALLOWED_EVALUATION_SPLITS
            )
        )
    )
    payload = {
        "benchmark_id": P3_REPAIR_BENCHMARK_ID,
        "base_benchmark_digest": dataset.benchmark.digest,
        "spec": spec.as_dict(),
        "thresholds": thresholds.as_dict(),
        "variants": [variant.value for variant in RepairVariant],
        "active_parameter_counts": active_parameter_counts,
        "parameter_mask_digests": parameter_mask_digests,
        "evaluated_splits": list(REPAIR_ALLOWED_EVALUATION_SPLITS),
        "embedding_diagnostic_source_codes": [
            code.as_tuple() for code in embedding_codes
        ],
        "test_transition_evaluations": 0,
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
class P3RepresentationRepairReport:
    """Machine-readable P3-2R validation report."""

    base_benchmark_digest: str
    experiment_digest: str
    spec: P3AblationSpec
    thresholds: RepairReadinessThresholds
    runs: Mapping[str, tuple[RepairSeedResult, ...]]
    summaries: Mapping[str, RepairVariantSummary]
    paired: Mapping[str, Mapping[str, SummaryStatistic]]
    gradient_audits: Mapping[str, ConstrainedGradientAudit]
    deterministic_replay_passed: bool
    selected_variant: str | None
    exact_state_pair_cache_size: int
    audit_errors: tuple[str, ...]
    benchmark_id: str = P3_REPAIR_BENCHMARK_ID
    evaluated_splits: tuple[str, ...] = REPAIR_ALLOWED_EVALUATION_SPLITS
    test_transition_evaluations: int = 0
    embedding_diagnostic_source_count: int = 54
    decoder_candidate_count: int = 81
    gate: str = "P3-2R"
    claim_status: str = "empirical-readiness"

    @property
    def passed(self) -> bool:
        return not self.audit_errors and self.selected_variant is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "claim_status": self.claim_status,
            "benchmark_id": self.benchmark_id,
            "base_benchmark_digest": self.base_benchmark_digest,
            "experiment_digest": self.experiment_digest,
            "spec": self.spec.as_dict(),
            "thresholds": self.thresholds.as_dict(),
            "evaluated_splits": list(self.evaluated_splits),
            "test_transition_evaluations": self.test_transition_evaluations,
            "embedding_diagnostic_source_count": (
                self.embedding_diagnostic_source_count
            ),
            "decoder_candidate_count": self.decoder_candidate_count,
            "runs": {
                variant: [run.as_dict() for run in variant_runs]
                for variant, variant_runs in self.runs.items()
            },
            "summaries": {
                variant: summary.as_dict()
                for variant, summary in self.summaries.items()
            },
            "paired_comparisons": {
                comparison: {
                    name: statistic.as_dict() for name, statistic in effects.items()
                }
                for comparison, effects in self.paired.items()
            },
            "gradient_audits": {
                variant: audit.as_dict()
                for variant, audit in self.gradient_audits.items()
            },
            "deterministic_replay_passed": self.deterministic_replay_passed,
            "selected_variant": self.selected_variant,
            "exact_state_pair_cache_size": self.exact_state_pair_cache_size,
            "audit_errors": list(self.audit_errors),
        }


def _audit_report(
    spec: P3AblationSpec,
    runs: Mapping[RepairVariant, tuple[RepairSeedResult, ...]],
    gradient_audits: Mapping[RepairVariant, ConstrainedGradientAudit],
    deterministic_replay_passed: bool,
    selected_variant: RepairVariant | None,
) -> tuple[str, ...]:
    errors: list[str] = []
    if len(spec.seeds) < 5:
        errors.append("P3-2R requires at least five paired seeds")
    if spec.seeds != DEFAULT_REPAIR_SEEDS:
        errors.append("P3-2R official holdout seed ledger changed")
    if REPAIR_ALLOWED_EVALUATION_SPLITS != ("train", "validation"):
        errors.append("P3-2R evaluation split ledger changed")
    total_counts: set[int] = set()
    active_counts: list[int] = []
    for variant in RepairVariant:
        variant_runs = runs.get(variant, ())
        if tuple(run.seed for run in variant_runs) != spec.seeds:
            errors.append(f"{variant.value} does not use the paired seed ledger")
            continue
        active_counts.append(variant_runs[0].active_parameter_count)
        for run in variant_runs:
            total_counts.add(run.total_parameter_count)
            if run.train.example_count != 108 or run.validation.example_count != 108:
                errors.append(f"{variant.value} has a wrong split size")
                break
            if (
                run.train.update_count != spec.training_steps
                or run.validation.update_count != spec.training_steps
            ):
                errors.append(f"{variant.value} has an unmatched update count")
                break
            if not run.mask_invariant_passed:
                errors.append(f"{variant.value} violates its parameter mask")
                break
            if not _all_finite(run):
                errors.append(f"{variant.value} produced a nonfinite metric")
                break
            if (
                run.train.post_projection_bridge_violation_rate != 0.0
                or run.validation.post_projection_bridge_violation_rate != 0.0
            ):
                errors.append(f"{variant.value} produced an invalid hard decode")
                break
    if len(total_counts) != 1:
        errors.append("repair variants do not share the allocated parameter count")
    if active_counts != sorted(active_counts, reverse=True):
        errors.append("repair active-parameter counts are not monotone")
    if any(not audit.passed for audit in gradient_audits.values()):
        errors.append("a constrained gradient audit failed")
    if not deterministic_replay_passed:
        errors.append("selected repair replay is not deterministic")
    if selected_variant is None:
        errors.append("no repair variant satisfies the readiness criteria")
    return tuple(dict.fromkeys(errors))


def run_p3_representation_repair(
    spec: P3AblationSpec | None = None,
    thresholds: RepairReadinessThresholds | None = None,
) -> P3RepresentationRepairReport:
    """Run cumulative repairs using train and validation transitions only."""

    resolved_spec = spec or P3AblationSpec(seeds=DEFAULT_REPAIR_SEEDS)
    resolved_thresholds = thresholds or RepairReadinessThresholds()
    dataset = build_p3_ablation_dataset()
    cache = ExactStatePairCache(dataset.benchmark)
    run_results: dict[RepairVariant, tuple[RepairSeedResult, ...]] = {}
    fingerprints: dict[tuple[RepairVariant, int], str] = {}
    for variant in RepairVariant:
        variant_runs: list[RepairSeedResult] = []
        for seed in resolved_spec.seeds:
            model = RepairStructuralJEPA(
                dataset,
                variant,
                seed,
                resolved_spec,
            ).fit()
            train = evaluate_repair_model(model, cache, "train")
            validation = evaluate_repair_model(model, cache, "validation")
            variant_runs.append(
                RepairSeedResult(
                    variant=variant.value,
                    seed=seed,
                    total_parameter_count=model.parameter_count,
                    active_parameter_count=model.active_parameter_count,
                    inactive_parameter_count=model.inactive_parameter_count,
                    mask_invariant_passed=model.mask_invariant_holds(),
                    train=train,
                    validation=validation,
                )
            )
            fingerprints[(variant, seed)] = _model_fingerprint(model)
        run_results[variant] = tuple(variant_runs)

    summaries = {
        variant: summarize_repair_variant(variant, run_results[variant])
        for variant in RepairVariant
    }
    paired = {
        "layer_routed_vs_reference": paired_repair_effects(
            run_results[RepairVariant.LAYER_ROUTED],
            run_results[RepairVariant.REFERENCE],
        ),
        "factorized_action_vs_layer_routed": paired_repair_effects(
            run_results[RepairVariant.FACTORIZED_ACTION],
            run_results[RepairVariant.LAYER_ROUTED],
        ),
        "factorized_action_vs_reference": paired_repair_effects(
            run_results[RepairVariant.FACTORIZED_ACTION],
            run_results[RepairVariant.REFERENCE],
        ),
    }
    selected = next(
        (
            variant
            for variant in RepairVariant
            if _variant_is_ready(
                variant,
                run_results,
                summaries,
                resolved_thresholds,
            )
        ),
        None,
    )
    replay_variant = selected or RepairVariant.FACTORIZED_ACTION
    replay = RepairStructuralJEPA(
        dataset,
        replay_variant,
        resolved_spec.seeds[0],
        resolved_spec,
    ).fit()
    deterministic_replay = (
        _model_fingerprint(replay)
        == fingerprints[(replay_variant, resolved_spec.seeds[0])]
    )
    gradient_audits = {
        variant: audit_repair_gradients(dataset, variant)
        for variant in (
            RepairVariant.LAYER_ROUTED,
            RepairVariant.FACTORIZED_ACTION,
        )
    }
    errors = _audit_report(
        resolved_spec,
        run_results,
        gradient_audits,
        deterministic_replay,
        selected,
    )
    return P3RepresentationRepairReport(
        base_benchmark_digest=dataset.benchmark.digest,
        experiment_digest=_experiment_digest(
            dataset,
            resolved_spec,
            resolved_thresholds,
        ),
        spec=resolved_spec,
        thresholds=resolved_thresholds,
        runs=MappingProxyType(
            {variant.value: run_results[variant] for variant in RepairVariant}
        ),
        summaries=MappingProxyType(
            {variant.value: summaries[variant] for variant in RepairVariant}
        ),
        paired=MappingProxyType(
            {name: MappingProxyType(dict(effects)) for name, effects in paired.items()}
        ),
        gradient_audits=MappingProxyType(
            {variant.value: audit for variant, audit in gradient_audits.items()}
        ),
        deterministic_replay_passed=deterministic_replay,
        selected_variant=selected.value if selected is not None else None,
        exact_state_pair_cache_size=cache.pair_count,
        audit_errors=errors,
    )
