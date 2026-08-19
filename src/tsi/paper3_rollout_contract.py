"""Frozen design contract for the P3-4A multihorizon rollout gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from statistics import NormalDist

from .paper3_independence_contract import MODEL_CONTROLS, WorldFamily


P3_ROLLOUT_CONTRACT_ID = "P3-4A-ROLLOUT-v1"
PARENT_P3B_EVIDENCE_DIGEST = (
    "57f2f7cadc9c7a38f19e79a52dbfb98aec62b111893314c8d6a3e7dbdba78c54"
)
PRIMARY_FAMILY = WorldFamily.CONTEXT_DEPENDENT
PRIMARY_MODEL = "signature_routed_oracle"
PRIMARY_CONTROLS = (
    "dense_active_matched",
    "random_routed_matched_sparsity",
    "permuted_or_wrong_routed",
)
MAX_HORIZON = 32
REPORT_HORIZONS = (1, 2, 4, 8, 16, 32)
TRAJECTORIES_PER_WORLD = 32
DEVELOPMENT_WORLDS = 24
OPTIMIZER_SEEDS = (0, 1, 2)
FAMILYWISE_ALPHA = 0.05
POWER_TARGET = 0.90
MINIMUM_TEST_WORLDS = 50
MAXIMUM_TEST_WORLDS = 128
PLANNING_SD_FLOOR = 0.10
PLANNING_SD_INFLATION = 1.25
POWER_ALTERNATIVE_EFFECT = 0.05
POWER_SIMULATION_ITERATIONS = 20_000

OPEN_LOOP_AUC_MAXIMUM = 0.10
TERMINAL_I0_MAXIMUM = 0.15
EXPOSURE_GAP_MARGIN = 0.05
TRACKING_ERROR_MAXIMUM = 0.05
LOCAL_LAW_VIOLATION_MAXIMUM = 0.05
DENSE_NONINFERIORITY_MARGIN = 0.05
SMALLEST_ROUTING_EFFECT = 0.05

SUCCESS_EFFECT_NAMES = (
    "absolute_open_loop_auc_margin",
    "terminal_i0_margin",
    "exposure_gap_noninferiority",
    "terminal_tracking_margin",
    "local_law_violation_margin",
    "dense_noninferiority",
    "random_routing_superiority",
    "wrong_routing_superiority",
)
PRIMARY_MODEL_SET = (PRIMARY_MODEL, *PRIMARY_CONTROLS)

REQUIRED_ARTIFACTS = (
    "rollout_contract_and_digest",
    "development_and_sealed_generator_audit",
    "fixed_metric_and_lipschitz_audit",
    "six_control_capacity_and_compute_audit",
    "development_rollout_pilot",
    "development_variance_and_power_report",
    "zero_access_rollout_seed_ledger",
    "frozen_confirmatory_analysis_plan",
    "one_shot_execution_lock",
)

NONNEGOTIABLE_POLICIES = (
    "P3-4A cannot promote evidence beyond level 3 by itself.",
    "The primary family is context-dependent and differs from the P3-3B bridge family.",
    "World is the independent unit and optimizer seeds are nested replicates.",
    "Teacher-forced and open-loop predictions are never pooled.",
    "All 32 rollout steps enter the primary AUC; report horizons cannot be selected post hoc.",
    "Absolute stability and relative routing specificity are conjunctive requirements.",
    "The fixed-carrier recursive bound uses self-conditioned local error and an exact finite Lipschitz constant.",
    "State-coherence bridge validity and dynamic transition-law adherence are reported separately.",
    "The sealed rollout seed is revealed once after all artifacts and sample size are frozen.",
    "A failed sealed result creates a new benchmark version and is not repaired in test.",
)


@dataclass(frozen=True)
class RolloutStatisticalPlan:
    independent_unit: str = "world"
    nested_replicate: str = "optimizer_seed"
    primary_endpoint: str = (
        "mean_over_trajectories_and_steps_1_to_32_open_loop_normalized_i0_error"
    )
    multiplicity_rule: str = "one_sided_student_t_holm_eight_coprimary_effects"
    cluster_interval: str = "world_cluster_bootstrap_bonferroni_alpha_over_eight"
    hierarchical_interval: str = (
        "world_random_intercept_seed_nested_variance_decomposition"
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "independent_unit": self.independent_unit,
            "nested_replicate": self.nested_replicate,
            "primary_endpoint": self.primary_endpoint,
            "multiplicity_rule": self.multiplicity_rule,
            "cluster_interval": self.cluster_interval,
            "hierarchical_interval": self.hierarchical_interval,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "power_target": POWER_TARGET,
            "development_worlds": DEVELOPMENT_WORLDS,
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "minimum_test_worlds": MINIMUM_TEST_WORLDS,
            "maximum_test_worlds": MAXIMUM_TEST_WORLDS,
            "planning_sd_floor": PLANNING_SD_FLOOR,
            "planning_sd_inflation": PLANNING_SD_INFLATION,
            "power_alternative_effect": POWER_ALTERNATIVE_EFFECT,
            "power_simulation_iterations": POWER_SIMULATION_ITERATIONS,
        }


FROZEN_ROLLOUT_STATISTICAL_PLAN = RolloutStatisticalPlan()


def holm_normal_criticals(
    effect_count: int = len(SUCCESS_EFFECT_NAMES),
) -> tuple[float, ...]:
    if type(effect_count) is not int or effect_count <= 0:
        raise ValueError("effect_count must be a positive integer")
    normal = NormalDist()
    return tuple(
        normal.inv_cdf(1.0 - FAMILYWISE_ALPHA / (effect_count - rank))
        for rank in range(effect_count)
    )


def analytic_world_floor(maximum_planning_sd: float) -> int:
    if maximum_planning_sd < 0.0:
        raise ValueError("planning SD must be nonnegative")
    normal = NormalDist()
    significance = normal.inv_cdf(1.0 - FAMILYWISE_ALPHA / len(SUCCESS_EFFECT_NAMES))
    power = normal.inv_cdf(POWER_TARGET)
    unconstrained = int(
        ((significance + power) * maximum_planning_sd / POWER_ALTERNATIVE_EFFECT) ** 2
    )
    if (
        unconstrained * POWER_ALTERNATIVE_EFFECT**2
        < ((significance + power) * maximum_planning_sd) ** 2
    ):
        unconstrained += 1
    required = max(MINIMUM_TEST_WORLDS, unconstrained)
    if required > MAXIMUM_TEST_WORLDS:
        raise ValueError("the maximum sealed-world budget is underpowered for P3-4A")
    return required


def _contract_payload() -> dict[str, object]:
    return {
        "identifier": P3_ROLLOUT_CONTRACT_ID,
        "parent_p3b_evidence_digest": PARENT_P3B_EVIDENCE_DIGEST,
        "evidence_level_before": 3,
        "evidence_level_after": 3,
        "primary_family": PRIMARY_FAMILY.value,
        "primary_model": PRIMARY_MODEL,
        "primary_controls": list(PRIMARY_CONTROLS),
        "all_model_controls": [model.identifier for model in MODEL_CONTROLS],
        "maximum_horizon": MAX_HORIZON,
        "report_horizons": list(REPORT_HORIZONS),
        "trajectories_per_world": TRAJECTORIES_PER_WORLD,
        "success_effect_names": list(SUCCESS_EFFECT_NAMES),
        "thresholds": {
            "open_loop_auc_maximum": OPEN_LOOP_AUC_MAXIMUM,
            "terminal_i0_maximum": TERMINAL_I0_MAXIMUM,
            "exposure_gap_margin": EXPOSURE_GAP_MARGIN,
            "tracking_error_maximum": TRACKING_ERROR_MAXIMUM,
            "local_law_violation_maximum": LOCAL_LAW_VIOLATION_MAXIMUM,
            "dense_noninferiority_margin": DENSE_NONINFERIORITY_MARGIN,
            "smallest_routing_effect": SMALLEST_ROUTING_EFFECT,
        },
        "statistical_plan": FROZEN_ROLLOUT_STATISTICAL_PLAN.as_dict(),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "policies": list(NONNEGOTIABLE_POLICIES),
    }


def rollout_contract_digest() -> str:
    return sha256(
        json.dumps(
            _contract_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def audit_rollout_contract() -> dict[str, object]:
    errors: list[str] = []
    expected_models = {model.identifier for model in MODEL_CONTROLS}
    if not set(PRIMARY_MODEL_SET).issubset(expected_models):
        errors.append("primary rollout models are not a subset of frozen controls")
    if REPORT_HORIZONS[-1] != MAX_HORIZON:
        errors.append("report horizons do not terminate at the maximum horizon")
    if tuple(sorted(set(REPORT_HORIZONS))) != REPORT_HORIZONS:
        errors.append("report horizons must be unique and increasing")
    if len(SUCCESS_EFFECT_NAMES) != 8:
        errors.append("P3-4A must retain all eight co-primary success effects")
    if DEVELOPMENT_WORLDS != 24 or len(OPTIMIZER_SEEDS) != 3:
        errors.append("development world/seed design changed")
    if TRAJECTORIES_PER_WORLD != 32 or MAX_HORIZON != 32:
        errors.append("trajectory count or horizon changed")
    thresholds = (
        OPEN_LOOP_AUC_MAXIMUM,
        TERMINAL_I0_MAXIMUM,
        EXPOSURE_GAP_MARGIN,
        TRACKING_ERROR_MAXIMUM,
        LOCAL_LAW_VIOLATION_MAXIMUM,
        DENSE_NONINFERIORITY_MARGIN,
        SMALLEST_ROUTING_EFFECT,
    )
    if any(value <= 0.0 for value in thresholds):
        errors.append("all rollout thresholds must be strictly positive")
    return {
        "identifier": P3_ROLLOUT_CONTRACT_ID,
        "contract_digest": rollout_contract_digest(),
        "effect_count": len(SUCCESS_EFFECT_NAMES),
        "model_count": len(MODEL_CONTROLS),
        "errors": errors,
        "passed": not errors,
    }
