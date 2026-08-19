"""Design contract for the P3-4B downstream predictive-validity gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from statistics import NormalDist

from .paper3_independence_contract import MODEL_CONTROLS, WorldFamily


P3_VALIDITY_CONTRACT_ID = "P3-4B-VALIDITY-v2"
PARENT_P3A_EVIDENCE_DIGEST = (
    "4ffd12cba71a0307332236e86dc197004d0a5c593fc0e3a948a1d89532eefcaf"
)
PRIMARY_FAMILY = WorldFamily.CONTEXT_DEPENDENT
DEVELOPMENT_WORLDS = 24
OPTIMIZER_SEEDS = (0, 1, 2)
UNITS_PER_WORLD = 32
PROBE_HORIZON = 8
TASKS_PER_UNIT = 8
TASK_HORIZONS = (1, 2, 4, 8, 1, 2, 4, 8)
PLAN_CANDIDATE_COUNT = 24
FAMILYWISE_ALPHA = 0.05
POWER_TARGET = 0.90
MINIMUM_TEST_WORLDS = 50
MAXIMUM_TEST_WORLDS = 90
PLANNING_SD_FLOOR = 0.01
PLANNING_SD_INFLATION = 1.25
POWER_ALTERNATIVE_EFFECT = 0.005
POWER_SIMULATION_ITERATIONS = 20_000
PREDICTIVE_SESOI = 0.0025
RIDGE_PENALTY = 0.01
LOGISTIC_MAX_ITERATIONS = 100
LOGISTIC_TOLERANCE = 1.0e-10

PRIMARY_EFFECT_NAMES = (
    "binary_brier_improvement",
    "first_failure_integrated_brier_improvement",
)
PRIMARY_PREDICTIVE_MODELS = (
    "layer_routed_dense_action",
    "strict_factorized_action",
    "random_routed_matched_sparsity",
    "permuted_or_wrong_routed",
)

REQUIRED_ARTIFACTS = (
    "validity_contract_and_digest",
    "noncircular_task_generator_audit",
    "six_control_capacity_and_compute_audit",
    "development_validity_benchmark",
    "frozen_baseline_and_tsi_predictors",
    "development_variance_and_power_report",
    "zero_access_validity_seed_ledger",
    "frozen_confirmatory_analysis_plan",
    "one_shot_execution_lock",
)

NONNEGOTIABLE_POLICIES = (
    "Downstream outcomes are defined only by exogenous structural-predicate goal utility and realized plan regret.",
    "No TSI distance, fixed-layer loss, or predictor output may enter task generation or outcome labels.",
    "Diagnostic probes and downstream tasks use domain-separated HMAC derivations.",
    "The generic baseline includes scalar training loss, scalar latent MSE, endpoint exactness, selected-versus-alternative scalar error contrasts, task margin, task difficulty, and model identity.",
    "The TSI model differs from the generic baseline only by prespecified task-local I0 and fixed-layer discrepancy levels and selected-versus-alternative contrasts.",
    "Discrete hazards use only the current task diagnostics and frozen time indicators.",
    "The four error-producing controls define the primary predictive population; exact signature and dense controls remain reported structural-zero controls.",
    "A layer-aware generic baseline is a mandatory non-gating sensitivity analysis.",
    "Prediction models are fitted on development worlds and frozen before any sealed seed reveal.",
    "No predictor coefficient, scaling parameter, threshold, or feature is refitted on sealed worlds.",
    "World is the independent confirmatory unit; optimizer seeds are nested replicates.",
    "Binary task failure and discrete first-failure time are separate co-primary outcomes.",
    "A failed sealed result creates a new benchmark version and is not repaired in test.",
)


@dataclass(frozen=True)
class ValidityStatisticalPlan:
    independent_unit: str = "world"
    nested_replicate: str = "optimizer_seed"
    binary_endpoint: str = "held_out_any_task_failure_brier_improvement"
    time_endpoint: str = (
        "held_out_discrete_first_failure_integrated_brier_improvement"
    )
    prediction_models: str = "development_fitted_frozen_discrete_hazard"
    multiplicity_rule: str = "one_sided_student_t_holm_two_coprimary_effects"
    cluster_interval: str = "world_cluster_bootstrap_bonferroni_alpha_over_two"
    hierarchical_interval: str = (
        "world_random_intercept_seed_nested_variance_decomposition"
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "independent_unit": self.independent_unit,
            "nested_replicate": self.nested_replicate,
            "binary_endpoint": self.binary_endpoint,
            "time_endpoint": self.time_endpoint,
            "prediction_models": self.prediction_models,
            "multiplicity_rule": self.multiplicity_rule,
            "cluster_interval": self.cluster_interval,
            "hierarchical_interval": self.hierarchical_interval,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "power_target": POWER_TARGET,
            "predictive_sesoi": PREDICTIVE_SESOI,
            "development_worlds": DEVELOPMENT_WORLDS,
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "minimum_test_worlds": MINIMUM_TEST_WORLDS,
            "maximum_test_worlds": MAXIMUM_TEST_WORLDS,
            "planning_sd_floor": PLANNING_SD_FLOOR,
            "planning_sd_inflation": PLANNING_SD_INFLATION,
            "power_alternative_effect": POWER_ALTERNATIVE_EFFECT,
            "power_simulation_iterations": POWER_SIMULATION_ITERATIONS,
            "ridge_penalty": RIDGE_PENALTY,
        }


FROZEN_VALIDITY_STATISTICAL_PLAN = ValidityStatisticalPlan()


def holm_normal_criticals(
    effect_count: int = len(PRIMARY_EFFECT_NAMES),
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
    significance = normal.inv_cdf(
        1.0 - FAMILYWISE_ALPHA / len(PRIMARY_EFFECT_NAMES)
    )
    power = normal.inv_cdf(POWER_TARGET)
    numerator = (significance + power) * maximum_planning_sd
    unconstrained = int((numerator / POWER_ALTERNATIVE_EFFECT) ** 2)
    if unconstrained * POWER_ALTERNATIVE_EFFECT**2 < numerator**2:
        unconstrained += 1
    required = max(MINIMUM_TEST_WORLDS, unconstrained)
    if required > MAXIMUM_TEST_WORLDS:
        raise ValueError("the remaining sealed-world support is underpowered")
    return required


def _contract_payload() -> dict[str, object]:
    return {
        "identifier": P3_VALIDITY_CONTRACT_ID,
        "parent_p3a_evidence_digest": PARENT_P3A_EVIDENCE_DIGEST,
        "evidence_level_before": 3,
        "evidence_level_after": 3,
        "level_4_requirement_before": "1/8",
        "level_4_requirement_after_if_passed": "2/8",
        "primary_family": PRIMARY_FAMILY.value,
        "all_model_controls": [model.identifier for model in MODEL_CONTROLS],
        "primary_predictive_models": list(PRIMARY_PREDICTIVE_MODELS),
        "units_per_world": UNITS_PER_WORLD,
        "probe_horizon": PROBE_HORIZON,
        "tasks_per_unit": TASKS_PER_UNIT,
        "task_horizons": list(TASK_HORIZONS),
        "plan_candidate_count": PLAN_CANDIDATE_COUNT,
        "primary_effect_names": list(PRIMARY_EFFECT_NAMES),
        "statistical_plan": FROZEN_VALIDITY_STATISTICAL_PLAN.as_dict(),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "policies": list(NONNEGOTIABLE_POLICIES),
    }


def validity_contract_digest() -> str:
    return sha256(
        json.dumps(
            _contract_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def audit_validity_contract() -> dict[str, object]:
    errors: list[str] = []
    if len(MODEL_CONTROLS) != 6:
        errors.append("the six frozen routing controls changed")
    if DEVELOPMENT_WORLDS != 24 or len(OPTIMIZER_SEEDS) != 3:
        errors.append("development world/seed design changed")
    if UNITS_PER_WORLD != 32 or PROBE_HORIZON != 8:
        errors.append("diagnostic panel dimensions changed")
    if len(TASK_HORIZONS) != TASKS_PER_UNIT:
        errors.append("task horizon schedule does not match the task count")
    if sorted(set(TASK_HORIZONS)) != [1, 2, 4, 8]:
        errors.append("task horizon support changed")
    if len(PRIMARY_EFFECT_NAMES) != 2:
        errors.append("P3-4B must retain both co-primary predictive effects")
    if len(PRIMARY_PREDICTIVE_MODELS) != 4:
        errors.append("the primary predictive population changed")
    if not 0.0 < PREDICTIVE_SESOI < POWER_ALTERNATIVE_EFFECT:
        errors.append("the predictive SESOI must be positive and conservative")
    if MAXIMUM_TEST_WORLDS > 90:
        errors.append("sealed world budget exceeds fresh mechanism support")
    return {
        "identifier": P3_VALIDITY_CONTRACT_ID,
        "contract_digest": validity_contract_digest(),
        "effect_count": len(PRIMARY_EFFECT_NAMES),
        "model_count": len(MODEL_CONTROLS),
        "errors": errors,
        "passed": not errors,
    }
