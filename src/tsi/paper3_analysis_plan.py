"""Frozen P3-3B confirmatory estimand and decision-rule specification."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .paper3_independence_contract import (
    FROZEN_STATISTICAL_PLAN,
    P3_INDEPENDENCE_CONTRACT_ID,
)
from .paper3_routing_controls import P3_ROUTING_CONTROL_ID


P3_ANALYSIS_PLAN_ID = "P3-3B-ANALYSIS-v2"
PRIMARY_MODEL = "signature_routed_oracle"
PRIMARY_CONTROLS = (
    "dense_active_matched",
    "random_routed_matched_sparsity",
    "permuted_or_wrong_routed",
)
PRIMARY_FAMILY = "bridge_coupled"
PRIMARY_OOD_SLICE = "bridge_consistent_shift"
PRIMARY_DECODER = "constructive_valid_primary"
PRIMARY_ENDPOINT = "world_mean_normalized_i0_quotient_error_constructive_valid_primary"
SMALLEST_EFFECT_OF_INTEREST = 0.05
DENSE_NONINFERIORITY_MARGIN = 0.05
FAMILYWISE_ALPHA = 0.05
POWER_TARGET = 0.90
MULTIPLICITY_RULE = "holm_fwer_three_primary_contrasts"
PLANNED_TEST_WORLDS = 50
PRIMARY_PVALUE_METHOD = "one_sided_student_t_on_seed_averaged_world_effects"
SIMULTANEOUS_INTERVAL_METHOD = (
    "world_cluster_bootstrap_bonferroni_alpha_over_3_conservative_for_holm"
)
HIERARCHICAL_METHOD = "world_random_intercept_seed_nested_variance_decomposition"
CONFIRMATORY_ANALYSIS_SEED = "tsi:p3-3b:analysis:2026-07-29:v1"


@dataclass(frozen=True)
class PrimaryContrast:
    control: str
    estimand: str
    direction: str
    minimum_point_effect: float | None
    noninferiority_margin: float | None
    adjusted_interval_rule: str

    def as_dict(self) -> dict[str, object]:
        return {
            "control": self.control,
            "estimand": self.estimand,
            "direction": self.direction,
            "minimum_point_effect": self.minimum_point_effect,
            "noninferiority_margin": self.noninferiority_margin,
            "adjusted_interval_rule": self.adjusted_interval_rule,
        }


PRIMARY_CONTRASTS = (
    PrimaryContrast(
        control="dense_active_matched",
        estimand="error_signature_routed_minus_error_control_by_world",
        direction="upper_bounded_establishes_signature_noninferiority",
        minimum_point_effect=None,
        noninferiority_margin=DENSE_NONINFERIORITY_MARGIN,
        adjusted_interval_rule="holm_adjusted_95pct_upper_bound_lt_margin",
    ),
    PrimaryContrast(
        control="random_routed_matched_sparsity",
        estimand="error_control_minus_error_signature_routed_by_world",
        direction="positive_favors_signature_routed",
        minimum_point_effect=SMALLEST_EFFECT_OF_INTEREST,
        noninferiority_margin=None,
        adjusted_interval_rule="holm_adjusted_95pct_lower_bound_gt_0",
    ),
    PrimaryContrast(
        control="permuted_or_wrong_routed",
        estimand="error_control_minus_error_signature_routed_by_world",
        direction="positive_favors_signature_routed",
        minimum_point_effect=SMALLEST_EFFECT_OF_INTEREST,
        noninferiority_margin=None,
        adjusted_interval_rule="holm_adjusted_95pct_lower_bound_gt_0",
    ),
)


KEY_SECONDARY_ENDPOINTS = (
    "fixed_joint_exact_rate",
    "fixed_layer_error_vector",
    "bridge_violation_rate",
    "tracking_exact_rate",
)


POSITIVE_CONTROL_RULES = (
    "strict_factorized_action mean quotient error must be at most 0.05 and "
    "fixed-joint exact rate at least 0.90 on separable-world unseen action "
    "composition before test reveal",
    "a failed separable positive control blocks P3-3B interpretation",
)


FAILURE_PRESERVATION_RULES = (
    "all six OOD slices are reported separately",
    "all failed runs and restarts remain in the ledger",
    "a failed primary contrast fails the conjunctive primary claim",
    "no architecture, decoder, threshold, sample size, or endpoint changes "
    "are allowed after test reveal",
    "full-codebook results remain oracle upper bounds and cannot rescue the "
    "primary decoder",
)


def analysis_plan_payload() -> dict[str, object]:
    return {
        "identifier": P3_ANALYSIS_PLAN_ID,
        "parent_contract": P3_INDEPENDENCE_CONTRACT_ID,
        "routing_contract": P3_ROUTING_CONTROL_ID,
        "primary_family": PRIMARY_FAMILY,
        "primary_ood_slice": PRIMARY_OOD_SLICE,
        "excluded_from_primary_endpoint": ["bridge_violating_control"],
        "negative_control_role": (
            "structural_specificity_diagnostic_not_prediction_ground_truth"
        ),
        "primary_model": PRIMARY_MODEL,
        "primary_controls": list(PRIMARY_CONTROLS),
        "primary_decoder": PRIMARY_DECODER,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "smallest_effect_of_interest": SMALLEST_EFFECT_OF_INTEREST,
        "dense_noninferiority_margin": DENSE_NONINFERIORITY_MARGIN,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "power_target": POWER_TARGET,
        "multiplicity_rule": MULTIPLICITY_RULE,
        "planned_test_worlds": PLANNED_TEST_WORLDS,
        "primary_pvalue_method": PRIMARY_PVALUE_METHOD,
        "simultaneous_interval_method": SIMULTANEOUS_INTERVAL_METHOD,
        "hierarchical_method": HIERARCHICAL_METHOD,
        "confirmatory_analysis_seed_commitment": sha256(
            CONFIRMATORY_ANALYSIS_SEED.encode("utf-8")
        ).hexdigest(),
        "primary_contrasts": [contrast.as_dict() for contrast in PRIMARY_CONTRASTS],
        "world_seed_aggregation": (
            "average paired optimizer seeds within world, then compare worlds"
        ),
        "uncertainty_methods": [
            "world_cluster_bootstrap",
            "hierarchical_world_seed_model",
        ],
        "discordant_uncertainty_rule": "use_the_more_conservative_decision",
        "key_secondary_endpoints": list(KEY_SECONDARY_ENDPOINTS),
        "positive_control_rules": list(POSITIVE_CONTROL_RULES),
        "failure_preservation_rules": list(FAILURE_PRESERVATION_RULES),
        "test_world_count_status": "frozen_after_p3_3a_power_gate",
    }


def analysis_plan_digest() -> str:
    return sha256(
        json.dumps(
            analysis_plan_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AnalysisPlanAudit:
    analysis_digest: str
    primary_contrast_count: int
    primary_controls_match_contract: bool
    statistical_constants_match_contract: bool
    test_world_count_frozen: bool
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "analysis_digest": self.analysis_digest,
            "primary_contrast_count": self.primary_contrast_count,
            "primary_controls_match_contract": (self.primary_controls_match_contract),
            "statistical_constants_match_contract": (
                self.statistical_constants_match_contract
            ),
            "test_world_count_frozen": self.test_world_count_frozen,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def audit_analysis_plan() -> AnalysisPlanAudit:
    errors: list[str] = []
    controls = tuple(contrast.control for contrast in PRIMARY_CONTRASTS)
    controls_match = controls == PRIMARY_CONTROLS
    if not controls_match:
        errors.append("primary contrasts do not match the frozen controls")
    constants_match = (
        SMALLEST_EFFECT_OF_INTEREST
        == FROZEN_STATISTICAL_PLAN.smallest_effect_of_interest
        and DENSE_NONINFERIORITY_MARGIN
        == FROZEN_STATISTICAL_PLAN.dense_noninferiority_margin
        and FAMILYWISE_ALPHA == FROZEN_STATISTICAL_PLAN.alpha
        and POWER_TARGET == FROZEN_STATISTICAL_PLAN.power
        and MULTIPLICITY_RULE == FROZEN_STATISTICAL_PLAN.multiplicity_rule
    )
    if not constants_match:
        errors.append("analysis constants differ from the P3-3A contract")
    if len(PRIMARY_CONTRASTS) != 3:
        errors.append("exactly three primary contrasts are required")
    test_world_count_frozen = (
        FROZEN_STATISTICAL_PLAN.minimum_test_worlds
        <= PLANNED_TEST_WORLDS
        <= FROZEN_STATISTICAL_PLAN.maximum_test_worlds
    )
    if not test_world_count_frozen:
        errors.append("the planned test world count is outside the frozen range")
    if PRIMARY_OOD_SLICE != "bridge_consistent_shift":
        errors.append("the primary OOD slice changed")
    if PRIMARY_CONTRASTS[0].adjusted_interval_rule != (
        "holm_adjusted_95pct_upper_bound_lt_margin"
    ):
        errors.append("the dense control must use frozen noninferiority")
    if any(
        contrast.adjusted_interval_rule != "holm_adjusted_95pct_lower_bound_gt_0"
        for contrast in PRIMARY_CONTRASTS[1:]
    ):
        errors.append("random and wrong controls must use frozen superiority")

    return AnalysisPlanAudit(
        analysis_digest=analysis_plan_digest(),
        primary_contrast_count=len(PRIMARY_CONTRASTS),
        primary_controls_match_contract=controls_match,
        statistical_constants_match_contract=constants_match,
        test_world_count_frozen=test_world_count_frozen,
        errors=tuple(errors),
    )
