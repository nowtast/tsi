"""Integrated v2 contract for the P3-5A blocker-remediation ablation.

The v2 contract treats split integrity, graph variation, joint gate learning,
non-saturated endpoints, matched selection, and observation/cardinality as
separate experimental factors.  No v2 result is confirmatory until all factors
and the four-way split are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json


P3_LEARNED_V2_CONTRACT_ID = "P3-5A-LEARNED-v2"
PARENT_V1_CONTRACT_DIGEST = (
    "e44966487dcf22ede16053228c74004efe2e28c402b54b932e1751ceca0156d6"
)

SPLITS = (
    "train",
    "routing_selection",
    "downstream_evaluation",
    "test",
)

GRAPH_VARIANTS = (
    "bridge_topology_to_relation",
    "context_order_to_metric",
    "independent_relation",
    "wrong_direction_negative_control",
)

OBSERVATION_REGIMES = (
    "exact_state",
    "noisy_recovered_structure",
    "pixel_object_observation",
    "held_out_entity_count",
)

MODEL_CONTROLS = (
    "oracle_joint_gate",
    "learned_joint_gate",
    "dense_joint_gate",
    "random_joint_gate",
    "wrong_joint_gate",
    "posthoc_ablation_gate",
)

ABLATION_FACTORS = (
    "nested_split_integrity",
    "graph_randomization",
    "joint_gate_learning",
    "non_saturated_endpoint",
    "matched_selection_budget",
    "observation_path",
    "cardinality_path",
)

ABLATION_ROWS = (
    "v1_fixed_graph_i0_same_selection_eval",
    "a_nested_split_only",
    "b_nested_split_plus_graph_randomization",
    "c_add_joint_gate_learning",
    "d_add_non_saturated_intervention_endpoint",
    "e_add_matched_control_selection",
    "f_add_noise_and_pixel_observation",
    "g_add_held_out_cardinality",
)

TRAIN_ENTITY_COUNT = 3
HELD_OUT_ENTITY_COUNTS = (2, 4)
NOISE_LEVELS = (0.10, 0.25)
DEVELOPMENT_WORLDS = 24
MINIMUM_TEST_WORLDS = 50
MAXIMUM_TEST_WORLDS = 128
OPTIMIZER_SEEDS = (0, 1, 2)
MECHANISM_SLOTS = (0, 1, 2, 3)
FAMILYWISE_ALPHA = 0.05

GRAPH_EDGE_F1_MINIMUM = 0.90
PERMUTATION_STABILITY_MINIMUM = 0.95
INTERVENTION_TARGET_LOGLOSS_SESOI = 0.05
INTERVENTION_REGRET_SESOI = 0.05
NOISY_RELATIVE_DEGRADATION_MAXIMUM = 0.15
CARDINALITY_RELATIVE_DEGRADATION_MAXIMUM = 0.20

PRIMARY_ENDPOINTS = (
    "held_out_graph_edge_f1",
    "held_out_permutation_stability",
    "held_out_intervention_target_logloss",
    "held_out_intervention_regret",
    "held_out_noisy_relative_degradation",
    "held_out_cardinality_relative_degradation",
)

REQUIRED_ARTIFACTS = (
    "v2_contract_and_digest",
    "four_way_split_manifest",
    "graph_randomized_generator_audit",
    "joint_gate_capacity_compute_audit",
    "matched_selection_budget_audit",
    "ablation_matrix_results",
    "noise_and_cardinality_observation_audit",
    "balanced_mechanism_factorial_audit",
    "development_variance_and_power_report",
    "zero_access_v2_seed_ledger",
    "frozen_v2_analysis_plan",
    "one_shot_v2_execution_lock",
)

NONNEGOTIABLE_POLICIES = (
    "Routing selection and downstream evaluation use disjoint data within every world.",
    "Graph variants are sampled independently across worlds and are not fixed to one family.",
    "Graph variants and mechanism slots are factorially crossed and reported by stratum.",
    "Joint gate, dense, random, wrong, and oracle controls use the same selection pipeline.",
    "Posthoc ablation is a named baseline and cannot be reported as joint learned routing.",
    "I0 quotient error is secondary; intervention relation log-loss and regret are primary utility endpoints.",
    "SESOI is frozen before confirmatory data and must be tied to endpoint units.",
    "Observation noise and cardinality changes pass through the actual model input path.",
    "Permutation stability is measured after graph freeze and before test aggregation.",
    "A failed ablation row is retained and cannot be hidden by cumulative averaging.",
    "No v2 sealed execution occurs before every required artifact and power condition passes.",
)


@dataclass(frozen=True)
class V2StatisticalPlan:
    independent_unit: str = "world_graph_variant"
    nested_replicate: str = "optimizer_seed"
    split_order: tuple[str, ...] = SPLITS
    primary_endpoints: tuple[str, ...] = PRIMARY_ENDPOINTS
    multiplicity_rule: str = "hierarchical_gate_then_holm_across_primary_endpoints"
    cluster_interval: str = "world_graph_cluster_bootstrap"
    hierarchical_interval: str = "graph_random_intercept_world_seed_nested_model"

    def as_dict(self) -> dict[str, object]:
        return {
            "independent_unit": self.independent_unit,
            "nested_replicate": self.nested_replicate,
            "split_order": list(self.split_order),
            "primary_endpoints": list(self.primary_endpoints),
            "multiplicity_rule": self.multiplicity_rule,
            "cluster_interval": self.cluster_interval,
            "hierarchical_interval": self.hierarchical_interval,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "development_worlds": DEVELOPMENT_WORLDS,
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "mechanism_slots": list(MECHANISM_SLOTS),
            "minimum_test_worlds": MINIMUM_TEST_WORLDS,
            "maximum_test_worlds": MAXIMUM_TEST_WORLDS,
        }


FROZEN_V2_STATISTICAL_PLAN = V2StatisticalPlan()


def _payload() -> dict[str, object]:
    return {
        "identifier": P3_LEARNED_V2_CONTRACT_ID,
        "parent_v1_contract_digest": PARENT_V1_CONTRACT_DIGEST,
        "evidence_level_before": 3,
        "evidence_level_after": 3,
        "splits": list(SPLITS),
        "graph_variants": list(GRAPH_VARIANTS),
        "observation_regimes": list(OBSERVATION_REGIMES),
        "model_controls": list(MODEL_CONTROLS),
        "ablation_factors": list(ABLATION_FACTORS),
        "ablation_rows": list(ABLATION_ROWS),
        "train_entity_count": TRAIN_ENTITY_COUNT,
        "held_out_entity_counts": list(HELD_OUT_ENTITY_COUNTS),
        "noise_levels": list(NOISE_LEVELS),
        "thresholds": {
            "graph_edge_f1_minimum": GRAPH_EDGE_F1_MINIMUM,
            "permutation_stability_minimum": PERMUTATION_STABILITY_MINIMUM,
            "intervention_target_logloss_sesoi": INTERVENTION_TARGET_LOGLOSS_SESOI,
            "intervention_regret_sesoi": INTERVENTION_REGRET_SESOI,
            "noisy_relative_degradation_maximum": NOISY_RELATIVE_DEGRADATION_MAXIMUM,
            "cardinality_relative_degradation_maximum": CARDINALITY_RELATIVE_DEGRADATION_MAXIMUM,
        },
        "statistical_plan": FROZEN_V2_STATISTICAL_PLAN.as_dict(),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "policies": list(NONNEGOTIABLE_POLICIES),
    }


def learned_v2_contract_digest() -> str:
    return sha256(
        json.dumps(_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def audit_learned_v2_contract() -> dict[str, object]:
    errors: list[str] = []
    if tuple(SPLITS) != tuple(dict.fromkeys(SPLITS)):
        errors.append("v2 splits must be unique")
    if SPLITS[:2] != ("train", "routing_selection"):
        errors.append("routing selection must follow training")
    if "downstream_evaluation" not in SPLITS or "test" not in SPLITS:
        errors.append("v2 requires downstream evaluation and test splits")
    if len(GRAPH_VARIANTS) < 3:
        errors.append("v2 requires multiple graph variants")
    if len(MODEL_CONTROLS) < 5:
        errors.append("v2 controls are incomplete")
    if len(ABLATION_ROWS) != len(ABLATION_FACTORS) + 1:
        errors.append("ablation rows must include v1 plus one row per factor")
    if TRAIN_ENTITY_COUNT in HELD_OUT_ENTITY_COUNTS:
        errors.append("held-out cardinalities must differ from train cardinality")
    if any(endpoint.startswith("held_out_world_normalized_i0") for endpoint in PRIMARY_ENDPOINTS):
        errors.append("I0 quotient cannot be the sole primary utility endpoint")
    if INTERVENTION_TARGET_LOGLOSS_SESOI <= 0.0 or INTERVENTION_REGRET_SESOI <= 0.0:
        errors.append("intervention SESOIs must be positive")
    if len(OPTIMIZER_SEEDS) < 3:
        errors.append("v2 requires at least three nested optimizer seeds")
    if len(MECHANISM_SLOTS) < 2:
        errors.append("v2 requires at least two mechanism slots for factorial crossing")
    return {
        "identifier": P3_LEARNED_V2_CONTRACT_ID,
        "contract_digest": learned_v2_contract_digest(),
        "split_count": len(SPLITS),
        "graph_variant_count": len(GRAPH_VARIANTS),
        "ablation_row_count": len(ABLATION_ROWS),
        "errors": errors,
        "passed": not errors,
    }
