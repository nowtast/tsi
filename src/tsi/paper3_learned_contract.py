"""Frozen design contract for the P3-5A learned-structure evidence gate.

P3-5A is deliberately narrower than a deployment claim.  It tests whether a
model can infer the declared structural routing from observations, and whether
that inferred structure remains useful under pre-specified observation noise
and held-out entity counts.  Oracle routing is an upper-bound control and is
never pooled with learned results.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from statistics import NormalDist


P3_LEARNED_CONTRACT_ID = "P3-5A-LEARNED-v1"
PARENT_P3_4B_EVIDENCE_DIGEST = (
    "0d5fc30d15c38e352470134889142ad42511318c8f22a30baae8a95416dbea32"
)

REGIMES = (
    "exact_state_oracle_routing",
    "exact_state_learned_routing",
    "noisy_recovered_structure_learned_routing",
    "pixel_object_observation_learned_routing",
    "held_out_entity_count_learned_routing",
)

PRIMARY_LEARNED_MODEL = "learned_signature_routing"
PRIMARY_CONTROLS = (
    "oracle_signature_routing_upper_bound",
    "dense_learned_routing",
    "random_learned_routing",
    "wrong_learned_routing",
)

INDEPENDENT_UNIT = "world"
NESTED_REPLICATE = "optimizer_seed"
OPTIMIZER_SEEDS = (0, 1, 2)
DEVELOPMENT_WORLDS = 24
MINIMUM_TEST_WORLDS = 50
MAXIMUM_TEST_WORLDS = 128
FAMILYWISE_ALPHA = 0.05
POWER_TARGET = 0.90
POWER_SIMULATION_ITERATIONS = 20_000

TRAIN_ENTITY_COUNT = 3
HELD_OUT_ENTITY_COUNTS = (2, 4)
NOISE_LEVELS = (0.10, 0.25)
PIXEL_IMAGE_SHAPE = (16, 16)
PIXEL_OBJECT_COUNT_RANGE = (2, 4)

GRAPH_EDGE_F1_MINIMUM = 0.90
LEARNED_DOWNSTREAM_SESOI = 0.05
NOISY_RELATIVE_DEGRADATION_MAXIMUM = 0.15
CARDINALITY_RELATIVE_DEGRADATION_MAXIMUM = 0.20

PRIMARY_ENDPOINTS = (
    "held_out_world_normalized_i0_quotient_error",
    "held_out_world_graph_edge_f1",
    "held_out_world_noisy_relative_error",
    "held_out_world_cardinality_relative_error",
)

REQUIRED_ARTIFACTS = (
    "learned_contract_and_digest",
    "observation_and_cardinality_generator_audit",
    "capacity_compute_and_information_audit",
    "development_learned_routing_pilot",
    "development_noise_and_cardinality_pilot",
    "development_variance_and_power_report",
    "zero_access_learned_seed_ledger",
    "frozen_confirmatory_analysis_plan",
    "one_shot_execution_lock",
)

NONNEGOTIABLE_POLICIES = (
    "Oracle and learned results are reported in separate regimes and are never pooled.",
    "The independent unit is world; optimizer seeds are nested replicates.",
    "The learned router cannot receive the oracle dependency graph or a target-state codebook.",
    "The primary learned-routing gate requires both graph recovery and downstream utility.",
    "Noise levels and held-out entity counts are fixed before any sealed result is observed.",
    "Pixel/object observation is an observation model, not evidence that pixels reveal exact structure.",
    "Failure of one regime is preserved and cannot be repaired by averaging across regimes.",
    "P3-5A does not claim temporal prognosis, births/deaths, public-benchmark validity, or cross-family replication.",
)


@dataclass(frozen=True)
class LearnedStatisticalPlan:
    independent_unit: str = INDEPENDENT_UNIT
    nested_replicate: str = NESTED_REPLICATE
    primary_endpoints: tuple[str, ...] = PRIMARY_ENDPOINTS
    multiplicity_rule: str = "one_sided_student_t_holm_across_four_regime_endpoints"
    cluster_interval: str = "world_cluster_bootstrap_bonferroni_alpha_over_four"
    hierarchical_interval: str = "world_random_intercept_seed_nested_variance_decomposition"

    def as_dict(self) -> dict[str, object]:
        return {
            "independent_unit": self.independent_unit,
            "nested_replicate": self.nested_replicate,
            "primary_endpoints": list(self.primary_endpoints),
            "multiplicity_rule": self.multiplicity_rule,
            "cluster_interval": self.cluster_interval,
            "hierarchical_interval": self.hierarchical_interval,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "power_target": POWER_TARGET,
            "development_worlds": DEVELOPMENT_WORLDS,
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "minimum_test_worlds": MINIMUM_TEST_WORLDS,
            "maximum_test_worlds": MAXIMUM_TEST_WORLDS,
            "power_simulation_iterations": POWER_SIMULATION_ITERATIONS,
        }


FROZEN_LEARNED_STATISTICAL_PLAN = LearnedStatisticalPlan()


def holm_normal_criticals(effect_count: int = len(PRIMARY_ENDPOINTS)) -> tuple[float, ...]:
    if type(effect_count) is not int or effect_count <= 0:
        raise ValueError("effect_count must be a positive integer")
    normal = NormalDist()
    return tuple(
        normal.inv_cdf(1.0 - FAMILYWISE_ALPHA / (effect_count - rank))
        for rank in range(effect_count)
    )


def _contract_payload() -> dict[str, object]:
    return {
        "identifier": P3_LEARNED_CONTRACT_ID,
        "parent_p3_4b_evidence_digest": PARENT_P3_4B_EVIDENCE_DIGEST,
        "evidence_level_before": 3,
        "evidence_level_after": 3,
        "regimes": list(REGIMES),
        "primary_learned_model": PRIMARY_LEARNED_MODEL,
        "primary_controls": list(PRIMARY_CONTROLS),
        "experimental_unit": INDEPENDENT_UNIT,
        "nested_replicate": NESTED_REPLICATE,
        "train_entity_count": TRAIN_ENTITY_COUNT,
        "held_out_entity_counts": list(HELD_OUT_ENTITY_COUNTS),
        "noise_levels": list(NOISE_LEVELS),
        "pixel_image_shape": list(PIXEL_IMAGE_SHAPE),
        "pixel_object_count_range": list(PIXEL_OBJECT_COUNT_RANGE),
        "thresholds": {
            "graph_edge_f1_minimum": GRAPH_EDGE_F1_MINIMUM,
            "learned_downstream_sesoi": LEARNED_DOWNSTREAM_SESOI,
            "noisy_relative_degradation_maximum": NOISY_RELATIVE_DEGRADATION_MAXIMUM,
            "cardinality_relative_degradation_maximum": CARDINALITY_RELATIVE_DEGRADATION_MAXIMUM,
        },
        "statistical_plan": FROZEN_LEARNED_STATISTICAL_PLAN.as_dict(),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "policies": list(NONNEGOTIABLE_POLICIES),
    }


def learned_contract_digest() -> str:
    return sha256(
        json.dumps(_contract_payload(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def audit_learned_contract() -> dict[str, object]:
    errors: list[str] = []
    if tuple(REGIMES) != tuple(dict.fromkeys(REGIMES)):
        errors.append("learned regimes must be unique")
    if TRAIN_ENTITY_COUNT in HELD_OUT_ENTITY_COUNTS:
        errors.append("held-out entity counts must differ from the training count")
    if len(OPTIMIZER_SEEDS) < 3:
        errors.append("at least three nested optimizer seeds are required")
    if not (MINIMUM_TEST_WORLDS <= MAXIMUM_TEST_WORLDS):
        errors.append("test-world bounds are inconsistent")
    if len(PRIMARY_ENDPOINTS) != 4:
        errors.append("P3-5A must retain four separate regime endpoints")
    if GRAPH_EDGE_F1_MINIMUM <= 0.0 or GRAPH_EDGE_F1_MINIMUM > 1.0:
        errors.append("graph F1 threshold must lie in (0, 1]")
    if any(level <= 0.0 for level in NOISE_LEVELS):
        errors.append("noise levels must be positive")
    if any(count <= 0 for count in HELD_OUT_ENTITY_COUNTS):
        errors.append("entity counts must be positive")
    return {
        "identifier": P3_LEARNED_CONTRACT_ID,
        "contract_digest": learned_contract_digest(),
        "regime_count": len(REGIMES),
        "endpoint_count": len(PRIMARY_ENDPOINTS),
        "errors": errors,
        "passed": not errors,
    }
