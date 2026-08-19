"""Frozen preregistration contract for the TSI P3-3A independence subgate.

This module freezes the design before any sealed-test result is generated.
A valid static contract does not pass P3-3A: the required generator, decoder,
control, and power artifacts must also be present with zero test access.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import ceil
from types import MappingProxyType
from typing import Mapping


P3_INDEPENDENCE_CONTRACT_ID = "P3-3A-INDEPENDENCE-v1"
PARENT_EVIDENCE_CONTRACT_ID = "P3-E0-EVIDENCE-v1"
PARENT_EVIDENCE_CONTRACT_DIGEST = (
    "0b2f41e4a078aed59160a157a92eed71d9e2428fdd9bb639e51589729f52f419"
)


class WorldFamily(str, Enum):
    SEPARABLE = "separable"
    BRIDGE_COUPLED = "bridge_coupled"
    CONTEXT_DEPENDENT = "context_dependent"


class BenchmarkSplit(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    SEALED_TEST = "sealed_test"


class DecoderRegime(str, Enum):
    RAW_UNCONSTRAINED = "raw_unconstrained"
    CONSTRUCTIVE_VALID_PRIMARY = "constructive_valid_primary"
    FULL_CODEBOOK_ORACLE_UPPER_BOUND = "full_codebook_oracle_upper_bound"


@dataclass(frozen=True)
class ModelControlSpec:
    identifier: str
    state_routing: str
    action_routing: str
    bridge_routing: str
    role: str

    def as_dict(self) -> dict[str, str]:
        return {
            "identifier": self.identifier,
            "state_routing": self.state_routing,
            "action_routing": self.action_routing,
            "bridge_routing": self.bridge_routing,
            "role": self.role,
        }


MODEL_CONTROLS = (
    ModelControlSpec(
        "dense_active_matched",
        "dense",
        "dense",
        "dense",
        "capacity-matched generic control",
    ),
    ModelControlSpec(
        "layer_routed_dense_action",
        "layerwise",
        "dense",
        "none",
        "state-routing control",
    ),
    ModelControlSpec(
        "strict_factorized_action",
        "layerwise",
        "block_diagonal",
        "none",
        "separable positive control",
    ),
    ModelControlSpec(
        "signature_routed_oracle",
        "declared_dependency_graph",
        "declared_dependency_graph",
        "correct",
        "primary oracle-routing model",
    ),
    ModelControlSpec(
        "random_routed_matched_sparsity",
        "random_matched_mask",
        "random_matched_mask",
        "random",
        "generic sparsity control",
    ),
    ModelControlSpec(
        "permuted_or_wrong_routed",
        "permuted_mask",
        "wrong_direction",
        "wrong",
        "structural specificity control",
    ),
)


OOD_SLICES = (
    "unseen_recombination",
    "unseen_structural_mode",
    "unseen_mechanism_parameter",
    "unseen_action_composition",
    "bridge_consistent_shift",
    "bridge_violating_control",
)


RELATION_GENERATORS = MappingProxyType(
    {
        "adjacent": "simplicial_1_skeleton_bridge_bound",
        "influences": "independent_directed_relation",
    }
)


DECODER_CONSTRUCTIONS = MappingProxyType(
    {
        "label": "per_entity_categorical_argmax",
        "topology": "predicted_simplices_then_downward_closure",
        "metric": "nonnegative_symmetric_edges_then_shortest_path_closure",
        "relation": "bridge_bound_adjacency_plus_independent_pair_logits",
        "order": "directed_generators_then_reflexive_transitive_closure",
        "tracking": "type_compatible_explicit_partial_bijection_assignment",
    }
)


PRIMARY_CONTROLS = (
    "dense_active_matched",
    "random_routed_matched_sparsity",
    "permuted_or_wrong_routed",
)


REQUIRED_ARTIFACTS = (
    "generator_implementation_and_digest",
    "development_validation_world_manifest",
    "sealed_test_seed_commitment",
    "ood_support_and_nonredundancy_audit",
    "constructive_decoder_and_validity_audit",
    "six_model_dependency_mask_manifest",
    "information_capacity_compute_tuning_ledger",
    "development_variance_and_power_report",
    "zero_access_test_ledger",
    "frozen_p3_3b_analysis_digest",
)


NONNEGOTIABLE_POLICIES = (
    "P3-3A has evidence ceiling 2 and cannot increase the evidence level.",
    "Sealed-test seeds are committed before reveal and results are never used for tuning.",
    "The primary decoder cannot receive or enumerate a global target-state codebook.",
    "Full-codebook projection is reported only as an oracle upper bound.",
    "Independent worlds are primary units and optimizer seeds are nested replicates.",
    "Correct routing is compared with dense, strict, random, permuted, and wrong controls.",
    "All models receive matched information, update count, and tuning budget.",
    "Active parameter count must match within five percent for the primary comparison.",
    "A failed confirmatory test requires a new benchmark version, not in-test repair.",
)


@dataclass(frozen=True)
class StatisticalPlan:
    primary_family: WorldFamily = WorldFamily.BRIDGE_COUPLED
    primary_endpoint: str = (
        "world_mean_normalized_i0_quotient_error_constructive_valid_primary"
    )
    smallest_effect_of_interest: float = 0.05
    dense_noninferiority_margin: float = 0.05
    alpha: float = 0.05
    power: float = 0.90
    development_worlds_per_family: int = 24
    optimizer_seeds_per_world: int = 3
    minimum_test_worlds: int = 36
    maximum_test_worlds: int = 128
    active_parameter_relative_tolerance: float = 0.05
    multiplicity_rule: str = "holm_fwer_three_primary_contrasts"
    uncertainty_methods: tuple[str, ...] = (
        "world_cluster_bootstrap",
        "hierarchical_world_seed_model",
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "primary_family": self.primary_family.value,
            "primary_endpoint": self.primary_endpoint,
            "smallest_effect_of_interest": self.smallest_effect_of_interest,
            "dense_noninferiority_margin": self.dense_noninferiority_margin,
            "alpha": self.alpha,
            "power": self.power,
            "development_worlds_per_family": (self.development_worlds_per_family),
            "optimizer_seeds_per_world": self.optimizer_seeds_per_world,
            "minimum_test_worlds": self.minimum_test_worlds,
            "maximum_test_worlds": self.maximum_test_worlds,
            "active_parameter_relative_tolerance": (
                self.active_parameter_relative_tolerance
            ),
            "multiplicity_rule": self.multiplicity_rule,
            "uncertainty_methods": list(self.uncertainty_methods),
        }


FROZEN_STATISTICAL_PLAN = StatisticalPlan()


def planned_test_world_count(development_effect_sd: float) -> int:
    """Compute the frozen pre-test world count or reject an underpowered plan."""

    if development_effect_sd < 0.0:
        raise ValueError("development effect SD must be nonnegative")
    plan = FROZEN_STATISTICAL_PLAN
    planning_sd = max(0.10, 1.25 * development_effect_sd)
    # One-sided alpha/3 is the least favorable first Holm threshold.
    normal_critical_sum = 2.128045234184983 + 1.2815515655446004
    unconstrained = ceil(
        (normal_critical_sum * planning_sd / plan.smallest_effect_of_interest) ** 2
    )
    required = max(plan.minimum_test_worlds, unconstrained)
    if required > plan.maximum_test_worlds:
        raise ValueError(
            "the frozen maximum of 128 test worlds is underpowered; "
            "revise the compute plan before revealing test seeds"
        )
    return required


def _contract_payload() -> dict[str, object]:
    return {
        "identifier": P3_INDEPENDENCE_CONTRACT_ID,
        "parent_evidence_contract": PARENT_EVIDENCE_CONTRACT_ID,
        "parent_evidence_contract_digest": PARENT_EVIDENCE_CONTRACT_DIGEST,
        "evidence_level_before": 2,
        "evidence_level_after": 2,
        "world_families": [family.value for family in WorldFamily],
        "splits": [split.value for split in BenchmarkSplit],
        "ood_slices": list(OOD_SLICES),
        "relation_generators": dict(RELATION_GENERATORS),
        "decoder_regimes": [regime.value for regime in DecoderRegime],
        "decoder_constructions": dict(DECODER_CONSTRUCTIONS),
        "model_controls": [model.as_dict() for model in MODEL_CONTROLS],
        "primary_controls": list(PRIMARY_CONTROLS),
        "statistical_plan": FROZEN_STATISTICAL_PLAN.as_dict(),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "policies": list(NONNEGOTIABLE_POLICIES),
    }


def independence_contract_digest() -> str:
    """Return a deterministic semantic digest of the preregistration."""

    encoded = json.dumps(
        _contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class IndependenceContractAudit:
    identifier: str
    evidence_level_before: int
    evidence_level_after: int
    test_seed_reveals: int
    test_result_evaluations: int
    static_errors: tuple[str, ...]
    artifact_blockers: tuple[str, ...]
    contract_digest: str

    @property
    def static_contract_passed(self) -> bool:
        return not self.static_errors

    @property
    def gate_passed(self) -> bool:
        return (
            self.static_contract_passed
            and not self.artifact_blockers
            and self.test_seed_reveals == 0
            and self.test_result_evaluations == 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "evidence_level_before": self.evidence_level_before,
            "evidence_level_after": self.evidence_level_after,
            "test_seed_reveals": self.test_seed_reveals,
            "test_result_evaluations": self.test_result_evaluations,
            "static_errors": list(self.static_errors),
            "artifact_blockers": list(self.artifact_blockers),
            "contract_digest": self.contract_digest,
            "static_contract_passed": self.static_contract_passed,
            "gate_passed": self.gate_passed,
        }


def audit_p3_3a_independence_contract(
    artifact_status: Mapping[str, bool] | None = None,
    *,
    test_seed_reveals: int = 0,
    test_result_evaluations: int = 0,
) -> IndependenceContractAudit:
    """Audit the frozen design while keeping unbuilt artifacts as blockers."""

    errors: list[str] = []
    if test_seed_reveals < 0 or test_result_evaluations < 0:
        errors.append("test access counts must be nonnegative")

    model_ids = tuple(model.identifier for model in MODEL_CONTROLS)
    if len(model_ids) != 6 or len(model_ids) != len(set(model_ids)):
        errors.append("the control matrix must contain six unique models")
    if "signature_routed_oracle" not in model_ids:
        errors.append("the primary signature-routed model is missing")
    if any(control not in model_ids for control in PRIMARY_CONTROLS):
        errors.append("a primary control is absent from the model matrix")

    if tuple(RELATION_GENERATORS) != ("adjacent", "influences"):
        errors.append("bridge-bound and independent relations must both be frozen")
    if len(DECODER_CONSTRUCTIONS) != 6:
        errors.append("all six constructive decoder components are required")
    if (
        DecoderRegime.CONSTRUCTIVE_VALID_PRIMARY.value
        not in _contract_payload()["decoder_regimes"]
    ):
        errors.append("the constructive decoder must remain primary")

    plan = FROZEN_STATISTICAL_PLAN
    if plan.primary_family is not WorldFamily.BRIDGE_COUPLED:
        errors.append("bridge-coupled worlds must remain the primary family")
    if plan.smallest_effect_of_interest != 0.05:
        errors.append("the preregistered SESOI changed")
    if plan.optimizer_seeds_per_world < 3:
        errors.append("at least three nested optimizer seeds are required")
    if plan.active_parameter_relative_tolerance > 0.05:
        errors.append("active parameter tolerance cannot exceed five percent")

    status = {} if artifact_status is None else dict(artifact_status)
    unknown_artifacts = sorted(set(status).difference(REQUIRED_ARTIFACTS))
    if unknown_artifacts:
        errors.append("unknown artifact status keys: " + ", ".join(unknown_artifacts))
    blockers = tuple(
        artifact for artifact in REQUIRED_ARTIFACTS if not status.get(artifact, False)
    )

    return IndependenceContractAudit(
        identifier=P3_INDEPENDENCE_CONTRACT_ID,
        evidence_level_before=2,
        evidence_level_after=2,
        test_seed_reveals=test_seed_reveals,
        test_result_evaluations=test_result_evaluations,
        static_errors=tuple(errors),
        artifact_blockers=blockers,
        contract_digest=independence_contract_digest(),
    )
