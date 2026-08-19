"""Operational evidence contract for the TSI Paper 3 empirical program.

The levels in this module are internal research-readiness categories, not
statistical measurements. The contract deliberately keeps P3-2R at level 2
and makes level 4 promotion conjunctive across independent OOD, rollout,
learned-structure, perception, and public-replication requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Iterable, Mapping


EVIDENCE_CONTRACT_ID = "P3-E0-EVIDENCE-v1"


class EvidenceLevel(IntEnum):
    """Internal empirical-evidence readiness levels."""

    ORACLE_CALIBRATION = 1
    DEVELOPMENT_DIAGNOSTIC = 2
    CONFIRMATORY_STRUCTURAL = 3
    MULTI_REGIME_VALIDATION = 4
    INDEPENDENT_REPLICATION = 5


@dataclass(frozen=True)
class EvidenceRequirement:
    key: str
    minimum_level: EvidenceLevel
    description: str

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "minimum_level": int(self.minimum_level),
            "description": self.description,
        }


@dataclass(frozen=True)
class EvidencePhase:
    identifier: str
    official_gate: str
    evidence_ceiling: EvidenceLevel
    purpose: str
    required_outputs: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "official_gate": self.official_gate,
            "evidence_ceiling": int(self.evidence_ceiling),
            "purpose": self.purpose,
            "required_outputs": list(self.required_outputs),
        }


LEVEL_3_REQUIREMENTS = (
    EvidenceRequirement(
        "sealed_independent_test",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "A new test generator and support are sealed before model selection.",
    ),
    EvidenceRequirement(
        "codebook_free_primary_decoder",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Primary decoding does not enumerate held-out target states.",
    ),
    EvidenceRequirement(
        "matched_information_capacity_compute_tuning",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Baselines receive matched information, active capacity, compute, and tuning budget.",
    ),
    EvidenceRequirement(
        "correct_random_wrong_dependency_controls",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Correct routing is compared with random, permuted, wrong, strict-factorized, and dense controls.",
    ),
    EvidenceRequirement(
        "independent_layer_information",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "At least one relation channel is not a deterministic duplicate of topology, metric, or order.",
    ),
    EvidenceRequirement(
        "world_level_replication",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Independent worlds or generator instances, not optimizer seeds alone, are experimental units.",
    ),
    EvidenceRequirement(
        "nested_uncertainty_analysis",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Uncertainty separates world variation from optimization-seed variation.",
    ),
    EvidenceRequirement(
        "confirmatory_structural_ood_effect",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "A preregistered structural OOD comparison meets its frozen success rule.",
    ),
)


LEVEL_4_REQUIREMENTS = (
    EvidenceRequirement(
        "open_loop_multihorizon_rollout",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "Open-loop and teacher-forced errors are reported over multiple horizons.",
    ),
    EvidenceRequirement(
        "downstream_predictive_validity",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "TSI discrepancies predict held-out task or planning failure beyond generic latent losses.",
    ),
    EvidenceRequirement(
        "learned_routing_or_structure",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "A non-oracle model learns routing or structural factors; exact masks are upper bounds only.",
    ),
    EvidenceRequirement(
        "noisy_perception",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "At least one regime begins from noisy or pixel observations rather than exact state input.",
    ),
    EvidenceRequirement(
        "variable_cardinality",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "Evaluation includes held-out entity counts or births and deaths.",
    ),
    EvidenceRequirement(
        "public_benchmark",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "At least one public benchmark is evaluated under a declared domain-specific signature.",
    ),
    EvidenceRequirement(
        "cross_family_replication",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "The main effect is tested in at least two non-identical environment families.",
    ),
    EvidenceRequirement(
        "artifact_reproducibility",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "Frozen artifacts reproduce tables, intervals, failures, and audit reports.",
    ),
)


LEVEL_5_REQUIREMENTS = (
    EvidenceRequirement(
        "independent_external_replication",
        EvidenceLevel.INDEPENDENT_REPLICATION,
        "An independent group or implementation reproduces the central effect.",
    ),
)


ALL_REQUIREMENTS = (
    *LEVEL_3_REQUIREMENTS,
    *LEVEL_4_REQUIREMENTS,
    *LEVEL_5_REQUIREMENTS,
)


NONNEGOTIABLE_POLICIES = (
    "P3-2R remains a development diagnostic and is never relabeled confirmatory.",
    "A complete held-out target-state codebook is forbidden for primary decoding.",
    "Exact layer masks and exact dependency masks are oracle upper bounds only.",
    "Test outputs cannot tune models, thresholds, sample sizes, or decoder policy.",
    "Worlds or environment instances are primary units; seeds are nested replicates.",
    "Correct, random, permuted, and wrong structural restrictions must be compared.",
    "Relation, topology, metric, and order need controlled independent information.",
    "Null results, instability, and slice-specific failures remain in the ledger.",
    "Paper 4 and a strong empirical claim are blocked below evidence level 4.",
)


EVIDENCE_PHASES = (
    EvidencePhase(
        "P3-E0-EVIDENCE-v1",
        "governance",
        EvidenceLevel.DEVELOPMENT_DIAGNOSTIC,
        "Freeze the level-4 promotion contract without increasing evidence.",
        (
            "machine-readable evidence contract",
            "Obsidian reinforcement roadmap",
            "deterministic contract digest",
        ),
    ),
    EvidencePhase(
        "P3-3A-INDEPENDENCE-v1",
        "P3-3",
        EvidenceLevel.DEVELOPMENT_DIAGNOSTIC,
        "Freeze independent generators, constructive decoding, controls, and statistics.",
        (
            "multi-world generator specification",
            "test-access ledger",
            "constructive decoder contract",
            "dependency-control matrix",
            "precision or power plan",
        ),
    ),
    EvidencePhase(
        "P3-3B-OOD-v1",
        "P3-3",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Run the sealed exact-state structural OOD comparison once.",
        (
            "confirmatory result ledger",
            "world-nested uncertainty analysis",
            "failure-preserving machine report",
        ),
    ),
    EvidencePhase(
        "P3-4A-ROLLOUT-v1",
        "P3-4",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Measure teacher-forced and open-loop structural error accumulation.",
        (
            "multi-horizon rollout report",
            "identity-switch and bridge-failure report",
        ),
    ),
    EvidencePhase(
        "P3-4B-VALIDITY-v1",
        "P3-4",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Test whether TSI discrepancies predict downstream failure.",
        (
            "held-out task-success report",
            "incremental predictive-validity analysis",
        ),
    ),
    EvidencePhase(
        "P3-5A-LEARNED-v1",
        "P3-5",
        EvidenceLevel.CONFIRMATORY_STRUCTURAL,
        "Remove oracle routing and exact-state access under noise and cardinality shift.",
        (
            "learned-routing comparison",
            "noisy or pixel perception report",
            "variable-cardinality report",
        ),
    ),
    EvidencePhase(
        "P3-5B-REPLICATION-v1",
        "P3-5",
        EvidenceLevel.MULTI_REGIME_VALIDATION,
        "Replicate the claim across a second family including a public benchmark.",
        (
            "public benchmark result",
            "cross-family effect audit",
            "reproducible artifact package",
        ),
    ),
    EvidencePhase(
        "P3-X-INDEPENDENT-v1",
        "post-P3",
        EvidenceLevel.INDEPENDENT_REPLICATION,
        "Obtain an implementation-independent replication.",
        ("external replication report",),
    ),
)


CURRENT_COMPLETED_PHASES = (
    "P3-0",
    "P3-1-ORACLE-v1",
    "P3-2-ABLATION-v1",
    "P3-2R-REPRESENTATION-v1",
    "P3-E0-EVIDENCE-v1",
)


def attained_evidence_level(
    satisfied_requirements: Iterable[str],
) -> EvidenceLevel:
    """Return the highest conjunctively satisfied internal evidence level."""

    satisfied = frozenset(satisfied_requirements)
    level = EvidenceLevel.DEVELOPMENT_DIAGNOSTIC
    if all(requirement.key in satisfied for requirement in LEVEL_3_REQUIREMENTS):
        level = EvidenceLevel.CONFIRMATORY_STRUCTURAL
    if level >= EvidenceLevel.CONFIRMATORY_STRUCTURAL and all(
        requirement.key in satisfied for requirement in LEVEL_4_REQUIREMENTS
    ):
        level = EvidenceLevel.MULTI_REGIME_VALIDATION
    if level >= EvidenceLevel.MULTI_REGIME_VALIDATION and all(
        requirement.key in satisfied for requirement in LEVEL_5_REQUIREMENTS
    ):
        level = EvidenceLevel.INDEPENDENT_REPLICATION
    return level


def missing_requirements(
    satisfied_requirements: Iterable[str],
    target: EvidenceLevel,
) -> tuple[str, ...]:
    """List unmet requirements up to and including ``target``."""

    satisfied = frozenset(satisfied_requirements)
    return tuple(
        requirement.key
        for requirement in ALL_REQUIREMENTS
        if requirement.minimum_level <= target and requirement.key not in satisfied
    )


def _contract_payload() -> dict[str, object]:
    return {
        "identifier": EVIDENCE_CONTRACT_ID,
        "target_level": int(EvidenceLevel.MULTI_REGIME_VALIDATION),
        "publication_floor": int(EvidenceLevel.MULTI_REGIME_VALIDATION),
        "requirements": [requirement.as_dict() for requirement in ALL_REQUIREMENTS],
        "policies": list(NONNEGOTIABLE_POLICIES),
        "phases": [phase.as_dict() for phase in EVIDENCE_PHASES],
        "current_completed_phases": list(CURRENT_COMPLETED_PHASES),
    }


def evidence_contract_digest() -> str:
    """Return a deterministic semantic digest of the frozen contract."""

    payload = json.dumps(
        _contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class EvidenceContractAudit:
    identifier: str
    current_level: EvidenceLevel
    target_level: EvidenceLevel
    publication_floor: EvidenceLevel
    next_phase: str
    publication_blocked: bool
    requirement_counts: Mapping[int, int]
    missing_for_target: tuple[str, ...]
    contract_digest: str
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "current_level": int(self.current_level),
            "target_level": int(self.target_level),
            "publication_floor": int(self.publication_floor),
            "next_phase": self.next_phase,
            "publication_blocked": self.publication_blocked,
            "requirement_counts": {
                str(level): count
                for level, count in sorted(self.requirement_counts.items())
            },
            "missing_for_target": list(self.missing_for_target),
            "contract_digest": self.contract_digest,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def audit_paper3_evidence_contract() -> EvidenceContractAudit:
    """Audit internal consistency without claiming empirical promotion."""

    errors: list[str] = []
    keys = tuple(requirement.key for requirement in ALL_REQUIREMENTS)
    if len(keys) != len(set(keys)):
        errors.append("evidence requirement keys must be unique")

    phase_ids = tuple(phase.identifier for phase in EVIDENCE_PHASES)
    if len(phase_ids) != len(set(phase_ids)):
        errors.append("evidence phase identifiers must be unique")
    if not phase_ids or phase_ids[0] != EVIDENCE_CONTRACT_ID:
        errors.append("the governance phase must be first")
    if len(EVIDENCE_PHASES) < 2 or EVIDENCE_PHASES[1].identifier != (
        "P3-3A-INDEPENDENCE-v1"
    ):
        errors.append("P3-3A independence must be the next phase")

    current_level = attained_evidence_level(())
    target_level = EvidenceLevel.MULTI_REGIME_VALIDATION
    publication_floor = EvidenceLevel.MULTI_REGIME_VALIDATION
    if current_level is not EvidenceLevel.DEVELOPMENT_DIAGNOSTIC:
        errors.append("P3-2R must remain at evidence level 2")
    if target_level < publication_floor:
        errors.append("target level cannot be below the publication floor")

    requirement_counts = MappingProxyType(
        {
            int(level): sum(
                requirement.minimum_level is level for requirement in ALL_REQUIREMENTS
            )
            for level in (
                EvidenceLevel.CONFIRMATORY_STRUCTURAL,
                EvidenceLevel.MULTI_REGIME_VALIDATION,
                EvidenceLevel.INDEPENDENT_REPLICATION,
            )
        }
    )
    expected_counts = {
        int(EvidenceLevel.CONFIRMATORY_STRUCTURAL): 8,
        int(EvidenceLevel.MULTI_REGIME_VALIDATION): 8,
        int(EvidenceLevel.INDEPENDENT_REPLICATION): 1,
    }
    if dict(requirement_counts) != expected_counts:
        errors.append("evidence requirement counts changed unexpectedly")

    policy_text = " ".join(NONNEGOTIABLE_POLICIES).lower()
    for required_phrase in (
        "target-state codebook",
        "oracle upper bounds",
        "primary units",
        "random",
        "paper 4",
    ):
        if required_phrase not in policy_text:
            errors.append(f"missing nonnegotiable policy: {required_phrase}")

    return EvidenceContractAudit(
        identifier=EVIDENCE_CONTRACT_ID,
        current_level=current_level,
        target_level=target_level,
        publication_floor=publication_floor,
        next_phase="P3-3A-INDEPENDENCE-v1",
        publication_blocked=current_level < publication_floor,
        requirement_counts=requirement_counts,
        missing_for_target=missing_requirements((), target_level),
        contract_digest=evidence_contract_digest(),
        errors=tuple(errors),
    )
