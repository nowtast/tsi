"""Conjunctive Level-3 evidence audit after the one-shot P3-3B run."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import stat
from typing import Mapping

from .paper3_constructive_decoder import audit_constructive_decoder
from .paper3_evidence_contract import (
    EvidenceLevel,
    LEVEL_3_REQUIREMENTS,
    attained_evidence_level,
)
from .paper3_independence_contract import MODEL_CONTROLS, WorldFamily
from .paper3_multiworld import (
    DEVELOPMENT_WORLDS_PER_FAMILY,
    VALIDATION_WORLDS_PER_FAMILY,
    _ranked_active_parameters,
)
from .paper3_multiworld_audit import audit_multiworld_generator
from .paper3_routing_controls import audit_routing_controls
from .paper3_sealed_access import (
    COMMITMENT_FILENAME,
    ESCROW_FILENAME,
    LEDGER_FILENAME,
)


P3_CONFIRMATORY_EVIDENCE_ID = "P3-3B-EVIDENCE-AUDIT-v1"
EXPECTED_ACCESS_SEQUENCE = (
    "seed_commitment_created",
    "test_seed_revealed",
    "test_prediction_started",
    "test_prediction_completed",
    "test_result_evaluated",
    "test_report_generated",
)


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _digest(payload: object) -> str:
    return sha256(_canonical(payload).encode("utf-8")).hexdigest()


def audit_confirmatory_access(sealed_root: Path) -> dict[str, object]:
    root = Path(sealed_root)
    descriptor = json.loads((root / COMMITMENT_FILENAME).read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (root / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    errors: list[str] = []
    previous_hash = "0" * 64
    reveal_count = 0
    result_count = 0
    for sequence, event in enumerate(events):
        payload = {key: value for key, value in event.items() if key != "event_hash"}
        expected_hash = _digest(payload)
        if event.get("sequence") != sequence:
            errors.append(f"event {sequence} sequence mismatch")
        if event.get("previous_event_hash") != previous_hash:
            errors.append(f"event {sequence} previous hash mismatch")
        if event.get("event_hash") != expected_hash:
            errors.append(f"event {sequence} hash mismatch")
        reveal_count += event.get("event") == "test_seed_revealed"
        result_count += event.get("event") == "test_result_evaluated"
        if event.get("test_seed_reveals_after_event") != reveal_count:
            errors.append(f"event {sequence} reveal count mismatch")
        if event.get("test_result_evaluations_after_event") != result_count:
            errors.append(f"event {sequence} result count mismatch")
        previous_hash = expected_hash
    event_names = tuple(event.get("event") for event in events)
    if event_names != EXPECTED_ACCESS_SEQUENCE:
        errors.append("test-access event sequence is not exactly one-shot")
    if reveal_count != 1:
        errors.append("sealed seed must be revealed exactly once")
    if result_count != 1:
        errors.append("test result must be evaluated exactly once")
    if descriptor.get("revealed") is not True:
        errors.append("commitment descriptor is not marked revealed")
    escrow_mode = stat.S_IMODE((root / ESCROW_FILENAME).stat().st_mode)
    if escrow_mode != 0:
        errors.append("escrow mode must return to 000 after execution")
    return {
        "event_count": len(events),
        "event_names": list(event_names),
        "test_seed_reveals": reveal_count,
        "test_result_evaluations": result_count,
        "latest_event_hash": previous_hash if events else None,
        "commitment": descriptor.get("commitment"),
        "escrow_mode_octal": oct(escrow_mode),
        "errors": errors,
        "passed": not errors,
    }


def evaluate_level3_requirements(
    manifest: Mapping[str, object],
    raw: Mapping[str, object],
    analysis: Mapping[str, object],
    access: Mapping[str, object],
) -> dict[str, bool]:
    worlds = manifest.get("worlds")
    if not isinstance(worlds, list):
        worlds = []
    public_count = DEVELOPMENT_WORLDS_PER_FAMILY + VALIDATION_WORLDS_PER_FAMILY
    public_signatures = {
        (candidate[0], candidate[1])
        for candidate in _ranked_active_parameters(WorldFamily.BRIDGE_COUPLED)[
            :public_count
        ]
    }
    sealed_signatures = {
        (
            tuple(world["active_parameter_signature"][0]),
            world["active_parameter_signature"][1],
        )
        for world in worlds
        if isinstance(world, dict)
        and isinstance(world.get("active_parameter_signature"), list)
    }
    sealed_independent = bool(
        access.get("passed")
        and manifest.get("world_count") == 50
        and len(sealed_signatures) == 50
        and not public_signatures.intersection(sealed_signatures)
    )

    decoder = audit_constructive_decoder()
    codebook_free = bool(
        decoder.passed
        and decoder.global_candidate_states == 0
        and raw.get("constructive_metric_cache", {}).get(
            "global_target_state_candidates"
        )
        == 0
    )
    routing = audit_routing_controls()
    matched = bool(
        routing.passed
        and routing.max_relative_parameter_difference <= 0.05
        and routing.compute_budgets_matched
        and routing.tuning_budgets_matched
    )
    expected_models = {model.identifier for model in MODEL_CONTROLS}
    observed_models = {
        run.get("model") for run in raw.get("runs", []) if isinstance(run, dict)
    }
    controls = bool(
        expected_models == observed_models
        and raw.get("run_count") == 50 * 3 * len(expected_models)
        and raw.get("failure_count") == 0
    )
    generator = audit_multiworld_generator()
    independent_layer = bool(
        generator.passed and generator.independent_relation_witnesses >= 2
    )
    world_replication = bool(
        raw.get("world_count") == 50 and raw.get("optimizer_seeds") == [0, 1, 2]
    )
    hierarchical = analysis.get("hierarchical_world_seed_analysis", {})
    bootstrap = analysis.get("world_cluster_bootstrap", {})
    nested_uncertainty = bool(
        isinstance(hierarchical, dict)
        and hierarchical.get("model")
        == "world_random_intercept_seed_nested_variance_decomposition"
        and isinstance(bootstrap, dict)
        and bootstrap.get("cluster_unit") == "world_with_all_nested_optimizer_seeds"
    )
    confirmatory_effect = analysis.get("passed") is True
    return {
        "sealed_independent_test": sealed_independent,
        "codebook_free_primary_decoder": codebook_free,
        "matched_information_capacity_compute_tuning": matched,
        "correct_random_wrong_dependency_controls": controls,
        "independent_layer_information": independent_layer,
        "world_level_replication": world_replication,
        "nested_uncertainty_analysis": nested_uncertainty,
        "confirmatory_structural_ood_effect": confirmatory_effect,
    }


def build_evidence_report(
    manifest: Mapping[str, object],
    raw: Mapping[str, object],
    analysis: Mapping[str, object],
    access: Mapping[str, object],
) -> dict[str, object]:
    requirements = evaluate_level3_requirements(
        manifest,
        raw,
        analysis,
        access,
    )
    expected_keys = tuple(requirement.key for requirement in LEVEL_3_REQUIREMENTS)
    if tuple(requirements) != expected_keys:
        raise RuntimeError("Level-3 requirement order changed")
    satisfied = tuple(key for key, value in requirements.items() if value)
    evidence_level = attained_evidence_level(satisfied)
    payload: dict[str, object] = {
        "identifier": P3_CONFIRMATORY_EVIDENCE_ID,
        "requirements": requirements,
        "satisfied_requirements": list(satisfied),
        "evidence_level_before": int(EvidenceLevel.DEVELOPMENT_DIAGNOSTIC),
        "evidence_level_after": int(evidence_level),
        "level_3_attained": (evidence_level >= EvidenceLevel.CONFIRMATORY_STRUCTURAL),
        "publication_floor": int(EvidenceLevel.MULTI_REGIME_VALIDATION),
        "publication_blocked": (evidence_level < EvidenceLevel.MULTI_REGIME_VALIDATION),
        "access_audit": dict(access),
        "sealed_manifest_digest": manifest.get("manifest_digest"),
        "raw_result_digest": raw.get("report_digest"),
        "confirmatory_analysis_digest": analysis.get("report_digest"),
    }
    return {**payload, "report_digest": _digest(payload)}


def write_evidence_report(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
