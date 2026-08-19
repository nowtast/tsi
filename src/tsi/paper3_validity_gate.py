"""Pre-reveal artifact gate for the one-shot P3-4B validity experiment."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .paper3_rollout_gate import implementation_source_digest as p3a_source_digest
from .paper3_validity_access import audit_validity_access
from .paper3_validity_contract import (
    REQUIRED_ARTIFACTS,
    audit_validity_contract,
)
from .paper3_validity_experiment import P3_VALIDITY_DEVELOPMENT_ID
from .paper3_validity_generator import (
    audit_validity_generator,
    normalized_exclusions,
)
from .paper3_validity_once import FAILURE_FILENAME, LOCK_FILENAME
from .paper3_validity_power import (
    P3_VALIDITY_ANALYSIS_PLAN_ID,
    P3_VALIDITY_POWER_ID,
)
from .paper3_validity_predictor import (
    P3_VALIDITY_PREDICTOR_ID,
    validate_frozen_predictors,
)
from .paper3_routing_controls import audit_routing_controls


P3_VALIDITY_GATE_ID = "P3-4B-FINAL-FREEZE-v2"
SOURCE_FILES = (
    "src/tsi/paper3_multiworld.py",
    "src/tsi/paper3_constructive_decoder.py",
    "src/tsi/paper3_routing_controls.py",
    "src/tsi/paper3_routing_model.py",
    "src/tsi/paper3_validity_contract.py",
    "src/tsi/paper3_validity_generator.py",
    "src/tsi/paper3_validity_evaluator.py",
    "src/tsi/paper3_validity_experiment.py",
    "src/tsi/paper3_validity_predictor.py",
    "src/tsi/paper3_validity_power.py",
    "src/tsi/paper3_validity_access.py",
    "src/tsi/paper3_validity_analysis.py",
    "src/tsi/paper3_validity_evidence.py",
    "src/tsi/paper3_validity_once.py",
    "src/tsi/paper3_validity_gate.py",
    "tools/run_paper3_validity_sealed.py",
)
SEALED_OUTPUT_FILENAMES = (
    "sealed_validity_manifest.json",
    "sealed_validity_raw_results.json",
    "sealed_validity_confirmatory_analysis.json",
    "validity_evidence_report.json",
)


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def report_digest_valid(
    report: Mapping[str, object],
    field: str = "report_digest",
) -> bool:
    payload = {key: value for key, value in report.items() if key != field}
    return report.get(field) == _canonical_digest(payload)


def implementation_source_digest(repository_root: Path) -> str:
    root = Path(repository_root)
    digest = sha256()
    for relative in SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _p3a_excluded_signatures(
    repository: Path,
) -> frozenset[tuple[object, ...]]:
    manifest = json.loads(
        (
            repository
            / "experiments"
            / "paper3_rollout"
            / "sealed_rollout_manifest.json"
        ).read_text(encoding="utf-8")
    )
    worlds = manifest.get("worlds")
    if not isinstance(worlds, list):
        raise ValueError("P3-4A sealed manifest has no worlds")
    return normalized_exclusions(
        world["active_parameter_signature"]
        for world in worlds
        if isinstance(world, dict)
    )


def build_validity_artifact_gate(
    repository_root: Path,
    output_root: Path,
) -> dict[str, object]:
    repository = Path(repository_root)
    output = Path(output_root)
    sealed = output / "sealed"
    development = json.loads(
        (output / "development_validity_results.json").read_text(encoding="utf-8")
    )
    predictors = json.loads(
        (output / "frozen_validity_predictors.json").read_text(encoding="utf-8")
    )
    power = json.loads(
        (output / "development_power_report.json").read_text(encoding="utf-8")
    )
    analysis_plan = json.loads(
        (output / "analysis_plan.json").read_text(encoding="utf-8")
    )
    p3a_evidence = json.loads(
        (
            repository
            / "experiments"
            / "paper3_rollout"
            / "rollout_evidence_report.json"
        ).read_text(encoding="utf-8")
    )
    p3a_lock = json.loads(
        (
            repository
            / "experiments"
            / "paper3_rollout"
            / "sealed"
            / "p3_4a_once.lock"
        ).read_text(encoding="utf-8")
    )
    exclusions = _p3a_excluded_signatures(repository)
    access = audit_validity_access(sealed, expected_phase="zero")
    contract = audit_validity_contract()
    generator = audit_validity_generator()
    routing = audit_routing_controls().as_dict()
    frozen = validate_frozen_predictors(predictors)

    plan_payload = {
        key: value
        for key, value in analysis_plan.items()
        if key != "analysis_plan_digest"
    }
    plan_valid = bool(
        analysis_plan.get("identifier") == P3_VALIDITY_ANALYSIS_PLAN_ID
        and analysis_plan.get("analysis_plan_digest") == _canonical_digest(plan_payload)
        and power.get("analysis_plan") == analysis_plan
        and analysis_plan.get("frozen_predictor_digest")
        == frozen.get("frozen_predictor_digest")
    )
    development_valid = bool(
        development.get("identifier") == P3_VALIDITY_DEVELOPMENT_ID
        and development.get("test_output_used") is False
        and development.get("failure_count") == 0
        and report_digest_valid(development)
    )
    predictors_valid = bool(
        predictors.get("identifier") == P3_VALIDITY_PREDICTOR_ID
        and predictors.get("test_output_used") is False
        and predictors.get("all_final_models_converged") is True
        and predictors.get("development_lowo_performed") is True
        and report_digest_valid(predictors)
    )
    power_valid = bool(
        power.get("identifier") == P3_VALIDITY_POWER_ID
        and power.get("passed") is True
        and power.get("test_output_used") is False
        and report_digest_valid(power)
    )
    current_p3a_source_digest = p3a_source_digest(repository)
    p3a_source_unchanged = current_p3a_source_digest == p3a_lock.get(
        "frozen_artifact_digests", {}
    ).get("implementation_source")
    p3a_valid = bool(
        p3a_evidence.get("level_3_retained") is True
        and p3a_evidence.get("evidence_level_after") == 3
        and p3a_evidence.get("level_4_requirements", {}).get(
            "open_loop_multihorizon_rollout"
        )
        is True
        and report_digest_valid(p3a_evidence)
        and p3a_source_unchanged
        and len(exclusions) == 62
    )
    planned_worlds = power.get("planned_test_worlds")
    fresh_support_valid = bool(
        type(planned_worlds) is int
        and planned_worlds <= 90
        and len(exclusions) == 62
    )
    lock_ready = bool(
        (repository / "tools" / "run_paper3_validity_sealed.py").is_file()
        and not (sealed / LOCK_FILENAME).exists()
        and not (sealed / FAILURE_FILENAME).exists()
        and not any((output / name).exists() for name in SEALED_OUTPUT_FILENAMES)
    )
    artifact_status = {
        "validity_contract_and_digest": contract["passed"],
        "noncircular_task_generator_audit": (
            generator["passed"]
            and generator["outcome_definition_uses_tsi_metric"] is False
        ),
        "six_control_capacity_and_compute_audit": routing["passed"],
        "development_validity_benchmark": development_valid,
        "frozen_baseline_and_tsi_predictors": predictors_valid,
        "development_variance_and_power_report": power_valid,
        "zero_access_validity_seed_ledger": access["passed"],
        "frozen_confirmatory_analysis_plan": plan_valid,
        "one_shot_execution_lock": lock_ready,
    }
    if tuple(artifact_status) != REQUIRED_ARTIFACTS:
        raise RuntimeError("P3-4B artifact order changed")
    source_digest = implementation_source_digest(repository)
    gate_audit = {
        "identifier": P3_VALIDITY_GATE_ID,
        "p3a_level_3_and_rollout_requirement_retained": p3a_valid,
        "p3a_frozen_source_digest_unchanged": p3a_source_unchanged,
        "p3a_excluded_mechanism_count": len(exclusions),
        "fresh_mechanism_support_valid": fresh_support_valid,
        "artifact_blockers": [
            name for name, passed in artifact_status.items() if not passed
        ],
        "test_seed_reveals": access["seed_reveals"],
        "test_result_evaluations": access["result_evaluations"],
        "planned_test_worlds": planned_worlds,
        "analysis_plan_digest": analysis_plan.get("analysis_plan_digest"),
        "frozen_predictor_digest": frozen.get("frozen_predictor_digest"),
        "implementation_source_digest": source_digest,
        "evidence_level_before": 3,
        "evidence_level_after": 3,
        "level_4_requirement_count_before": 1,
        "level_4_requirement_count_after_if_passed": 2,
    }
    gate_audit["gate_passed"] = bool(
        p3a_valid
        and fresh_support_valid
        and all(artifact_status.values())
        and access["seed_reveals"] == 0
        and access["result_evaluations"] == 0
    )
    payload: dict[str, object] = {
        "identifier": P3_VALIDITY_GATE_ID,
        "artifact_status": artifact_status,
        "gate_audit": gate_audit,
        "subreports": {
            "contract": contract,
            "generator": generator,
            "routing_controls": routing,
            "access": access,
        },
        "frozen_artifact_digests": {
            "p3a_evidence": p3a_evidence.get("report_digest"),
            "p3a_implementation_source": current_p3a_source_digest,
            "development_benchmark": development.get("report_digest"),
            "frozen_predictor_report": predictors.get("report_digest"),
            "frozen_predictors": frozen.get("frozen_predictor_digest"),
            "development_power": power.get("report_digest"),
            "analysis_plan": analysis_plan.get("analysis_plan_digest"),
            "implementation_source": source_digest,
        },
    }
    return {**payload, "combined_digest": _canonical_digest(payload)}


def write_validity_artifact_gate(
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
