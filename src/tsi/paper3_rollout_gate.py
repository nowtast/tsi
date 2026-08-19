"""Pre-reveal artifact gate for the one-shot P3-4A rollout experiment."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .paper3_rollout_access import audit_rollout_access
from .paper3_rollout_contract import (
    REQUIRED_ARTIFACTS,
    audit_rollout_contract,
)
from .paper3_rollout_evaluator import audit_fixed_metric_and_lipschitz
from .paper3_rollout_experiment import P3_ROLLOUT_DEVELOPMENT_ID
from .paper3_rollout_generator import (
    audit_rollout_generator,
    development_rollout_worlds,
)
from .paper3_rollout_once import FAILURE_FILENAME, LOCK_FILENAME
from .paper3_rollout_power import (
    P3_ROLLOUT_ANALYSIS_PLAN_ID,
    P3_ROLLOUT_POWER_ID,
)
from .paper3_routing_controls import audit_routing_controls


P3_ROLLOUT_GATE_ID = "P3-4A-FINAL-FREEZE-v1"
SOURCE_FILES = (
    "src/tsi/paper3_multiworld.py",
    "src/tsi/paper3_constructive_decoder.py",
    "src/tsi/paper3_routing_controls.py",
    "src/tsi/paper3_routing_model.py",
    "src/tsi/paper3_rollout_contract.py",
    "src/tsi/paper3_rollout_generator.py",
    "src/tsi/paper3_rollout_evaluator.py",
    "src/tsi/paper3_rollout_experiment.py",
    "src/tsi/paper3_rollout_access.py",
    "src/tsi/paper3_rollout_power.py",
    "src/tsi/paper3_rollout_analysis.py",
    "src/tsi/paper3_rollout_evidence.py",
    "src/tsi/paper3_rollout_once.py",
    "src/tsi/paper3_rollout_gate.py",
    "tools/run_paper3_rollout_sealed.py",
)
SEALED_OUTPUT_FILENAMES = (
    "sealed_rollout_manifest.json",
    "sealed_rollout_raw_results.json",
    "sealed_rollout_confirmatory_analysis.json",
    "rollout_evidence_report.json",
)
P3B_SOURCE_FILES = (
    "src/tsi/paper3_multiworld.py",
    "src/tsi/paper3_sealed_worlds.py",
    "src/tsi/paper3_constructive_decoder.py",
    "src/tsi/paper3_routing_controls.py",
    "src/tsi/paper3_routing_model.py",
    "src/tsi/paper3_analysis_plan.py",
    "src/tsi/paper3_confirmatory_experiment.py",
    "src/tsi/paper3_confirmatory_analysis.py",
    "src/tsi/paper3_confirmatory_evidence.py",
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


def _named_source_digest(
    repository_root: Path,
    relative_paths: tuple[str, ...],
) -> str:
    digest = sha256()
    for relative in relative_paths:
        digest.update(relative.encode("utf-8"))
        digest.update((repository_root / relative).read_bytes())
    return digest.hexdigest()


def build_rollout_artifact_gate(
    repository_root: Path,
    output_root: Path,
) -> dict[str, object]:
    repository = Path(repository_root)
    output = Path(output_root)
    sealed = output / "sealed"
    pilot = json.loads(
        (output / "development_rollout_results.json").read_text(encoding="utf-8")
    )
    power = json.loads(
        (output / "development_power_report.json").read_text(encoding="utf-8")
    )
    analysis_plan = json.loads(
        (output / "analysis_plan.json").read_text(encoding="utf-8")
    )
    p3b_evidence = json.loads(
        (
            repository
            / "experiments"
            / "paper3_independence_contract"
            / "evidence_level_report.json"
        ).read_text(encoding="utf-8")
    )
    p3b_lock = json.loads(
        (
            repository
            / "experiments"
            / "paper3_independence_contract"
            / "sealed"
            / "p3_3b_once.lock"
        ).read_text(encoding="utf-8")
    )
    p3b_source_digest = _named_source_digest(
        repository,
        P3B_SOURCE_FILES,
    )
    p3b_source_unchanged = p3b_source_digest == p3b_lock.get(
        "frozen_artifact_digests", {}
    ).get("implementation_source")
    access = audit_rollout_access(sealed, expected_phase="zero")
    contract = audit_rollout_contract()
    generator = audit_rollout_generator()
    metric = audit_fixed_metric_and_lipschitz(development_rollout_worlds()[0])
    routing = audit_routing_controls().as_dict()

    plan_payload = {
        key: value
        for key, value in analysis_plan.items()
        if key != "analysis_plan_digest"
    }
    plan_valid = bool(
        analysis_plan.get("identifier") == P3_ROLLOUT_ANALYSIS_PLAN_ID
        and analysis_plan.get("analysis_plan_digest") == _canonical_digest(plan_payload)
        and power.get("analysis_plan") == analysis_plan
    )
    pilot_valid = bool(
        pilot.get("identifier") == P3_ROLLOUT_DEVELOPMENT_ID
        and pilot.get("test_output_used") is False
        and pilot.get("failure_count") == 0
        and report_digest_valid(pilot)
    )
    power_valid = bool(
        power.get("identifier") == P3_ROLLOUT_POWER_ID
        and power.get("passed") is True
        and power.get("test_output_used") is False
        and report_digest_valid(power)
    )
    p3b_valid = bool(
        p3b_evidence.get("level_3_attained") is True
        and p3b_evidence.get("evidence_level_after") == 3
        and report_digest_valid(p3b_evidence)
        and p3b_source_unchanged
    )
    lock_ready = bool(
        (repository / "tools" / "run_paper3_rollout_sealed.py").is_file()
        and not (sealed / LOCK_FILENAME).exists()
        and not (sealed / FAILURE_FILENAME).exists()
        and not any((output / name).exists() for name in SEALED_OUTPUT_FILENAMES)
    )
    artifact_status = {
        "rollout_contract_and_digest": contract["passed"],
        "development_and_sealed_generator_audit": generator["passed"],
        "fixed_metric_and_lipschitz_audit": metric["passed"],
        "six_control_capacity_and_compute_audit": routing["passed"],
        "development_rollout_pilot": pilot_valid,
        "development_variance_and_power_report": power_valid,
        "zero_access_rollout_seed_ledger": access["passed"],
        "frozen_confirmatory_analysis_plan": plan_valid,
        "one_shot_execution_lock": lock_ready,
    }
    if tuple(artifact_status) != REQUIRED_ARTIFACTS:
        raise RuntimeError("P3-4A artifact order changed")
    source_digest = implementation_source_digest(repository)
    gate_audit = {
        "identifier": P3_ROLLOUT_GATE_ID,
        "p3b_level_3_retained": p3b_valid,
        "p3b_frozen_source_digest_unchanged": p3b_source_unchanged,
        "artifact_blockers": [
            name for name, passed in artifact_status.items() if not passed
        ],
        "test_seed_reveals": access["seed_reveals"],
        "test_result_evaluations": access["result_evaluations"],
        "planned_test_worlds": power.get("planned_test_worlds"),
        "analysis_plan_digest": analysis_plan.get("analysis_plan_digest"),
        "implementation_source_digest": source_digest,
        "evidence_level_before": 3,
        "evidence_level_after": 3,
    }
    gate_audit["gate_passed"] = bool(
        p3b_valid
        and all(artifact_status.values())
        and access["seed_reveals"] == 0
        and access["result_evaluations"] == 0
    )
    payload: dict[str, object] = {
        "identifier": P3_ROLLOUT_GATE_ID,
        "artifact_status": artifact_status,
        "gate_audit": gate_audit,
        "subreports": {
            "contract": contract,
            "generator": generator,
            "fixed_metric_and_lipschitz": metric,
            "routing_controls": routing,
            "access": access,
        },
        "frozen_artifact_digests": {
            "p3b_evidence": p3b_evidence.get("report_digest"),
            "p3b_implementation_source": p3b_source_digest,
            "development_pilot": pilot.get("report_digest"),
            "development_power": power.get("report_digest"),
            "analysis_plan": analysis_plan.get("analysis_plan_digest"),
            "implementation_source": source_digest,
        },
    }
    return {**payload, "combined_digest": _canonical_digest(payload)}


def write_rollout_artifact_gate(
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
