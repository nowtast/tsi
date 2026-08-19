#!/usr/bin/env python3
"""Execute the frozen P3-3B sealed test exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_analysis_plan import (  # noqa: E402
    PLANNED_TEST_WORLDS,
    analysis_plan_digest,
)
from tsi.paper3_confirmatory_analysis import (  # noqa: E402
    analyze_confirmatory_experiment,
    write_confirmatory_analysis,
)
from tsi.paper3_confirmatory_evidence import (  # noqa: E402
    audit_confirmatory_access,
    build_evidence_report,
    write_evidence_report,
)
from tsi.paper3_confirmatory_experiment import (  # noqa: E402
    run_confirmatory_experiment,
    write_confirmatory_experiment,
)
from tsi.paper3_constructive_decoder import (  # noqa: E402
    constructive_decoder_digest,
)
from tsi.paper3_multiworld import multiworld_generator_digest  # noqa: E402
from tsi.paper3_routing_controls import routing_control_digest  # noqa: E402
from tsi.paper3_routing_model import routing_model_digest  # noqa: E402
from tsi.paper3_sealed_access import (  # noqa: E402
    append_test_access_event,
    reveal_sealed_test_seed,
)
from tsi.paper3_sealed_worlds import (  # noqa: E402
    P3_SEALED_WORLD_GENERATOR_ID,
    sealed_world_manifest_digest,
    sealed_world_mechanisms,
)


LOCK_FILENAME = "p3_3b_once.lock"
FAILURE_FILENAME = "p3_3b_execution_failure.json"
SOURCE_FILES = (
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


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_digest() -> str:
    digest = sha256()
    for relative in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_frozen_gate(output_root: Path) -> tuple[dict[str, object], dict[str, str]]:
    gate = json.loads((output_root / "artifact_gate.json").read_text(encoding="utf-8"))
    gate_audit = gate.get("gate_audit", {})
    if (
        gate_audit.get("gate_passed") is not True
        or gate_audit.get("test_seed_reveals") != 0
        or gate_audit.get("test_result_evaluations") != 0
        or not all(gate.get("artifact_status", {}).values())
    ):
        raise RuntimeError("P3-3A gate is not a passing zero-access gate")
    power = json.loads(
        (output_root / "development_power_report.json").read_text(encoding="utf-8")
    )
    if (
        power.get("passed") is not True
        or power.get("planned_test_worlds") != PLANNED_TEST_WORLDS
        or power.get("analysis_plan_digest") != analysis_plan_digest()
    ):
        raise RuntimeError("power report does not match the frozen analysis")
    frozen = {
        "p3a_gate": str(gate["combined_digest"]),
        "analysis_plan": analysis_plan_digest(),
        "power_report": str(power["report_digest"]),
        "multiworld_generator": multiworld_generator_digest(),
        "constructive_decoder": constructive_decoder_digest(),
        "routing_controls": routing_control_digest(),
        "routing_model": routing_model_digest(),
        "implementation_source": _source_digest(),
    }
    return gate, frozen


def _acquire_once_lock(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        encoded = (f"{json.dumps(payload, indent=2, sort_keys=True)}\n").encode("utf-8")
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("failed to write complete one-shot lock")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(REPOSITORY_ROOT / "experiments" / "paper3_independence_contract"),
    )
    args = parser.parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    sealed_root = output_root / "sealed"
    gate, frozen = _load_frozen_gate(output_root)
    lock_path = sealed_root / LOCK_FILENAME
    started_at = datetime.now(timezone.utc).isoformat()
    _acquire_once_lock(
        lock_path,
        {
            "identifier": "P3-3B-ONE-SHOT-LOCK-v1",
            "started_at_utc": started_at,
            "p3a_gate_digest": gate["combined_digest"],
            "frozen_artifact_digests": frozen,
        },
    )

    try:
        secret, commitment = reveal_sealed_test_seed(
            sealed_root,
            gate_digest=str(gate["combined_digest"]),
            frozen_artifact_digests=frozen,
        )
        append_test_access_event(
            sealed_root,
            "test_prediction_started",
            {
                "planned_test_worlds": PLANNED_TEST_WORLDS,
                "analysis_plan_digest": analysis_plan_digest(),
            },
        )
        mechanisms = sealed_world_mechanisms(
            secret,
            commitment,
            world_count=PLANNED_TEST_WORLDS,
        )
        del secret
        manifest_digest = sealed_world_manifest_digest(mechanisms)
        manifest = {
            "identifier": P3_SEALED_WORLD_GENERATOR_ID,
            "commitment": commitment,
            "world_count": len(mechanisms),
            "manifest_digest": manifest_digest,
            "active_signature_count": len(
                {world.active_parameter_signature for world in mechanisms}
            ),
            "worlds": [world.as_dict() for world in mechanisms],
        }
        _write_json_atomic(output_root / "sealed_world_manifest.json", manifest)

        raw = run_confirmatory_experiment(
            mechanisms,
            commitment,
            p3a_gate_digest=str(gate["combined_digest"]),
            frozen_artifact_digests=frozen,
            progress=lambda message: print(message, flush=True),
        )
        raw_path = output_root / "sealed_test_raw_results.json"
        write_confirmatory_experiment(raw_path, raw)
        append_test_access_event(
            sealed_root,
            "test_prediction_completed",
            {
                "raw_result_digest": raw["report_digest"],
                "run_count": raw["run_count"],
                "failure_count": raw["failure_count"],
            },
        )

        analysis = analyze_confirmatory_experiment(raw)
        analysis_path = output_root / "sealed_test_confirmatory_analysis.json"
        write_confirmatory_analysis(analysis_path, analysis)
        append_test_access_event(
            sealed_root,
            "test_result_evaluated",
            {
                "confirmatory_analysis_digest": analysis["report_digest"],
                "passed": analysis["passed"],
            },
        )
        append_test_access_event(
            sealed_root,
            "test_report_generated",
            {
                "raw_result_path": raw_path.name,
                "analysis_path": analysis_path.name,
            },
        )

        access = audit_confirmatory_access(sealed_root)
        evidence = build_evidence_report(manifest, raw, analysis, access)
        evidence_path = output_root / "evidence_level_report.json"
        write_evidence_report(evidence_path, evidence)
        _write_json_atomic(
            lock_path,
            {
                "identifier": "P3-3B-ONE-SHOT-LOCK-v1",
                "started_at_utc": started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "p3a_gate_digest": gate["combined_digest"],
                "frozen_artifact_digests": frozen,
                "raw_result_digest": raw["report_digest"],
                "confirmatory_analysis_digest": analysis["report_digest"],
                "evidence_report_digest": evidence["report_digest"],
                "evidence_level_after": evidence["evidence_level_after"],
            },
        )
        print(
            json.dumps(
                {
                    "raw_result_digest": raw["report_digest"],
                    "confirmatory_analysis_digest": analysis["report_digest"],
                    "confirmatory_passed": analysis["passed"],
                    "evidence_report_digest": evidence["report_digest"],
                    "evidence_level_after": evidence["evidence_level_after"],
                    "publication_blocked": evidence["publication_blocked"],
                    "test_seed_reveals": access["test_seed_reveals"],
                    "test_result_evaluations": (access["test_result_evaluations"]),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if evidence["level_3_attained"] else 2
    except Exception as error:
        _write_json_atomic(
            output_root / FAILURE_FILENAME,
            {
                "identifier": "P3-3B-ONE-SHOT-FAILURE-v1",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "one_shot_lock": str(lock_path),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
