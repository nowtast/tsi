#!/usr/bin/env python3
"""Execute the frozen P3-4A rollout test exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_rollout_access import (  # noqa: E402
    append_rollout_access_event,
    audit_rollout_access,
    reveal_rollout_seed,
)
from tsi.paper3_rollout_analysis import (  # noqa: E402
    analyze_rollout_confirmatory,
    write_rollout_confirmatory_analysis,
)
from tsi.paper3_rollout_evidence import (  # noqa: E402
    build_rollout_evidence_report,
    write_rollout_evidence_report,
)
from tsi.paper3_rollout_experiment import (  # noqa: E402
    run_rollout_sealed_experiment,
    write_rollout_experiment,
)
from tsi.paper3_rollout_gate import (  # noqa: E402
    implementation_source_digest,
    report_digest_valid,
)
from tsi.paper3_rollout_generator import (  # noqa: E402
    rollout_manifest,
    sealed_rollout_trajectories,
    sealed_rollout_worlds,
)
from tsi.paper3_rollout_once import (  # noqa: E402
    FAILURE_FILENAME,
    LOCK_FILENAME,
    acquire_once_lock,
    write_json_atomic,
)


def _load_frozen_artifacts(
    output_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    gate = json.loads((output_root / "artifact_gate.json").read_text(encoding="utf-8"))
    power = json.loads(
        (output_root / "development_power_report.json").read_text(encoding="utf-8")
    )
    analysis_plan = json.loads(
        (output_root / "analysis_plan.json").read_text(encoding="utf-8")
    )
    if (
        gate.get("gate_audit", {}).get("gate_passed") is not True
        or not all(gate.get("artifact_status", {}).values())
        or gate.get("gate_audit", {}).get("test_seed_reveals") != 0
        or gate.get("gate_audit", {}).get("test_result_evaluations") != 0
        or not report_digest_valid(gate, "combined_digest")
    ):
        raise RuntimeError("P3-4A artifact gate is not a passing zero-access gate")
    if (
        power.get("passed") is not True
        or power.get("analysis_plan") != analysis_plan
        or power.get("planned_test_worlds") != analysis_plan.get("planned_test_worlds")
    ):
        raise RuntimeError("P3-4A power report and analysis plan disagree")
    source_digest = implementation_source_digest(REPOSITORY_ROOT)
    if source_digest != gate["gate_audit"]["implementation_source_digest"]:
        raise RuntimeError("P3-4A implementation changed after the final freeze")
    return gate, power, analysis_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "paper3_rollout",
    )
    args = parser.parse_args()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else REPOSITORY_ROOT / args.output_root
    )
    sealed_root = output_root / "sealed"
    lock_path = sealed_root / LOCK_FILENAME
    failure_path = sealed_root / FAILURE_FILENAME
    gate, _power, analysis_plan = _load_frozen_artifacts(output_root)
    frozen_digests = dict(gate["frozen_artifact_digests"])
    started_at = datetime.now(timezone.utc).isoformat()
    acquire_once_lock(
        lock_path,
        {
            "identifier": "P3-4A-ONE-SHOT-LOCK-v1",
            "started_at_utc": started_at,
            "gate_digest": gate["combined_digest"],
            "frozen_artifact_digests": frozen_digests,
        },
    )

    try:
        secret, commitment = reveal_rollout_seed(
            sealed_root,
            gate_digest=str(gate["combined_digest"]),
            frozen_artifact_digests=frozen_digests,
        )
        planned_worlds = int(analysis_plan["planned_test_worlds"])
        append_rollout_access_event(
            sealed_root,
            "rollout_prediction_started",
            {
                "planned_test_worlds": planned_worlds,
                "analysis_plan_digest": analysis_plan["analysis_plan_digest"],
            },
        )
        worlds = sealed_rollout_worlds(
            secret,
            commitment,
            planned_worlds,
        )
        trajectories = {
            world.world_index: sealed_rollout_trajectories(
                secret,
                commitment,
                world.world_index,
            )
            for world in worlds
        }
        del secret
        manifest = rollout_manifest(worlds, trajectories)
        manifest_path = output_root / "sealed_rollout_manifest.json"
        write_json_atomic(manifest_path, manifest)

        raw = run_rollout_sealed_experiment(
            worlds,
            trajectories,
            analysis_plan_digest=str(analysis_plan["analysis_plan_digest"]),
            progress=lambda message: print(message, flush=True),
        )
        raw_path = output_root / "sealed_rollout_raw_results.json"
        write_rollout_experiment(raw_path, raw)
        append_rollout_access_event(
            sealed_root,
            "rollout_prediction_completed",
            {
                "raw_result_digest": raw["report_digest"],
                "run_count": raw["run_count"],
                "failure_count": raw["failure_count"],
            },
        )

        analysis = analyze_rollout_confirmatory(raw, analysis_plan)
        analysis_path = output_root / "sealed_rollout_confirmatory_analysis.json"
        write_rollout_confirmatory_analysis(analysis_path, analysis)
        append_rollout_access_event(
            sealed_root,
            "rollout_result_evaluated",
            {
                "confirmatory_analysis_digest": analysis["report_digest"],
                "passed": analysis["passed"],
            },
        )
        append_rollout_access_event(
            sealed_root,
            "rollout_report_generated",
            {
                "manifest_path": manifest_path.name,
                "raw_result_path": raw_path.name,
                "analysis_path": analysis_path.name,
            },
        )
        access = audit_rollout_access(sealed_root, expected_phase="final")
        p3b_evidence = json.loads(
            (
                REPOSITORY_ROOT
                / "experiments"
                / "paper3_independence_contract"
                / "evidence_level_report.json"
            ).read_text(encoding="utf-8")
        )
        evidence = build_rollout_evidence_report(
            p3b_evidence,
            raw,
            analysis,
            access,
        )
        evidence_path = output_root / "rollout_evidence_report.json"
        write_rollout_evidence_report(evidence_path, evidence)
        completed_at = datetime.now(timezone.utc).isoformat()
        write_json_atomic(
            lock_path,
            {
                "identifier": "P3-4A-ONE-SHOT-LOCK-v1",
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "gate_digest": gate["combined_digest"],
                "frozen_artifact_digests": frozen_digests,
                "manifest_digest": manifest["manifest_digest"],
                "raw_result_digest": raw["report_digest"],
                "confirmatory_analysis_digest": analysis["report_digest"],
                "evidence_report_digest": evidence["report_digest"],
                "confirmatory_passed": analysis["passed"],
                "evidence_level_after": evidence["evidence_level_after"],
            },
        )
        print(
            json.dumps(
                {
                    "manifest_digest": manifest["manifest_digest"],
                    "raw_result_digest": raw["report_digest"],
                    "confirmatory_analysis_digest": analysis["report_digest"],
                    "confirmatory_passed": analysis["passed"],
                    "evidence_report_digest": evidence["report_digest"],
                    "evidence_level_after": evidence["evidence_level_after"],
                    "level_4_attained": evidence["level_4_attained"],
                    "publication_blocked": evidence["publication_blocked"],
                    "seed_reveals": access["seed_reveals"],
                    "result_evaluations": access["result_evaluations"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if analysis["passed"] and access["passed"] else 1
    except Exception as error:
        write_json_atomic(
            failure_path,
            {
                "identifier": "P3-4A-EXECUTION-FAILURE-v1",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "gate_digest": gate["combined_digest"],
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "rerun_permitted": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
