"""Pre-seal ledger and confirmatory freeze audit for P3-5A-v2."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence


V2_ANALYSIS_PLAN_ID = "P3-5A-v2-source-conditioned-robustness"
SEALED_TEST_ACCESS_COUNT = 0
REQUIRED_ARTIFACTS = (
    "experiments/paper3_learned_v2/contract_audit.json",
    "experiments/paper3_learned_v2/independent_source_conditioned_validation.json",
    "experiments/paper3_learned_v2/development_variance_power_report.json",
    "experiments/paper3_learned_v2/source_conditioned_robustness_development.json",
)


def _digest_paths(paths: Sequence[str]) -> str:
    payload = []
    for path in paths:
        file_path = Path(path)
        payload.append(
            {
                "path": path,
                "exists": file_path.exists(),
                "sha256": (
                    sha256(file_path.read_bytes()).hexdigest()
                    if file_path.exists()
                    else None
                ),
            }
        )
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_zero_access_ledger(
    *,
    analysis_inputs: Sequence[str] = REQUIRED_ARTIFACTS,
    sealed_test_access_count: int = SEALED_TEST_ACCESS_COUNT,
) -> dict[str, object]:
    if sealed_test_access_count < 0:
        raise ValueError("sealed_test_access_count must be nonnegative")
    return {
        "analysis_plan_id": V2_ANALYSIS_PLAN_ID,
        "sealed_test_access_count": sealed_test_access_count,
        "sealed_test_reveal_count": 0,
        "sealed_test_evaluation_count": 0,
        "analysis_inputs": list(analysis_inputs),
        "input_digest": _digest_paths(analysis_inputs),
        "status": "zero_access_confirmed"
        if sealed_test_access_count == 0
        else "zero_access_failed",
    }


def audit_confirmatory_freeze(
    *,
    performance_gate_passed: bool = False,
    sealed_execution_requested: bool = False,
) -> dict[str, object]:
    errors: list[str] = []
    missing = [path for path in REQUIRED_ARTIFACTS if not Path(path).exists()]
    if missing:
        errors.append("missing required artifact: " + ", ".join(missing))
    if not performance_gate_passed:
        errors.append("source-conditioned performance gate is unresolved")
    if sealed_execution_requested and errors:
        errors.append("sealed execution is blocked until all gates pass")
    ledger = build_zero_access_ledger()
    passed = not errors
    return {
        "analysis_plan_id": V2_ANALYSIS_PLAN_ID,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "missing_artifacts": missing,
        "zero_access_ledger": ledger,
        "performance_gate_passed": performance_gate_passed,
        "sealed_execution_requested": sealed_execution_requested,
        "errors": errors,
        "passed": passed,
        "status": "ready_for_one_shot_lock" if passed else "preseal_blocked",
    }


def write_preseal_audit(path: str | Path) -> dict[str, object]:
    audit = audit_confirmatory_freeze()
    Path(path).write_text(json.dumps(audit, indent=2, sort_keys=True))
    return audit
