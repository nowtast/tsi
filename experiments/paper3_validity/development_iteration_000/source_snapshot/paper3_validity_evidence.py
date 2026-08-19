"""Cumulative evidence audit after P3-4B without premature Level-4 promotion."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .paper3_rollout_evidence import (
    LEVEL_4_REQUIREMENTS,
    P3_ROLLOUT_EVIDENCE_ID,
)
from .paper3_validity_analysis import P3_VALIDITY_CONFIRMATORY_ANALYSIS_ID
from .paper3_validity_experiment import P3_VALIDITY_SEALED_RAW_ID


P3_VALIDITY_EVIDENCE_ID = "P3-4B-CUMULATIVE-EVIDENCE-v1"


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _report_digest_valid(report: Mapping[str, object]) -> bool:
    payload = {
        key: value for key, value in report.items() if key != "report_digest"
    }
    return report.get("report_digest") == _canonical_digest(payload)


def build_validity_evidence_report(
    p3a_evidence: Mapping[str, object],
    raw: Mapping[str, object],
    analysis: Mapping[str, object],
    access: Mapping[str, object],
) -> dict[str, object]:
    p3a_requirements = p3a_evidence.get("level_4_requirements")
    level_3_retained = bool(
        p3a_evidence.get("identifier") == P3_ROLLOUT_EVIDENCE_ID
        and p3a_evidence.get("level_3_retained") is True
        and p3a_evidence.get("evidence_level_after") == 3
        and isinstance(p3a_requirements, dict)
        and p3a_requirements.get("open_loop_multihorizon_rollout") is True
        and _report_digest_valid(p3a_evidence)
    )
    downstream = bool(
        raw.get("identifier") == P3_VALIDITY_SEALED_RAW_ID
        and raw.get("test_output_used") is True
        and raw.get("failure_count") == 0
        and analysis.get("identifier") == P3_VALIDITY_CONFIRMATORY_ANALYSIS_ID
        and analysis.get("passed") is True
        and access.get("passed") is True
        and access.get("seed_reveals") == 1
        and access.get("result_evaluations") == 1
    )
    requirements = {
        "open_loop_multihorizon_rollout": bool(
            isinstance(p3a_requirements, dict)
            and p3a_requirements.get("open_loop_multihorizon_rollout")
        ),
        "downstream_predictive_validity": downstream,
        "learned_routing_or_structure": False,
        "noisy_perception": False,
        "variable_cardinality": False,
        "public_benchmark": False,
        "cross_family_replication": False,
        "artifact_reproducibility": False,
    }
    if tuple(requirements) != LEVEL_4_REQUIREMENTS:
        raise RuntimeError("Level-4 requirement order changed")
    level_4_attained = bool(level_3_retained and all(requirements.values()))
    evidence_level = 4 if level_4_attained else (3 if level_3_retained else 2)
    payload: dict[str, object] = {
        "identifier": P3_VALIDITY_EVIDENCE_ID,
        "level_3_retained": level_3_retained,
        "level_4_requirements": requirements,
        "satisfied_requirement_count": sum(requirements.values()),
        "newly_satisfied_requirements": (
            ["downstream_predictive_validity"] if downstream else []
        ),
        "evidence_level_before": 3,
        "evidence_level_after": evidence_level,
        "level_4_attained": level_4_attained,
        "publication_floor": 4,
        "publication_blocked": evidence_level < 4,
        "p3a_evidence_digest": p3a_evidence.get("report_digest"),
        "validity_raw_digest": raw.get("report_digest"),
        "validity_analysis_digest": analysis.get("report_digest"),
        "validity_access_audit_digest": access.get("audit_digest"),
        "scope_boundary": (
            "exact-state oracle-routing synthetic context family with frozen "
            "development-fitted diagnostic predictors; not learned routing, "
            "perception, cardinality, public benchmark, or independent "
            "environment-family replication"
        ),
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def write_validity_evidence_report(
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
