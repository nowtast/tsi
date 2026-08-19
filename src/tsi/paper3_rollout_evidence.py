"""Cumulative evidence audit after P3-4A without premature Level-4 promotion."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .paper3_rollout_analysis import P3_ROLLOUT_CONFIRMATORY_ANALYSIS_ID
from .paper3_rollout_contract import MAX_HORIZON
from .paper3_rollout_experiment import P3_ROLLOUT_SEALED_RAW_ID


P3_ROLLOUT_EVIDENCE_ID = "P3-4A-CUMULATIVE-EVIDENCE-v1"
LEVEL_4_REQUIREMENTS = (
    "open_loop_multihorizon_rollout",
    "downstream_predictive_validity",
    "learned_routing_or_structure",
    "noisy_perception",
    "variable_cardinality",
    "public_benchmark",
    "cross_family_replication",
    "artifact_reproducibility",
)


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_rollout_evidence_report(
    p3b_evidence: Mapping[str, object],
    raw: Mapping[str, object],
    analysis: Mapping[str, object],
    access: Mapping[str, object],
) -> dict[str, object]:
    level_3_retained = bool(
        p3b_evidence.get("level_3_attained") is True
        and p3b_evidence.get("evidence_level_after") == 3
        and all(p3b_evidence.get("requirements", {}).values())
    )
    open_loop = bool(
        raw.get("identifier") == P3_ROLLOUT_SEALED_RAW_ID
        and raw.get("test_output_used") is True
        and raw.get("failure_count") == 0
        and analysis.get("identifier") == P3_ROLLOUT_CONFIRMATORY_ANALYSIS_ID
        and analysis.get("passed") is True
        and analysis.get("maximum_horizon") == MAX_HORIZON
        and access.get("passed") is True
        and access.get("seed_reveals") == 1
        and access.get("result_evaluations") == 1
    )
    requirements = {
        "open_loop_multihorizon_rollout": open_loop,
        "downstream_predictive_validity": False,
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
        "identifier": P3_ROLLOUT_EVIDENCE_ID,
        "level_3_retained": level_3_retained,
        "level_4_requirements": requirements,
        "newly_satisfied_requirements": (
            ["open_loop_multihorizon_rollout"] if open_loop else []
        ),
        "evidence_level_before": 3,
        "evidence_level_after": evidence_level,
        "level_4_attained": level_4_attained,
        "publication_floor": 4,
        "publication_blocked": evidence_level < 4,
        "p3b_evidence_digest": p3b_evidence.get("report_digest"),
        "rollout_raw_digest": raw.get("report_digest"),
        "rollout_analysis_digest": analysis.get("report_digest"),
        "rollout_access_audit_digest": access.get("audit_digest"),
        "scope_boundary": (
            "exact-state oracle-routing synthetic context family; "
            "not learned routing, perception, cardinality, public benchmark, "
            "or independent environment-family replication"
        ),
    }
    return {**payload, "report_digest": _canonical_digest(payload)}


def write_rollout_evidence_report(
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
