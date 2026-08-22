#!/usr/bin/env python3
"""Freeze Research A2 sources after review and clean-room dry run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from tsi.research_a2_contract import audit_contract, contract_digest


FROZEN_FILES = (
    "research/research_a/research_a2_preregistration_draft.json",
    "research/research_a/A2_THEORY.md",
    "research/research_a/A2_THEORY_KO.md",
    "research/research_a/A2_PREREGISTRATION_DRAFT.md",
    "research/research_a/A2_PREREGISTRATION_DRAFT_KO.md",
    "research/research_a/A2_CLEANROOM_REPLAY_SPEC.md",
    "research/research_a/A2_CLEANROOM_REPLAY_SPEC_KO.md",
    "src/tsi/research_a2_features.py",
    "src/tsi/research_a2_populations.py",
    "src/tsi/research_a2_design.py",
    "src/tsi/research_a2_development.py",
    "src/tsi/research_a2_power.py",
    "src/tsi/research_a2_seed.py",
    "src/tsi/research_a2_contract.py",
    "src/tsi/research_a2_confirmatory.py",
    "src/tsi/research_a2_analysis.py",
    "tests/test_research_a2_features.py",
    "tests/test_research_a2_populations.py",
    "tests/test_research_a2_design.py",
    "tests/test_research_a2_development.py",
    "tests/test_research_a2_power.py",
    "tests/test_research_a2_seed.py",
    "tests/test_research_a2_contract.py",
    "tests/test_research_a2_confirmatory.py",
    "tests/test_research_a2_analysis.py",
    "tools/run_research_a2_confirmatory.py",
    "tools/commit_research_a2_seed.py",
    "tools/freeze_research_a2.py",
    "tools/replay_research_a2_cleanroom.mjs",
    "research/research_a/A2_SEED_CUSTODIAN_PROTOCOL.md",
    "research/research_a/A2_SEED_CUSTODIAN_PROTOCOL_KO.md",
    "research/research_a/A2_SEED_CUSTODIAN_ATTESTATION_TEMPLATE.json",
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/research_a2_v1/freeze_manifest.json"),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("experiments/research_a2_v1/development_report.json"),
    )
    parser.add_argument(
        "--power",
        type=Path,
        default=Path("experiments/research_a2_v1/prospective_power.json"),
    )
    parser.add_argument(
        "--cleanroom-audit",
        type=Path,
        default=Path("experiments/research_a2_v1/cleanroom_dry_run_audit.json"),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate freeze prerequisites without writing a manifest",
    )
    parser.add_argument(
        "--seed-custodian-id",
        required=True,
        help="external custodian identifier fixed in the public freeze manifest",
    )
    args = parser.parse_args()
    custodian_id = args.seed_custodian_id.strip()
    if not custodian_id or any(
        marker in custodian_id.upper()
        for marker in ("REPLACE", "PLACEHOLDER", "PENDING")
    ):
        raise RuntimeError("a non-placeholder external seed custodian is required")
    contract_audit = audit_contract()
    if not contract_audit["passed"]:
        raise RuntimeError(f"contract audit failed: {contract_audit['errors']}")
    root = Path(__file__).resolve().parents[1]
    files = {relative: digest(root / relative) for relative in FROZEN_FILES}
    preregistration = json.loads(
        (root / "research/research_a/research_a2_preregistration_draft.json").read_text(
            encoding="utf-8"
        )
    )
    development = json.loads(args.development.read_text(encoding="utf-8"))
    power = json.loads(args.power.read_text(encoding="utf-8"))
    cleanroom = json.loads(args.cleanroom_audit.read_text(encoding="utf-8"))
    if preregistration.get("contract_digest_at_draft") != contract_digest():
        raise RuntimeError("A2 preregistration and executable contract diverged")
    if development.get("status") != "development_only_not_confirmatory":
        raise RuntimeError("invalid A2 development status")
    if power.get("uses_confirmatory_results") is not False:
        raise RuntimeError("A2 prospective power touched confirmatory results")
    if power.get("selected_world_count_per_axis_or_scope_condition") != 135:
        raise RuntimeError("A2 power selected a different world count")
    if cleanroom.get("passed") is not True or cleanroom.get("fixture_only") is not True:
        raise RuntimeError("A2 clean-room fixture dry run did not pass")
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "a2_freeze_prerequisites_passed_no_manifest_written",
                    "contract_digest": contract_digest(),
                    "file_count": len(files),
                    "confirmatory_seed_created": False,
                    "seed_custodian_id": custodian_id,
                }
            )
        )
        return
    payload = {
        "status": "frozen_before_a2_confirmatory_seed_generation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_digest": contract_digest(),
        "development_report_sha256": digest(args.development),
        "prospective_power_sha256": digest(args.power),
        "cleanroom_dry_run_audit_sha256": digest(args.cleanroom_audit),
        "seed_custodian_id": custodian_id,
        "seed_selection_policy": "exactly one externally generated 32-byte draw after this freeze commit is public; author generation, selection, and reroll are forbidden",
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["freeze_digest"] = sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {"freeze_digest": payload["freeze_digest"], "file_count": len(files)}
        )
    )


if __name__ == "__main__":
    main()
