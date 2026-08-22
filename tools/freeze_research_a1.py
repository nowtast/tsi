#!/usr/bin/env python3
"""Freeze Research A1 sources before any confirmatory seed is generated."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from tsi.research_a_contract import audit_contract, contract_digest


FROZEN_FILES = (
    "research/research_a/preregistration_draft.json",
    "research/research_a/theory.md",
    "research/research_a/theory_KO.md",
    "src/tsi/research_a_sample_complexity.py",
    "src/tsi/research_a_contract.py",
    "src/tsi/research_a_design.py",
    "src/tsi/research_a_confirmatory.py",
    "src/tsi/research_a_analysis.py",
    "tests/test_research_a_sample_complexity.py",
    "tests/test_research_a_contract.py",
    "tests/test_research_a_design.py",
    "tests/test_research_a_confirmatory.py",
    "tests/test_research_a_analysis.py",
    "tools/run_research_a1_confirmatory.py",
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/research_a_v1/freeze_manifest.json"),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("experiments/research_a_v1/development_report_low_grid.json"),
    )
    parser.add_argument(
        "--power",
        type=Path,
        default=Path("experiments/research_a_v1/prospective_power.json"),
    )
    args = parser.parse_args()
    contract_audit = audit_contract()
    if not contract_audit["passed"]:
        raise RuntimeError(f"contract audit failed: {contract_audit['errors']}")
    root = Path(__file__).resolve().parents[1]
    files = {relative: digest(root / relative) for relative in FROZEN_FILES}
    preregistration = json.loads(
        (root / "research/research_a/preregistration_draft.json").read_text(
            encoding="utf-8"
        )
    )
    development = json.loads(args.development.read_text(encoding="utf-8"))
    power = json.loads(args.power.read_text(encoding="utf-8"))
    if preregistration.get("contract_digest_at_draft") != contract_digest():
        raise RuntimeError("preregistration and executable contract diverged")
    if development.get("status") != "development_only_not_confirmatory":
        raise RuntimeError("invalid development status")
    if power.get("uses_confirmatory_results") is not False:
        raise RuntimeError("prospective power touched confirmatory results")
    if power.get("selected_world_count") != 126:
        raise RuntimeError("prospective power selected a different world count")
    payload = {
        "status": "frozen_before_confirmatory_seed_generation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_digest": contract_digest(),
        "development_report_sha256": digest(args.development),
        "prospective_power_sha256": digest(args.power),
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["freeze_digest"] = sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"freeze_digest": payload["freeze_digest"], "file_count": len(files)}))


if __name__ == "__main__":
    main()
