"""Freeze the prospective resolution implementation before seed generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from tsi.paper34_resolution_contract import audit_contract, contract_digest


FROZEN_FILES = (
    "src/tsi/paper34_resolution_contract.py",
    "src/tsi/paper34_resolution_benchmark.py",
    "src/tsi/paper34_resolution_analysis.py",
    "tests/test_paper34_resolution.py",
    "tools/run_paper34_resolution_development.py",
    "tools/run_paper34_resolution_confirmatory.py",
)


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--development-report", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_contract()
    if not audit["passed"]:
        raise RuntimeError(f"contract audit failed: {audit['errors']}")
    root = Path(__file__).resolve().parents[1]
    files = {relative: file_digest(root / relative) for relative in FROZEN_FILES}
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    if development.get("contract_digest") != contract_digest():
        raise RuntimeError("development report was produced under a different contract")
    payload = {
        "status": "frozen_before_confirmatory_seed_generation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_digest": contract_digest(),
        "development_report_sha256": file_digest(args.development_report),
        "criterion_calibration": development["analysis"]["criterion_calibration"],
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["freeze_digest"] = sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"freeze_digest": payload["freeze_digest"], "file_count": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
