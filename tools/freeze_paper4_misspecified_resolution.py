from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from tsi.paper4_misspecified_contract import contract_digest


FILES = (
    "src/tsi/paper34_resolution_benchmark.py",
    "src/tsi/paper34_resolution_contract.py",
    "src/tsi/paper4_misspecified_contract.py",
    "src/tsi/paper4_misspecified_resolution.py",
    "tests/test_paper4_misspecified_resolution.py",
    "tools/run_paper4_misspecified_confirmatory.py",
)


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--development-report", type=Path, required=True)
    args = parser.parse_args()
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    if development["contract_digest"] != contract_digest() or not all(development["analysis"]["gates"].values()):
        raise RuntimeError("development stress gates or contract failed")
    root = Path(__file__).resolve().parents[1]
    payload = {
        "status": "frozen_before_confirmatory_seed_generation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_digest": contract_digest(),
        "development_report_sha256": sha(args.development_report),
        "files": {relative: sha(root / relative) for relative in FILES},
    }
    payload["freeze_digest"] = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"freeze_digest": payload["freeze_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
