"""One-shot execution of the prospective Paper 3/4 resolution cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import secrets

from tsi.paper34_resolution_analysis import summarize_cohort
from tsi.paper34_resolution_benchmark import run_world
from tsi.paper34_resolution_contract import CONFIRMATORY_WORLD_COUNT, contract_digest


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("contract_digest") != contract_digest():
        raise RuntimeError("freeze contract digest mismatch")
    root = Path(__file__).resolve().parents[1]
    for relative, expected in freeze["files"].items():
        if digest(root / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / "ONE_SHOT.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(datetime.now(timezone.utc).isoformat() + "\n")
    root_seed = secrets.token_bytes(32)
    commitment = sha256(root_seed).hexdigest()
    seeds = [
        int.from_bytes(sha256(root_seed + index.to_bytes(4, "little")).digest()[:8], "little")
        for index in range(CONFIRMATORY_WORLD_COUNT)
    ]
    rows = [run_world(index, seed) for index, seed in enumerate(seeds)]
    calibration = tuple(float(value) for value in freeze["criterion_calibration"])
    analysis = summarize_cohort(rows, calibration=calibration, confirmatory=True)
    raw = {
        "status": "prospective_one_shot_confirmatory",
        "contract_digest": contract_digest(),
        "freeze_digest": freeze["freeze_digest"],
        "root_seed_commitment": commitment,
        "world_count": CONFIRMATORY_WORLD_COUNT,
        "rows": rows,
    }
    raw_path = args.output_dir / "raw_results.json"
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "prospective_one_shot_confirmatory_analysis",
        "contract_digest": contract_digest(),
        "freeze_digest": freeze["freeze_digest"],
        "root_seed_commitment": commitment,
        "raw_results_sha256": digest(raw_path),
        "analysis": analysis,
    }
    report_path = args.output_dir / "confirmatory_analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ledger = {
        "root_seed_hex_revealed_after_execution": root_seed.hex(),
        "root_seed_commitment": commitment,
        "commitment_verified": sha256(bytes.fromhex(root_seed.hex())).hexdigest() == commitment,
        "world_seed_derivation": "sha256(root_seed || uint32_le(world_index))[:8]",
        "confirmatory_analysis_sha256": digest(report_path),
    }
    (args.output_dir / "seed_and_integrity_ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "world_count": analysis["world_count"],
        "all_gates_passed": analysis["all_gates_passed"],
        "gates": analysis["gates"],
        "root_seed_commitment": commitment,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
