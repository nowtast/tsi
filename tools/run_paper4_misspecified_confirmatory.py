from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import secrets

from tsi.paper4_misspecified_contract import CONFIRMATORY_WORLDS, contract_digest
from tsi.paper4_misspecified_resolution import run_stress_world, summarize_stress


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze["contract_digest"] != contract_digest():
        raise RuntimeError("contract digest mismatch")
    root = Path(__file__).resolve().parents[1]
    for relative, expected in freeze["files"].items():
        if sha(root / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ONE_SHOT.lock").open("x", encoding="utf-8") as handle:
        handle.write(datetime.now(timezone.utc).isoformat() + "\n")
    root_seed = secrets.token_bytes(32)
    commitment = sha256(root_seed).hexdigest()
    seeds = [int.from_bytes(sha256(root_seed + index.to_bytes(4, "little")).digest()[:8], "little") for index in range(CONFIRMATORY_WORLDS)]
    rows = tuple(run_stress_world(index, seed) for index, seed in enumerate(seeds))
    analysis = summarize_stress(rows)
    payload = {
        "status": "prospective_one_shot_confirmatory",
        "contract_digest": contract_digest(),
        "freeze_digest": freeze["freeze_digest"],
        "root_seed_commitment": commitment,
        "analysis": analysis,
        "rows": rows,
    }
    result = args.output_dir / "confirmatory_analysis.json"
    result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ledger = {
        "root_seed_hex_revealed_after_execution": root_seed.hex(),
        "root_seed_commitment": commitment,
        "commitment_verified": sha256(bytes.fromhex(root_seed.hex())).hexdigest() == commitment,
        "confirmatory_analysis_sha256": sha(result),
    }
    (args.output_dir / "seed_and_integrity_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_passed": all(analysis["gates"].values()), "analysis": analysis, "root_seed_commitment": commitment}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
