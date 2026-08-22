#!/usr/bin/env python3
"""Create A2 seed escrow and commitment after the source freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import secrets

from tsi.research_a2_contract import contract_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument(
        "--escrow",
        type=Path,
        default=Path("experiments/research_a2_v1/confirmatory/root_seed.escrow"),
    )
    parser.add_argument(
        "--commitment",
        type=Path,
        default=Path("experiments/research_a2_v1/seed_commitment.json"),
    )
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("contract_digest") != contract_digest():
        raise RuntimeError("A2 freeze contract digest mismatch")
    root = Path(__file__).resolve().parents[1]
    for relative, expected in freeze["files"].items():
        if sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen A2 source changed: {relative}")
    args.escrow.parent.mkdir(parents=True, exist_ok=True)
    with args.escrow.open("xb") as handle:
        root_seed = secrets.token_bytes(32)
        handle.write(root_seed)
    payload = {
        "status": "a2_seed_committed_before_one_shot_execution",
        "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_digest": contract_digest(),
        "freeze_digest": freeze["freeze_digest"],
        "root_seed_commitment": sha256(root_seed).hexdigest(),
        "escrow_path_not_public": str(args.escrow),
        "execution_policy": "commit and push this file before one-shot execution",
    }
    with args.commitment.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"root_seed_commitment": payload["root_seed_commitment"]}))


if __name__ == "__main__":
    main()
