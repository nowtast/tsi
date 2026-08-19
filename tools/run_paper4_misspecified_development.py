from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tsi.paper4_misspecified_contract import DEVELOPMENT_WORLDS, contract_digest
from tsi.paper4_misspecified_resolution import run_stress_world, summarize_stress


ROOT = "TSI-P4-OUTSIDE-MODEL-FAMILY-v1-development"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    seeds = [int.from_bytes(sha256(f"{ROOT}:{i}".encode()).digest()[:8], "little") for i in range(DEVELOPMENT_WORLDS)]
    rows = tuple(run_stress_world(i, seed) for i, seed in enumerate(seeds))
    payload = {"status": "development_only", "contract_digest": contract_digest(), "analysis": summarize_stress(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["analysis"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
