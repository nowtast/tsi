"""Run the public development cohort for the Paper 3/4 resolution design."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tsi.paper34_resolution_analysis import summarize_cohort
from tsi.paper34_resolution_benchmark import run_world
from tsi.paper34_resolution_contract import (
    DEVELOPMENT_WORLD_COUNT,
    audit_contract,
    contract_digest,
)


DEVELOPMENT_ROOT = "TSI-P34-RESOLUTION-v1-public-development"


def world_seed(index: int) -> int:
    digest = sha256(f"{DEVELOPMENT_ROOT}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = audit_contract()
    if not audit["passed"]:
        raise RuntimeError(f"resolution contract failed: {audit['errors']}")
    rows = [run_world(index, world_seed(index)) for index in range(DEVELOPMENT_WORLD_COUNT)]
    analysis = summarize_cohort(rows)
    payload = {
        "status": "development_only_not_confirmatory",
        "contract_digest": contract_digest(),
        "development_root_commitment": sha256(DEVELOPMENT_ROOT.encode()).hexdigest(),
        "analysis": analysis,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "world_count": analysis["world_count"],
        "identification_rate": analysis["identification_rate"],
        "gates": analysis["gates"],
        "effect_intervals": analysis["effect_intervals"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
