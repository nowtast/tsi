"""Export portable cases for a second-language clean-room implementation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tsi.paper34_resolution_benchmark import generate_cases, world_spec
from tsi.paper34_resolution_contract import (
    CONFIRMATORY_WORLD_COUNT,
    OOD_CASES_PER_WORLD,
    OOD_NOISE_PROBABILITY,
    SELECTION_CASES_PER_WORLD,
    TRAIN_CASES_PER_WORLD,
    TRAIN_NOISE_PROBABILITY,
    contract_digest,
)


def _case(case) -> list[object]:
    return [list(case.source), list(case.action), list(case.observed)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    root_seed = bytes.fromhex(ledger["root_seed_hex_revealed_after_execution"])
    if sha256(root_seed).hexdigest() != raw["root_seed_commitment"]:
        raise RuntimeError("revealed root seed does not match the commitment")
    worlds = []
    for index in range(CONFIRMATORY_WORLD_COUNT):
        seed = int.from_bytes(
            sha256(root_seed + index.to_bytes(4, "little")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        spec = world_spec(index, rng)
        train = generate_cases(spec, TRAIN_CASES_PER_WORLD, rng, composition=False, noise_probability=TRAIN_NOISE_PROBABILITY)
        selection = generate_cases(spec, SELECTION_CASES_PER_WORLD, rng, composition=False, noise_probability=TRAIN_NOISE_PROBABILITY)
        test = generate_cases(spec, OOD_CASES_PER_WORLD, rng, composition=True, noise_probability=OOD_NOISE_PROBABILITY)
        worlds.append({
            "world_index": index,
            "graph": [spec.graph[0], list(spec.graph[1])],
            "families": list(spec.families),
            "train": [_case(case) for case in train],
            "selection": [_case(case) for case in selection],
            "test": [_case(case) for case in test],
            "expected_row": raw["rows"][index],
        })
    payload = {
        "status": "portable_inputs_for_cleanroom_reimplementation",
        "contract_digest": contract_digest(),
        "root_seed_commitment": raw["root_seed_commitment"],
        "worlds": worlds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"world_count": len(worlds), "output_sha256": sha256(args.output.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
