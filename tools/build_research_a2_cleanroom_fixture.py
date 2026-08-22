#!/usr/bin/env python3
"""Build public fixed-seed A2 fixtures for the pre-freeze clean-room dry run."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tsi.research_a2_analysis import analyze_a2_axes
from tsi.research_a2_confirmatory import run_a2_cohort
from tsi.research_a2_contract import contract_digest


FIXTURE_SEED_LABEL = "TSI-RESEARCH-A2-CLEANROOM-FIXTURE-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portable",
        type=Path,
        default=Path("/tmp/research_a2_cleanroom_portable.json"),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("/tmp/research_a2_cleanroom_raw.json"),
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("/tmp/research_a2_cleanroom_analysis.json"),
    )
    args = parser.parse_args()
    seed = sha256(FIXTURE_SEED_LABEL.encode()).digest()
    axes, portable, derivation_audit = run_a2_cohort(
        seed,
        world_count=45,
        test_case_count=60,
    )
    portable["status"] = "developmental_cleanroom_fixture_with_answers"
    portable["fixture_seed_label"] = FIXTURE_SEED_LABEL
    portable_bytes = (json.dumps(portable) + "\n").encode()
    raw = {
        "status": "developmental_cleanroom_fixture",
        "contract_digest": contract_digest(),
        "fixture_seed_label": FIXTURE_SEED_LABEL,
        "derivation_audit": derivation_audit,
        "axes": axes,
    }
    analysis = {
        "status": "developmental_cleanroom_fixture_analysis",
        "contract_digest": contract_digest(),
        "portable_replay_sha256": sha256(portable_bytes).hexdigest(),
        "analysis": analyze_a2_axes(axes),
    }
    for path in (args.portable, args.raw, args.analysis):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.portable.write_bytes(portable_bytes)
    args.raw.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    args.analysis.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "fixture_seed_label": FIXTURE_SEED_LABEL,
                "world_count_per_condition": 45,
                "portable_sha256": analysis["portable_replay_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
