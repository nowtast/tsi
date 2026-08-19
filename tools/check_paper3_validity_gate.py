#!/usr/bin/env python3
"""Build and report the final zero-access P3-4B artifact gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_validity_gate import (  # noqa: E402
    build_validity_artifact_gate,
    write_validity_artifact_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "paper3_validity_v2",
    )
    args = parser.parse_args()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else REPOSITORY_ROOT / args.output_root
    )
    gate = build_validity_artifact_gate(REPOSITORY_ROOT, output_root)
    write_validity_artifact_gate(output_root / "artifact_gate.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["gate_audit"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
