#!/usr/bin/env python3
"""Create the zero-access P3-4A rollout seed commitment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_rollout_access import (  # noqa: E402
    audit_rollout_access,
    initialize_rollout_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=(REPOSITORY_ROOT / "experiments" / "paper3_rollout" / "sealed"),
    )
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPOSITORY_ROOT / args.root
    initialize_rollout_seed(root)
    audit = audit_rollout_access(root, expected_phase="zero")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
