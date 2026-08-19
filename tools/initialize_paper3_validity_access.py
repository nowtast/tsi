#!/usr/bin/env python3
"""Initialize the zero-access P3-4B sealed seed escrow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_validity_access import (  # noqa: E402
    audit_validity_access,
    initialize_validity_seed,
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
    sealed_root = output_root / "sealed"
    initialize_validity_seed(sealed_root)
    audit = audit_validity_access(sealed_root, expected_phase="zero")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
