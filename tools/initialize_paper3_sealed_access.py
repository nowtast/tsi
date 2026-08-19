#!/usr/bin/env python3
"""Initialize and audit opaque P3-3 sealed-test seed material."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_sealed_access import (  # noqa: E402
    audit_sealed_test_material,
    initialize_sealed_test_material,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            REPOSITORY_ROOT / "experiments" / "paper3_independence_contract" / "sealed"
        ),
    )
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    initialize_sealed_test_material(args.root)
    audit = audit_sealed_test_material(args.root)
    rendered = json.dumps(audit.as_dict(), indent=2, sort_keys=True)
    print(rendered)
    if args.write_report is not None:
        report_path = args.write_report
        if not report_path.is_absolute():
            report_path = REPOSITORY_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
