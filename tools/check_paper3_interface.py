#!/usr/bin/env python3
"""Audit and serialize the frozen TSI Paper 3 structural interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_interface import audit_frozen_paper3_interface  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    report = audit_frozen_paper3_interface()
    rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    print(rendered)

    if args.write_report is not None:
        output = args.write_report
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
