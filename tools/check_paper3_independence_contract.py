#!/usr/bin/env python3
"""Audit and serialize the frozen TSI P3-3A preregistration contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_independence_contract import (  # noqa: E402
    audit_p3_3a_independence_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    report = audit_p3_3a_independence_contract()
    rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    print(rendered)

    if args.write_report is not None:
        output = args.write_report
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n")

    return 0 if report.static_contract_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
