#!/usr/bin/env python3
"""Run the deterministic Paper 3 oracle benchmark and persist its report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from tsi.paper3_oracle_benchmark import run_p3_oracle_benchmark


def main() -> int:
    report = run_p3_oracle_benchmark()
    output = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "paper3_oracle_benchmark"
        / "results.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
