#!/usr/bin/env python3
"""Run the explicitly nonconfirmatory Research A development pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsi.research_a_development import run_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/research_a_v1/development_report.json"),
    )
    parser.add_argument("--world-count", type=int, default=36)
    parser.add_argument("--test-case-count", type=int, default=1200)
    parser.add_argument("--sample-sizes", type=int, nargs="+")
    args = parser.parse_args()
    report = run_development(
        world_count=args.world_count,
        sample_sizes=args.sample_sizes or (50, 100, 200, 300, 400, 800, 1600, 3200, 6400, 12800),
        test_case_count=args.test_case_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "world_count", "sample_sizes")}, indent=2))


if __name__ == "__main__":
    main()
