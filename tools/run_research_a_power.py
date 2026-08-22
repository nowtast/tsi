#!/usr/bin/env python3
"""Compute Research A prospective power from development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsi.research_a_power import estimate_prospective_power


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("experiments/research_a_v1/development_report_low_grid.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/research_a_v1/prospective_power.json"),
    )
    parser.add_argument("--iterations", type=int, default=20_000)
    args = parser.parse_args()
    development = json.loads(args.development.read_text(encoding="utf-8"))
    report = estimate_prospective_power(development, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_world_count": report["selected_world_count"],
                "selected_ordered_transition_power": report[
                    "selected_ordered_transition_power"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
