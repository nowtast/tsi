#!/usr/bin/env python3
"""Calculate A2 prospective power from the development artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsi.research_a2_power import estimate_a2_prospective_power


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("experiments/research_a2_v1/development_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/research_a2_v1/prospective_power.json"),
    )
    parser.add_argument("--iterations", type=int, default=20_000)
    args = parser.parse_args()
    development = json.loads(args.development.read_text(encoding="utf-8"))
    report = estimate_a2_prospective_power(development, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "confirmatory_seed_created": report["confirmatory_seed_created"],
                "selected_world_count": report[
                    "selected_world_count_per_axis_or_scope_condition"
                ],
                "selected_operating_characteristics": report[
                    "selected_operating_characteristics"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
