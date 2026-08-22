#!/usr/bin/env python3
"""Run the explicitly nonconfirmatory Research A2 development study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsi.research_a2_development import run_a2_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/research_a2_v1/development_report.json"),
    )
    parser.add_argument("--matched-world-count", type=int, default=36)
    parser.add_argument("--misspecification-world-count", type=int, default=45)
    parser.add_argument("--test-case-count", type=int, default=600)
    args = parser.parse_args()
    report = run_a2_development(
        matched_world_count=args.matched_world_count,
        misspecification_world_count=args.misspecification_world_count,
        test_case_count=args.test_case_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "confirmatory_seed_created": report["confirmatory_seed_created"],
                "matched_world_count": report[
                    "matched_world_count_per_efficiency_axis"
                ],
                "misspecification_world_count": report[
                    "world_count_per_misspecification_condition"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
