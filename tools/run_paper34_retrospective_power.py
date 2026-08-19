#!/usr/bin/env python3
"""Generate the retrospective Paper 3/4 sample-size justification."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tsi.paper34_retrospective_power import estimate_retrospective_power


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("development_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=20_000)
    args = parser.parse_args()

    development = json.loads(
        args.development_report.read_text(encoding="utf-8")
    )
    report = estimate_retrospective_power(
        development, iterations=args.iterations
    )
    report["development_report_sha256"] = sha256(
        args.development_report.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_world_count": report["selected_world_count"],
                "selected_conjunctive_gate_power": report[
                    "selected_conjunctive_gate_power"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
