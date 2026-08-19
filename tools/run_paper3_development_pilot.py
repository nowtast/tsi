#!/usr/bin/env python3
"""Run the public P3-3A pilot and freeze its world-level power report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_development_experiment import (  # noqa: E402
    run_development_pilot,
    write_development_pilot,
)
from tsi.paper3_power_analysis import (  # noqa: E402
    build_power_report,
    write_power_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(REPOSITORY_ROOT / "experiments" / "paper3_independence_contract"),
    )
    args = parser.parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root

    pilot = run_development_pilot(progress=lambda message: print(message, flush=True))
    pilot_path = output_root / "development_pilot_results.json"
    write_development_pilot(pilot_path, pilot)
    power = build_power_report(pilot)
    power_path = output_root / "development_power_report.json"
    write_power_report(power_path, power)
    print(
        json.dumps(
            {
                "pilot_path": str(pilot_path),
                "pilot_digest": pilot["report_digest"],
                "run_count": pilot["run_count"],
                "failure_count": pilot["failure_count"],
                "power_path": str(power_path),
                "power_digest": power["report_digest"],
                "passed": power["passed"],
                "planned_test_worlds": power["planned_test_worlds"],
                "minimum_simulation_power": power["minimum_simulation_power"],
                "simulation_power_95pct_lower_bound": (
                    power["simulation_power_95pct_lower_bound"]
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if power["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
