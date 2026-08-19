#!/usr/bin/env python3
"""Run the development-only P3-4B benchmark and freeze predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_routing_controls import audit_routing_controls  # noqa: E402
from tsi.paper3_validity_contract import audit_validity_contract  # noqa: E402
from tsi.paper3_validity_experiment import (  # noqa: E402
    run_validity_development_benchmark,
    write_validity_experiment,
)
from tsi.paper3_validity_generator import audit_validity_generator  # noqa: E402
from tsi.paper3_validity_power import (  # noqa: E402
    build_validity_power_report,
    write_validity_analysis_plan,
    write_validity_power_report,
)
from tsi.paper3_validity_predictor import (  # noqa: E402
    fit_frozen_validity_predictors,
    write_validity_predictors,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "paper3_validity_v2",
    )
    args = parser.parse_args()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else REPOSITORY_ROOT / args.output_root
    )
    if (output_root / "artifact_gate.json").exists():
        raise RuntimeError("P3-4B development is frozen by an existing artifact gate")
    if (output_root / "sealed" / "p3_4b_once.lock").exists():
        raise RuntimeError("P3-4B sealed execution has already started")
    output_root.mkdir(parents=True, exist_ok=True)

    preflight = {
        "contract": audit_validity_contract(),
        "generator": audit_validity_generator(),
        "routing_controls": audit_routing_controls().as_dict(),
    }
    preflight["passed"] = all(
        report["passed"]
        for name, report in preflight.items()
        if name != "passed"
    )
    _write_json(output_root / "development_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("P3-4B development preflight failed")

    development = run_validity_development_benchmark(
        progress=lambda message: print(message, flush=True),
    )
    write_validity_experiment(
        output_root / "development_validity_results.json",
        development,
    )
    if development["failure_count"] != 0:
        raise RuntimeError("P3-4B development benchmark has failed runs")

    predictors = fit_frozen_validity_predictors(
        development,
        perform_lowo=True,
    )
    write_validity_predictors(
        output_root / "frozen_validity_predictors.json",
        predictors,
    )
    power = build_validity_power_report(predictors)
    write_validity_power_report(
        output_root / "development_power_report.json",
        power,
    )
    write_validity_analysis_plan(
        output_root / "analysis_plan.json",
        power,
    )
    print(
        json.dumps(
            {
                "development_digest": development["report_digest"],
                "predictor_report_digest": predictors["report_digest"],
                "frozen_predictor_digest": predictors["frozen_predictors"][
                    "frozen_predictor_digest"
                ],
                "development_event_rate_primary": predictors["development_event_rate_primary"],
                "development_lowo_mean_effects": predictors[
                    "development_lowo_mean_effects"
                ],
                "power_passed": power["passed"],
                "planned_test_worlds": power["planned_test_worlds"],
                "selected_conjunctive_power": power[
                    "selected_conjunctive_power"
                ],
                "selected_monte_carlo_95pct_lower_bound": power[
                    "selected_monte_carlo_95pct_lower_bound"
                ],
                "analysis_plan_digest": power["analysis_plan"][
                    "analysis_plan_digest"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if power["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
