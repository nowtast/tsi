#!/usr/bin/env python3
"""Run the public P3-4A rollout pilot and freeze its power plan."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_rollout_access import audit_rollout_access  # noqa: E402
from tsi.paper3_rollout_contract import audit_rollout_contract  # noqa: E402
from tsi.paper3_rollout_evaluator import (  # noqa: E402
    audit_fixed_metric_and_lipschitz,
)
from tsi.paper3_rollout_experiment import (  # noqa: E402
    run_rollout_development_pilot,
    write_rollout_experiment,
)
from tsi.paper3_rollout_generator import (  # noqa: E402
    audit_rollout_generator,
    development_rollout_worlds,
)
from tsi.paper3_rollout_power import (  # noqa: E402
    build_rollout_power_report,
    write_rollout_analysis_plan,
    write_rollout_power_report,
)
from tsi.paper3_routing_controls import audit_routing_controls  # noqa: E402


def _canonical_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        default=REPOSITORY_ROOT / "experiments" / "paper3_rollout",
    )
    args = parser.parse_args()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else REPOSITORY_ROOT / args.output_root
    )
    sealed_root = output_root / "sealed"

    access = audit_rollout_access(sealed_root, expected_phase="zero")
    contract = audit_rollout_contract()
    generator = audit_rollout_generator()
    metric = audit_fixed_metric_and_lipschitz(development_rollout_worlds()[0])
    routing = audit_routing_controls().as_dict()
    preflight = {
        "identifier": "P3-4A-DEVELOPMENT-PREFLIGHT-v1",
        "test_output_used": False,
        "access": access,
        "contract": contract,
        "generator": generator,
        "fixed_metric_and_lipschitz": metric,
        "routing_controls": routing,
    }
    preflight["passed"] = all(
        report["passed"] for report in (access, contract, generator, metric, routing)
    )
    preflight["report_digest"] = _canonical_digest(preflight)
    _write_json(output_root / "development_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("P3-4A development preflight failed")

    pilot = run_rollout_development_pilot(
        progress=lambda message: print(message, flush=True)
    )
    write_rollout_experiment(
        output_root / "development_rollout_results.json",
        pilot,
    )
    power = build_rollout_power_report(pilot)
    write_rollout_power_report(
        output_root / "development_power_report.json",
        power,
    )
    write_rollout_analysis_plan(
        output_root / "analysis_plan.json",
        power,
    )
    summary = {
        "development_preflight_digest": preflight["report_digest"],
        "development_pilot_digest": pilot["report_digest"],
        "development_runs": pilot["run_count"],
        "development_failures": pilot["failure_count"],
        "power_report_digest": power["report_digest"],
        "power_passed": power["passed"],
        "planned_test_worlds": power["planned_test_worlds"],
        "selected_conjunctive_power": power["selected_conjunctive_power"],
        "selected_monte_carlo_95pct_lower_bound": power[
            "selected_monte_carlo_95pct_lower_bound"
        ],
        "analysis_plan_digest": power["analysis_plan"]["analysis_plan_digest"],
        "test_seed_reveals": access["seed_reveals"],
        "test_result_evaluations": access["result_evaluations"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if power["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
