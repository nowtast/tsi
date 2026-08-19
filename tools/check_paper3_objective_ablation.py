#!/usr/bin/env python3
"""Run the matched P3-2 objective-ablation gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from tsi.paper3_ablation_experiment import run_p3_objective_ablation


def main() -> int:
    report = run_p3_objective_ablation()
    output = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "paper3_objective_ablation"
        / "results.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "gate": report.gate,
        "passed": report.passed,
        "benchmark_id": report.benchmark_id,
        "benchmark_digest": report.benchmark_digest,
        "experiment_digest": report.experiment_digest,
        "parameter_count": report.parameter_count,
        "gradient_audit": report.gradient_audit.as_dict(),
        "deterministic_replay_passed": report.deterministic_replay_passed,
        "audit_errors": list(report.audit_errors),
        "primary_test_means": {
            condition: {
                metric: summary.metrics[metric].mean
                for metric in (
                    "fixed_joint_exact_rate",
                    "mean_quotient_distance",
                    "mean_tracking_error",
                    "mean_soft_bridge_defect",
                    "mean_projection_correction",
                )
            }
            for condition, summary in report.summaries.items()
        },
        "paired_primary_effects": {
            condition: {
                name: statistic.as_dict() for name, statistic in effects.items()
            }
            for condition, effects in report.paired.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
