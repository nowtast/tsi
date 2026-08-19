#!/usr/bin/env python3
"""Run and persist the validation-only P3-2R representation-repair gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from tsi.paper3_repair_experiment import run_p3_representation_repair


def main() -> int:
    report = run_p3_representation_repair()
    output = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "paper3_representation_repair"
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
        "base_benchmark_digest": report.base_benchmark_digest,
        "experiment_digest": report.experiment_digest,
        "evaluated_splits": list(report.evaluated_splits),
        "test_transition_evaluations": report.test_transition_evaluations,
        "embedding_diagnostic_source_count": (report.embedding_diagnostic_source_count),
        "decoder_candidate_count": report.decoder_candidate_count,
        "selected_variant": report.selected_variant,
        "gradient_audits": {
            variant: audit.as_dict()
            for variant, audit in report.gradient_audits.items()
        },
        "deterministic_replay_passed": report.deterministic_replay_passed,
        "audit_errors": list(report.audit_errors),
        "readiness_metrics": {
            variant: {
                metric: summary.metrics[metric].mean
                for metric in (
                    "train_fixed_joint_exact_rate",
                    "validation_fixed_joint_exact_rate",
                    "validation_quotient_distance",
                    "validation_tracking_exact_rate",
                    "validation_embedding_collision_count",
                    "validation_post_projection_bridge_violation_rate",
                )
            }
            for variant, summary in report.summaries.items()
        },
        "paired_primary_effects": {
            comparison: {
                name: statistic.as_dict() for name, statistic in effects.items()
            }
            for comparison, effects in report.paired.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
