#!/usr/bin/env python3
"""Build and audit all currently available P3-3A preregistration artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper3_analysis_plan import (  # noqa: E402
    analysis_plan_payload,
    audit_analysis_plan,
)
from tsi.paper3_constructive_decoder import (  # noqa: E402
    audit_constructive_decoder,
)
from tsi.paper3_independence_contract import (  # noqa: E402
    REQUIRED_ARTIFACTS,
    audit_p3_3a_independence_contract,
)
from tsi.paper3_multiworld import (  # noqa: E402
    development_validation_world_manifest,
)
from tsi.paper3_multiworld_audit import (  # noqa: E402
    audit_multiworld_generator,
)
from tsi.paper3_routing_controls import (  # noqa: E402
    audit_routing_controls,
    routing_control_manifests,
)
from tsi.paper3_sealed_access import (  # noqa: E402
    audit_sealed_test_material,
)
from tsi.paper3_independence_contract import WorldFamily  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _audit_power_report(path: Path) -> tuple[bool, dict[str, object]]:
    if not path.is_file():
        return False, {
            "status": "pending",
            "reason": "development variance and simulation-power report is absent",
            "test_output_used": False,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, {"status": "invalid", "reason": str(error)}
    required = {
        "identifier": "P3-3A-POWER-v1",
        "passed": True,
        "test_output_used": False,
    }
    errors = [
        f"{key} must equal {expected!r}"
        for key, expected in required.items()
        if payload.get(key) != expected
    ]
    planned_worlds = payload.get("planned_test_worlds")
    if type(planned_worlds) is not int or planned_worlds < 36 or planned_worlds > 128:
        errors.append("planned_test_worlds must be an integer in [36, 128]")
    simulation_power = payload.get("minimum_simulation_power")
    if not isinstance(simulation_power, (int, float)) or float(simulation_power) < 0.90:
        errors.append("minimum_simulation_power must be at least 0.90")
    return not errors, {**payload, "audit_errors": errors}


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

    generator = audit_multiworld_generator()
    decoder = audit_constructive_decoder()
    routing = audit_routing_controls()
    sealed = audit_sealed_test_material(output_root / "sealed")
    analysis = audit_analysis_plan()
    power_passed, power = _audit_power_report(
        output_root / "development_power_report.json"
    )

    artifact_status = {
        "generator_implementation_and_digest": generator.passed,
        "development_validation_world_manifest": (
            generator.passed and generator.manifest_world_count == 108
        ),
        "sealed_test_seed_commitment": sealed.passed,
        "ood_support_and_nonredundancy_audit": generator.passed,
        "constructive_decoder_and_validity_audit": decoder.passed,
        "six_model_dependency_mask_manifest": routing.passed,
        "information_capacity_compute_tuning_ledger": routing.passed,
        "development_variance_and_power_report": power_passed,
        "zero_access_test_ledger": (
            sealed.passed
            and sealed.test_seed_reveals == 0
            and sealed.test_result_evaluations == 0
        ),
        "frozen_p3_3b_analysis_digest": analysis.passed,
    }
    if tuple(artifact_status) != REQUIRED_ARTIFACTS:
        raise RuntimeError("artifact status order differs from the frozen contract")
    gate = audit_p3_3a_independence_contract(
        artifact_status,
        test_seed_reveals=sealed.test_seed_reveals,
        test_result_evaluations=sealed.test_result_evaluations,
    )

    manifest_payload = {
        "world_count": len(development_validation_world_manifest()),
        "sealed_test_worlds": 0,
        "worlds": [
            mechanism.as_dict() for mechanism in development_validation_world_manifest()
        ],
    }
    routing_payload = {
        "families": {
            family.value: [
                manifest.as_dict() for manifest in routing_control_manifests(family)
            ]
            for family in WorldFamily
        }
    }
    subreports = {
        "generator_audit": generator.as_dict(),
        "decoder_audit": decoder.as_dict(),
        "routing_audit": routing.as_dict(),
        "sealed_access_audit": sealed.as_dict(),
        "analysis_plan_audit": analysis.as_dict(),
        "power_audit": power,
    }
    combined_payload = {
        "artifact_status": artifact_status,
        "gate_audit": gate.as_dict(),
        "subreports": subreports,
    }
    combined_digest = sha256(
        json.dumps(
            combined_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report = {**combined_payload, "combined_digest": combined_digest}

    _write_json(output_root / "world_manifest.json", manifest_payload)
    _write_json(output_root / "generator_audit.json", generator.as_dict())
    _write_json(output_root / "decoder_audit.json", decoder.as_dict())
    _write_json(output_root / "routing_manifest.json", routing_payload)
    _write_json(output_root / "routing_audit.json", routing.as_dict())
    _write_json(output_root / "analysis_plan.json", analysis_plan_payload())
    _write_json(output_root / "analysis_plan_audit.json", analysis.as_dict())
    _write_json(output_root / "artifact_gate.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))

    static_subaudits_passed = (
        generator.passed
        and decoder.passed
        and routing.passed
        and sealed.passed
        and analysis.passed
        and gate.static_contract_passed
    )
    return 0 if static_subaudits_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
