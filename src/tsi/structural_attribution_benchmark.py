"""Integrity and submission checks for the structural attribution benchmark."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


BENCHMARK_ID = "tsi-structural-attribution"
BENCHMARK_VERSION = "0.1.0"
BENCHMARK_DIRECTORY = Path("benchmarks/structural_attribution_v0_1")


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark artifact or submission violates the contract."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkValidationError(f"expected a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise BenchmarkValidationError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def verify_release(repo_root: Path) -> dict[str, Any]:
    """Verify artifact hashes and reference values without recomputing results."""

    benchmark_root = repo_root / BENCHMARK_DIRECTORY
    manifest = _read_json(benchmark_root / "benchmark.json")
    reference = _read_json(benchmark_root / "reference_results.json")

    _require_equal(manifest.get("benchmark_id"), BENCHMARK_ID, "benchmark id")
    _require_equal(manifest.get("version"), BENCHMARK_VERSION, "benchmark version")
    _require_equal(reference.get("benchmark_id"), BENCHMARK_ID, "reference id")
    _require_equal(reference.get("version"), BENCHMARK_VERSION, "reference version")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BenchmarkValidationError("benchmark artifact list is empty")

    by_role: dict[str, Path] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise BenchmarkValidationError("artifact entries must be objects")
        role = artifact.get("role")
        relative = artifact.get("path")
        if not isinstance(role, str) or not isinstance(relative, str):
            raise BenchmarkValidationError("artifact role and path must be strings")
        if role in by_role:
            raise BenchmarkValidationError(f"duplicate artifact role: {role}")
        path = repo_root / relative
        if not path.is_file():
            raise BenchmarkValidationError(f"missing artifact: {relative}")
        _require_equal(path.stat().st_size, artifact.get("size"), f"{role} size")
        _require_equal(_file_sha256(path), artifact.get("sha256"), f"{role} sha256")
        by_role[role] = path

    attribution = _read_json(by_role["attribution_analysis"])
    attribution_freeze = _read_json(by_role["attribution_freeze"])
    stress = _read_json(by_role["stress_analysis"])
    stress_freeze = _read_json(by_role["stress_freeze"])

    contracts = manifest["contracts"]
    _require_equal(
        attribution["contract_digest"], contracts["attribution"], "attribution contract"
    )
    _require_equal(
        attribution_freeze["contract_digest"],
        contracts["attribution"],
        "attribution freeze contract",
    )
    _require_equal(
        stress["contract_digest"],
        contracts["outside_family_stress"],
        "stress contract",
    )
    _require_equal(
        stress_freeze["contract_digest"],
        contracts["outside_family_stress"],
        "stress freeze contract",
    )

    analysis = attribution["analysis"]
    intervals = analysis["effect_intervals"]
    ref_attr = reference["attribution"]
    _require_equal(analysis["world_count"], ref_attr["world_count"], "world count")
    _require_equal(
        analysis["identification_rate"],
        ref_attr["identification_rate"],
        "identification rate",
    )
    _require_equal(
        analysis["identification_simultaneous_wilson_lower"],
        ref_attr["identification_simultaneous_wilson_lower"],
        "identification Wilson lower bound",
    )

    interval_pairs = (
        ("factorized_graph_nll", "factorized_wrong_minus_correct_composition_nll"),
        ("generic_graph_nll", "generic_sparse_wrong_minus_correct_composition_nll"),
        ("large_generic_graph_nll", "generic_dense_wrong_minus_correct_composition_nll"),
    )
    for source_name, reference_name in interval_pairs:
        source = intervals[source_name]
        expected = ref_attr[reference_name]
        for field in ("mean", "simultaneous_lower", "simultaneous_upper"):
            _require_equal(source[field], expected[field], f"{reference_name}.{field}")

    _require_equal(
        intervals["factorized_head_nll_correct_graph"]["mean"],
        ref_attr["factorized_minus_generic_sparse_correct_graph_nll"],
        "matched sparse equivalence",
    )

    stress_analysis = stress["analysis"]
    ref_stress = reference["outside_family_stress"]
    _require_equal(stress_analysis["world_count"], ref_stress["world_count"], "stress worlds")
    _require_equal(
        stress_analysis["worlds_with_nonexact_learned_center"],
        ref_stress["worlds_with_nonexact_learned_center"],
        "nonexact stress worlds",
    )
    for field in ("mean", "lower_95", "upper_95"):
        _require_equal(
            stress_analysis["graph_nll_effect"][field],
            ref_stress["wrong_minus_learned_composition_nll"][field],
            f"stress graph effect {field}",
        )

    return {
        "benchmark_id": BENCHMARK_ID,
        "version": BENCHMARK_VERSION,
        "artifact_count": len(artifacts),
        "world_count": analysis["world_count"],
        "passed": True,
    }


def load_participant_worlds(repo_root: Path) -> list[dict[str, Any]]:
    """Load a leakage-reduced view of the public attribution worlds.

    Training and selection retain observed next states. Test cases expose only
    current state and action. Ground-truth graph/head metadata and archived
    expected outputs are omitted.
    """

    manifest = _read_json(repo_root / BENCHMARK_DIRECTORY / "benchmark.json")
    portable_entry = next(
        artifact for artifact in manifest["artifacts"] if artifact["role"] == "portable_inputs"
    )
    payload = _read_json(repo_root / portable_entry["path"])
    worlds = payload.get("worlds")
    if not isinstance(worlds, list):
        raise BenchmarkValidationError("portable input worlds must be a list")

    participant_worlds: list[dict[str, Any]] = []
    for world in worlds:
        if not isinstance(world, dict):
            raise BenchmarkValidationError("portable world entries must be objects")
        test_inputs: list[list[Any]] = []
        for case in world["test"]:
            if not isinstance(case, list) or len(case) != 3:
                raise BenchmarkValidationError("test cases must be state/action/target triples")
            test_inputs.append([case[0], case[1]])
        participant_worlds.append(
            {
                "world_index": world["world_index"],
                "train": world["train"],
                "selection": world["selection"],
                "test_inputs": test_inputs,
            }
        )

    _require_equal(len(participant_worlds), manifest["world_count"], "participant worlds")
    return participant_worlds


def _valid_graph(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    target, sources = value
    if not isinstance(target, int) or not 0 <= target < 5:
        return False
    if not isinstance(sources, list) or len(sources) != 2:
        return False
    if any(not isinstance(source, int) or not 0 <= source < 5 for source in sources):
        return False
    return len({target, *sources}) == 3


def validate_submission(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a portable report and its declared information policy."""

    _require_equal(payload.get("benchmark_id"), BENCHMARK_ID, "submission benchmark id")
    _require_equal(
        payload.get("benchmark_version"), BENCHMARK_VERSION, "submission benchmark version"
    )
    if not isinstance(payload.get("submission_id"), str) or not payload["submission_id"].strip():
        raise BenchmarkValidationError("submission_id must be a nonempty string")

    method = payload.get("method")
    if not isinstance(method, dict):
        raise BenchmarkValidationError("method must be an object")
    for field in ("name", "version"):
        if not isinstance(method.get(field), str) or not method[field].strip():
            raise BenchmarkValidationError(f"method.{field} must be a nonempty string")

    policy = payload.get("information_policy")
    if not isinstance(policy, dict):
        raise BenchmarkValidationError("information_policy must be an object")
    _require_equal(policy.get("fit_partitions"), ["train"], "fit partitions")
    _require_equal(policy.get("selection_partitions"), ["selection"], "selection partitions")
    _require_equal(
        policy.get("test_targets_used_for_fit_or_selection"),
        False,
        "test-target access declaration",
    )
    _require_equal(
        policy.get("reference_answers_used"), False, "reference-answer access declaration"
    )

    allowed_tasks = {"graph_head_recovery", "held_out_composition_prediction"}
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or set(tasks) != allowed_tasks or len(tasks) != 2:
        raise BenchmarkValidationError("submissions must report both benchmark tasks")

    coverage = payload.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("kind") not in {"smoke", "full"}:
        raise BenchmarkValidationError("coverage.kind must be smoke or full")
    expected_count = coverage.get("world_count")
    if not isinstance(expected_count, int) or not 1 <= expected_count <= 120:
        raise BenchmarkValidationError("coverage.world_count must be in [1, 120]")
    if coverage["kind"] == "full" and expected_count != 120:
        raise BenchmarkValidationError("full coverage requires 120 worlds")

    worlds = payload.get("worlds")
    if not isinstance(worlds, list) or len(worlds) != expected_count:
        raise BenchmarkValidationError("world rows do not match declared coverage")
    indices: set[int] = set()
    for row in worlds:
        if not isinstance(row, dict):
            raise BenchmarkValidationError("world rows must be objects")
        index = row.get("world_index")
        if not isinstance(index, int) or not 0 <= index < 120 or index in indices:
            raise BenchmarkValidationError("world indices must be unique integers in [0, 119]")
        indices.add(index)
        if not _valid_graph(row.get("selected_graph")):
            raise BenchmarkValidationError(f"invalid selected_graph for world {index}")
        families = row.get("selected_head_families")
        if (
            not isinstance(families, list)
            or len(families) != 2
            or any(not isinstance(family, str) or not family for family in families)
        ):
            raise BenchmarkValidationError(f"invalid selected_head_families for world {index}")
        nll = row.get("composition_nll")
        if not isinstance(nll, (int, float)) or isinstance(nll, bool):
            raise BenchmarkValidationError(f"composition_nll must be numeric for world {index}")
        if not math.isfinite(float(nll)) or float(nll) < 0.0:
            raise BenchmarkValidationError(f"composition_nll must be finite and nonnegative")

    return {
        "benchmark_id": BENCHMARK_ID,
        "version": BENCHMARK_VERSION,
        "submission_id": payload["submission_id"],
        "coverage": coverage["kind"],
        "world_count": len(worlds),
        "passed": True,
    }


def validate_submission_file(path: Path) -> dict[str, Any]:
    return validate_submission(_read_json(path))
