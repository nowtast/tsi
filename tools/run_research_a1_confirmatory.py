#!/usr/bin/env python3
"""Execute Research A1 once from a previously published seed commitment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from tsi.research_a_analysis import analyze_confirmatory_rows
from tsi.research_a_confirmatory import run_cohort
from tsi.research_a_contract import WORLD_COUNT, contract_digest


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _commit_containing(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root)
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(relative)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("seed commitment has not been committed")
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=root
    )
    if dirty.returncode != 0:
        raise RuntimeError("seed commitment differs from its committed version")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--escrow", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    commitment = json.loads(args.commitment.read_text(encoding="utf-8"))
    if freeze.get("contract_digest") != contract_digest():
        raise RuntimeError("freeze contract digest mismatch")
    if commitment.get("freeze_digest") != freeze.get("freeze_digest"):
        raise RuntimeError("seed commitment freeze digest mismatch")
    for relative, expected in freeze["files"].items():
        if digest(root / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    commitment_commit = _commit_containing(args.commitment, root)
    root_seed = args.escrow.read_bytes()
    if sha256(root_seed).hexdigest() != commitment.get("root_seed_commitment"):
        raise RuntimeError("seed escrow does not match public commitment")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / "ONE_SHOT.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(datetime.now(timezone.utc).isoformat() + "\n")
    rows, portable, derivation_audit = run_cohort(root_seed)
    analysis = analyze_confirmatory_rows(rows)
    if analysis["world_count"] != WORLD_COUNT:
        raise RuntimeError("confirmatory world count changed")
    raw = {
        "status": "prospective_one_shot_confirmatory",
        "contract_digest": contract_digest(),
        "freeze_digest": freeze["freeze_digest"],
        "root_seed_commitment": commitment["root_seed_commitment"],
        "commitment_git_commit": commitment_commit,
        "derivation_audit": derivation_audit,
        "rows": rows,
    }
    raw_path = args.output_dir / "raw_results.json"
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    portable_path = args.output_dir / "portable_replay.json"
    portable_path.write_text(json.dumps(portable) + "\n", encoding="utf-8")
    report = {
        "status": "prospective_one_shot_confirmatory_analysis",
        "contract_digest": contract_digest(),
        "freeze_digest": freeze["freeze_digest"],
        "root_seed_commitment": commitment["root_seed_commitment"],
        "raw_results_sha256": digest(raw_path),
        "portable_replay_sha256": digest(portable_path),
        "analysis": analysis,
    }
    report_path = args.output_dir / "confirmatory_analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ledger = {
        "root_seed_hex_revealed_after_execution": root_seed.hex(),
        "root_seed_commitment": commitment["root_seed_commitment"],
        "commitment_verified": sha256(root_seed).hexdigest()
        == commitment["root_seed_commitment"],
        "commitment_git_commit": commitment_commit,
        "confirmatory_analysis_sha256": digest(report_path),
    }
    (args.output_dir / "seed_and_integrity_ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"world_count": WORLD_COUNT, **analysis}, indent=2))


if __name__ == "__main__":
    main()
