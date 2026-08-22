#!/usr/bin/env python3
"""Commit one externally custodied A2 seed after a public source freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from tsi.research_a2_contract import contract_digest
from tsi.research_a2_seed import validate_custodian_attestation


def _public_commit_containing(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root)
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=root
    )
    if dirty.returncode != 0:
        raise RuntimeError("A2 freeze manifest differs from its committed version")
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(relative)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("A2 freeze manifest has not been committed")
    remote = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if remote.returncode != 0:
        raise RuntimeError("origin/main is unavailable for public freeze verification")
    published = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=root
    )
    if published.returncode != 0:
        raise RuntimeError("A2 freeze commit is not published on origin/main")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--custodian-seed", type=Path, required=True)
    parser.add_argument("--custodian-attestation", type=Path, required=True)
    parser.add_argument(
        "--escrow",
        type=Path,
        default=Path("experiments/research_a2_v1/confirmatory/root_seed.escrow"),
    )
    parser.add_argument(
        "--commitment",
        type=Path,
        default=Path("experiments/research_a2_v1/seed_commitment.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("contract_digest") != contract_digest():
        raise RuntimeError("A2 freeze contract digest mismatch")
    for relative, expected in freeze["files"].items():
        if sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen A2 source changed: {relative}")
    freeze_git_commit = _public_commit_containing(args.freeze, root)
    root_seed = args.custodian_seed.read_bytes()
    attestation = json.loads(args.custodian_attestation.read_text(encoding="utf-8"))
    selection = validate_custodian_attestation(
        root_seed, attestation, freeze, freeze_git_commit
    )
    args.escrow.parent.mkdir(parents=True, exist_ok=True)
    with args.escrow.open("xb") as handle:
        handle.write(root_seed)
    payload = {
        "status": "a2_seed_committed_before_one_shot_execution",
        "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_digest": contract_digest(),
        "freeze_digest": freeze["freeze_digest"],
        "freeze_git_commit": freeze_git_commit,
        "root_seed_commitment": sha256(root_seed).hexdigest(),
        "seed_selection_control": selection,
        "custodian_attestation": attestation,
        "escrow_path_not_public": str(args.escrow),
        "execution_policy": "commit and push this file before one-shot execution; no seed reroll or replacement is permitted",
    }
    with args.commitment.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"root_seed_commitment": payload["root_seed_commitment"]}))


if __name__ == "__main__":
    main()
