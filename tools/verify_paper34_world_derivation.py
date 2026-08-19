#!/usr/bin/env python3
"""Verify exported Paper 3/4 worlds against the revealed root seed."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tsi.paper34_world_derivation_audit import audit_world_derivation


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("portable_inputs", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    portable = json.loads(args.portable_inputs.read_text(encoding="utf-8"))
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    audit = audit_world_derivation(portable, ledger)
    audit["portable_inputs_sha256"] = _digest(args.portable_inputs)
    audit["ledger_sha256"] = _digest(args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "world_count": audit["world_count"],
                "passed": audit["passed"],
            },
            indent=2,
        )
    )
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
