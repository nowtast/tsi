#!/usr/bin/env python3
"""Write the post-review Paper 3/4 multiplicity sensitivity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsi.paper34_multiplicity_audit import audit_multiplicity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--divisor", type=int, default=10)
    args = parser.parse_args()

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    audit = audit_multiplicity(
        payload["analysis"], sensitivity_divisor=args.divisor
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "all_sensitivity_gates_passed": audit[
                    "all_sensitivity_gates_passed"
                ],
                "sensitivity_critical": audit["sensitivity_critical"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
