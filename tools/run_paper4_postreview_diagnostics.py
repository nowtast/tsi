"""Run the post-review Paper 4 attribution and misspecification diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsi.paper4_postreview_diagnostics import run_full_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run_full_diagnostic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cell_count": result["cell_count"], "panels": result["panels"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
