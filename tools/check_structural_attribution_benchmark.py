#!/usr/bin/env python3
"""Check the structural attribution benchmark release and an optional report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsi.structural_attribution_benchmark import (
    validate_submission_file,
    verify_release,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report = {"release": verify_release(repo_root)}
    if args.submission is not None:
        report["submission"] = validate_submission_file(args.submission)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
