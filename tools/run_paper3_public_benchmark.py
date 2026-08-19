#!/usr/bin/env python3
"""Run the deterministic public P3 factorization benchmark."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsi.paper3_public_benchmark import write_public_benchmark


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/public_benchmark/results.json")
    )
    args = parser.parse_args()
    result = write_public_benchmark(Path.cwd(), args.output, smoke=args.smoke)
    print(result["result_digest"])
    print(result["v3"]["cell_count"], result["replication"]["cell_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
