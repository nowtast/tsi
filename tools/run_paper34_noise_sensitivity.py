#!/usr/bin/env python3
"""Run the post-review Paper 3/4 noise sensitivity grid."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from tsi.paper34_noise_sensitivity import (
    OOD_NOISE_GRID,
    TRAIN_NOISE_GRID,
    build_noise_sensitivity,
    run_noise_world,
)


def _job(arguments: tuple[int, float, float]) -> dict[str, object]:
    return run_noise_world(*arguments)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--worlds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.worlds < 2:
        raise ValueError("at least two worlds per cell are required")

    jobs = [
        (world_index, train_noise, ood_noise)
        for train_noise in TRAIN_NOISE_GRID
        for ood_noise in OOD_NOISE_GRID
        for world_index in range(args.worlds)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_job, jobs, chunksize=2))
    report = build_noise_sensitivity(rows)
    report["worlds_per_cell"] = args.worlds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cell_count": len(report["cells"]),
                "worlds_per_cell": args.worlds,
                "minimum_identification_rate": report[
                    "all_cells_identification_rate"
                ],
                "minimum_graph_effect_by_head": report[
                    "minimum_graph_effect_by_head"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
