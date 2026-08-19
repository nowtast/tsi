#!/usr/bin/env python3
"""Fail when paired TSI TeX packages drift structurally."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.paper_parity import check_bilingual_parity  # noqa: E402


def main() -> int:
    errors = check_bilingual_parity(REPOSITORY_ROOT)
    if errors:
        for error in errors:
            print(f"{error.paper}:{error.location}: {error.message}")
        return 1
    print("Bilingual TeX structural and display-math parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
