"""Atomic one-shot lock helpers for the sealed P3-4A execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


LOCK_FILENAME = "p3_4a_once.lock"
FAILURE_FILENAME = "p3_4a_execution_failure.json"


def write_json_atomic(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def acquire_once_lock(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    """Create the lock with O_EXCL so a second execution cannot start."""

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        encoded = (f"{json.dumps(payload, indent=2, sort_keys=True)}\n").encode("utf-8")
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("failed to write complete P3-4A one-shot lock")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
