"""One-shot seed escrow and hash-chained access ledger for P3-4B."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping

from .paper3_validity_contract import (
    P3_VALIDITY_CONTRACT_ID,
    validity_contract_digest,
)


P3_VALIDITY_ACCESS_ID = "P3-4B-VALIDITY-SEALED-ACCESS-v1"
COMMITMENT_FILENAME = "validity_seed_commitment.json"
ESCROW_FILENAME = "validity_seed.escrow"
LEDGER_FILENAME = "validity_access_ledger.jsonl"
SECRET_BYTES = 32
ZERO_HASH = "0" * 64
EVENT_SEQUENCE = (
    "seed_commitment_created",
    "validity_seed_revealed",
    "validity_prediction_started",
    "validity_prediction_completed",
    "validity_result_evaluated",
    "validity_report_generated",
)
ALLOWED_APPEND_EVENTS = frozenset(EVENT_SEQUENCE[1:])


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _digest(payload: object) -> str:
    return sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def initialize_validity_seed(root: Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    commitment_path = root / COMMITMENT_FILENAME
    escrow_path = root / ESCROW_FILENAME
    ledger_path = root / LEDGER_FILENAME
    existing = tuple(
        path.exists() for path in (commitment_path, escrow_path, ledger_path)
    )
    if all(existing):
        return
    if any(existing):
        raise RuntimeError("validity sealed material is only partially initialized")

    secret = secrets.token_bytes(SECRET_BYTES)
    commitment = sha256(secret).hexdigest()
    unsigned = {
        "identifier": P3_VALIDITY_ACCESS_ID,
        "parent_contract": P3_VALIDITY_CONTRACT_ID,
        "parent_contract_digest": validity_contract_digest(),
        "algorithm": "sha256",
        "secret_bytes": SECRET_BYTES,
        "commitment": commitment,
        "revealed": False,
    }
    descriptor = {**unsigned, "descriptor_digest": _digest(unsigned)}
    descriptor_handle = os.open(
        escrow_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        written = os.write(descriptor_handle, secret)
        if written != len(secret):
            raise OSError("failed to write complete validity seed")
        os.fsync(descriptor_handle)
    finally:
        os.close(descriptor_handle)
    os.chmod(escrow_path, 0o000)
    _write_json_atomic(commitment_path, descriptor)

    payload = {
        "sequence": 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": EVENT_SEQUENCE[0],
        "commitment": commitment,
        "previous_event_hash": ZERO_HASH,
        "seed_reveals_after_event": 0,
        "result_evaluations_after_event": 0,
    }
    event = {**payload, "event_hash": _digest(payload)}
    ledger_path.write_text(f"{_canonical(event)}\n", encoding="utf-8")


def append_validity_access_event(
    root: Path,
    event: str,
    metadata: Mapping[str, object],
) -> str:
    if event not in ALLOWED_APPEND_EVENTS:
        raise ValueError("unknown validity access event")
    root = Path(root)
    ledger_path = root / LEDGER_FILENAME
    events = _read_events(ledger_path)
    if not events:
        raise RuntimeError("validity access ledger is empty")
    expected_event = EVENT_SEQUENCE[len(events)] if len(events) < 6 else None
    if event != expected_event:
        raise RuntimeError(
            f"expected validity event {expected_event!r}, received {event!r}"
        )
    reveal_count = sum(item.get("event") == "validity_seed_revealed" for item in events)
    result_count = sum(
        item.get("event") == "validity_result_evaluated" for item in events
    )
    reveal_count += event == "validity_seed_revealed"
    result_count += event == "validity_result_evaluated"
    payload = {
        "sequence": len(events),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "previous_event_hash": events[-1]["event_hash"],
        "seed_reveals_after_event": reveal_count,
        "result_evaluations_after_event": result_count,
        "metadata": dict(metadata),
    }
    event_hash = _digest(payload)
    with ledger_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_canonical({**payload, 'event_hash': event_hash})}\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event_hash


def audit_validity_access(
    root: Path,
    *,
    expected_phase: str,
) -> dict[str, object]:
    if expected_phase not in ("zero", "final"):
        raise ValueError("expected_phase must be zero or final")
    root = Path(root)
    commitment_path = root / COMMITMENT_FILENAME
    escrow_path = root / ESCROW_FILENAME
    ledger_path = root / LEDGER_FILENAME
    errors: list[str] = []
    try:
        descriptor = json.loads(commitment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        descriptor = {}
        errors.append(f"cannot parse validity commitment: {error}")
    unsigned = {
        key: value for key, value in descriptor.items() if key != "descriptor_digest"
    }
    if descriptor.get("descriptor_digest") != _digest(unsigned):
        errors.append("validity commitment descriptor digest mismatch")
    if descriptor.get("identifier") != P3_VALIDITY_ACCESS_ID:
        errors.append("validity access identifier changed")
    if descriptor.get("parent_contract_digest") != validity_contract_digest():
        errors.append("validity access parent contract digest changed")
    commitment = descriptor.get("commitment")
    if not isinstance(commitment, str) or len(commitment) != 64:
        errors.append("validity commitment is not a SHA-256 digest")

    try:
        events = _read_events(ledger_path)
    except (OSError, json.JSONDecodeError) as error:
        events = []
        errors.append(f"cannot parse validity access ledger: {error}")
    previous_hash = ZERO_HASH
    reveal_count = 0
    result_count = 0
    for sequence, event in enumerate(events):
        payload = {key: value for key, value in event.items() if key != "event_hash"}
        expected_hash = _digest(payload)
        if event.get("sequence") != sequence:
            errors.append(f"validity event {sequence} sequence mismatch")
        if event.get("previous_event_hash") != previous_hash:
            errors.append(f"validity event {sequence} breaks the hash chain")
        if event.get("event_hash") != expected_hash:
            errors.append(f"validity event {sequence} hash mismatch")
        reveal_count += event.get("event") == "validity_seed_revealed"
        result_count += event.get("event") == "validity_result_evaluated"
        if event.get("seed_reveals_after_event") != reveal_count:
            errors.append(f"validity event {sequence} reveal count mismatch")
        if event.get("result_evaluations_after_event") != result_count:
            errors.append(f"validity event {sequence} result count mismatch")
        previous_hash = expected_hash
    event_names = tuple(event.get("event") for event in events)
    expected_names = EVENT_SEQUENCE[:1] if expected_phase == "zero" else EVENT_SEQUENCE
    if event_names != expected_names:
        errors.append("validity access event sequence does not match its phase")
    expected_reveals = 0 if expected_phase == "zero" else 1
    expected_results = 0 if expected_phase == "zero" else 1
    if reveal_count != expected_reveals:
        errors.append("validity seed reveal count is invalid")
    if result_count != expected_results:
        errors.append("validity result evaluation count is invalid")
    if descriptor.get("revealed") is not (expected_phase == "final"):
        errors.append("validity commitment reveal flag is invalid")

    escrow_mode: int | None = None
    escrow_size: int | None = None
    try:
        escrow_stat = escrow_path.stat()
    except OSError as error:
        errors.append(f"cannot stat validity escrow: {error}")
    else:
        escrow_mode = stat.S_IMODE(escrow_stat.st_mode)
        escrow_size = escrow_stat.st_size
        if escrow_mode != 0:
            errors.append("validity escrow must have mode 000")
        if escrow_size != SECRET_BYTES:
            errors.append("validity escrow has the wrong byte length")
    payload = {
        "identifier": P3_VALIDITY_ACCESS_ID,
        "expected_phase": expected_phase,
        "commitment": commitment,
        "event_names": list(event_names),
        "latest_event_hash": previous_hash if events else None,
        "seed_reveals": reveal_count,
        "result_evaluations": result_count,
        "escrow_mode_octal": (None if escrow_mode is None else oct(escrow_mode)),
        "escrow_size": escrow_size,
        "errors": errors,
        "passed": not errors,
    }
    return {**payload, "audit_digest": _digest(payload)}


def reveal_validity_seed(
    root: Path,
    *,
    gate_digest: str,
    frozen_artifact_digests: Mapping[str, str],
) -> tuple[bytes, str]:
    root = Path(root)
    audit = audit_validity_access(root, expected_phase="zero")
    if not audit["passed"]:
        raise RuntimeError("validity seed is not in a valid zero-access state")
    commitment_path = root / COMMITMENT_FILENAME
    escrow_path = root / ESCROW_FILENAME
    descriptor = json.loads(commitment_path.read_text(encoding="utf-8"))
    commitment = descriptor["commitment"]
    try:
        os.chmod(escrow_path, 0o400)
        secret = escrow_path.read_bytes()
    finally:
        os.chmod(escrow_path, 0o000)
    if len(secret) != SECRET_BYTES or sha256(secret).hexdigest() != commitment:
        raise RuntimeError("revealed validity seed does not match its commitment")

    unsigned = {
        key: value for key, value in descriptor.items() if key != "descriptor_digest"
    }
    unsigned["revealed"] = True
    unsigned["revealed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(
        commitment_path,
        {**unsigned, "descriptor_digest": _digest(unsigned)},
    )
    append_validity_access_event(
        root,
        "validity_seed_revealed",
        {
            "gate_digest": gate_digest,
            "frozen_artifact_digests": dict(frozen_artifact_digests),
        },
    )
    return secret, commitment
