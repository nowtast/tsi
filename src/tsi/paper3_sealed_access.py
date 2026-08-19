"""Sealed-test seed commitment and hash-chained zero-access ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping

from .paper3_independence_contract import P3_INDEPENDENCE_CONTRACT_ID


P3_SEALED_ACCESS_ID = "P3-3A-SEALED-ACCESS-v1"
COMMITMENT_FILENAME = "sealed_test_seed_commitment.json"
ESCROW_FILENAME = "sealed_test_seed.escrow"
LEDGER_FILENAME = "test_access_ledger.jsonl"
SECRET_BYTES = 32
ZERO_HASH = "0" * 64
REVEAL_EVENTS = frozenset({"test_seed_revealed"})
RESULT_EVENTS = frozenset({"test_result_evaluated"})
TEST_ACTIVITY_EVENTS = frozenset(
    {
        "test_prediction_started",
        "test_prediction_completed",
        "test_result_evaluated",
        "test_report_generated",
    }
)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _event_hash(payload: Mapping[str, object]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_ledger(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("ledger rows must be JSON objects")
            events.append(event)
    return events


def append_test_access_event(
    root: Path,
    event: str,
    metadata: Mapping[str, object],
) -> str:
    """Append one hash-chained reveal/test event without exposing seed bytes."""

    if event not in REVEAL_EVENTS.union(TEST_ACTIVITY_EVENTS):
        raise ValueError("unknown test-access event")
    root = Path(root)
    ledger_path = root / LEDGER_FILENAME
    events = _read_ledger(ledger_path)
    if not events:
        raise RuntimeError("test-access ledger is empty")
    previous_hash = events[-1].get("event_hash")
    if not isinstance(previous_hash, str):
        raise RuntimeError("latest ledger event has no hash")
    reveal_count = sum(item.get("event") in REVEAL_EVENTS for item in events)
    result_count = sum(item.get("event") in RESULT_EVENTS for item in events)
    reveal_count += event in REVEAL_EVENTS
    result_count += event in RESULT_EVENTS
    payload = {
        "sequence": len(events),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "previous_event_hash": previous_hash,
        "test_seed_reveals_after_event": reveal_count,
        "test_result_evaluations_after_event": result_count,
        "metadata": dict(metadata),
    }
    event_hash = _event_hash(payload)
    with ledger_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_canonical_json({**payload, 'event_hash': event_hash})}\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event_hash


def reveal_sealed_test_seed(
    root: Path,
    *,
    gate_digest: str,
    frozen_artifact_digests: Mapping[str, str],
) -> tuple[bytes, str]:
    """Reveal once after a passing P3-3A audit and record the access first."""

    root = Path(root)
    audit = audit_sealed_test_material(root)
    if not audit.passed:
        raise RuntimeError("sealed material is not in the zero-access state")
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
        raise RuntimeError("revealed seed does not match its commitment")

    unsigned = {
        key: value for key, value in descriptor.items() if key != "descriptor_digest"
    }
    unsigned["revealed"] = True
    unsigned["revealed_at_utc"] = datetime.now(timezone.utc).isoformat()
    revealed_descriptor = {
        **unsigned,
        "descriptor_digest": sha256(
            _canonical_json(unsigned).encode("utf-8")
        ).hexdigest(),
    }
    _write_json_atomic(commitment_path, revealed_descriptor)
    append_test_access_event(
        root,
        "test_seed_revealed",
        {
            "gate_digest": gate_digest,
            "frozen_artifact_digests": dict(frozen_artifact_digests),
        },
    )
    return secret, commitment


def initialize_sealed_test_material(root: Path) -> None:
    """Create one opaque seed escrow, commitment, and initial ledger event."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    commitment_path = root / COMMITMENT_FILENAME
    escrow_path = root / ESCROW_FILENAME
    ledger_path = root / LEDGER_FILENAME
    paths = (commitment_path, escrow_path, ledger_path)
    existing = tuple(path.exists() for path in paths)
    if all(existing):
        return
    if any(existing):
        raise RuntimeError("sealed-test material is only partially initialized")

    secret = secrets.token_bytes(SECRET_BYTES)
    commitment = sha256(secret).hexdigest()
    descriptor = {
        "identifier": P3_SEALED_ACCESS_ID,
        "parent_contract": P3_INDEPENDENCE_CONTRACT_ID,
        "algorithm": "sha256",
        "secret_bytes": SECRET_BYTES,
        "commitment": commitment,
        "revealed": False,
    }

    descriptor = {
        **descriptor,
        "descriptor_digest": sha256(
            _canonical_json(descriptor).encode("utf-8")
        ).hexdigest(),
    }
    file_descriptor = os.open(
        escrow_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        written = os.write(file_descriptor, secret)
        if written != len(secret):
            raise OSError("failed to write the complete escrow secret")
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    os.chmod(escrow_path, 0o000)
    _write_json_atomic(commitment_path, descriptor)

    event_payload = {
        "sequence": 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "seed_commitment_created",
        "commitment": commitment,
        "previous_event_hash": ZERO_HASH,
        "test_seed_reveals_after_event": 0,
        "test_result_evaluations_after_event": 0,
    }
    event = {**event_payload, "event_hash": _event_hash(event_payload)}
    ledger_path.write_text(
        f"{_canonical_json(event)}\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class SealedAccessAudit:
    identifier: str
    commitment: str | None
    descriptor_digest: str | None
    event_count: int
    latest_event_hash: str | None
    test_seed_reveals: int
    test_result_evaluations: int
    escrow_exists: bool
    escrow_size: int | None
    escrow_mode: int | None
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def artifact_digest(self) -> str | None:
        if (
            self.commitment is None
            or self.descriptor_digest is None
            or self.latest_event_hash is None
        ):
            return None
        payload = {
            "identifier": self.identifier,
            "commitment": self.commitment,
            "descriptor_digest": self.descriptor_digest,
            "latest_event_hash": self.latest_event_hash,
            "test_seed_reveals": self.test_seed_reveals,
            "test_result_evaluations": self.test_result_evaluations,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "commitment": self.commitment,
            "descriptor_digest": self.descriptor_digest,
            "event_count": self.event_count,
            "latest_event_hash": self.latest_event_hash,
            "test_seed_reveals": self.test_seed_reveals,
            "test_result_evaluations": self.test_result_evaluations,
            "escrow_exists": self.escrow_exists,
            "escrow_size": self.escrow_size,
            "escrow_mode_octal": (
                None if self.escrow_mode is None else oct(self.escrow_mode)
            ),
            "artifact_digest": self.artifact_digest,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def audit_sealed_test_material(root: Path) -> SealedAccessAudit:
    """Audit commitment and ledger metadata without reading the escrow secret."""

    root = Path(root)
    commitment_path = root / COMMITMENT_FILENAME
    escrow_path = root / ESCROW_FILENAME
    ledger_path = root / LEDGER_FILENAME
    errors: list[str] = []

    commitment: str | None = None
    descriptor_digest: str | None = None
    if not commitment_path.is_file():
        errors.append("sealed-test commitment descriptor is missing")
    else:
        try:
            descriptor = json.loads(commitment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot parse commitment descriptor: {error}")
        else:
            commitment = descriptor.get("commitment")
            descriptor_digest = descriptor.get("descriptor_digest")
            unsigned = {
                key: value
                for key, value in descriptor.items()
                if key != "descriptor_digest"
            }
            expected_descriptor_digest = sha256(
                _canonical_json(unsigned).encode("utf-8")
            ).hexdigest()
            if descriptor.get("identifier") != P3_SEALED_ACCESS_ID:
                errors.append("sealed-access identifier changed")
            if descriptor.get("parent_contract") != P3_INDEPENDENCE_CONTRACT_ID:
                errors.append("sealed-access parent contract changed")
            if descriptor.get("algorithm") != "sha256":
                errors.append("sealed commitment algorithm must be sha256")
            if descriptor.get("secret_bytes") != SECRET_BYTES:
                errors.append("sealed escrow length changed")
            if descriptor.get("revealed") is not False:
                errors.append("commitment descriptor is marked revealed")
            if not isinstance(commitment, str) or len(commitment) != 64:
                errors.append("commitment must be a 64-character hex digest")
            else:
                try:
                    int(commitment, 16)
                except ValueError:
                    errors.append("commitment is not hexadecimal")
            if descriptor_digest != expected_descriptor_digest:
                errors.append("commitment descriptor digest mismatch")

    escrow_exists = escrow_path.is_file()
    escrow_size: int | None = None
    escrow_mode: int | None = None
    if not escrow_exists:
        errors.append("sealed seed escrow is missing")
    else:
        escrow_stat = escrow_path.stat()
        escrow_size = escrow_stat.st_size
        escrow_mode = stat.S_IMODE(escrow_stat.st_mode)
        if escrow_size != SECRET_BYTES:
            errors.append("sealed seed escrow has the wrong byte length")
        if escrow_mode != 0:
            errors.append("sealed seed escrow must have mode 000 before reveal")

    events: list[dict[str, object]] = []
    if not ledger_path.is_file():
        errors.append("test-access ledger is missing")
    else:
        try:
            for line_number, line in enumerate(
                ledger_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line:
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"line {line_number} is not an object")
                events.append(event)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"cannot parse test-access ledger: {error}")

    previous_hash = ZERO_HASH
    reveal_count = 0
    result_count = 0
    latest_event_hash: str | None = None
    for sequence, event in enumerate(events):
        event_hash = event.get("event_hash")
        payload = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("sequence") != sequence:
            errors.append(f"ledger event {sequence} has the wrong sequence")
        if event.get("previous_event_hash") != previous_hash:
            errors.append(f"ledger event {sequence} breaks the hash chain")
        expected_event_hash = _event_hash(payload)
        if event_hash != expected_event_hash:
            errors.append(f"ledger event {sequence} hash mismatch")
        event_name = event.get("event")
        if event_name in REVEAL_EVENTS:
            reveal_count += 1
        if event_name in RESULT_EVENTS:
            result_count += 1
        if event.get("test_seed_reveals_after_event") != reveal_count:
            errors.append(f"ledger event {sequence} reveal count mismatch")
        if event.get("test_result_evaluations_after_event") != result_count:
            errors.append(f"ledger event {sequence} result count mismatch")
        if sequence == 0:
            if event_name != "seed_commitment_created":
                errors.append("first ledger event must create the commitment")
            if event.get("commitment") != commitment:
                errors.append("ledger commitment does not match descriptor")
        previous_hash = (
            event_hash if isinstance(event_hash, str) else expected_event_hash
        )
        latest_event_hash = previous_hash

    if len(events) != 1:
        errors.append("P3-3A zero-access ledger must contain one event")
    if reveal_count != 0:
        errors.append("test seed was revealed during P3-3A")
    if result_count != 0:
        errors.append("test result was evaluated during P3-3A")

    return SealedAccessAudit(
        identifier=P3_SEALED_ACCESS_ID,
        commitment=commitment,
        descriptor_digest=descriptor_digest,
        event_count=len(events),
        latest_event_hash=latest_event_hash,
        test_seed_reveals=reveal_count,
        test_result_evaluations=result_count,
        escrow_exists=escrow_exists,
        escrow_size=escrow_size,
        escrow_mode=escrow_mode,
        errors=tuple(errors),
    )
