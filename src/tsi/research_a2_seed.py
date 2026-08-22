"""Validation for externally custodied Research A2 seed material."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Mapping


ATTESTATION_STATUS = "external_custodian_single_seed_attestation"
SEED_ORIGIN = "external_custodian_single_draw"


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def validate_custodian_attestation(
    seed: bytes,
    attestation: Mapping[str, object],
    freeze: Mapping[str, object],
    freeze_git_commit: str,
) -> dict[str, object]:
    """Validate one externally generated seed against its public freeze."""

    if len(seed) != 32:
        raise ValueError("the external A2 seed must contain exactly 32 bytes")
    if attestation.get("status") != ATTESTATION_STATUS:
        raise ValueError("invalid A2 custodian attestation status")
    custodian = freeze.get("seed_custodian_id")
    if not isinstance(custodian, str) or not custodian.strip():
        raise ValueError("the freeze manifest must name an external seed custodian")
    if attestation.get("custodian_id") != custodian:
        raise ValueError("the attestation custodian does not match the freeze manifest")
    if attestation.get("freeze_digest") != freeze.get("freeze_digest"):
        raise ValueError("the attestation is not bound to this freeze digest")
    if attestation.get("freeze_git_commit") != freeze_git_commit:
        raise ValueError("the attestation is not bound to the public freeze commit")
    seed_digest = sha256(seed).hexdigest()
    if attestation.get("seed_sha256") != seed_digest:
        raise ValueError("the external seed does not match the custodian attestation")
    if attestation.get("single_draw") is not True:
        raise ValueError("the custodian must attest that exactly one draw was made")
    if attestation.get("author_generated_seed") is not False:
        raise ValueError("an author-generated A2 seed is forbidden")
    if attestation.get("author_selected_seed") is not False:
        raise ValueError("an author-selected A2 seed is forbidden")
    method = attestation.get("generation_method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("the custodian must state a seed generation method")
    generated = _timestamp(attestation.get("generated_at_utc"), "generated_at_utc")
    frozen = _timestamp(freeze.get("frozen_at_utc"), "frozen_at_utc")
    if generated <= frozen:
        raise ValueError("the custodian seed must be generated after source freeze")
    return {
        "seed_origin": SEED_ORIGIN,
        "custodian_id": custodian,
        "seed_sha256": seed_digest,
        "generated_at_utc": generated.isoformat(),
        "generation_method": method,
        "single_draw": True,
        "author_generated_seed": False,
        "author_selected_seed": False,
        "freeze_git_commit": freeze_git_commit,
    }
