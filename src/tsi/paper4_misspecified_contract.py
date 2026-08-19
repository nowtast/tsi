"""Contract for the prospective outside-model-family stress test."""

from __future__ import annotations

from hashlib import sha256
import json


CONTRACT_ID = "TSI-P4-OUTSIDE-MODEL-FAMILY-v1"
DEVELOPMENT_WORLDS = 24
CONFIRMATORY_WORLDS = 120
GRAPH_NLL_SESOI = 0.03
FAMILYWISE_ALPHA = 0.05
POLICIES = (
    "The synergy term is zero on every primitive training and selection action.",
    "The synergy term is absent from every fitted factorized and generic feature dictionary.",
    "The term activates only when both true graph sources are intervened on together.",
    "All model fitting and graph/head selection precede construction of the stress-test outcomes.",
    "Every world must lose noiseless exactness under the outside-family term.",
    "The primary effect is wrong-routing minus learned-routing stochastic composition NLL.",
)


def payload() -> dict[str, object]:
    return {
        "identifier": CONTRACT_ID,
        "development_worlds": DEVELOPMENT_WORLDS,
        "confirmatory_worlds": CONFIRMATORY_WORLDS,
        "graph_nll_sesoi": GRAPH_NLL_SESOI,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "policies": list(POLICIES),
    }


def contract_digest() -> str:
    return sha256(json.dumps(payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
