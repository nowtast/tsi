from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tsi.paper3_rollout_access import (
    LEDGER_FILENAME,
    append_rollout_access_event,
    audit_rollout_access,
    initialize_rollout_seed,
    reveal_rollout_seed,
)


class Paper3RolloutAccessTest(unittest.TestCase):
    def test_zero_access_initialization_is_idempotent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_rollout_seed(root)
            first = audit_rollout_access(root, expected_phase="zero")
            initialize_rollout_seed(root)
            second = audit_rollout_access(root, expected_phase="zero")

            self.assertTrue(first["passed"])
            self.assertEqual(first["commitment"], second["commitment"])
            self.assertEqual(first["seed_reveals"], 0)
            self.assertEqual(first["result_evaluations"], 0)

    def test_reveal_and_events_are_one_shot_and_hash_chained(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_rollout_seed(root)
            secret, commitment = reveal_rollout_seed(
                root,
                gate_digest="a" * 64,
                frozen_artifact_digests={"analysis": "b" * 64},
            )
            self.assertEqual(len(secret), 32)
            self.assertEqual(len(commitment), 64)
            for event in (
                "rollout_prediction_started",
                "rollout_prediction_completed",
                "rollout_result_evaluated",
                "rollout_report_generated",
            ):
                append_rollout_access_event(root, event, {})

            final = audit_rollout_access(root, expected_phase="final")
            self.assertTrue(final["passed"])
            self.assertEqual(final["seed_reveals"], 1)
            self.assertEqual(final["result_evaluations"], 1)
            with self.assertRaisesRegex(RuntimeError, "zero-access"):
                reveal_rollout_seed(
                    root,
                    gate_digest="a" * 64,
                    frozen_artifact_digests={},
                )

    def test_ledger_tampering_is_detected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_rollout_seed(root)
            ledger = root / LEDGER_FILENAME
            event = json.loads(ledger.read_text())
            event["seed_reveals_after_event"] = 1
            ledger.write_text(f"{json.dumps(event)}\n")

            audit = audit_rollout_access(root, expected_phase="zero")
            self.assertFalse(audit["passed"])
            self.assertTrue(any("hash mismatch" in item for item in audit["errors"]))


if __name__ == "__main__":
    unittest.main()
