from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tsi.paper3_sealed_access import (
    COMMITMENT_FILENAME,
    LEDGER_FILENAME,
    audit_sealed_test_material,
    initialize_sealed_test_material,
    reveal_sealed_test_seed,
)


class SealedAccessTest(unittest.TestCase):
    def test_initialization_creates_commitment_without_access(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            initialize_sealed_test_material(root)
            audit = audit_sealed_test_material(root)

            self.assertTrue(audit.passed)
            self.assertEqual(audit.event_count, 1)
            self.assertEqual(audit.test_seed_reveals, 0)
            self.assertEqual(audit.test_result_evaluations, 0)
            self.assertEqual(audit.escrow_size, 32)
            self.assertEqual(audit.escrow_mode, 0)
            self.assertIsNotNone(audit.artifact_digest)

    def test_initialization_is_idempotent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_sealed_test_material(root)
            first = audit_sealed_test_material(root)

            initialize_sealed_test_material(root)
            second = audit_sealed_test_material(root)

            self.assertEqual(first.commitment, second.commitment)
            self.assertEqual(first.latest_event_hash, second.latest_event_hash)

    def test_partial_material_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / COMMITMENT_FILENAME).write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "partially"):
                initialize_sealed_test_material(root)

    def test_ledger_tampering_is_detected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_sealed_test_material(root)
            ledger_path = root / LEDGER_FILENAME
            event = json.loads(ledger_path.read_text(encoding="utf-8"))
            event["test_seed_reveals_after_event"] = 1
            ledger_path.write_text(
                f"{json.dumps(event, sort_keys=True)}\n",
                encoding="utf-8",
            )

            audit = audit_sealed_test_material(root)

            self.assertFalse(audit.passed)
            self.assertTrue(any("hash mismatch" in error for error in audit.errors))

    def test_reveal_is_one_shot_and_hash_chained(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_sealed_test_material(root)

            secret, commitment = reveal_sealed_test_seed(
                root,
                gate_digest="a" * 64,
                frozen_artifact_digests={"analysis": "b" * 64},
            )

            self.assertEqual(len(secret), 32)
            self.assertEqual(len(commitment), 64)
            events = [
                json.loads(line)
                for line in (root / LEDGER_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["event"], "test_seed_revealed")
            self.assertEqual(events[-1]["test_seed_reveals_after_event"], 1)
            with self.assertRaisesRegex(RuntimeError, "zero-access"):
                reveal_sealed_test_seed(
                    root,
                    gate_digest="a" * 64,
                    frozen_artifact_digests={"analysis": "b" * 64},
                )


if __name__ == "__main__":
    unittest.main()
