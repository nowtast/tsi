from pathlib import Path
import tempfile
import unittest

from tsi.paper3_validity_access import (
    append_validity_access_event,
    audit_validity_access,
    initialize_validity_seed,
    reveal_validity_seed,
)


class Paper3ValidityAccessTests(unittest.TestCase):
    def test_zero_to_final_access_sequence_is_single_reveal_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_validity_seed(root)
            zero = audit_validity_access(root, expected_phase="zero")
            self.assertTrue(zero["passed"])
            self.assertEqual(zero["seed_reveals"], 0)
            self.assertEqual(zero["result_evaluations"], 0)
            with self.assertRaises(RuntimeError):
                append_validity_access_event(
                    root,
                    "validity_prediction_started",
                    {},
                )
            secret, commitment = reveal_validity_seed(
                root,
                gate_digest="a" * 64,
                frozen_artifact_digests={"source": "b" * 64},
            )
            self.assertEqual(len(secret), 32)
            self.assertEqual(len(commitment), 64)
            append_validity_access_event(
                root,
                "validity_prediction_started",
                {},
            )
            append_validity_access_event(
                root,
                "validity_prediction_completed",
                {},
            )
            append_validity_access_event(
                root,
                "validity_result_evaluated",
                {},
            )
            append_validity_access_event(
                root,
                "validity_report_generated",
                {},
            )
            final = audit_validity_access(root, expected_phase="final")
            self.assertTrue(final["passed"])
            self.assertEqual(final["seed_reveals"], 1)
            self.assertEqual(final["result_evaluations"], 1)


if __name__ == "__main__":
    unittest.main()
