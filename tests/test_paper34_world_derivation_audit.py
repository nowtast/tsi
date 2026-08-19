import json
from pathlib import Path
import unittest

from tsi.paper34_world_derivation_audit import (
    audit_world_derivation,
    derive_world_seed,
)


class Paper34WorldDerivationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.portable = json.loads(
            (
                root
                / "experiments/paper34_resolution_v1/cleanroom/portable_inputs.json"
            ).read_text(encoding="utf-8")
        )
        cls.ledger = json.loads(
            (
                root
                / "experiments/paper34_resolution_v1/confirmatory/"
                "seed_and_integrity_ledger.json"
            ).read_text(encoding="utf-8")
        )

    def test_declared_seed_derivation_matches_exported_seed(self) -> None:
        root_seed = bytes.fromhex(
            self.ledger["root_seed_hex_revealed_after_execution"]
        )
        expected = self.portable["worlds"][0]["expected_row"]["seed"]
        self.assertEqual(derive_world_seed(root_seed, 0), expected)

    def test_regenerated_world_matches_portable_export(self) -> None:
        audit = audit_world_derivation(
            self.portable, self.ledger, maximum_worlds=2
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["checks"]["test_cases_match"], 2)


if __name__ == "__main__":
    unittest.main()
