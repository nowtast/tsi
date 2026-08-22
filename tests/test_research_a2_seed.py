import unittest
from pathlib import Path

from tsi.research_a2_seed import validate_custodian_attestation


class ResearchA2SeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = bytes(range(32))
        self.freeze = {
            "freeze_digest": "f" * 64,
            "frozen_at_utc": "2026-08-22T00:00:00+00:00",
            "seed_custodian_id": "reviewer-custodian-01",
        }
        self.commit = "a" * 40
        from hashlib import sha256

        self.attestation = {
            "status": "external_custodian_single_seed_attestation",
            "custodian_id": "reviewer-custodian-01",
            "freeze_digest": "f" * 64,
            "freeze_git_commit": self.commit,
            "generated_at_utc": "2026-08-22T00:01:00+00:00",
            "seed_sha256": sha256(self.seed).hexdigest(),
            "generation_method": "operating-system CSPRNG",
            "single_draw": True,
            "author_generated_seed": False,
            "author_selected_seed": False,
        }

    def test_valid_single_external_draw_is_accepted(self) -> None:
        result = validate_custodian_attestation(
            self.seed, self.attestation, self.freeze, self.commit
        )
        self.assertEqual(result["seed_origin"], "external_custodian_single_draw")

    def test_author_selection_or_reroll_is_rejected(self) -> None:
        for field, value in (
            ("author_selected_seed", True),
            ("author_generated_seed", True),
            ("single_draw", False),
        ):
            with self.subTest(field=field):
                changed = {**self.attestation, field: value}
                with self.assertRaises(ValueError):
                    validate_custodian_attestation(
                        self.seed, changed, self.freeze, self.commit
                    )

    def test_attestation_must_postdate_and_match_the_public_freeze(self) -> None:
        stale = {
            **self.attestation,
            "generated_at_utc": "2026-08-21T23:59:00+00:00",
        }
        with self.assertRaises(ValueError):
            validate_custodian_attestation(
                self.seed, stale, self.freeze, self.commit
            )
        wrong_commit = {**self.attestation, "freeze_git_commit": "b" * 40}
        with self.assertRaises(ValueError):
            validate_custodian_attestation(
                self.seed, wrong_commit, self.freeze, self.commit
            )

    def test_commit_tool_contains_no_author_side_random_generator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "tools/commit_research_a2_seed.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("import secrets", "token_bytes", "os.urandom"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
