from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

NINE_GATE_LABELS = (
    "graph--head identification",
    "learned-routing NLL",
    "factorized-head graph NLL",
    "generic-head graph NLL",
    "dense-head graph NLL",
    "matched-head predictive equivalence",
    "rollout Hamming",
    "criterion Brier",
    "outside-family noninferiority",
)


class Package22ReviewResolutionTest(unittest.TestCase):
    def test_paper34_nine_gate_conjunction_is_explicit_in_both_languages(self) -> None:
        sections = (
            "papers/paper3/sections/resolution_results.tex",
            "papers/paper3_ko/sections/resolution_results.tex",
            "papers/paper4/sections/method.tex",
            "papers/paper4_ko/sections/method.tex",
        )
        for section in sections:
            source = (REPOSITORY_ROOT / section).read_text()
            for label in NINE_GATE_LABELS:
                self.assertIn(label, source, f"{label!r} is missing from {section}")

    def test_paper1_audits_use_the_current_counterexample_and_zero_criterion(self) -> None:
        english = (REPOSITORY_ROOT / "papers/paper1/proof_audit.md").read_text()
        korean = (REPOSITORY_ROOT / "papers/paper1_ko/proof_audit.md").read_text()
        readme = (REPOSITORY_ROOT / "papers/paper1/README.md").read_text()

        for source in (english, korean, readme):
            self.assertNotIn("torus", source)
            self.assertNotIn("wedge", source)
        self.assertIn("six zero conditions", english)
        self.assertIn("여섯 zero condition", korean)
        self.assertIn("extended pseudometric", korean)

    def test_paper1_korean_limits_include_current_layers_and_plural_evidence(self) -> None:
        english = (
            REPOSITORY_ROOT
            / "papers/paper1/sections/interpretation_and_limits.tex"
        ).read_text()
        korean = (
            REPOSITORY_ROOT
            / "papers/paper1_ko/sections/interpretation_and_limits.tex"
        ).read_text()

        for token in ("Filtration", "mass", "temporal"):
            self.assertIn(token, korean)
        self.assertIn("empirical papers", english)
        self.assertIn("empirical paper들은", korean)


if __name__ == "__main__":
    unittest.main()
