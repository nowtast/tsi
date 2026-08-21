from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tsi.paper_parity import check_bilingual_parity


class BilingualPaperParityTest(unittest.TestCase):
    def test_all_paired_packages_have_matching_structure(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        self.assertEqual(check_bilingual_parity(repository_root), [])

    def _write_pair(
        self,
        root: Path,
        english_section: str,
        korean_section: str,
        english_bibliography: str,
        korean_bibliography: str | None = None,
    ) -> None:
        for package in ("sample", "sample_ko"):
            (root / "papers" / package / "sections").mkdir(parents=True)
            (root / "papers" / package / "main.tex").write_text(
                "\\input{sections/body}\n"
            )
        (root / "papers" / "sample" / "sections" / "body.tex").write_text(
            english_section
        )
        (root / "papers" / "sample_ko" / "sections" / "body.tex").write_text(
            korean_section
        )
        (root / "papers" / "sample" / "references.bib").write_text(
            english_bibliography
        )
        (root / "papers" / "sample_ko" / "references.bib").write_text(
            korean_bibliography
            if korean_bibliography is not None
            else english_bibliography
        )

    def test_bibliographies_must_be_byte_identical(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_pair(
                root,
                "\\cite{source}\n",
                "\\cite{source}\n",
                "@article{source, title={A}}\n",
                "@article{source, title={B}}\n",
            )
            errors = check_bilingual_parity(root, papers=("sample",))
            self.assertTrue(
                any("not byte-identical" in error.message for error in errors)
            )

    def test_missing_citation_key_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_pair(
                root,
                "\\cite{missing}\n",
                "\\cite{missing}\n",
                "@article{source, title={A}}\n",
            )
            errors = check_bilingual_parity(root, papers=("sample",))
            self.assertTrue(
                any("missing from bibliography" in error.message for error in errors)
            )

    def test_uncited_bibliography_entry_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_pair(
                root,
                "\\cite{source}\n",
                "\\cite{source}\n",
                "@article{source, title={A}}\n"
                "@article{unused, title={B}}\n",
            )
            errors = check_bilingual_parity(root, papers=("sample",))
            self.assertTrue(
                any("uncited bibliography entries" in error.message for error in errors)
            )


if __name__ == "__main__":
    unittest.main()
