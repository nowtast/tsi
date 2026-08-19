from pathlib import Path
import unittest

from tsi.paper_parity import check_bilingual_parity


class BilingualPaperParityTest(unittest.TestCase):
    def test_all_paired_packages_have_matching_structure(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        self.assertEqual(check_bilingual_parity(repository_root), [])


if __name__ == "__main__":
    unittest.main()
