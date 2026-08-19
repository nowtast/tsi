from pathlib import Path
import subprocess
import sys
import unittest

from tools.fetch_zenodo_release import load_manifest


class PublicReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_zenodo_manifest_is_pinned_to_published_version(self) -> None:
        manifest = load_manifest(
            self.root / "artifacts" / "paper03-04-v1.0.0.json"
        )
        self.assertEqual(manifest["version_doi"], "10.5281/zenodo.22004526")
        self.assertEqual(manifest["concept_doi"], "10.5281/zenodo.22004525")
        artifact = manifest["files"][0]
        self.assertEqual(artifact["size_bytes"], 44242790)
        self.assertEqual(
            artifact["sha256"],
            "11e3fd40b623a46c3ebab1ed03e0125329fef087e88191de6b73e6229d141e07",
        )

    def test_downloader_cli_is_available(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/fetch_zenodo_release.py", "--help"],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--extract", completed.stdout)

    def test_public_docs_reference_the_version_doi(self) -> None:
        for name in ("README.md", "README_KO.md", "REPRODUCE.md", "REPRODUCE_KO.md"):
            text = (self.root / name).read_text(encoding="utf-8")
            self.assertIn("10.5281/zenodo.22004526", text, name)


if __name__ == "__main__":
    unittest.main()
