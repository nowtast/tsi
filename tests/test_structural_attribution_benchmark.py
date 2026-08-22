from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from tsi.structural_attribution_benchmark import (
    BENCHMARK_DIRECTORY,
    BenchmarkValidationError,
    load_participant_worlds,
    validate_submission,
    verify_release,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class StructuralAttributionBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        example_path = REPO_ROOT / BENCHMARK_DIRECTORY / "examples/minimal_submission.json"
        self.example = json.loads(example_path.read_text(encoding="utf-8"))

    def test_release_hashes_and_reference_values(self) -> None:
        report = verify_release(REPO_ROOT)
        self.assertTrue(report["passed"])
        self.assertEqual(report["artifact_count"], 8)
        self.assertEqual(report["world_count"], 120)

    def test_participant_view_removes_answers_and_test_targets(self) -> None:
        worlds = load_participant_worlds(REPO_ROOT)
        self.assertEqual(len(worlds), 120)
        self.assertEqual(
            set(worlds[0]), {"world_index", "train", "selection", "test_inputs"}
        )
        self.assertTrue(worlds[0]["train"])
        self.assertTrue(worlds[0]["selection"])
        self.assertTrue(worlds[0]["test_inputs"])
        self.assertTrue(all(len(case) == 2 for case in worlds[0]["test_inputs"]))

    def test_minimal_submission_is_valid(self) -> None:
        report = validate_submission(self.example)
        self.assertTrue(report["passed"])
        self.assertEqual(report["coverage"], "smoke")

    def test_submission_rejects_reference_answer_access(self) -> None:
        invalid = copy.deepcopy(self.example)
        invalid["information_policy"]["reference_answers_used"] = True
        with self.assertRaises(BenchmarkValidationError):
            validate_submission(invalid)

    def test_submission_rejects_invalid_graph(self) -> None:
        invalid = copy.deepcopy(self.example)
        invalid["worlds"][0]["selected_graph"] = [2, [2, 4]]
        with self.assertRaises(BenchmarkValidationError):
            validate_submission(invalid)


if __name__ == "__main__":
    unittest.main()
