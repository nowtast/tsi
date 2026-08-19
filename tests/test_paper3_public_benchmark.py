from pathlib import Path
import unittest

from tsi.paper3_public_benchmark import BENCHMARK_ID, run_public_benchmark


class PublicBenchmarkTests(unittest.TestCase):
    def test_smoke_benchmark_is_complete_and_deterministic(self) -> None:
        root = Path(__file__).resolve().parents[1]
        first = run_public_benchmark(root, smoke=True)
        second = run_public_benchmark(root, smoke=True)
        self.assertEqual(first, second)
        self.assertEqual(first["benchmark_id"], BENCHMARK_ID)
        self.assertEqual(first["v3"]["cell_count"], 16)
        self.assertEqual(first["replication"]["cell_count"], 12)
        self.assertEqual(first["v3"]["transition_exact_cells"], 16)
        self.assertEqual(first["replication"]["transition_exact_cells"], 12)


if __name__ == "__main__":
    unittest.main()
