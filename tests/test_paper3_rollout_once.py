from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tsi.paper3_rollout_once import acquire_once_lock


class Paper3RolloutOnceTest(unittest.TestCase):
    def test_lock_is_created_atomically_and_cannot_be_reacquired(self) -> None:
        with TemporaryDirectory() as temporary:
            lock = Path(temporary) / "once.lock"
            acquire_once_lock(lock, {"identifier": "test"})

            self.assertTrue(lock.is_file())
            with self.assertRaises(FileExistsError):
                acquire_once_lock(lock, {"identifier": "second"})


if __name__ == "__main__":
    unittest.main()
