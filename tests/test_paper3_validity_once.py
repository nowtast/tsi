import json
from pathlib import Path
import tempfile
import unittest

from tsi.paper3_validity_once import acquire_once_lock


class Paper3ValidityOnceTests(unittest.TestCase):
    def test_lock_is_created_exclusively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "p3_4b_once.lock"
            payload = {"identifier": "test"}
            acquire_once_lock(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)
            with self.assertRaises(FileExistsError):
                acquire_once_lock(path, payload)


if __name__ == "__main__":
    unittest.main()
