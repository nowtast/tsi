import unittest

from tsi import StructuralImaginationSpace, StructuralState


class StructuralImaginationSpaceTest(unittest.TestCase):
    def test_add_state_and_transition(self) -> None:
        source = StructuralState(objects=frozenset({"a"}))
        target = StructuralState(objects=frozenset({"a", "b"}))
        space = StructuralImaginationSpace()

        source_index = space.add_state(source)
        target_index = space.add_state(target)
        space.add_transition(source_index, target_index, "add-object")

        self.assertEqual(space.neighbors(source_index), (target_index,))

    def test_invalid_state_index(self) -> None:
        space = StructuralImaginationSpace()

        with self.assertRaises(IndexError):
            space.neighbors(0)


if __name__ == "__main__":
    unittest.main()
