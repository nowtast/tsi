from __future__ import annotations

from hashlib import sha256
import unittest

from tsi.paper3_independence_contract import BenchmarkSplit, WorldFamily
from tsi.paper3_multiworld import (
    DEVELOPMENT_WORLDS_PER_FAMILY,
    VALIDATION_WORLDS_PER_FAMILY,
    _ranked_active_parameters,
)
from tsi.paper3_sealed_worlds import (
    sealed_world_manifest_digest,
    sealed_world_mechanisms,
)


class SealedWorldTest(unittest.TestCase):
    def test_seeded_worlds_are_deterministic_unique_and_public_disjoint(self) -> None:
        secret = bytes(range(32))
        commitment = sha256(secret).hexdigest()

        first = sealed_world_mechanisms(secret, commitment, world_count=50)
        second = sealed_world_mechanisms(secret, commitment, world_count=50)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual(
            {mechanism.cohort for mechanism in first},
            {BenchmarkSplit.SEALED_TEST},
        )
        self.assertEqual(
            len({mechanism.active_parameter_signature for mechanism in first}),
            50,
        )
        public_count = DEVELOPMENT_WORLDS_PER_FAMILY + VALIDATION_WORLDS_PER_FAMILY
        public = {
            (candidate[0], candidate[1])
            for candidate in _ranked_active_parameters(WorldFamily.BRIDGE_COUPLED)[
                :public_count
            ]
        }
        self.assertFalse(
            public.intersection(
                mechanism.active_parameter_signature for mechanism in first
            )
        )
        self.assertEqual(
            sealed_world_manifest_digest(first),
            sealed_world_manifest_digest(second),
        )

    def test_wrong_commitment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "commitment"):
            sealed_world_mechanisms(bytes(range(32)), "0" * 64)


if __name__ == "__main__":
    unittest.main()
