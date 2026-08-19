from __future__ import annotations

from hashlib import sha256
import unittest

from tsi.paper3_rollout_contract import (
    MAX_HORIZON,
    TRAJECTORIES_PER_WORLD,
)
from tsi.paper3_rollout_generator import (
    ACTION_BLOCK_NAMES,
    TRAJECTORIES_PER_STRATUM,
    audit_rollout_generator,
    development_rollout_manifest,
    development_rollout_trajectories,
    development_rollout_worlds,
    sealed_rollout_trajectories,
    sealed_rollout_worlds,
)


class Paper3RolloutGeneratorTest(unittest.TestCase):
    def test_development_manifest_is_balanced_and_deterministic(self) -> None:
        first = development_rollout_manifest()
        second = development_rollout_manifest()
        specs = development_rollout_trajectories(0)

        self.assertEqual(first, second)
        self.assertEqual(len(specs), TRAJECTORIES_PER_WORLD)
        self.assertEqual(
            sum(spec.stratum == "unseen_structural_mode" for spec in specs),
            TRAJECTORIES_PER_STRATUM,
        )
        self.assertEqual(len({spec.initial_code for spec in specs}), len(specs))
        for spec in specs:
            self.assertEqual(len(spec.actions), MAX_HORIZON)
            for offset in range(0, MAX_HORIZON, len(ACTION_BLOCK_NAMES)):
                self.assertEqual(
                    sorted(
                        action.name
                        for action in spec.actions[
                            offset : offset + len(ACTION_BLOCK_NAMES)
                        ]
                    ),
                    sorted(ACTION_BLOCK_NAMES),
                )

    def test_sealed_generator_uses_fresh_context_mechanisms(self) -> None:
        secret = b"r" * 32
        commitment = sha256(secret).hexdigest()
        worlds = sealed_rollout_worlds(secret, commitment, 10)
        trajectories = sealed_rollout_trajectories(secret, commitment, 0)
        public = {
            world.active_parameter_signature for world in development_rollout_worlds()
        }

        self.assertEqual(len(worlds), 10)
        self.assertFalse(
            public.intersection(world.active_parameter_signature for world in worlds)
        )
        self.assertEqual(len(trajectories), TRAJECTORIES_PER_WORLD)
        with self.assertRaisesRegex(ValueError, "commitment"):
            sealed_rollout_worlds(secret, "0" * 64, 10)

    def test_generator_audit_passes_without_materializing_test(self) -> None:
        audit = audit_rollout_generator()

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["sealed_worlds_materialized"], 0)


if __name__ == "__main__":
    unittest.main()
