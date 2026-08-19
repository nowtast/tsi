from hashlib import sha256
import unittest

from tsi.paper3_validity_generator import (
    UNIT_STRATA,
    development_validity_units,
    development_validity_worlds,
    goal_utility,
    sealed_validity_units,
    sealed_validity_worlds,
)


class Paper3ValidityGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = development_validity_worlds()[0]
        cls.units = development_validity_units(cls.world)

    def test_panel_is_deterministic_and_balanced(self) -> None:
        self.assertEqual(self.units, development_validity_units(self.world))
        counts = {
            stratum: sum(unit.stratum == stratum for unit in self.units)
            for stratum in UNIT_STRATA
        }
        self.assertEqual(counts, {stratum: 16 for stratum in UNIT_STRATA})

    def test_tasks_have_reproducible_nonzero_oracle_gap(self) -> None:
        for unit in self.units:
            self.assertNotIn(
                unit.probe_initial_code,
                {task.start_code for task in unit.tasks},
            )
            for task in unit.tasks:
                endpoints = []
                for plan in task.candidate_plans:
                    current = task.start_code
                    from tsi.paper3_multiworld import successor_code

                    for action in plan:
                        current = successor_code(current, action, self.world)
                    endpoints.append(current)
                utilities = tuple(
                    goal_utility(
                        endpoint,
                        task.goal_layers,
                        task.goal_queries,
                        task.goal_values,
                    )
                    for endpoint in endpoints
                )
                self.assertEqual(utilities, task.oracle_utilities)
                self.assertNotEqual(utilities[0], utilities[1])
                self.assertEqual(
                    utilities[task.oracle_best_index],
                    max(utilities),
                )

    def test_sealed_worlds_exclude_prior_signatures(self) -> None:
        secret = bytes(range(32))
        commitment = sha256(secret).hexdigest()
        exclusions = [
            world.active_parameter_signature
            for world in development_validity_worlds()[:4]
        ]
        worlds = sealed_validity_worlds(
            secret,
            commitment,
            3,
            excluded_active_signatures=exclusions,
        )
        self.assertEqual(len(worlds), 3)
        self.assertTrue(
            set(world.active_parameter_signature for world in worlds).isdisjoint(
                set(exclusions)
            )
        )
        units = sealed_validity_units(secret, commitment, worlds[0])
        self.assertEqual(len(units), 32)

    def test_bad_sealed_commitment_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sealed_validity_worlds(
                bytes(32),
                "0" * 64,
                1,
                excluded_active_signatures=(),
            )


if __name__ == "__main__":
    unittest.main()
