import unittest

import numpy as np

from tsi.research_a2_populations import (
    GENERIC_MISSPECIFIED,
    MATCHED,
    TYPED_MISSPECIFIED,
    balanced_world_specs,
    family_pairs_for_condition,
    generate_a2_cases,
    generate_coupled_noise_cases,
    paired_misspecification_specs,
)


class ResearchA2PopulationTests(unittest.TestCase):
    def test_condition_pair_manifests_have_intended_support(self) -> None:
        self.assertEqual(len(family_pairs_for_condition(MATCHED)), 9)
        typed_misspecified = family_pairs_for_condition(TYPED_MISSPECIFIED)
        generic_misspecified = family_pairs_for_condition(GENERIC_MISSPECIFIED)
        self.assertEqual(len(typed_misspecified), 5)
        self.assertEqual(len(generic_misspecified), 5)
        self.assertTrue(all("cubic_target" in pair for pair in typed_misspecified))
        self.assertTrue(
            all("quadratic_target" in pair for pair in generic_misspecified)
        )

    def test_balanced_specs_and_case_stream_are_deterministic(self) -> None:
        first = np.random.default_rng(1234)
        second = np.random.default_rng(1234)
        specs_a = balanced_world_specs(45, MATCHED, first)
        specs_b = balanced_world_specs(45, MATCHED, second)
        self.assertEqual(specs_a, specs_b)
        self.assertEqual(
            generate_a2_cases(
                specs_a[0], 12, first, composition=True, noise_probability=0.3
            ),
            generate_a2_cases(
                specs_b[0], 12, second, composition=True, noise_probability=0.3
            ),
        )

    def test_composition_cases_cycle_over_three_strata(self) -> None:
        rng = np.random.default_rng(5)
        spec = balanced_world_specs(1, MATCHED, rng)[0]
        cases = generate_a2_cases(
            spec, 12, rng, composition=True, noise_probability=0.0
        )
        counts = {}
        for case in cases:
            counts[case.composition_stratum] = (
                counts.get(case.composition_stratum, 0) + 1
            )
        self.assertEqual(set(counts.values()), {4})

    def test_coupled_noise_masks_are_nested(self) -> None:
        rng = np.random.default_rng(11)
        spec = balanced_world_specs(1, MATCHED, rng)[0]
        streams = generate_coupled_noise_cases(spec, 100, rng, (0.08, 0.3, 0.6, 0.8))
        previous = set()
        for level in (0.08, 0.3, 0.6, 0.8):
            corruptions = {
                (row, layer)
                for row, case in enumerate(streams[level])
                for layer, (observed, center) in enumerate(
                    zip(case.observed, case.center, strict=True)
                )
                if observed != center
            }
            self.assertTrue(previous <= corruptions)
            previous = corruptions

    def test_misspecification_pairs_share_all_nonfamily_parameters(self) -> None:
        rng = np.random.default_rng(17)
        cubic, quadratic = paired_misspecification_specs(10, rng)
        for first, second in zip(cubic, quadratic, strict=True):
            self.assertEqual(first.graph, second.graph)
            self.assertEqual(first.multipliers, second.multipliers)
            self.assertEqual(first.coefficients, second.coefficients)
            self.assertEqual(
                tuple(
                    "quadratic_target" if family == "cubic_target" else family
                    for family in first.families
                ),
                second.families,
            )


if __name__ == "__main__":
    unittest.main()
