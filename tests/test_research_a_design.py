import unittest
from dataclasses import replace

from tsi.paper34_resolution_benchmark import (
    TransitionCase,
    WorldSpec,
    deterministic_successor,
    fit_generic_sparse,
)
from tsi.research_a_design import (
    exact_generic_support_recovered,
    fit_isomorphic_generic,
    fit_typed_structured,
    fit_unstructured_generic,
    isomorphic_prediction_audit,
)


class ResearchADesignTests(unittest.TestCase):
    @staticmethod
    def _covered_world() -> tuple[WorldSpec, tuple[TransitionCase, ...]]:
        spec = WorldSpec(
            world_index=0,
            graph=(0, (1, 2)),
            families=("quadratic_target", "source_target"),
            multipliers=(1, 2, 3, 1, 2),
            coefficients=(2, 3),
        )
        cases = []
        for target_state in range(7):
            for first_source_state in range(7):
                for second_source_state in range(7):
                    source = (
                        target_state,
                        first_source_state,
                        second_source_state,
                        0,
                        0,
                    )
                    for active in range(5):
                        for magnitude in (1, 2):
                            action = tuple(
                                magnitude if layer == active else 0
                                for layer in range(5)
                            )
                            center = deterministic_successor(
                                source,
                                action,
                                spec.graph,
                                spec.families,
                                spec.multipliers,
                                spec.coefficients,
                            )
                            cases.append(
                                TransitionCase(
                                    source=source,
                                    action=action,
                                    observed=center,
                                    center=center,
                                    composition_stratum="primitive",
                                )
                            )
        return spec, tuple(cases)

    def test_typed_and_isomorphic_generic_recover_same_function(self) -> None:
        spec, cases = self._covered_world()
        typed = fit_typed_structured(cases, spec.graph)
        generic = fit_isomorphic_generic(cases, spec.graph)
        self.assertEqual(typed.families, spec.families)
        self.assertEqual(typed.multipliers, spec.multipliers)
        self.assertEqual(typed.coefficients, spec.coefficients)
        self.assertTrue(exact_generic_support_recovered(generic, spec))
        self.assertTrue(isomorphic_prediction_audit(typed, generic, cases)["passed"])

    def test_unstructured_greedy_recovers_covered_noiseless_world(self) -> None:
        spec, cases = self._covered_world()
        historical = fit_generic_sparse(cases, spec.graph, 7, name="historical")
        generic = fit_unstructured_generic(cases, spec.graph)
        self.assertEqual(generic.generic_terms, historical.generic_terms)
        self.assertTrue(exact_generic_support_recovered(generic, spec))

    def test_isomorphic_control_uses_same_rows_under_noise(self) -> None:
        spec, cases = self._covered_world()
        noisy = tuple(
            replace(
                case,
                observed=tuple(
                    (value + 1) % 7 if (index + layer) % 17 == 0 else value
                    for layer, value in enumerate(case.observed)
                ),
            )
            for index, case in enumerate(cases)
        )
        typed = fit_typed_structured(noisy, spec.graph)
        generic = fit_isomorphic_generic(noisy, spec.graph)
        self.assertTrue(isomorphic_prediction_audit(typed, generic, cases)["passed"])
        historical = fit_generic_sparse(noisy, spec.graph, 7, name="historical")
        vectorized = fit_unstructured_generic(noisy, spec.graph)
        self.assertEqual(vectorized.generic_terms, historical.generic_terms)


if __name__ == "__main__":
    unittest.main()
