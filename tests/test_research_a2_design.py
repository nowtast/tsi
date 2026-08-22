import unittest

import numpy as np

from tsi.paper34_resolution_benchmark import (
    center_accuracy,
    coordinate_nll,
    fit_generic_sparse,
)
from tsi.research_a2_design import fit_width_generic
from tsi.research_a2_design import (
    exact_catalog_support_recovered,
    fit_catalog_generic,
    fit_typed_catalog,
    typed_parameters_recovered,
)
from tsi.research_a2_features import (
    ALTERNATIVE_FAMILY_CATALOG,
    TYPED_FAMILY_CATALOG,
)
from tsi.research_a2_populations import (
    GENERIC_MISSPECIFIED,
    TYPED_MISSPECIFIED,
    balanced_world_specs,
    generate_a2_cases,
)
from tests.test_research_a_design import ResearchADesignTests


class ResearchA2DesignTests(unittest.TestCase):
    def test_width_55_reproduces_a1_selector_term_for_term(self) -> None:
        spec, cases = ResearchADesignTests._covered_world()
        a1 = fit_generic_sparse(cases, spec.graph, 7, name="a1")
        a2 = fit_width_generic(cases, spec.graph, 55)
        self.assertEqual(a2.generic_terms, a1.generic_terms)

    def test_all_widths_return_exactly_seven_terms(self) -> None:
        spec, cases = ResearchADesignTests._covered_world()
        for width in (55, 100, 300):
            model = fit_width_generic(cases[:200], spec.graph, width)
            self.assertEqual(len(model.generic_terms), 7)
            self.assertEqual(len(model.predict(cases[0].source, cases[0].action)), 5)

    def test_bidirectional_misspecification_changes_which_class_can_represent_truth(
        self,
    ) -> None:
        rng = np.random.default_rng(20260822)
        for condition, typed_expected, generic_expected in (
            (TYPED_MISSPECIFIED, False, True),
            (GENERIC_MISSPECIFIED, True, False),
        ):
            spec = balanced_world_specs(1, condition, rng)[0]
            train = generate_a2_cases(
                spec, 5000, rng, composition=False, noise_probability=0.0
            )
            test = generate_a2_cases(
                spec, 600, rng, composition=True, noise_probability=0.0
            )
            typed = fit_typed_catalog(
                train, spec.graph, family_catalog=TYPED_FAMILY_CATALOG
            )
            generic = fit_catalog_generic(
                train, spec.graph, family_catalog=ALTERNATIVE_FAMILY_CATALOG
            )
            self.assertEqual(typed_parameters_recovered(typed, spec), typed_expected)
            self.assertEqual(
                exact_catalog_support_recovered(
                    generic, spec, ALTERNATIVE_FAMILY_CATALOG
                ),
                generic_expected,
            )
            representable = typed if typed_expected else generic
            misspecified = generic if typed_expected else typed
            self.assertEqual(center_accuracy(representable, test), 1.0)
            self.assertGreater(
                coordinate_nll(misspecified, test),
                coordinate_nll(representable, test),
            )


if __name__ == "__main__":
    unittest.main()
