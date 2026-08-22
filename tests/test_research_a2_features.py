import unittest

from tsi.paper34_resolution_benchmark import generic_features
from tsi.research_a2_features import (
    ALTERNATIVE_FAMILY_CATALOG,
    NUISANCE_FEATURES,
    TYPED_FAMILY_CATALOG,
    WIDTH_FEATURE_COUNTS,
    WIDTH_POSITION_COUNTS,
    audit_fourth_family_separation,
    audit_misspecification_catalogs,
    audit_width_feature_libraries,
    augmented_generic_features,
)


class ResearchA2FeatureTests(unittest.TestCase):
    def test_declared_widths_have_exact_feature_counts(self) -> None:
        source = (0, 1, 2, 3, 4)
        action = (0, 2, 0, 0, 0)
        graph = (0, (1, 2))
        for positions, features in zip(
            WIDTH_POSITION_COUNTS, WIDTH_FEATURE_COUNTS, strict=True
        ):
            self.assertEqual(
                len(augmented_generic_features(source, action, graph, positions)),
                features,
            )
        self.assertEqual(len(NUISANCE_FEATURES), 49)

    def test_width_55_is_exactly_the_a1_dictionary(self) -> None:
        source = (6, 5, 4, 3, 2)
        action = (0, 0, 1, 0, 0)
        graph = (0, (1, 2))
        self.assertEqual(
            augmented_generic_features(source, action, graph, 55),
            generic_features(source, action, graph),
        )

    def test_all_width_features_are_projectively_collision_free(self) -> None:
        audit = audit_width_feature_libraries()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["state_count"], 7**5)
        self.assertEqual(audit["graph_count"], 30)

    def test_cubic_family_is_separated_from_each_typed_family(self) -> None:
        audit = audit_fourth_family_separation()
        self.assertTrue(audit["passed"])
        self.assertEqual(
            [
                row["minimum_disagreement_count_over_nonzero_scalings"]
                for row in audit["comparisons"]
            ],
            [28, 35, 42],
        )
        self.assertEqual(
            audit["reverse_generic_span_audit"]["minimum_disagreement_count"],
            28,
        )

    def test_misspecification_catalogs_swap_one_family_at_equal_width(self) -> None:
        audit = audit_misspecification_catalogs()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["output_feature_positions"], 55)
        self.assertEqual(len(TYPED_FAMILY_CATALOG), len(ALTERNATIVE_FAMILY_CATALOG))


if __name__ == "__main__":
    unittest.main()
