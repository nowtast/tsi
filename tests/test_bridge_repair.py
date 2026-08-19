from itertools import combinations, product
import unittest

from tsi.bridge_repair import (
    BinaryObservation,
    FiniteBridgeCode,
    joint_bridge_repairs,
    normalized_bridge_defect,
    one_sided_relation_repair,
    relation_word,
    threshold_profile,
    thresholds_separate_distance_alphabet,
)


def subsets(values):
    values = tuple(values)
    return tuple(
        frozenset(
            values[index]
            for index in range(len(values))
            if mask & (1 << index)
        )
        for mask in range(1 << len(values))
    )


def error_erasure_neighborhood(word, max_errors, max_erasures):
    neighborhood = set()
    for observation in product((0, 1, None), repeat=len(word)):
        erasures = sum(value is None for value in observation)
        errors = sum(
            value is not None and value != bit
            for value, bit in zip(observation, word, strict=True)
        )
        if erasures <= max_erasures and errors <= max_errors:
            neighborhood.add(observation)
    return neighborhood


class LocalBridgeRepairTest(unittest.TestCase):
    def test_one_sided_repair_is_unique_and_equals_the_bridge_defect(self) -> None:
        cells = ("00", "01", "10", "11")
        relation = {"00", "01", "11"}
        induced = {"00", "10"}
        weights = {"00": 1.0, "01": 2.0, "10": 3.0, "11": 4.0}

        repair = one_sided_relation_repair(
            cells,
            relation,
            induced,
            relation_weights=weights,
        )

        self.assertEqual(repair.consensus, frozenset(induced))
        self.assertEqual(repair.relation_flips, frozenset({"01", "10", "11"}))
        self.assertEqual(repair.cost, 9.0)
        self.assertEqual(repair.normalized_defect, 0.75)
        self.assertEqual(
            repair.normalized_defect,
            normalized_bridge_defect(cells, relation, induced),
        )

    def test_joint_closed_form_matches_brute_force_on_all_three_cell_inputs(
        self,
    ) -> None:
        cells = (0, 1, 2)
        relations = subsets(cells)
        for relation, induced, relation_weight, induced_weight in product(
            relations,
            relations,
            (1.0, 2.0),
            (1.0, 2.0),
        ):
            repairs = joint_bridge_repairs(
                cells,
                relation,
                induced,
                relation_weights=relation_weight,
                induced_weights=induced_weight,
            )
            brute_costs = {
                consensus: (
                    relation_weight * len(consensus.symmetric_difference(relation))
                    + induced_weight
                    * len(consensus.symmetric_difference(induced))
                )
                for consensus in relations
            }
            minimum = min(brute_costs.values())
            brute_minimizers = {
                consensus
                for consensus, cost in brute_costs.items()
                if cost == minimum
            }

            self.assertEqual(
                {repair.consensus for repair in repairs},
                brute_minimizers,
            )
            self.assertTrue(all(repair.cost == minimum for repair in repairs))

            conflicts = len(relation.symmetric_difference(induced))
            expected_count = (
                2**conflicts
                if relation_weight == induced_weight
                else 1
            )
            self.assertEqual(len(repairs), expected_count)

    def test_cellwise_reliability_and_minimal_joint_ambiguity(self) -> None:
        ambiguous = joint_bridge_repairs(
            ("cell",),
            (),
            ("cell",),
        )
        self.assertEqual(
            {repair.consensus for repair in ambiguous},
            {frozenset(), frozenset({"cell"})},
        )
        self.assertEqual({repair.cost for repair in ambiguous}, {1.0})

        trusted_relation = joint_bridge_repairs(
            ("cell",),
            (),
            ("cell",),
            relation_weights=3.0,
            induced_weights=1.0,
        )
        self.assertEqual(
            trusted_relation[0].consensus,
            frozenset(),
        )
        self.assertEqual(trusted_relation[0].cost, 1.0)

        trusted_induced = joint_bridge_repairs(
            ("cell",),
            (),
            ("cell",),
            relation_weights=1.0,
            induced_weights=3.0,
        )
        self.assertEqual(
            trusted_induced[0].consensus,
            frozenset({"cell"}),
        )
        self.assertEqual(trusted_induced[0].cost, 1.0)

        repaired_again = joint_bridge_repairs(
            ("cell",),
            trusted_induced[0].consensus,
            trusted_induced[0].consensus,
        )
        self.assertEqual(len(repaired_again), 1)
        self.assertEqual(
            repaired_again[0].consensus,
            trusted_induced[0].consensus,
        )
        self.assertEqual(repaired_again[0].cost, 0.0)

    def test_invalid_local_repair_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty"):
            joint_bridge_repairs((), (), ())
        with self.assertRaisesRegex(ValueError, "outside"):
            one_sided_relation_repair((0,), (1,), ())
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            joint_bridge_repairs((0,), (), (), relation_weights=0.0)
        with self.assertRaisesRegex(ValueError, "restricted"):
            joint_bridge_repairs(
                (0, 1, 2),
                (),
                (0, 1, 2),
                max_repairs=4,
            )


class BridgeObservationCodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.code = FiniteBridgeCode(
            probes=("p0", "p1", "p2"),
            codewords=(
                ("a", (0, 0, 0)),
                ("b", (0, 1, 1)),
                ("c", (1, 0, 1)),
            ),
        )

    def test_identifiability_is_exactly_pairwise_difference_hitting(self) -> None:
        supports = [
            set(support)
            for _, _, support in self.code.pairwise_difference_supports()
        ]
        for size in range(4):
            for selected in combinations(self.code.probes, size):
                hitting = all(set(selected).intersection(support) for support in supports)
                self.assertEqual(self.code.is_identifiable(selected), hitting)

        self.assertEqual(
            set(self.code.minimum_identifying_probe_sets()),
            {
                ("p0", "p1"),
                ("p0", "p2"),
                ("p1", "p2"),
            },
        )
        self.assertEqual(self.code.information_lower_bound, 2)
        self.assertEqual(self.code.minimum_distance(), 2.0)

    def test_duplicate_full_words_are_not_identifiable(self) -> None:
        code = FiniteBridgeCode(
            probes=("p",),
            codewords=(("left", (1,)), ("right", (1,))),
        )
        self.assertFalse(code.is_identifiable())
        self.assertEqual(code.minimum_distance(), 0.0)
        self.assertEqual(code.minimum_identifying_probe_sets(), ())
        self.assertFalse(code.error_erasure_identifiable(0, 0))

    def test_nearest_repair_and_margin_stability(self) -> None:
        code = FiniteBridgeCode(
            probes=("p0", "p1", "p2", "p3"),
            codewords=(
                ("zero", (0, 0, 0, 0)),
                ("one", (1, 1, 1, 1)),
                ("mixed", (1, 0, 1, 0)),
            ),
        )
        weights = {"p0": 1.0, "p1": 2.0, "p2": 4.0, "p3": 8.0}
        observations = tuple(
            BinaryObservation(tuple(word))
            for word in product((0, 1), repeat=4)
        )
        certified_pairs = 0
        for left, right in product(observations, repeat=2):
            result = code.nearest_repair(left, weights=weights)
            budget = code.observation_distance(left, right, weights=weights)
            if result.is_unique and 2 * budget < result.margin:
                certified_pairs += 1
                perturbed = code.nearest_repair(right, weights=weights)
                self.assertTrue(perturbed.is_unique)
                self.assertEqual(perturbed.candidates, result.candidates)
        self.assertGreater(certified_pairs, 0)

        erased = code.nearest_repair({"p1": 0})
        self.assertFalse(erased.is_unique)
        self.assertEqual(erased.margin, 0.0)
        with self.assertRaisesRegex(ValueError, "same erasure pattern"):
            code.observation_distance({"p0": 0}, {"p1": 0})

    def test_invalid_codes_and_observations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one probe"):
            FiniteBridgeCode((), (("candidate", ()),))
        with self.assertRaisesRegex(ValueError, "unique"):
            FiniteBridgeCode(("p", "p"), (("candidate", (0, 0)),))
        with self.assertRaisesRegex(ValueError, "one bit"):
            FiniteBridgeCode(("p",), (("candidate", (0, 1)),))
        with self.assertRaisesRegex(ValueError, "binary"):
            FiniteBridgeCode(("p",), (("candidate", (2,)),))
        with self.assertRaisesRegex(ValueError, "unknown probes"):
            self.code.nearest_repair({"unknown": 0})
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            self.code.error_erasure_identifiable(True, 0)


class ErrorErasureTheoremTest(unittest.TestCase):
    def test_minimum_distance_criterion_is_necessary_and_sufficient_exhaustively(
        self,
    ) -> None:
        words = tuple(product((0, 1), repeat=3))
        checked = 0
        for candidate_count in (2, 3):
            for selected_words in combinations(words, candidate_count):
                code = FiniteBridgeCode(
                    probes=("p0", "p1", "p2"),
                    codewords=tuple(
                        (f"c{index}", word)
                        for index, word in enumerate(selected_words)
                    ),
                )
                for max_errors, max_erasures in product(range(3), repeat=2):
                    neighborhoods = [
                        error_erasure_neighborhood(
                            word,
                            max_errors,
                            max_erasures,
                        )
                        for word in selected_words
                    ]
                    disjoint = all(
                        left.isdisjoint(right)
                        for left, right in combinations(neighborhoods, 2)
                    )
                    self.assertEqual(
                        code.error_erasure_identifiable(
                            max_errors,
                            max_erasures,
                        ),
                        disjoint,
                    )
                    checked += 1
        self.assertEqual(checked, 756)


class StructuralSpecializationTest(unittest.TestCase):
    def test_adjacency_does_not_identify_a_complex_without_a_flag_assumption(
        self,
    ) -> None:
        code = FiniteBridgeCode(
            probes=("edge01", "edge02", "edge12", "triangle012"),
            codewords=(
                ("boundary", (1, 1, 1, 0)),
                ("filled", (1, 1, 1, 1)),
            ),
        )
        self.assertFalse(
            code.is_identifiable(("edge01", "edge02", "edge12"))
        )
        self.assertTrue(code.is_identifiable())
        self.assertEqual(
            code.minimum_identifying_probe_sets(),
            (("triangle012",),),
        )

    def test_threshold_profiles_identify_only_a_separated_distance_alphabet(
        self,
    ) -> None:
        alphabet = (1.0, 2.0, 4.0)
        thresholds = (1.0, 2.0)
        self.assertTrue(
            thresholds_separate_distance_alphabet(alphabet, thresholds)
        )
        self.assertEqual(threshold_profile(1.0, thresholds), (1, 1))
        self.assertEqual(threshold_profile(2.0, thresholds), (0, 1))
        self.assertEqual(threshold_profile(4.0, thresholds), (0, 0))

        self.assertFalse(
            thresholds_separate_distance_alphabet(alphabet, (5.0,))
        )
        finite_thresholds = (0.5, 1.0, 2.0)
        self.assertEqual(
            threshold_profile(3.0, finite_thresholds),
            threshold_profile(4.0, finite_thresholds),
        )

    def test_category_composability_and_dynamic_reachability_are_lossy(
        self,
    ) -> None:
        category_code = FiniteBridgeCode(
            probes=("composable_a_a", "a_squared_is_e"),
            codewords=(("C4", (1, 0)), ("V4", (1, 1))),
        )
        self.assertFalse(
            category_code.is_identifiable(("composable_a_a",))
        )
        self.assertTrue(category_code.is_identifiable())

        dynamics_code = FiniteBridgeCode(
            probes=("reach01", "reach12", "reach02", "step02"),
            codewords=(("path", (1, 1, 1, 0)), ("shortcut", (1, 1, 1, 1))),
        )
        self.assertFalse(
            dynamics_code.is_identifiable(
                ("reach01", "reach12", "reach02")
            )
        )
        self.assertTrue(dynamics_code.is_identifiable())

    def test_order_restriction_needs_additional_probes(self) -> None:
        code = FiniteBridgeCode(
            probes=("cross_a0_b", "cross_a1_b", "within_a0_a1"),
            codewords=(
                ("equality_order", (0, 0, 0)),
                ("within_type_order", (0, 0, 1)),
            ),
        )
        self.assertFalse(
            code.is_identifiable(("cross_a0_b", "cross_a1_b"))
        )
        self.assertTrue(code.is_identifiable())

    def test_relation_words_follow_the_declared_cell_order(self) -> None:
        self.assertEqual(
            relation_word(("a", "b", "c"), {"b", "c"}),
            (0, 1, 1),
        )


if __name__ == "__main__":
    unittest.main()
