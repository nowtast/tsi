import unittest

import numpy as np

from tsi.coherent import bridge_defects
from tsi.paper3_ablation_experiment import (
    ExactStatePairCache,
    SummaryStatistic,
    audit_manual_gradients,
    evaluate_ablation_model,
)
from tsi.paper3_objective_ablation import (
    DEFAULT_ABLATION_SEEDS,
    OBJECTIVE_MASKS,
    P3_ABLATION_BENCHMARK_ID,
    ObjectiveCondition,
    P3AblationSpec,
    TrainableStructuralJEPA,
    build_p3_ablation_benchmark,
    build_p3_ablation_dataset,
    decode_ablation_predictions,
    interaction_residue,
)
from tsi.paper3_oracle_benchmark import SPLIT_NAMES, SyntheticAction


EXPECTED_BENCHMARK_DIGEST = (
    "1a6e9651a7f7e94ef3815b7d51d75083e03ba0d4cf9fe5c1882701f5ed8e348b"
)


class P3AblationBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = build_p3_ablation_benchmark()
        cls.dataset = build_p3_ablation_dataset(cls.benchmark)

    def test_objective_ledger_has_the_eight_frozen_conditions(self) -> None:
        self.assertEqual(tuple(OBJECTIVE_MASKS), tuple(ObjectiveCondition))
        self.assertEqual(
            OBJECTIVE_MASKS[ObjectiveCondition.JEPA_ONLY].jepa_latent,
            1.0,
        )
        self.assertEqual(
            sum(OBJECTIVE_MASKS[ObjectiveCondition.JEPA_ONLY].as_dict().values()),
            1.0,
        )
        for condition, omitted in (
            (ObjectiveCondition.NO_TOPOLOGY, "simplicial_surrogate"),
            (ObjectiveCondition.NO_METRIC, "metric_surrogate"),
            (ObjectiveCondition.NO_RELATION, "relation_surrogate"),
            (ObjectiveCondition.NO_ORDER, "order_surrogate"),
            (ObjectiveCondition.NO_BRIDGE, "bridge_surrogate"),
            (ObjectiveCondition.NO_TRACKING, "tracking_surrogate"),
        ):
            mask = OBJECTIVE_MASKS[condition].as_dict()
            self.assertEqual(mask[omitted], 0.0)
            self.assertEqual(sum(value == 0.0 for value in mask.values()), 1)

    def test_interaction_split_is_balanced_and_source_disjoint(self) -> None:
        self.assertEqual(self.benchmark.state_count, 81)
        self.assertEqual(self.benchmark.transition_count, 324)
        source_sets = {
            split: self.benchmark.source_codes(split) for split in SPLIT_NAMES
        }
        self.assertTrue(source_sets["train"].isdisjoint(source_sets["validation"]))
        self.assertTrue(source_sets["train"].isdisjoint(source_sets["test"]))
        self.assertTrue(source_sets["validation"].isdisjoint(source_sets["test"]))
        for expected_residue, split in enumerate(SPLIT_NAMES):
            self.assertEqual(len(source_sets[split]), 27)
            self.assertEqual(
                {interaction_residue(code) for code in source_sets[split]},
                {expected_residue},
            )
            self.assertEqual(len(self.benchmark.splits[split]), 108)

    def test_every_source_keeps_all_four_actions_in_one_split(self) -> None:
        for split in SPLIT_NAMES:
            actions_by_source = {}
            for case in self.benchmark.splits[split]:
                actions_by_source.setdefault(case.source_code, set()).add(case.action)
            self.assertTrue(
                all(
                    actions == set(SyntheticAction)
                    for actions in actions_by_source.values()
                )
            )

    def test_benchmark_digest_is_semantic_and_deterministic(self) -> None:
        repeated = build_p3_ablation_benchmark()
        self.assertEqual(P3_ABLATION_BENCHMARK_ID, "P3-2-ABLATION-v1")
        self.assertEqual(self.benchmark.digest, repeated.digest)
        self.assertEqual(self.benchmark.digest, EXPECTED_BENCHMARK_DIGEST)

    def test_numeric_layout_is_injective_and_train_fitted(self) -> None:
        self.assertEqual(self.dataset.input_dimension, 24)
        self.assertEqual(self.dataset.candidate_features.shape, (81, 31))
        self.assertEqual(
            np.unique(self.dataset.candidate_features, axis=0).shape[0],
            81,
        )
        self.assertEqual(len(self.dataset.bridge_links), 6)
        for split in SPLIT_NAMES:
            numeric = self.dataset.splits[split]
            self.assertEqual(numeric.source_inputs.shape, (108, 24))
            self.assertEqual(numeric.target_features.shape, (108, 31))
            self.assertEqual(numeric.tracking_targets.shape, (108, 9))

    def test_invalid_optimization_specs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            P3AblationSpec(seeds=())
        with self.assertRaises(ValueError):
            P3AblationSpec(seeds=(1, 1))
        with self.assertRaises(ValueError):
            P3AblationSpec(training_steps=0)
        with self.assertRaises(ValueError):
            P3AblationSpec(latent_dimension=1)
        with self.assertRaises(ValueError):
            P3AblationSpec(ema_momentum=1.0)


class P3AblationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_p3_ablation_dataset()

    def test_manual_gradients_match_finite_differences(self) -> None:
        audit = audit_manual_gradients(self.dataset)
        self.assertTrue(audit.passed)
        self.assertEqual(audit.checked_coordinates, 32)
        self.assertLess(audit.maximum_absolute_error, 2.0e-5)

    def test_all_conditions_have_identical_parameter_counts(self) -> None:
        spec = P3AblationSpec(seeds=(1,), training_steps=1)
        counts = {
            TrainableStructuralJEPA(
                self.dataset,
                condition,
                1,
                spec,
            ).parameter_count
            for condition in ObjectiveCondition
        }
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts.pop(), 2168)

    def test_short_training_is_deterministic_and_reduces_full_loss(self) -> None:
        spec = P3AblationSpec(seeds=(17,), training_steps=80)
        first = TrainableStructuralJEPA(
            self.dataset,
            ObjectiveCondition.FULL,
            17,
            spec,
        ).fit()
        second = TrainableStructuralJEPA(
            self.dataset,
            ObjectiveCondition.FULL,
            17,
            spec,
        ).fit()
        self.assertLess(first.final_snapshot.total, first.initial_snapshot.total)
        self.assertEqual(first.update_count, 80)
        for name in first.parameters:
            np.testing.assert_array_equal(
                first.parameters[name],
                second.parameters[name],
            )
        np.testing.assert_array_equal(first.target_weight, second.target_weight)

    def test_jepa_only_leaves_every_soft_head_unchanged(self) -> None:
        spec = P3AblationSpec(seeds=(3,), training_steps=25)
        model = TrainableStructuralJEPA(
            self.dataset,
            ObjectiveCondition.JEPA_ONLY,
            3,
            spec,
        )
        before = {
            name: value.copy()
            for name, value in model.parameters.items()
            if any(
                name.startswith(prefix)
                for prefix in (
                    "label_",
                    "simplicial_",
                    "metric_",
                    "relation_",
                    "order_",
                    "tracking_",
                )
            )
        }
        model.fit()
        for name, value in before.items():
            np.testing.assert_array_equal(model.parameters[name], value)

    def test_no_tracking_leaves_tracking_head_unchanged(self) -> None:
        spec = P3AblationSpec(seeds=(5,), training_steps=25)
        model = TrainableStructuralJEPA(
            self.dataset,
            ObjectiveCondition.NO_TRACKING,
            5,
            spec,
        )
        weight = model.parameters["tracking_weight"].copy()
        bias = model.parameters["tracking_bias"].copy()
        model.fit()
        np.testing.assert_array_equal(model.parameters["tracking_weight"], weight)
        np.testing.assert_array_equal(model.parameters["tracking_bias"], bias)

    def test_ema_target_is_not_an_independently_optimized_parameter(self) -> None:
        spec = P3AblationSpec(seeds=(7,), training_steps=20)
        model = TrainableStructuralJEPA(
            self.dataset,
            ObjectiveCondition.FULL,
            7,
            spec,
        )
        initial = model.target_weight.copy()
        model.fit()
        self.assertNotIn("target_weight", model.parameters)
        self.assertFalse(np.array_equal(model.target_weight, initial))

    def test_hard_decode_is_coherent_and_tracking_endpoints_match(self) -> None:
        spec = P3AblationSpec(seeds=(11,), training_steps=30)
        model = TrainableStructuralJEPA(
            self.dataset,
            ObjectiveCondition.FULL,
            11,
            spec,
        ).fit()
        predictions = decode_ablation_predictions(model, "test")
        self.assertEqual(len(predictions), 108)
        for case, prediction in zip(
            self.dataset.splits["test"].cases,
            predictions,
            strict=True,
        ):
            self.assertFalse(
                any(
                    bridge_defects(
                        prediction.target.core,
                        prediction.target.order,
                        prediction.target.signature,
                    ).values()
                )
            )
            self.assertEqual(
                prediction.tracking.source,
                case.example.source.core,
            )
            self.assertEqual(
                prediction.tracking.target,
                prediction.target.core,
            )
            self.assertTrue(prediction.tracking.is_full)

    def test_exact_evaluation_returns_finite_primary_metrics(self) -> None:
        spec = P3AblationSpec(seeds=(13,), training_steps=40)
        model = TrainableStructuralJEPA(
            self.dataset,
            ObjectiveCondition.FULL,
            13,
            spec,
        ).fit()
        result = evaluate_ablation_model(
            model,
            ExactStatePairCache(self.dataset.benchmark),
        )
        self.assertEqual(result.example_count, 108)
        self.assertEqual(result.post_projection_bridge_violation_rate, 0.0)
        self.assertTrue(
            all(np.isfinite(value) for value in result.scalar_metrics().values())
        )


class P3AblationStatisticsTests(unittest.TestCase):
    def test_five_seed_summary_uses_the_registered_t_interval(self) -> None:
        statistic = SummaryStatistic.from_values((1, 2, 3, 4, 5))
        self.assertEqual(statistic.mean, 3.0)
        self.assertAlmostEqual(
            statistic.sample_standard_deviation,
            np.sqrt(2.5),
        )
        self.assertLess(statistic.confidence_95_low, 3.0)
        self.assertGreater(statistic.confidence_95_high, 3.0)
        self.assertTrue(statistic.excludes_zero)

    def test_default_seed_ledger_has_five_unique_paired_seeds(self) -> None:
        self.assertEqual(len(DEFAULT_ABLATION_SEEDS), 5)
        self.assertEqual(len(set(DEFAULT_ABLATION_SEEDS)), 5)
        self.assertEqual(P3AblationSpec().seeds, DEFAULT_ABLATION_SEEDS)


if __name__ == "__main__":
    unittest.main()
