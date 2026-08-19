from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from tsi.paper3_ablation_experiment import (
    ExactStatePairCache,
    SummaryStatistic,
    embedding_diagnostics,
)
from tsi.paper3_objective_ablation import (
    P3AblationSpec,
    SyntheticAction,
    TrainableStructuralJEPA,
    ObjectiveCondition,
    build_p3_ablation_dataset,
)
from tsi.paper3_repair_experiment import (
    RepairReadinessThresholds,
    RepairVariantSummary,
    _meets_numeric_thresholds,
    audit_repair_gradients,
    evaluate_repair_model,
)
from tsi.paper3_representation_repair import (
    DEFAULT_REPAIR_SEEDS,
    P3_REPAIR_BENCHMARK_ID,
    REPAIR_ALLOWED_EVALUATION_SPLITS,
    LatentFactorLayout,
    RepairStructuralJEPA,
    RepairVariant,
)


class Paper3RepresentationRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_p3_ablation_dataset()

    def test_stable_gate_and_variant_ledger(self) -> None:
        self.assertEqual(P3_REPAIR_BENCHMARK_ID, "P3-2R-REPRESENTATION-v1")
        self.assertEqual(
            tuple(variant.value for variant in RepairVariant),
            ("reference", "layer_routed", "factorized_action"),
        )
        self.assertEqual(
            REPAIR_ALLOWED_EVALUATION_SPLITS,
            ("train", "validation"),
        )
        self.assertEqual(
            DEFAULT_REPAIR_SEEDS,
            (20_260_733, 20_260_734, 20_260_735, 20_260_736, 20_260_737),
        )

    def test_latent_factor_layout_is_a_partition(self) -> None:
        layout = LatentFactorLayout.default(16)
        covered: list[int] = []
        for _, block in layout.items():
            covered.extend(range(block.start, block.stop))
        self.assertEqual(covered, list(range(16)))
        with self.assertRaises(ValueError):
            LatentFactorLayout.default(15)

    def test_train_has_complete_layer_action_marginal_support(self) -> None:
        cases = self.dataset.benchmark.splits["train"]
        attributes = (
            "label_phase",
            "topology_mode",
            "metric_mode",
            "order_mode",
        )
        expected = set(product(range(3), SyntheticAction))
        for attribute in attributes:
            observed = {
                (getattr(case.source_code, attribute), case.action) for case in cases
            }
            self.assertEqual(observed, expected)

    def test_train_validation_interactions_are_disjoint(self) -> None:
        train_pairs = {
            (code.label_phase, code.topology_mode)
            for code in self.dataset.benchmark.source_codes("train")
        }
        validation_pairs = {
            (code.label_phase, code.topology_mode)
            for code in self.dataset.benchmark.source_codes("validation")
        }
        self.assertFalse(train_pairs & validation_pairs)
        self.assertEqual(len(train_pairs), 3)
        self.assertEqual(len(validation_pairs), 3)

    def test_active_parameter_counts_are_fixed(self) -> None:
        spec = P3AblationSpec(seeds=(7,), training_steps=1)
        counts = {}
        for variant in RepairVariant:
            model = RepairStructuralJEPA(self.dataset, variant, 7, spec)
            counts[variant] = model.active_parameter_count
            self.assertEqual(model.parameter_count, 2168)
            self.assertTrue(model.mask_invariant_holds())
        self.assertEqual(
            counts,
            {
                RepairVariant.REFERENCE: 2168,
                RepairVariant.LAYER_ROUTED: 1465,
                RepairVariant.FACTORIZED_ACTION: 649,
            },
        )

    def test_layer_routing_masks_encoder_and_structural_heads(self) -> None:
        spec = P3AblationSpec(seeds=(8,), training_steps=1)
        model = RepairStructuralJEPA(
            self.dataset,
            RepairVariant.LAYER_ROUTED,
            8,
            spec,
        )
        self.assertGreater(
            np.count_nonzero(model.parameter_masks["encoder_weight"] == 0.0),
            0,
        )
        self.assertTrue(np.all(model.parameter_masks["predictor_weight"] == 1.0))
        for name, block in model.factors.items():
            mask = model.parameter_masks[f"{name}_weight"]
            self.assertTrue(np.all(mask[block, :] == 1.0))
            outside = np.ones(mask.shape[0], dtype=bool)
            outside[block] = False
            self.assertTrue(np.all(mask[outside, :] == 0.0))
        self.assertTrue(np.all(model.parameter_masks["tracking_weight"] == 1.0))

    def test_factorized_action_mask_is_block_diagonal(self) -> None:
        spec = P3AblationSpec(seeds=(9,), training_steps=1)
        model = RepairStructuralJEPA(
            self.dataset,
            RepairVariant.FACTORIZED_ACTION,
            9,
            spec,
        )
        mask = model.parameter_masks["predictor_weight"]
        for action_index in range(mask.shape[0]):
            for row in range(mask.shape[1]):
                for column in range(mask.shape[2]):
                    same_factor = any(
                        row in range(block.start, block.stop)
                        and column in range(block.start, block.stop)
                        for _, block in model.factors.items()
                    )
                    self.assertEqual(mask[action_index, row, column], same_factor)

    def test_reference_is_exactly_the_original_full_model(self) -> None:
        spec = P3AblationSpec(seeds=(10,), training_steps=5)
        reference = RepairStructuralJEPA(
            self.dataset,
            RepairVariant.REFERENCE,
            10,
            spec,
        ).fit()
        original = TrainableStructuralJEPA(
            self.dataset,
            ObjectiveCondition.FULL,
            10,
            spec,
        ).fit()
        for name in original.parameters:
            np.testing.assert_array_equal(
                reference.parameters[name],
                original.parameters[name],
            )
        np.testing.assert_array_equal(reference.target_weight, original.target_weight)
        np.testing.assert_array_equal(reference.target_bias, original.target_bias)

    def test_mask_invariant_survives_optimization(self) -> None:
        spec = P3AblationSpec(seeds=(11,), training_steps=20)
        for variant in (
            RepairVariant.LAYER_ROUTED,
            RepairVariant.FACTORIZED_ACTION,
        ):
            model = RepairStructuralJEPA(
                self.dataset,
                variant,
                11,
                spec,
            ).fit()
            self.assertTrue(model.mask_invariant_holds())

    def test_repair_evaluator_rejects_test_split(self) -> None:
        spec = P3AblationSpec(seeds=(12,), training_steps=1)
        model = RepairStructuralJEPA(
            self.dataset,
            RepairVariant.REFERENCE,
            12,
            spec,
        )
        cache = ExactStatePairCache(self.dataset.benchmark)
        with self.assertRaisesRegex(ValueError, "forbids test"):
            evaluate_repair_model(model, cache, "test")

    def test_embedding_diagnostic_scope_is_validated(self) -> None:
        spec = P3AblationSpec(seeds=(13,), training_steps=1)
        model = RepairStructuralJEPA(
            self.dataset,
            RepairVariant.REFERENCE,
            13,
            spec,
        )
        with self.assertRaises(ValueError):
            embedding_diagnostics(
                model,
                state_features=self.dataset.candidate_features[:1],
            )
        with self.assertRaises(ValueError):
            embedding_diagnostics(
                model,
                state_features=np.zeros((2, 30)),
            )

    def test_constrained_gradient_audits_pass(self) -> None:
        for variant in (
            RepairVariant.LAYER_ROUTED,
            RepairVariant.FACTORIZED_ACTION,
        ):
            audit = audit_repair_gradients(
                self.dataset,
                variant,
                coordinates_per_parameter=1,
            )
            self.assertTrue(audit.passed)
            self.assertEqual(audit.checked_coordinates, 16)
            self.assertLess(audit.maximum_absolute_error, 2.0e-5)
            self.assertEqual(audit.inactive_gradient_maximum, 0.0)

    def test_gradient_audit_rejects_reference_and_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            audit_repair_gradients(self.dataset, RepairVariant.REFERENCE)
        with self.assertRaises(ValueError):
            audit_repair_gradients(
                self.dataset,
                RepairVariant.LAYER_ROUTED,
                epsilon=0.0,
            )
        with self.assertRaises(ValueError):
            audit_repair_gradients(
                self.dataset,
                RepairVariant.LAYER_ROUTED,
                coordinates_per_parameter=0,
            )

    def test_readiness_thresholds_are_validated(self) -> None:
        thresholds = RepairReadinessThresholds()
        self.assertEqual(thresholds.validation_quotient_maximum, 0.20)
        with self.assertRaises(ValueError):
            RepairReadinessThresholds(validation_fixed_joint_minimum=1.1)

    def test_numeric_readiness_requires_every_endpoint(self) -> None:
        passing = {
            "train_fixed_joint_exact_rate": SummaryStatistic.from_values([0.96]),
            "validation_fixed_joint_exact_rate": SummaryStatistic.from_values([0.30]),
            "validation_quotient_distance": SummaryStatistic.from_values([0.10]),
            "validation_tracking_exact_rate": SummaryStatistic.from_values([0.60]),
        }
        summary = RepairVariantSummary("probe", passing)
        self.assertTrue(_meets_numeric_thresholds(summary, RepairReadinessThresholds()))
        failing = dict(passing)
        failing["validation_quotient_distance"] = SummaryStatistic.from_values([0.21])
        self.assertFalse(
            _meets_numeric_thresholds(
                RepairVariantSummary("probe", failing),
                RepairReadinessThresholds(),
            )
        )

    def test_factorized_action_meets_readiness_on_first_seed(self) -> None:
        spec = P3AblationSpec(seeds=(20_260_728,))
        model = RepairStructuralJEPA(
            self.dataset,
            RepairVariant.FACTORIZED_ACTION,
            spec.seeds[0],
            spec,
        ).fit()
        cache = ExactStatePairCache(self.dataset.benchmark)
        train = evaluate_repair_model(model, cache, "train")
        validation = evaluate_repair_model(model, cache, "validation")
        self.assertEqual(train.fixed_joint_exact_rate, 1.0)
        self.assertEqual(validation.fixed_joint_exact_rate, 1.0)
        self.assertEqual(validation.mean_quotient_distance, 0.0)
        self.assertEqual(validation.tracking_exact_rate, 1.0)
        self.assertEqual(validation.embedding.collision_count, 0)
        self.assertEqual(
            validation.post_projection_bridge_violation_rate,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
