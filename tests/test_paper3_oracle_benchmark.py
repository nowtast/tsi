import unittest

import numpy as np

from tsi.coherent import bridge_defects
from tsi.paper3_interface import FROZEN_PAPER3_INTERFACE
from tsi.paper3_oracle_benchmark import (
    ACTION_AGNOSTIC_FIXED_EXACT_UPPER_BOUND,
    ACTION_RULES,
    ENTITY_IDS,
    SPLIT_NAMES,
    IdentityTransitionBaseline,
    LinearStructuralJEPA,
    P3OracleBenchmarkSpec,
    StructuralFeatureLayout,
    SyntheticAction,
    SyntheticStateCode,
    all_state_codes,
    build_oracle_state,
    build_oracle_transition_benchmark,
    evaluate_transition_predictor,
    run_p3_oracle_benchmark,
    successor_code,
)


class SyntheticStateTests(unittest.TestCase):
    def test_state_code_requires_four_ternary_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticStateCode(3, 0, 0, 0)
        with self.assertRaises(ValueError):
            SyntheticStateCode(True, 0, 0, 0)

    def test_state_family_has_81_distinct_coherent_members(self) -> None:
        codes = all_state_codes()
        self.assertEqual(len(codes), 81)
        states = [build_oracle_state(code) for code in codes]
        self.assertTrue(
            all(
                not any(
                    bridge_defects(
                        state.core,
                        state.order,
                        state.signature,
                    ).values()
                )
                for state in states
            )
        )

        layout = StructuralFeatureLayout.from_states(states)
        encoded = layout.encode_many(states)
        self.assertEqual(encoded.shape, (81, 31))
        self.assertEqual(np.unique(encoded, axis=0).shape[0], 81)

    def test_every_action_has_the_declared_modular_successor(self) -> None:
        source = SyntheticStateCode(2, 2, 2, 2)
        targets = set()
        for action, rule in ACTION_RULES.items():
            target = successor_code(source, action)
            targets.add(target)
            self.assertEqual(
                target.as_tuple(),
                (
                    (2 + rule.label_delta) % 3,
                    (2 + rule.topology_delta) % 3,
                    (2 + rule.metric_delta) % 3,
                    (2 + rule.order_delta) % 3,
                ),
            )
        self.assertEqual(len(targets), len(SyntheticAction))


class OracleBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = build_oracle_transition_benchmark()

    def test_benchmark_is_deterministic(self) -> None:
        repeated = build_oracle_transition_benchmark()
        self.assertEqual(self.benchmark.digest, repeated.digest)
        self.assertEqual(len(self.benchmark.digest), 64)
        self.assertEqual(
            self.benchmark.source_codes("train"),
            repeated.source_codes("train"),
        )

    def test_seed_changes_the_split_and_digest(self) -> None:
        alternative = build_oracle_transition_benchmark(
            P3OracleBenchmarkSpec(seed=20_260_729)
        )
        self.assertNotEqual(self.benchmark.digest, alternative.digest)
        self.assertNotEqual(
            self.benchmark.source_codes("test"),
            alternative.source_codes("test"),
        )

    def test_source_states_do_not_cross_split_boundaries(self) -> None:
        source_sets = {
            split: self.benchmark.source_codes(split) for split in SPLIT_NAMES
        }
        self.assertTrue(source_sets["train"].isdisjoint(source_sets["validation"]))
        self.assertTrue(source_sets["train"].isdisjoint(source_sets["test"]))
        self.assertTrue(source_sets["validation"].isdisjoint(source_sets["test"]))
        self.assertEqual(
            set().union(*source_sets.values()),
            set(all_state_codes()),
        )

    def test_all_actions_are_present_for_every_source(self) -> None:
        self.assertEqual(self.benchmark.state_count, 81)
        self.assertEqual(self.benchmark.transition_count, 324)
        for split in SPLIT_NAMES:
            by_source = {}
            for case in self.benchmark.splits[split]:
                by_source.setdefault(case.source_code, set()).add(case.action)
            self.assertTrue(
                all(actions == set(SyntheticAction) for actions in by_source.values())
            )

    def test_oracle_tracking_is_total_and_label_preserving(self) -> None:
        for split in SPLIT_NAMES:
            for case in self.benchmark.splits[split]:
                component = case.example.tracking.components["entity"]
                self.assertEqual(len(component.pairs), len(ENTITY_IDS))
                for source_id, target_id in component.pairs:
                    self.assertEqual(
                        case.example.source.core.relational.labels["entity"][source_id],
                        case.example.target.core.relational.labels["entity"][target_id],
                    )

    def test_invalid_benchmark_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            P3OracleBenchmarkSpec(train_fraction=0.9, validation_fraction=0.2)
        with self.assertRaises(ValueError):
            P3OracleBenchmarkSpec(ridge=0.0)
        with self.assertRaises(ValueError):
            P3OracleBenchmarkSpec(target_momentum=1.0)


class OraclePredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = build_oracle_transition_benchmark()
        cls.full = LinearStructuralJEPA(condition_on_action=True).fit(cls.benchmark)
        cls.pooled = LinearStructuralJEPA(condition_on_action=False).fit(cls.benchmark)

    def test_encoder_is_noncollapsed_and_separates_codebook(self) -> None:
        self.assertGreater(self.full.latent_dimension, 0)
        embeddings = np.stack(
            [
                self.full.target_embedding(self.benchmark.states[code])
                for code in sorted(self.benchmark.states)
            ]
        )
        self.assertEqual(np.unique(np.round(embeddings, 12), axis=0).shape[0], 81)
        self.assertEqual(self.full.target_ema_updates, 1)

    def test_action_conditioned_predictions_are_valid(self) -> None:
        for case in self.benchmark.splits["test"]:
            prediction = self.full.predict(case.example.source, case.action)
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
            self.assertEqual(prediction.tracking.target, prediction.target.core)

    def test_action_conditioned_model_is_exact_on_held_out_sources(self) -> None:
        result = evaluate_transition_predictor(
            self.full,
            self.benchmark,
            "test",
        )
        self.assertEqual(result.fixed_joint_exact_rate, 1.0)
        self.assertEqual(result.bridge_violation_rate, 0.0)
        self.assertAlmostEqual(result.mean_fixed_total, 0.0)
        self.assertAlmostEqual(result.mean_tracking_error, 0.0)

    def test_action_conditioning_beats_pooled_and_identity_baselines(self) -> None:
        full = evaluate_transition_predictor(self.full, self.benchmark, "test")
        pooled = evaluate_transition_predictor(self.pooled, self.benchmark, "test")
        identity = evaluate_transition_predictor(
            IdentityTransitionBaseline(),
            self.benchmark,
            "test",
        )
        self.assertGreater(
            full.fixed_joint_exact_rate,
            pooled.fixed_joint_exact_rate,
        )
        self.assertGreater(
            full.fixed_joint_exact_rate,
            identity.fixed_joint_exact_rate,
        )

    def test_training_and_prediction_are_deterministic(self) -> None:
        repeated = LinearStructuralJEPA(condition_on_action=True).fit(self.benchmark)
        for case in self.benchmark.splits["test"]:
            first = self.full.predict(case.example.source, case.action)
            second = repeated.predict(case.example.source, case.action)
            self.assertEqual(first.target, second.target)
            self.assertEqual(first.tracking, second.tracking)
            np.testing.assert_allclose(first.latent, second.latent)

    def test_unknown_action_and_unfitted_use_are_rejected(self) -> None:
        source = next(iter(self.benchmark.states.values()))
        with self.assertRaises(ValueError):
            self.full.predict(source, "undeclared")
        with self.assertRaises(RuntimeError):
            LinearStructuralJEPA().predict(source, SyntheticAction.HOLD)

    def test_gate_report_is_empirical_and_machine_audited(self) -> None:
        report = run_p3_oracle_benchmark()
        self.assertTrue(report.passed, report.audit_errors)
        self.assertEqual(
            report.interface_id,
            FROZEN_PAPER3_INTERFACE.identifier,
        )
        self.assertEqual(report.claim_status, "empirical")
        self.assertEqual(report.gate, "P3-1")
        self.assertGreater(report.action_conditioning_gain, 0.0)
        self.assertEqual(
            report.action_agnostic_fixed_exact_upper_bound,
            ACTION_AGNOSTIC_FIXED_EXACT_UPPER_BOUND,
        )
        self.assertEqual(
            report.action_agnostic_fixed_exact_upper_bound,
            0.25,
        )
        self.assertIn(
            "complete declared 81-state",
            report.method_contract["decoder"],
        )
        self.assertIn("not evidence", report.method_contract["scope"])


if __name__ == "__main__":
    unittest.main()
