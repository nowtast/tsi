from __future__ import annotations

import unittest

import numpy as np

from tsi.coherent import bridge_defects
from tsi.paper3_constructive_decoder import (
    ConstructiveStructuralDecoder,
    audit_constructive_decoder,
    build_multiworld_feature_layout,
    constructive_decoder_digest,
)
from tsi.paper3_multiworld import (
    MultiworldStateCode,
    build_multiworld_state,
)


class ConstructiveDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = build_multiworld_feature_layout()
        self.decoder = ConstructiveStructuralDecoder(self.layout)

    def test_layout_is_local_and_has_expected_dimension(self) -> None:
        self.assertEqual(self.layout.dimension, 40)
        self.assertEqual(len(self.layout.label_vocabulary), 2)
        self.assertEqual(len(self.layout.relation_cells), 18)

    def test_exact_features_reconstruct_a_state_without_candidates(self) -> None:
        state = build_multiworld_state(MultiworldStateCode(2, 1, 2, 3, 1))

        decoded = self.decoder.decode_state(self.layout.encode(state))

        self.assertEqual(decoded, state)

    def test_arbitrary_finite_features_decode_to_a_coherent_state(self) -> None:
        raw = np.linspace(-3.0, 4.0, self.layout.dimension)

        decoded = self.decoder.decode_state(raw)

        self.assertTrue(
            all(
                value == 0.0
                for value in bridge_defects(
                    decoded.core,
                    decoded.order,
                    decoded.signature,
                ).values()
            )
        )

    def test_nonfinite_and_wrong_shape_features_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            self.decoder.decode_state(np.zeros(self.layout.dimension - 1))
        raw = np.zeros(self.layout.dimension)
        raw[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.decoder.decode_state(raw)

    def test_digest_is_deterministic(self) -> None:
        self.assertEqual(
            constructive_decoder_digest(),
            constructive_decoder_digest(),
        )

    def test_exhaustive_machine_audit_passes_without_codebook(self) -> None:
        audit = audit_constructive_decoder()

        self.assertTrue(audit.passed)
        self.assertEqual(audit.exact_state_decodes, 324)
        self.assertEqual(audit.exact_tracking_decodes, 1_944)
        self.assertEqual(audit.target_state_collection_parameters, 0)
        self.assertEqual(audit.global_candidate_states, 0)
        self.assertEqual(audit.bridge_violations, 0)


if __name__ == "__main__":
    unittest.main()
