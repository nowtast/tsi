# Stage 1: Theory and Formalization Integration

Objective: produce one rigorous theory paper connecting Paper 1's Structural Imagination Space with topology, geometry, category, and dynamics.

Sources: `papers/paper1`, `papers/paper2a`, `paper2b`, `paper2c`, `paper2d`, and the Stage 2 theorem-audit registers.

Required work: define the carrier, structural maps, transition semantics, and admissible invariants; separate definitions, propositions, theorems, proofs, examples, and conjectures; audit hidden assumptions and counterexamples; distinguish mathematical claims from cognitive and AI interpretation.

Completion gate: matching theorem structure and notation in English/Korean, complete proof audit, and an explicit formal-versus-interpretive boundary.

## Current status

The bilingual integrated core draft is present in `main.tex` and `sections/`, builds to 6-page PDFs, and has a paired proof-audit record. Stage 2A persistent topology has now been integrated as sections/topological_extension.tex; both language versions build to 6-page PDFs. The filtered stability theorem is explicitly imported, while the levelwise inclusion, isomorphism-invariance proof, tolerance corollary, and Betti-number counterexample are checked locally. Stage 2B metric-measure is now integrated as sections/metric_measure_extension.tex; both language versions build to 8-page PDFs. The zero-discrepancy exactness theorem and label-mass necessary condition are audited with full-support and coupling assumptions. Stage 2C categorical descent is now integrated as sections/category_extension.tex; both language versions build to 9-page PDFs. Path-equation descent and the generator natural-equivalence criterion are audited. Stage 2D dynamics is now integrated as sections/dynamics_extension.tex; both language versions build to 11-page PDFs. Action-history extension, trajectory preservation, turnover defects, and the empty-tracking counterexample are audited. Stage 2A--2D are complete; the next workstream is empirical-validation integration.


## Reviewer-driven revision status (2026-08-11)

M1/M2/M5 are addressed by the presented schema, the explicit category of
label-preserving tracked transitions, and the corrected six-component zero
criterion. M3/M4/M6 are addressed by attaching filtration and mass to the state,
adding state-level invariance results, using an extended pseudometric for rollout,
and separating Lipschitz and quotient-equivariance failures. M7/M8 are addressed
by separating standard imported mathematics from framework-specific construction
and adding a finite running example.

The English build passes at 18 pages. A 2026-08-20 attribution pass added a
dedicated related-work section and 20 verified, fully cited references. The
English/Korean parity gate now also requires byte-identical bibliographies and
rejects missing or uncited citation keys.
