# Research A: Structured Support-Recovery Efficiency

Research A tests whether a typed support restriction improves finite-sample
support recovery and held-out composition prediction after both arms receive the
same observable transitions and condition on the same graph.

This directory is a design workspace, not a frozen preregistration. No
confirmatory seed has been generated and no sealed experiment may run until the
theory, estimands, SESOI, multiplicity family, power calculation, and stopping
rule pass review.

- `theory.md`: finite-sample envelope and its proofs.
- `preregistration_draft.json`: machine-readable design draft.
- `DESIGN_HISTORY.md`: transparent development-grid correction and power basis.
- `ROBUSTNESS_PLAN.md`: A2 directions recorded before the A1 confirmatory seed.
- `RESULTS.md`: sealed A1 decision, simultaneous intervals, and scope boundary.
- `EXTERNAL_REPLAY.md`: package integrity, execution, and independence rules.
- `README_KO.md` and `theory_KO.md`: structurally matched Korean documents.

The development pilot is written to
`experiments/research_a_v1/development_report.json`. It may calibrate power and
decision thresholds but cannot enter the confirmatory cohort.

The main comparison is not notation versus notation. It is a structured search
over the nine generator-compatible edge-family supports versus an unstructured
seven-step search over 55 output-feature positions. An isomorphic generic
nine-support control must tie the typed arm and separates notation from search
restriction.
