# Stage 1 Proof Audit

## Current scope

The integrated draft contains structural equivalence, survivor-level composition closure, six defect criteria, extension results, and the quotient-level rollout bound.

## Checks completed

- All symbols used in the definitions are declared in the preamble or the local section.
- The partial transition has an explicit tagged extension before it is used.
- The target simplicial complex is consistently named `L` for `\Sigma'`.
- The structural discrepancy is defined as an extended pseudometric on equivalence classes, with a concrete discrete metric instance.
- The composition proof names domain, intermediate survivors, and image before
  restricting each layer equality.
- The rollout proof uses only the triangle inequality, the one-step error bound,
  and the true-update Lipschitz assumption.
- The one-step counterexample is explicitly separated from the rollout theorem
  assumptions.

## Stage 2 audit rule

Each imported Stage 2 theorem was verified before admission to the consolidated
paper. Persistent topology, metric-measure discrepancy, categorical descent,
action-conditioned dynamics, and turnover were admitted only with their exact
assumptions, proof obligations, boundaries, and counterexamples recorded below.

## Stage 2A persistent topology audit

- The finite carrier, field, monotone filtration, sublevel complex, persistence diagram, and truncated signature are defined before use.
- The filtered stability result is explicitly marked as an imported theorem. Its hypotheses are fixed carrier, common field, monotone real-valued filtrations, and common units.
- The proof checks both sublevel inclusions and constructs the interleaving; only algebraic stability is externally imported.
- The filtration-preserving simplicial-isomorphism proposition is proved by levelwise homology isomorphisms and a commuting persistence-module diagram.
- The two-vertex counterexample gives identical persistence diagrams after
  swapping filtration values on distinctly labeled isolated vertices, while no
  label-preserving bijection preserves those filtration values.
- The extension states its boundary: it does not prove completeness, cross-carrier comparison without correspondence, or neural carrier discovery.

Gate result: Stage 2A topology is admitted into the bilingual core with the
imported-theorem dependency and the labeled-carrier incompleteness boundary
recorded.

## Stage 2B metric-measure audit

- The finite metric-measure state, probability normalization, full support, and label map are defined before the coupling objective.
- The coupling marginal constraints and hard label compatibility are explicit; infeasible label-mass profiles have discrepancy infinity.
- The zero-discrepancy proof uses compactness of the finite coupling polytope, continuity, full-support coverage, pairwise nonnegative summands, and the diagonal arguments that force the support to be a bijection graph.
- The resulting bijection is separately shown to preserve distances, labels, and masses.
- The label-mass equality proposition is proved by summing the coupling marginals.
- The boundary is explicit: no sampling robustness, empirical convergence rate, arbitrary-carrier existence, or coordinate-loss superiority is claimed.

Gate result: Stage 2B metric-measure extension is admitted into the bilingual
core with its full-support and hard-label assumptions recorded.

## Stage 2C categorical descent audit

- The finite quiver, free path category, path congruence, presented schema, and generator realization are defined before descent.
- The finite-relation category proposition checks associativity and both identity laws directly from existential relational composition.
- The descent theorem proves both directions: factorization implies equation preservation, and equation preservation induces a well-defined quotient functor through the smallest path congruence.
- Uniqueness follows because every quotient morphism has a path representative.
- The natural-equivalence generator criterion is proved by transporting relation witnesses and extending the result by path-length induction.
- The boundary is explicit: equation-free generator agreement is not categorical validity, and schema discovery is not claimed.

Gate result: Stage 2C categorical descent is admitted into the bilingual core
with its equation-preservation boundary recorded.

## Stage 2D dynamics audit

- The finite action-prefix category and tracked action-history functor are defined before trajectory claims.
- The unique-extension theorem proves existence, functoriality, and uniqueness from one-step action edges.
- Trajectory preservation is derived by repeated application of the previously proved composition-closure theorem.
- Topology, geometry, relation, turnover, filtration, and mass defects are
  defined on finite survivor structures.
- The zero criterion separately proves six zero conditions and their conjunction as full integrated structural isomorphism.
- The empty-tracking proposition records the vacuity of survivor-only preservation and motivates turnover as a separate quantity.
- The boundary excludes endpoint-only identity inference, causal-counterfactual claims, and unconstrained mechanism discovery.

Gate result: Stage 2D dynamics is admitted and the Stage 2A--2D mathematical
extensions are complete. Papers 03 and 04 document the bounded empirical
operationalizations and the portions of the formal state that they do not test.

## Reviewer-driven revision audit (2026-08-11)

- Schema: finite quiver and path equations precede generator use.
- State: filtration f, full-support mass mu, and relation-to-simplex compatibility are explicit.
- Isomorphism: labels, simplices, distances, filtration, mass, and generator relations are preserved.
- Tracked transitions are label-preserving partial injections with explicit category composition.
- Zero criterion: topology, geometry, relations, turnover, filtration, and mass.
- Rollout discrepancy: extended pseudometric; discrete metric instance; separate Lipschitz and quotient-equivariance propositions.
- Running example: nonempty compatible state, invariance, and survivor-defect/turnover distinction.
- Standard imported mathematics is separated from framework-specific construction.

Pre-attribution build status (2026-08-11): English PDF 15 pages; Korean PDF 14
pages with XeLaTeX. This historical count is superseded by the audit below.

## Bibliographic attribution audit (2026-08-20)

- The manuscript now contains a dedicated related-work and positioning section
  and 20 cited references; there are no uncited bibliography entries or missing
  citation keys.
- The finite-relation and presented-schema ingredients are attributed to the
  calculus of relations and functorial data literature. The text distinguishes
  TSI's `FinRel`-valued carrier and added structural layers from set-valued
  categorical database instances.
- The filtered-stability proof identifies its external dependency at the exact
  proof step. Finiteness of the carrier is used to establish pointwise finite
  dimensionality and q-tameness before algebraic stability is invoked.
- The hard-label discrepancy is positioned as a constrained
  Gromov--Wasserstein-type objective. Its exactness result remains proved
  internally, while the coupling and gluing background is attributed.
- The quotient rollout result is explicitly separated from reward- and
  transition-defined bisimulation metrics and from policy-performance claims.
- English and Korean citation keys are structurally checked, and the two
  `references.bib` files must be byte-identical. The checker now fails on
  missing, unmatched, or uncited bibliography keys.

Gate result: the missing-attribution blocker is closed. The clean English PDF
builds to 18 pages and the matched Korean PDF to 17 pages.
