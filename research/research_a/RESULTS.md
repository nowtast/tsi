# Research A1 Confirmatory Results

## Decision

The prospective one-shot A1 cohort supports a finite-sample efficiency
advantage for typed support restriction in the declared matched-dictionary
population. It does not support universal representational superiority or a
point-valued crossing.

Joint advantage was detected at training sizes 10, 15, 20, and 25. Joint
equivalence was first detected at size 50. Sizes 30 and 40 were indeterminate
under the frozen simultaneous rules, so the reported transition band is
\([25,50]\), not a point estimate of \(n^*\).

## Primary results

All intervals are two-sided Bonferroni-simultaneous intervals over the 16 named
world-level endpoints. Positive NLL means worse composition prediction by the
unstructured generic selector. Positive recovery difference means more exact
recoveries by the typed selector.

| Train size | Generic minus typed NLL | Typed minus generic exact recovery | Typed rate | Generic rate | Decision |
|---:|---:|---:|---:|---:|---|
| 5 | 0.53189 [0.44609, 0.61769] | 0.01587 [-0.01716, 0.04891] | 0.016 | 0.000 | Neither |
| 10 | 0.32427 [0.25547, 0.39308] | 0.16667 [0.06816, 0.26517] | 0.222 | 0.056 | Joint advantage |
| 15 | 0.17773 [0.10812, 0.24733] | 0.16667 [0.06816, 0.26517] | 0.492 | 0.325 | Joint advantage |
| 20 | 0.09687 [0.04406, 0.14968] | 0.15079 [0.05621, 0.24538] | 0.714 | 0.563 | Joint advantage |
| 25 | 0.05263 [0.01584, 0.08943] | 0.11905 [0.03345, 0.20465] | 0.849 | 0.730 | Joint advantage |
| 30 | 0.02047 [-0.00397, 0.04491] | 0.04762 [-0.00867, 0.10391] | 0.897 | 0.849 | Indeterminate |
| 40 | 0.00660 [-0.00469, 0.01789] | 0.02381 [-0.01649, 0.06411] | 0.984 | 0.960 | Indeterminate |
| 50 | -0.00002 [-0.00008, 0.00004] | 0.00000 [0.00000, 0.00000] | 0.992 | 0.992 | Joint equivalence |

The typed and isomorphic generic controls had exactly equal predictions and
composition NLL in every world at every size. The observed advantage is
therefore attributed to support-search restriction rather than notation.

## Provenance

- Independent unit: 126 worlds.
- Family balance: 14 worlds in each of nine ordered family-pair strata.
- Graph coverage: all 30 graph motifs, with four or five worlds per graph.
- Contract digest: `9a37d3f2d9e424dd9e6a00f1235b4f3fa07b2dfc8d26633f06dd632cba04cdee`.
- Freeze digest: `9ea1b9b75cea3d51dcf401f62fee812cea4c9e374569df5161433c389b234339`.
- Seed commitment: `49956b58a7cb4e129304cbb621596946c0175c165607758cbe30a33c9053da0e`.
- Public commitment commit: `cc99dfc05614084ecd89460d24d5f5a958f3e8c0`.
- Confirmatory analysis SHA-256: `0cca53e69c63ffacaccbf7a66064eedc84bb6bf6479c0a9eada4e4cf2b4c1e5d`.

The source freeze and seed commitment were pushed before execution. The root
seed was revealed after the one-shot run. A zero-project-import Node.js replay
recomputed all 16 endpoints with zero failures. Both implementations were
created in the same author workflow, and no external person has completed an
independent replay; this result must not be described as independently
replicated.

## Boundary and next gate

The result is limited to a supplied correct graph, the declared modulo-7
primitive-intervention population, a matched dictionary, and the specific
factorized and seven-step greedy selectors. A2 will test candidate width,
training-noise degradation, and misspecification in both directions under a new
contract and seed. A1 cannot be altered or repaired by A2.
