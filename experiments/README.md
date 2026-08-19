# Experiments

Experiments test whether learned representations instantiate the mathematical
objects defined by TSI. They do not prove the definitions or theorems.

## Stage 2A Topology Protocol

1. Generate finite labeled simplicial carriers and transitions with ground-truth
   adjacency, stars, links, Betti sequences, filtrations, and induced maps.
2. Separate state-level errors from transition-map errors.
3. Report adjacency and star/link agreement, Betti error, persistence-diagram
   distance on an aligned parameter scale, and induced-homology-map error.
4. Include terminal states that agree while the true and predicted transition
   maps differ.

## Stage 2B Geometry Protocol

1. Generate finite metric and embedded states with known labels, masses,
   correspondences, and task symmetry group.
2. Evaluate absolute-pose, `SE(m)`-invariant, and `E(m)`-invariant regimes
   separately.
3. Compare coordinate, pairwise-distance, and structured metric predictors.
4. Report coordinate error, aligned ambient discrepancy, exact or approximated
   label-compatible `Delta_g`, diameter error, Rips persistence discrepancy, and
   metric-measure discrepancy.
5. Include translations, rotations, reflected chiral states, intrinsic
   distortions, missing entities, incompatible label mass, and support noise.

## Stage 2C Category/Relation Protocol

1. Generate functorial finite relation states from quivers with declared path
   equations, then create controlled insertion, deletion, endpoint-swap, label,
   and equation violations.
2. Separate carrier accuracy, generator relation accuracy, and path-equation
   consistency.
3. Report exact or approximated `Delta_rel`, composition defect, generator
   precision/recall, and held-out path precision/recall by carrier size and path
   length.
4. Test arbitrary entity relabelings and include path-versus-fork and
   swapped-composite counterexamples.
5. Ablate typed alignment, path-equation loss, labels, and explicit relation
   prediction one at a time.

## Stage 2D Dynamics Protocol

1. Generate action-conditioned structural trajectories with ground-truth typed
   identity, births, deaths, topology, metric, generator relations, and
   intervention context.
2. Report endpoint integrated discrepancy, tracking distance, turnover, and
   survivor-level topology, geometry, and relation defects separately.
3. Compare direct and segmented transition composition, and report rollout
   error by horizon together with actionwise amplification estimates.
4. Validate causal claims only against a known simulator, randomized
   interventions, or an audited identification theorem.
5. Evaluate collision claims from complete motion traces or certified
   reachability evidence, never from endpoint states alone.
6. Include empty-tracking, endpoint-correct-but-identity-wrong,
   observationally equivalent causal, and equal-endpoint/different-path
   counterexamples.

## Statistical Requirements

Every learned-model result must state the data-generating process, train/test
split, estimator, approximation method, uncertainty interval, ablation, failure
cases, and out-of-distribution regime. Sampling consistency and learnability are
open until separately proved or empirically supported; finite exact theorems are
not substitutes for those results.

## Paper 3 Evidence-Level Contract

Paper 3 uses the machine-readable `P3-E0-EVIDENCE-v1` contract. Its levels are
internal readiness categories rather than statistical measurements.

1. P3-2R remains a level-2 development diagnostic.
2. Level 3 requires a sealed independent structural-OOD test, codebook-free
   primary decoding, matched baselines, wrong-structure controls, independent
   layer information, world-level replication, and nested uncertainty.
3. Level 4 additionally requires open-loop rollout, downstream predictive
   validity, learned routing or structure, noisy perception, variable
   cardinality, a public benchmark, cross-family replication, and complete
   artifact replay.
4. Paper 4 and a strong empirical claim remain blocked below level 4.

The next phase is `P3-3A-INDEPENDENCE-v1`, which freezes the generator, decoder,
control, and statistical contracts without inspecting P3-3 test outputs.

## P3-3A Independence Preregistration

The static contract is implemented in
`src/tsi/paper3_independence_contract.py`; its machine report is
`experiments/paper3_independence_contract/results.json`.

- static contract: passed
- evidence level before/after: `2 / 2`
- sealed-test seed reveals: `0`
- sealed-test result evaluations: `0`
- implemented artifacts: `9 / 10`
- P3-3A gate: blocked on `development_variance_and_power_report`
- contract digest:
  `a5c1767ffd2744b15b24e25e4e2cbb1ba6895fff9b05b145af2c1bbee30627e9`
- aggregate artifact digest:
  `69f7ce95270171a989b676fe1bc00e1f10f0e13e0e6e893226d89f50506f9f84`

A passing static preregistration is not an empirical result. P3-3A remains open
until the development-world model pilot fixes a test-world count with at least
0.90 Holm-adjusted simulation power. The current public artifacts include 108
development/validation worlds, a 324-state codebook-free constructive decoder
audit, six 420-active-parameter routing controls, and a hash-chained zero-access
ledger. No sealed-test world or result has been materialized.
