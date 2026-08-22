# Benchmark Card

## Population

The attribution cohort contains 120 independent finite worlds. Each world has
five coordinates modulo 7, one of 30 two-edge motifs, and two edge heads drawn
from three declared families. Training uses 800 primitive-action transitions,
selection uses 400 disjoint transitions, and evaluation uses 1,200 held-out
two-coordinate-action transitions.

## Tasks and Estimands

1. Recover the common-target graph and its two head families from train and
   selection data only.
2. Predict held-out transition targets and report world-level composition NLL.

Reference graph effects are wrong-routing minus correct-routing NLL. They are
not universal model rankings. The stress cohort is a separate 120-world family
with a nonadditive term dormant during training and active in designated test
compositions.

## Information Symmetry

Methods compared for attribution must receive the same observable states,
actions, partitions, and corruption process. A method may encode a different
inductive bias, but it may not receive identities, graph answers, typed
relations, test targets, or reference outputs unavailable to another arm.

## Evaluation Unit

Worlds are independent. Transitions are nested observations and must not be
treated as independent samples for inferential intervals. Hyperparameter and
model selection must finish before test-target access.

## Reference Scope

The archived reference arms are factorized typed, matched seven-parameter
generic sparse, and 55-parameter generic dense heads under correct and wrong
graphs. Their values are integrity anchors, not a requirement that future
methods reproduce a preferred ranking.

## Limitations

- The answers are public, so v0.1 cannot support a blind leaderboard claim.
- Entity discovery, cross-time identity, visual input, continuous state, and
  real-world validity are absent.
- The reference cohorts share source families and do not constitute external
  or between-laboratory replication.
- Submission-policy declarations are machine-checked for shape, not audited for
  truth. Independent execution remains necessary.
