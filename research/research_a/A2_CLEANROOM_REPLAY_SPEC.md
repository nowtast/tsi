# Research A2 Clean-Room Replay Specification

## 1. Required independence

The replay implementation must not import Python project modules or copy the
Python selector source. It may consume only the portable JSON artifact, this
specification, and the numeric preregistration. Running it on the same machine is
a software replay, not an independent external replication.

## 2. Inputs and arithmetic

The portable artifact contains world specifications, training rows, held-out
rows, and noiseless centers for audit. All arithmetic is integer arithmetic
modulo seven. NLL uses probability 0.88 for a matched coordinate and 0.02 for
each of six mismatches. Exact support compares unordered
output-feature-coefficient triples.

## 3. Selectors

The typed selector independently estimates five nonzero direct coefficients and
selects one of the ordered declared families and one nonzero coefficient for
each supplied edge. Ties are broken by error count, family order, then
coefficient order.

The generic selector starts with zero increments. At each of exactly seven
moves, it evaluates every still-available output-feature position and every
coefficient 1 through 6. It chooses the lexicographically first minimum in
output, feature, coefficient order and never reuses an output-feature position.

The width feature order, nuisance descriptor order, typed catalog, and
alternative catalog are exactly those in `research_a2_features.py` and
`research_a2_preregistration_draft.json`.

## 4. Required replay outputs

The replay must reproduce every world-level endpoint within absolute tolerance
`1e-12`, every exact-recovery Boolean exactly, all three simultaneous-interval
analyses within `1e-12`, and all gate decisions exactly. It must also verify row
counts, family-pair balance, composition-stratum cycling, and paired
misspecification metadata.

## 5. Pre-freeze dry run

Before source freeze, a fixture generated from a public fixed test seed must be
replayed end to end. The fixture and audit are developmental software tests and
are excluded from confirmation. The confirmatory portable artifact is not
created until one-shot execution.
