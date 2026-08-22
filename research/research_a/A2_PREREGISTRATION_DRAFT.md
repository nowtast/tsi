# Research A2 Preregistration Draft

## 1. Status

This is a numeric draft ready for review. It is not frozen. No confirmatory seed
has been generated. The executable contract digest is
`fb4247528af568aaddd12c2e38b972cf84c7044d8992e92e17d3f541b5326f58`.

Research A2 was directionally recorded before the A1 confirmatory seed. The
development label `TSI-RESEARCH-A2-DEVELOPMENT-v1` and power label
`TSI-RESEARCH-A2-PROSPECTIVE-POWER-v1` are deterministic and nonconfirmatory.

## 2. Independent populations

Each efficiency axis contains 135 independent worlds. Each scope condition
also contains 135 worlds. The count is the smallest candidate at or above the
pre-A1 minimum of 126 that is divisible by both nine matched family-pair strata
and five special-family-pair strata. Matched strata contain 15 worlds each;
special strata contain 27 worlds each.

Every world supplies the same graph and raw rows to its two arms. Training uses
primitive actions. Testing uses 1,200 held-out two-action composition cases at
noise probability 0.12. The held-out set is not used for fitting or selection.

## 3. Width axis

Generic output-feature position counts are 55, 100, and 300. Train prefixes are
10, 15, 20, 25, 30, and 40. Every dictionary includes the true seven-term
support and every generic fit takes seven greedy moves. The typed search is
unchanged. Training noise is 0.08.

At every width-by-prefix cell the paired outcomes are generic-minus-typed
composition NLL and typed-minus-generic exact recovery. The 36 named outcomes
form one Bonferroni family at familywise alpha 0.05. A cell has joint advantage
only when both simultaneous lower bounds exceed zero and both point estimates
meet their endpoint SESOI, 0.01 NLL and 0.10 recovery. The width gate requires at
least one such cell at every width.

## 4. Noise axis

Training noise probabilities are 0.08, 0.30, 0.60, and 0.80. Train prefixes are
15, 20, 30, 40, 80, and 160. The generic arm uses 55 positions and seven moves.
Within a world the four training streams share clean rows and nested corruption
masks.

The same two outcomes produce 48 named endpoints and a separate Bonferroni
family at familywise alpha 0.05. The noise gate requires at least one joint
SESOI-qualified advantage anywhere on the fixed probability-by-prefix grid. It
does not require an advantage at 0.80; that level tests degradation near the
unique-mode boundary.

## 5. Bidirectional scope axis

The confirmatory prefix is 320. The matched condition compares two
representable catalogs. In typed misspecification every generator pair contains
cubic and only the generic catalog contains cubic. In generic misspecification
the aligned cubic occurrences are replaced by quadratic, retained only by the
typed catalog. Each catalog has 55 positions.

Six outcomes form a third Bonferroni family: NLL and center-accuracy differences
for each condition. The scope gate requires all of the following:

1. Matched simultaneous intervals are inside NLL margin 0.01 and accuracy
   margin 0.025.
2. In typed misspecification, both interval upper bounds are below zero and
   both point estimates favor generic by at least 0.10.
3. In generic misspecification, both interval lower bounds are above zero and
   both point estimates favor typed by at least 0.10.

The scope gate cannot change the efficiency decision.

## 6. Development-only power

The stratified bootstrap used 36 matched development worlds and 45 scope worlds
for 20,000 iterations. At 135 worlds, estimated power was 1.0 for the all-width
gate, 1.0 for the any-noise gate, and 1.0 for the conjunctive scope gate. Noise
group power at 0.80 was 0.0 and is reported explicitly; it is not silently
removed or converted into a required advantage.

The development report and power report are
`experiments/research_a2_v1/development_report.json` and
`experiments/research_a2_v1/prospective_power.json`. They cannot enter the
confirmatory cohort.

## 7. Freeze and execution order

The required order is external review of this draft, full tests and clean-room
dry run, source freeze, commit and public recording of the freeze digest,
generation of a 32-byte seed escrow and public commitment, commit and push of
that commitment, and one-shot execution. No threshold, endpoint, or cohort
repair is allowed after seed commitment.
