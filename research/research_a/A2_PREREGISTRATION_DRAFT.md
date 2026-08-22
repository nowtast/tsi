# Research A2 Preregistration Draft

## 1. Status

This is a numeric draft ready for review. It is not frozen. No confirmatory seed
has been generated. The executable contract digest is
`c3c07798ef01406974640b743603b8e49174879b5e466c386fae249c141a967b`.

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
unchanged. Training noise is 0.08. After the eleven A1 features, nuisance
descriptors are ordered by increasing state coordinate, then degree 1 before 2,
then increasing action coordinate. Width 100 uses the first nine descriptors.
Width 300 uses 49 of the 50 possible descriptors and excludes only
\(a_4x_4^2\).

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
SESOI-qualified advantage at each of 0.08, 0.30, and 0.60. All six sample-size
summaries and simultaneous intervals at 0.80 are mandatory boundary-stress
outputs, but 0.80 does not enter the advantage gate. A passed gate therefore
supports robustness through 0.60 in the declared population, not robustness at
0.80 or at arbitrary noise levels. The complete degradation curve is reported
regardless of the gate.

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
gate, 0.9994 for the conjunctive through-0.60 noise gate, and 1.0 for the
conjunctive scope gate. Noise group power at 0.80 was 0.0 and is reported
explicitly; it is not silently removed or converted into a required advantage.

The development report and power report are
`experiments/research_a2_v1/development_report.json` and
`experiments/research_a2_v1/prospective_power.json`. They cannot enter the
confirmatory cohort.

## 7. Freeze and execution order

The required order is external review of this draft, full tests and clean-room
dry run, source freeze naming one external seed custodian, commit and public
recording of the freeze digest, one post-freeze 32-byte draw by that custodian,
validation of the custodian attestation, commit and push of the resulting seed
commitment, and one-shot execution. The author is prohibited from generating,
screening, selecting, rerolling, or replacing the confirmatory seed. No
threshold, endpoint, or cohort repair is allowed after seed commitment. The
remaining trust boundary is the truth of the named custodian's single-draw
attestation; its digest, freeze binding, and eventual seed reveal are auditable.
