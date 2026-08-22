# Theory of Structural Imagination

This repository contains the publishable manuscripts, reference code, tests, and
experiment specifications for the Theory of Structural Imagination (TSI).
A Korean guide is available in [README_KO.md](README_KO.md).

The archived Paper 03/04 evidence release is permanently available at:

```text
Version DOI: 10.5281/zenodo.22004526
Concept DOI: 10.5281/zenodo.22004525
```

For a clean reproduction of the Paper 3/4 confirmatory chain and post-review
analyses, see [REPRODUCE.md](REPRODUCE.md). A Korean version is provided in
[REPRODUCE_KO.md](REPRODUCE_KO.md).

Large sealed and development outputs are not stored in Git history. The
versioned Zenodo ZIP is pinned by size and SHA-256 in
`artifacts/paper03-04-v1.0.0.json` and can be downloaded and verified with:

```bash
python3 tools/fetch_zenodo_release.py --extract
```

## Evidence and Claims Ledger

This table is the repository-level statement of what the current program does
and does not establish. Papers 03 and 04 share one frozen prospective cohort;
their overlapping endpoints are not independent replications.

| Claim | Status | Evidence | Population and boundary | Decisive extension test |
|---|---|---|---|---|
| Train-only recovery of the declared graph motif and head families | Established in the frozen attribution cohort | 120/120 worlds; simultaneous Wilson lower bound 0.941 | Five-coordinate modulo-7 worlds, 30 two-edge motifs, three declared head families | Independent environments, larger graph families, and external teams |
| Correct graph information improves held-out two-mechanism composition NLL | Established in the frozen attribution cohort | Wrong-minus-correct NLL: factorized 0.35933 [0.35431, 0.36435], generic sparse 0.36160 [0.35683, 0.36637], generic dense 0.03893 [0.02527, 0.05259] | Declared finite stochastic population with train-only graph/head selection | New generators, continuous states, perceptual inputs, and independent replication |
| The graph effect survives removal of exact representability | Established in the frozen stress cohort | 0.18545 [0.18388, 0.18702]; 120/120 worlds nonexact | One preregistered nonadditive-synergy stress family | Additional misspecification families and independently implemented generators |
| Factorized and matched generic-sparse heads are predictively equivalent after recovery of the same support | Proved conditionally and observed in the attribution cohort | Seven-sparse embedding proposition; correct-graph NLL difference 0.00000 in every world | Requires a generic dictionary containing the selected edge functions and successful support recovery | Misspecified dictionaries and finite-sample support-recovery regimes |
| Typed structure reduces sample or search complexity | Untested | No confirmatory evidence | Not implied by the representational-equivalence result | Research A1: source-frozen matched sample-size study; A2 width direction recorded but not numerically frozen |
| The richer five-layer state adds predictive value beyond the correct minimal support | Untested | No confirmatory evidence | Current attribution isolates graph information and minimal support | Turnover and model-audit studies with information-symmetric arms |
| Entity discovery and cross-time identity maintenance from raw observation | Outside current estimands | Entity sets and identities are supplied by the generators | No visual or perceptual population is sampled | A separately frozen perception study after the preceding gates |

The manuscripts are release candidates. As of 22 August 2026, arXiv submission
is blocked by endorsement only, not by a scientific or packaging defect.
Research A and the model-audit study are separate future outputs and will not be
folded into the current Papers 01, 03, or 04 before their first public versions.

## Public Manuscript Set

- Paper 01 is the companion theory and structural-specification report.
- Paper 03 is the sealed evaluation record for OOD generalization, rollout, and
  learned-routing criterion validation.
- Paper 04 is the prospective graph-information and head-factorization
  attribution study.

Each manuscript has a structurally matched Korean edition. Developmental Paper
02 components and submission/review snapshots are retained locally but are not
part of this public release. See [papers/README.md](papers/README.md) for the
canonical inventory and status of each manuscript.

## Repository Layout

```text
papers/
  README.md           Canonical public-paper inventory
  README_KO.md        Korean inventory
  paper1/             Companion theory report, English
  paper1_ko/          Companion theory report, Korean
  paper3/             Sealed empirical evaluation, English
  paper3_ko/          Sealed empirical evaluation, Korean
  paper4/             Graph-by-head attribution study, English
  paper4_ko/          Graph-by-head attribution study, Korean
src/tsi/
  README.md          Module map and retention policy
  README_KO.md       Korean module map
  structured_space.py
  coherent.py         Canonical finite state, bridges, and common metric
  geometric.py        Exact small-state implementation of Paper 2B Delta_g
  geometric_validation.py
  labeled_topology.py X2 labeled filtrations and map audits
  relational.py       Exact small-state implementation of Paper 2C
  dynamical.py        Exact finite implementation of Paper 2D
  topological.py      GF(2) homology and finite persistence audit
  paper_parity.py     English/Korean public-manuscript parity checks
tools/
  check_bilingual_parity.py
experiments/
  stage2_i0/          Finite oracle and fixed-seed learned pilot
  paper34_resolution_v1/
                       Prospective Paper 3/4 cohort and post-review audits
artifacts/
  paper03-04-v1.0.0.json
                       Zenodo DOI, URL, size, and checksums
benchmarks/
  structural_attribution_v0_1/
                       Public audit/development benchmark and checker contract
research/
  research_a/          Prospective support-recovery efficiency design
reproduction/
  README.md           Reproduction-asset scope and handling rules
  README_KO.md        Korean reproduction guide
  frozen_source/       Historical confirmatory source snapshot
tests/
```

See [src/tsi/README.md](src/tsi/README.md) for the live-module map and
[reproduction/README.md](reproduction/README.md) for the frozen-source and
clean-room retention rules. The benchmark scope and non-blind limitation are
documented in [benchmarks/README.md](benchmarks/README.md).
Prospective-study status and separation rules are documented in
[research/README.md](research/README.md).

## Working Principle

TSI treats imagination as construction, transformation, and simulation in an
internal structural space. A coordinate encoding may implement such a space, but
coordinates alone do not specify its equivalence relation, observables,
transformations, or validation criteria.

## Rigor Rule

An intuition is not a final result. Each claim must be assigned a domain and
status, supported by relevant literature, converted where possible into a
precise definition or theorem, tested against counterexamples, and separated
from empirical hypotheses. Open claims remain explicitly open.

## Bilingual Rule

Every paper is maintained as paired English and Korean TeX packages with the same
section order, formal-environment order, citation keys, and byte-identical
bibliographies. The checker also rejects missing and uncited bibliography keys:

```bash
PYTHONPATH=src python3 tools/check_bilingual_parity.py
```

Run the reference-code tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Scope

The empirical evidence uses declared synthetic world families. It does not
establish real-world validity, unconstrained neural structure discovery,
universal model superiority, or external independent replication. Endpoints
shared by Papers 03 and 04 belong to one frozen analysis and are not independent
evidence.

## License and Citation

Source code is licensed under MIT; data and documentation are licensed under CC
BY 4.0. See `LICENSE`, `LICENSE-DATA.md`, and `CITATION.cff`. Exact reuse of the
Paper 03/04 evidence should cite version DOI `10.5281/zenodo.22004526`.
