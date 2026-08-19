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
reproduction/
  README.md           Reproduction-asset scope and handling rules
  README_KO.md        Korean reproduction guide
  frozen_source/       Historical confirmatory source snapshot
tests/
```

See [src/tsi/README.md](src/tsi/README.md) for the live-module map and
[reproduction/README.md](reproduction/README.md) for the frozen-source and
clean-room retention rules.

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
section order, formal-environment order, and citation keys. Check this invariant
with:

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
