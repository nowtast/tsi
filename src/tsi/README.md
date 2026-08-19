# TSI Python Modules

This directory contains the live Python package used by the public theory
audits and Paper 03/04 evidence workflows. It is not a single production model:
some modules implement mathematical reference objects, some execute frozen or
prospective studies, and some preserve development and negative-result paths.

## Module Groups

| Group | Files | Role |
|---|---|---|
| Structural reference implementation | `structured_space.py`, `topological.py`, `geometric.py`, `geometric_validation.py`, `relational.py`, `dynamical.py`, `coherent.py`, `order_topology.py`, `labeled_topology.py`, `metric_graph.py`, `attribute_geometry.py`, `coherence_spectrum.py`, `bridge_repair.py` | Finite constructions and executable audits associated with the theory report |
| Historical Paper 03 chains | `paper3_*` excluding the final `paper34_*` family | Preregistration, sealed OOD/rollout/validity workflows, supporting benchmarks, ablations, and preserved development paths |
| Prospective Paper 03/04 resolution | `paper34_*` | Final graph-information/head-factorization cohort, multiplicity audit, noise sensitivity, retrospective power analysis, and world-derivation audit |
| Paper 04 attribution and stress studies | `paper4_*` | Comparator, capacity-matching, misspecification, diagnostic, and statistical-analysis components |
| Publication checks | `paper_parity.py` | English/Korean TeX structure and display-math parity |

The `paper3_learned_*` modules include exploratory and failed neural or routing
designs. Their presence does not promote those designs to confirmatory evidence
and does not support a claim of unconstrained neural structure discovery. They
are retained because their tests and recorded outputs document how the final
claim boundary was reached.

## How to Run the Package

User-facing commands live in `tools/`; most `src/tsi` files are importable
modules rather than command-line entry points. From the repository root:

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Use the exact Paper 03/04 workflow in `REPRODUCE.md`. Do not rerun one-shot or
sealed-access tools merely to test installation; their historical locks and
ledgers are evidence artifacts.

## Live Source Versus Frozen Source

Files here are the maintained live implementation. The byte-preserved source
used for the prospective confirmatory run is under
`reproduction/frozen_source/paper34_resolution_v1`. The frozen copy is retained
even when a live module later receives reporting-only or audit changes.

## Retention Rule

A module remains in the public tree when it is required by another module, a
public tool, a test, an artifact/freeze manifest, or a documented reproduction
path. Test-only development modules may be removed only together with their
tests and provenance artifacts after confirming that no manuscript or archived
evidence claim depends on them. No such removal is made silently.
