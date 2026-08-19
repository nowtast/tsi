# TSI Reference-Implementation Test Matrix

This matrix records executable audits for the finite reference implementation.
It does not identify a finite audit with a universal mathematical proof.

| Implemented invariant | Reference module | Executable audit |
|---|---|---|
| Finite Betti numbers and H0 persistence | `src/tsi/topological.py` | `tests/test_topological.py` |
| Finite preorder/Alexandrov correspondence | `src/tsi/order_topology.py` | `tests/test_order_topology.py` |
| Label subcomplexes, filtrations, and contiguity | `src/tsi/labeled_topology.py` | `tests/test_labeled_topology.py` |
| Finite correspondence discrepancy | `src/tsi/geometric.py` | `tests/test_geometric.py` |
| Ambient and metric-measure fixtures | `src/tsi/geometric_validation.py` | `tests/test_geometric_validation.py` |
| Metric-graph realization and transport curvature | `src/tsi/metric_graph.py` | `tests/test_metric_graph.py` |
| Attributed correspondence and sampling fixtures | `src/tsi/attribute_geometry.py` | `tests/test_attribute_geometry.py` |
| Finite relational composition | `src/tsi/relational.py` | `tests/test_relational.py` |
| Partial persistence and rollout recurrence | `src/tsi/dynamical.py` | `tests/test_dynamical.py` |
| Integrated finite-state and bridge invariants | `src/tsi/coherent.py` | `tests/test_coherent.py` |

These tests can expose implementation defects and counterexamples in the tested
finite domains. They are supporting audits, not substitutes for the proofs and
scope statements in the public manuscripts.
