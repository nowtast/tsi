# Structural Attribution Benchmark v0.1

This directory packages the existing Paper 03/04 attribution cohort as a
versioned, machine-checkable audit benchmark. It introduces no new experiment
and does not turn the frozen reference results into independent evidence.

## Intended Use

The benchmark separates graph/head recovery from held-out two-mechanism
composition prediction. The independent unit is a world. Fitting may read the
`train` partition, graph/head selection may read `selection`, and neither may
read test targets or the archived answers.

Use the participant loader instead of opening `portable_inputs.json` directly:

```python
from pathlib import Path
from tsi.structural_attribution_benchmark import load_participant_worlds

worlds = load_participant_worlds(Path.cwd())
```

The loader removes `graph`, `families`, and `expected_row`, and strips target
states from test cases. The original portable file remains public for frozen
reproduction, so v0.1 is an audit and development benchmark, not a blind
leaderboard. Self-reported compliance is not independent validation.

## Files

- `benchmark.json`: benchmark contract and hashes of all source artifacts.
- `reference_results.json`: exact values linked to the frozen analyses.
- `submission_schema.json`: portable report format.
- `examples/minimal_submission.json`: one-world format smoke test only.
- `BENCHMARK_CARD.md`: estimands, information policy, limitations, and reuse.

Run:

```bash
PYTHONPATH=src python3 tools/check_structural_attribution_benchmark.py
PYTHONPATH=src python3 tools/check_structural_attribution_benchmark.py \
  benchmarks/structural_attribution_v0_1/examples/minimal_submission.json
```

The public evidence payload is archived at DOI
`10.5281/zenodo.22004526`. Exact reuse must preserve the contract digests and
artifact hashes recorded in `benchmark.json`.
