# Reproduction Assets

This directory preserves two assets required to audit the Paper 03/04
prospective evidence. These are final reproducibility materials, not manuscript
drafts or general development outputs.

## Contents

- `frozen_source/paper34_resolution_v1/`: byte-preserved Python source, tests,
  and runners used for the confirmatory cohort. Its file hashes are fixed by
  `experiments/paper34_resolution_v1/freeze_manifest.json`.
- `paper34_resolution_cleanroom.mjs`: a zero-project-import Node.js
  reimplementation of graph/head search, factorized fitting, NLL evaluation,
  and reported effect means.

The Node.js path consumes exported worlds. It does not independently regenerate
those worlds from the root seed. The separate Python derivation audit checks the
seed-to-export lineage. Both implementations were written by the same author,
so this is an implementation reproducibility check, not an independent
research-group replication.

## Running the Checks

Run commands from the repository root. The exact inputs, expected statuses, and
digests are documented in `REPRODUCE.md`. The clean-room command is:

```bash
node reproduction/paper34_resolution_cleanroom.mjs \
  experiments/paper34_resolution_v1/cleanroom/portable_inputs.json \
  experiments/paper34_resolution_v1/confirmatory/confirmatory_analysis.json \
  experiments/paper34_resolution_v1/confirmatory/seed_and_integrity_ledger.json \
  /tmp/paper34_cleanroom_audit.json
```

## Handling Rule

Do not edit files under `frozen_source/`. Changes to maintained implementations
belong in `src/tsi`; a new confirmatory study requires a new versioned freeze
rather than modifying this historical snapshot. Regenerated outputs should be
written outside this directory.
