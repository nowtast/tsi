# Reproducing the Paper 3/4 Evidence

Archived evidence release:

```text
Version DOI: 10.5281/zenodo.22004526
Concept DOI: 10.5281/zenodo.22004525
ZIP SHA-256: 11e3fd40b623a46c3ebab1ed03e0125329fef087e88191de6b73e6229d141e07
```

Run every command from the repository root. The commands below do not modify
the frozen confirmatory artifacts; regenerated reports are written to `/tmp`.

## Requirements

- Python 3.10 or later and NumPy 1.24 or later
- Node.js 18 or later for the clean-room implementation
- XeLaTeX, BibTeX, and `latexmk` for the papers
- `pdftotext` for rendered-text checks

An isolated Python environment can be prepared with:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

Download, verify, and extract the complete archived evidence release with:

```bash
python3 tools/fetch_zenodo_release.py --extract
```

The downloader reads `artifacts/paper03-04-v1.0.0.json`, verifies the exact
byte count and SHA-256, and writes only under the ignored `.artifacts/`
directory. The repository retains the smaller confirmatory inputs needed by
the commands below; the 1.13 GiB sealed validity result is available only in
the Zenodo release.

## 1. Verify the frozen artifacts

The historical source snapshot is in
`reproduction/frozen_source/paper34_resolution_v1`. The complete Zenodo release
contains `SHA256SUMS` for every archived file. The
principal frozen digests in this checkout are:

```text
2e6880df054327a56164bad98532dcb22efe7e3e69e3c6426558e5d9bd945501  confirmatory_analysis.json
cd89c1ff6055fa7b74ab8c2c135115677cb808244788e78e68df39baf6a1c532  raw_results.json
3f438a1cfb7d139cfc717730d081178f309102dec92ba7ab88ff36263b4996f2  portable_inputs.json
```

## 2. Re-derive all 120 worlds from the revealed root seed

```bash
PYTHONPATH=src python3 tools/verify_paper34_world_derivation.py \
  experiments/paper34_resolution_v1/cleanroom/portable_inputs.json \
  experiments/paper34_resolution_v1/confirmatory/seed_and_integrity_ledger.json \
  /tmp/paper34_world_derivation_audit.json
sha256sum /tmp/paper34_world_derivation_audit.json
```

Expected status: `passed: true`; all six check counts are 120. Expected SHA-256:

```text
fcc0c16346be4941d161a1007fbc778c8d9735eaf4b198b0a2733a40b3e33758
```

This step verifies the root-seed derivation, typed graph and head family, and
every train, selection, and OOD case. It uses the frozen Python generator and
is therefore a lineage audit, not an independent implementation.

## 3. Recompute predictions and effect means in Node.js

```bash
node reproduction/paper34_resolution_cleanroom.mjs \
  experiments/paper34_resolution_v1/cleanroom/portable_inputs.json \
  experiments/paper34_resolution_v1/confirmatory/confirmatory_analysis.json \
  experiments/paper34_resolution_v1/confirmatory/seed_and_integrity_ledger.json \
  /tmp/paper34_cleanroom_audit.json
```

Expected status: `passed: true`, 120 learned-factorized worlds reproduced, ten
effect means reproduced, and no project imports. This Node implementation
consumes the exported worlds; Step 2 is what verifies their derivation.

## 4. Recompute the post-review analyses

Multiplicity sensitivity at a divisor of ten:

```bash
PYTHONPATH=src python3 tools/audit_paper34_multiplicity.py \
  experiments/paper34_resolution_v1/confirmatory/confirmatory_analysis.json \
  /tmp/multiplicity_sensitivity_divisor10.json
```

Retrospective power analysis using only the 24 development worlds:

```bash
PYTHONPATH=src python3 tools/run_paper34_retrospective_power.py \
  experiments/paper34_resolution_v1/development_report.json \
  /tmp/retrospective_power_report.json --iterations 20000
```

Separate 3-by-3 noise sensitivity experiment:

```bash
PYTHONPATH=src python3 tools/run_paper34_noise_sensitivity.py \
  /tmp/noise_sensitivity_3x3.json --worlds 120 --workers 8
```

Expected SHA-256 values, in the order above:

```text
6dce71a264109469d8b2d607de06ac9d166c5da73eced4128e6141ca064be3ad
c3b0358fec2039b6eb9004e0bb8737bede27a5a37aa7ba8fb2c56608d82e52cd
5acf46dcdf6c965ed0cda91061cbc0a3f9380dcf8771c3710417b4a4373ac044
```

On an 8-core development workstation, Steps 2 and 4 usually complete in under
three minutes in total; the noise grid dominates runtime. Runtime is not an
integrity condition.

## 5. Run tests and build both language editions

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Build each paper from its own directory:

```bash
(cd papers/paper3 && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)
(cd papers/paper3_ko && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)
(cd papers/paper4 && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)
(cd papers/paper4_ko && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)
PYTHONPATH=src python3 -m unittest -q tests.test_paper_parity
```

The clean-room implementation and the derivation audit were written by the
same author as the benchmark. They are reproducibility checks, not an external
independent replication. An external replication must be reported separately.
