# Research A1 External Replay Instructions

## Scope

This package permits a third party to recompute all 16 primary Research A1
endpoints from exported world data without importing the TSI Python package.
The replay does not regenerate worlds from the root seed; seed-to-export
lineage is covered by the published commitment, reveal ledger, and file hashes.

## Integrity checks

From the unpacked package root, verify the files against
`confirmatory/release_manifest.json`. In particular:

```bash
sha256sum confirmatory/portable_replay.json
sha256sum confirmatory/confirmatory_analysis.json
sha256sum reproduction/research_a1_cleanroom.mjs
```

Expected SHA-256 values are respectively:

- `c0de0594f9febc318a1edcca84341c2af541e44fa77d9862e9a22b55fd66bd3e`
- `0cca53e69c63ffacaccbf7a66064eedc84bb6bf6479c0a9eada4e4cf2b4c1e5d`
- `45ae1069543dabbdd883098f87f61b542a824649bc8612268c174b07bbe72998`

## Replay

Use a current Node.js runtime. No package installation is required.

```bash
node reproduction/research_a1_cleanroom.mjs \
  confirmatory/portable_replay.json \
  confirmatory/confirmatory_analysis.json \
  cleanroom_audit_external.json
```

The expected terminal status is `passed: true` with zero failures. Compare the
generated file with `confirmatory/cleanroom_audit.json`; runtime version text
may differ, while endpoint values and decisions must agree within the encoded
numerical tolerance.

## Reporting an external replay

Please record the following:

- replayer name or persistent pseudonym;
- affiliation, if any;
- operating system and `node --version`;
- package SHA-256 and acquisition source;
- execution date;
- generated audit SHA-256;
- whether any file or command was changed;
- full failure records if the audit did not pass.

The original author will call the result independently replayed only when the
replayer is outside the author workflow and executed the package without
private assistance or modified inputs.
