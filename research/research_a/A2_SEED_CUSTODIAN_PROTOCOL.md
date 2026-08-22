# Research A2 External Seed-Custodian Protocol

## Purpose

The author must not generate, screen, select, reroll, or replace the Research A2
confirmatory seed. A single external custodian supplies the seed only after the
reviewed source freeze is committed and publicly visible on `origin/main`.

## Required order

1. Complete external review and all pre-freeze tests.
2. Generate the freeze manifest with the custodian identifier fixed by
   `--seed-custodian-id`.
3. Commit and push the freeze manifest. Give the custodian the public commit hash
   and freeze digest.
4. The custodian generates exactly one 32-byte value with an operating-system
   cryptographic random-number generator. No candidate set or reroll is allowed.
5. The custodian computes its SHA-256 and completes
   `A2_SEED_CUSTODIAN_ATTESTATION_TEMPLATE.json`, binding the draw to the freeze
   digest and public commit.
6. The custodian delivers the 32-byte file and attestation once. The author runs
   `tools/commit_research_a2_seed.py`; the tool contains no random generator and
   rejects an unattested, stale, mismatched, author-generated, or rerolled seed.
7. Commit and push `seed_commitment.json` before the one-shot execution.
8. Execute once. The revealed seed, attestation, freeze commit, commitment commit,
   and result hashes enter the integrity ledger.

## Custodian command example

The custodian may use the following on a private machine:

```bash
openssl rand 32 > research_a2_root_seed.bin
sha256sum research_a2_root_seed.bin
```

The binary seed is not public before execution. The attestation and its digest
may be made public with the commitment. The custodian should retain the original
message or transmission record so that a later audit can establish that only one
draw was delivered.

## Trust boundary

The executable checks file length, digest binding, timestamps, freeze identity,
and all required statements. It cannot prove that a person told the truth. The
remaining trust is therefore explicit and narrow: the named custodian attests to
one post-freeze draw and no author choice. This is materially stronger than an
author-created commitment, which cannot exclude private candidate-seed screening.
