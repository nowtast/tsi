# Research A2 외부 Seed Custodian Protocol

## 목적

저자는 Research A2 confirmatory seed를 생성, 선별, 선택, 재추출 또는 교체할 수 없다.
외부 custodian 한 명이 review를 마친 source freeze가 commit되어 `origin/main`에 공개된
뒤에만 seed 하나를 제공한다.

## 필수 순서

1. 외부 review와 freeze 전 test를 모두 완료한다.
2. `--seed-custodian-id`로 custodian 식별자를 고정하여 freeze manifest를 생성한다.
3. Freeze manifest를 commit하고 push한다. 공개 commit hash와 freeze digest를
   custodian에게 전달한다.
4. Custodian은 운영체제의 cryptographic random-number generator로 정확히 한 번
   32-byte 값을 생성한다. 후보 집합 생성이나 재추출은 허용하지 않는다.
5. Custodian은 SHA-256을 계산하고
   `A2_SEED_CUSTODIAN_ATTESTATION_TEMPLATE.json`을 작성하여 draw를 freeze digest와
   공개 commit에 결박한다.
6. Custodian은 32-byte 파일과 attestation을 한 번만 전달한다. 저자는
   `tools/commit_research_a2_seed.py`를 실행한다. 이 도구에는 난수 생성기가 없으며,
   attestation이 없거나 오래되었거나 일치하지 않거나 저자가 생성·선택·재추출한 seed를
   거부한다.
7. One-shot 실행 전에 `seed_commitment.json`을 commit하고 push한다.
8. 한 번만 실행한다. 공개된 seed, attestation, freeze commit, commitment commit,
   결과 hash를 integrity ledger에 기록한다.

## Custodian 명령 예시

Custodian은 자신의 비공개 machine에서 다음 명령을 사용할 수 있다.

```bash
openssl rand 32 > research_a2_root_seed.bin
sha256sum research_a2_root_seed.bin
```

Binary seed는 실행 전에 공개하지 않는다. Attestation과 그 digest는 commitment와 함께
공개할 수 있다. Custodian은 한 번의 draw만 전달했다는 사실을 사후 감사할 수 있도록
원본 메시지 또는 전달 기록을 보존한다.

## 신뢰 경계

실행 코드는 파일 길이, digest 결박, timestamp, freeze identity와 모든 필수 진술을
검사한다. 그러나 사람이 진실을 말했다는 사실까지 코드로 증명할 수는 없다. 따라서
잔여 신뢰는 명시적이고 좁다. 이름을 고정한 custodian이 freeze 이후 한 번만 draw했고
저자의 선택이 없었다고 attest한다. 이는 저자의 비공개 후보 seed 선별을 배제하지 못하는
저자 자체 commitment보다 강한 통제다.
