# Research A1 외부 Replay 지침

## 범위

이 package로 제3자는 TSI Python package를 import하지 않고 export된 world data에서
Research A1 primary endpoint 16개를 모두 재계산할 수 있다. Root seed에서 world를 다시
생성하지는 않는다. Seed-to-export 계보는 공개 commitment, reveal ledger 및 file hash로
감사한다.

## 무결성 검사

압축을 푼 package root에서 `confirmatory/release_manifest.json`과 파일을 대조한다.
특히 다음을 확인한다.

```bash
sha256sum confirmatory/portable_replay.json
sha256sum confirmatory/confirmatory_analysis.json
sha256sum reproduction/research_a1_cleanroom.mjs
```

각각의 예상 SHA-256은 다음과 같다.

- `c0de0594f9febc318a1edcca84341c2af541e44fa77d9862e9a22b55fd66bd3e`
- `0cca53e69c63ffacaccbf7a66064eedc84bb6bf6479c0a9eada4e4cf2b4c1e5d`
- `45ae1069543dabbdd883098f87f61b542a824649bc8612268c174b07bbe72998`

## Replay

현재 Node.js runtime을 사용한다. Package 설치는 필요하지 않다.

```bash
node reproduction/research_a1_cleanroom.mjs \
  confirmatory/portable_replay.json \
  confirmatory/confirmatory_analysis.json \
  cleanroom_audit_external.json
```

예상 terminal status는 failure 0개와 `passed: true`다. 생성 파일을
`confirmatory/cleanroom_audit.json`과 비교한다. Runtime version text는 달라도 되지만
endpoint value와 decision은 코드에 명시한 numerical tolerance 안에서 같아야 한다.

## 외부 replay 보고 항목

다음을 기록한다.

- Replayer 이름 또는 지속적으로 사용하는 pseudonym
- 소속이 있다면 affiliation
- Operating system과 `node --version`
- Package SHA-256과 입수 경로
- 실행 날짜
- 생성한 audit의 SHA-256
- 파일 또는 명령 변경 여부
- Audit 실패 시 전체 failure record

원저자는 replayer가 저자 workflow 밖에 있고 private assistance나 변경한 input 없이
package를 실행한 경우에만 independently replayed라고 기술한다.
