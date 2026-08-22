# 재현성 자산

이 디렉터리는 Paper 03/04 prospective 증거와 Research A1의 구현 독립 audit 경로를
보존한다. 논문 초안이나 일반 개발 산출물이 아니라 재현성 자료다.

## 구성

- `frozen_source/paper34_resolution_v1/`: confirmatory cohort에 사용한 Python source,
  test 및 runner의 byte-preserved snapshot이다. 파일 hash는
  `experiments/paper34_resolution_v1/freeze_manifest.json`에 고정되어 있다.
- `paper34_resolution_cleanroom.mjs`: graph/head search, factorized fitting, NLL
  evaluation 및 보고된 effect mean을 project import 없이 재구현한 Node.js 코드다.
- `research_a1_cleanroom.mjs`: A1 selector 3개와 primary endpoint decision 16개를
  project import 없이 재구현한 Node.js 코드다.

Node.js 경로는 export된 world를 입력으로 사용하며 root seed에서 world를 독립적으로
재생성하지 않는다. 별도의 Python derivation audit가 seed-to-export 계보를 검사한다.
두 구현 모두 같은 저자가 작성했으므로 이는 구현 재현성 검사이지 독립 연구그룹의
재현 연구가 아니다.

## 검사 실행

모든 명령은 저장소 루트에서 실행한다. 정확한 입력, 예상 상태와 digest는
`REPRODUCE_KO.md`에 있다. Clean-room 명령은 다음과 같다.

```bash
node reproduction/paper34_resolution_cleanroom.mjs \
  experiments/paper34_resolution_v1/cleanroom/portable_inputs.json \
  experiments/paper34_resolution_v1/confirmatory/confirmatory_analysis.json \
  experiments/paper34_resolution_v1/confirmatory/seed_and_integrity_ledger.json \
  /tmp/paper34_cleanroom_audit.json
```

Research A1 명령은 다음과 같다.

```bash
node reproduction/research_a1_cleanroom.mjs \
  experiments/research_a_v1/confirmatory/portable_replay.json \
  experiments/research_a_v1/confirmatory/confirmatory_analysis.json \
  /tmp/research_a1_cleanroom_audit.json
```

## 취급 규칙

`frozen_source/` 아래 파일은 수정하지 않는다. 유지보수 변경은 `src/tsi`에 반영한다.
새 confirmatory 연구에는 이 snapshot을 고치는 대신 새로운 versioned freeze를 만든다.
재생성한 출력도 이 디렉터리 밖에 기록한다.
