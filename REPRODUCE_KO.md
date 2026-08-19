# Paper 3/4 증거 재현

보존된 증거 릴리스:

```text
버전 DOI: 10.5281/zenodo.22004526
개념 DOI: 10.5281/zenodo.22004525
ZIP SHA-256: 11e3fd40b623a46c3ebab1ed03e0125329fef087e88191de6b73e6229d141e07
```

모든 명령은 저장소 루트에서 실행한다. 아래 명령은 동결된 confirmatory 산출물을
수정하지 않으며, 재생성한 보고서는 `/tmp`에 기록한다.

## 요구 환경

- Python 3.10 이상과 NumPy 1.24 이상
- clean-room 구현 실행용 Node.js 18 이상
- 논문 빌드용 XeLaTeX, BibTeX, `latexmk`
- 렌더링된 본문 검사용 `pdftotext`

격리된 Python 환경은 다음과 같이 준비할 수 있다.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

전체 보존 증거 릴리스는 다음 명령으로 내려받고 검증한 뒤 압축을 푼다.

```bash
python3 tools/fetch_zenodo_release.py --extract
```

다운로더는 `artifacts/paper03-04-v1.0.0.json`을 읽고 정확한 byte 수와 SHA-256을
검증하며, Git에서 제외된 `.artifacts/` 아래에만 기록한다. 아래 명령에 필요한 작은
확증 입력은 저장소에 남아 있지만, 1.13 GiB 봉인 validity 결과는 Zenodo 릴리스에서만
제공한다.

## 1. 동결 산출물 검증

역사적 source snapshot은
`reproduction/frozen_source/paper34_resolution_v1`에 있다. 전체 Zenodo 릴리스의
`SHA256SUMS`는 모든 보존 파일을 검증한다. 현재 checkout의 주요
동결 digest는 다음과 같다.

```text
2e6880df054327a56164bad98532dcb22efe7e3e69e3c6426558e5d9bd945501  confirmatory_analysis.json
cd89c1ff6055fa7b74ab8c2c135115677cb808244788e78e68df39baf6a1c532  raw_results.json
3f438a1cfb7d139cfc717730d081178f309102dec92ba7ab88ff36263b4996f2  portable_inputs.json
```

## 2. 공개된 root seed에서 120개 world 재유도

```bash
PYTHONPATH=src python3 tools/verify_paper34_world_derivation.py \
  experiments/paper34_resolution_v1/cleanroom/portable_inputs.json \
  experiments/paper34_resolution_v1/confirmatory/seed_and_integrity_ledger.json \
  /tmp/paper34_world_derivation_audit.json
sha256sum /tmp/paper34_world_derivation_audit.json
```

예상 상태는 `passed: true`이며 여섯 검사의 count가 모두 120이어야 한다. 예상
SHA-256은 다음과 같다.

```text
fcc0c16346be4941d161a1007fbc778c8d9735eaf4b198b0a2733a40b3e33758
```

이 단계는 root-seed 유도, typed graph와 head family, 모든 train, selection, OOD
case를 검증한다. 동결된 Python generator를 사용하므로 계보 검증이지 독립 구현은
아니다.

## 3. Node.js에서 예측과 효과 평균 재계산

```bash
node reproduction/paper34_resolution_cleanroom.mjs \
  experiments/paper34_resolution_v1/cleanroom/portable_inputs.json \
  experiments/paper34_resolution_v1/confirmatory/confirmatory_analysis.json \
  experiments/paper34_resolution_v1/confirmatory/seed_and_integrity_ledger.json \
  /tmp/paper34_cleanroom_audit.json
```

예상 상태는 `passed: true`이고, learned-factorized world 120개와 effect mean 10개를
재현하며 project import는 0이어야 한다. 이 Node 구현은 export된 world를 소비하며,
그 world의 유도는 2단계가 검증한다.

## 4. 사후 리뷰 분석 재계산

나눗수 10의 다중성 민감도 분석:

```bash
PYTHONPATH=src python3 tools/audit_paper34_multiplicity.py \
  experiments/paper34_resolution_v1/confirmatory/confirmatory_analysis.json \
  /tmp/multiplicity_sensitivity_divisor10.json
```

24개 development world만 이용한 retrospective power 분석:

```bash
PYTHONPATH=src python3 tools/run_paper34_retrospective_power.py \
  experiments/paper34_resolution_v1/development_report.json \
  /tmp/retrospective_power_report.json --iterations 20000
```

별도의 3-by-3 잡음 민감도 실험:

```bash
PYTHONPATH=src python3 tools/run_paper34_noise_sensitivity.py \
  /tmp/noise_sensitivity_3x3.json --worlds 120 --workers 8
```

위 순서에 따른 예상 SHA-256은 다음과 같다.

```text
6dce71a264109469d8b2d607de06ac9d166c5da73eced4128e6141ca064be3ad
c3b0358fec2039b6eb9004e0bb8737bede27a5a37aa7ba8fb2c56608d82e52cd
5acf46dcdf6c965ed0cda91061cbc0a3f9380dcf8771c3710417b4a4373ac044
```

8-core 개발 workstation에서 2단계와 4단계는 통상 합계 3분 이내에 완료되며 잡음
grid가 실행시간의 대부분을 차지한다. 실행시간은 무결성 조건이 아니다.

## 5. test 실행과 두 언어판 빌드

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

각 논문 directory에서 다음을 실행한다.

```bash
(cd papers/paper3 && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)
(cd papers/paper3_ko && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)
(cd papers/paper4 && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)
(cd papers/paper4_ko && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)
PYTHONPATH=src python3 -m unittest -q tests.test_paper_parity
```

clean-room 구현과 derivation audit는 benchmark와 같은 저자가 작성했다. 이들은
재현성 검사이지 외부 독립 재현이 아니다. 외부 재현은 별도로 보고해야 한다.
