# 구조적 상상 이론

이 저장소는 Theory of Structural Imagination(TSI)의 공개 가능한 원고, 참조 코드,
테스트, 실험 명세를 제공한다. 영문 안내는 [README.md](README.md)에 있다.

Paper 03/04 증거 릴리스의 영구 보존 위치는 다음과 같다.

```text
버전 DOI: 10.5281/zenodo.22004526
개념 DOI: 10.5281/zenodo.22004525
```

Paper 03/04 확증 계보와 리뷰 후 분석의 재현 절차는
[REPRODUCE_KO.md](REPRODUCE_KO.md)에 있다. 대용량 봉인 결과와 development 출력은
Git history에 저장하지 않는다. Zenodo ZIP의 URL, 크기, SHA-256은
`artifacts/paper03-04-v1.0.0.json`에 고정되어 있으며 다음 명령으로 내려받고 검증한다.

```bash
python3 tools/fetch_zenodo_release.py --extract
```

## 증거 및 주장 원장

다음 표는 현재 연구 프로그램이 확립한 것과 확립하지 않은 것을 저장소 수준에서
규정한다. Paper 03과 04는 하나의 동결 prospective cohort를 공유하므로 겹치는
endpoint는 독립 replication이 아니다.

| 주장 | 지위 | 근거 | 대상 population과 경계 | 결정적 확장 검정 |
|---|---|---|---|---|
| Train-only 절차가 선언한 graph motif와 head family를 회수한다 | 동결 attribution cohort에서 확립 | 120/120 world; simultaneous Wilson 하한 0.941 | 5-coordinate modulo-7 world, 30개 two-edge motif, 선언한 세 head family | 독립 환경, 더 큰 graph family 및 외부 연구자 재현 |
| Correct graph information이 held-out two-mechanism composition NLL을 개선한다 | 동결 attribution cohort에서 확립 | Wrong-minus-correct NLL: factorized 0.35933 [0.35431, 0.36435], generic sparse 0.36160 [0.35683, 0.36637], generic dense 0.03893 [0.02527, 0.05259] | Train-only graph/head selection을 사용하는 선언한 finite stochastic population | 새 generator, continuous state, perceptual input 및 독립 replication |
| Exact representability를 제거해도 graph effect가 생존한다 | 동결 stress cohort에서 확립 | 0.18545 [0.18388, 0.18702]; 120/120 world nonexact | 사전등록한 하나의 nonadditive-synergy stress family | 추가 misspecification family와 독립 구현 generator |
| 동일 support를 회수하면 factorized head와 matched generic-sparse head가 predictively equivalent하다 | 조건부로 증명되고 attribution cohort에서 관측 | Seven-sparse embedding 명제; 모든 world에서 correct-graph NLL 차이 0.00000 | Generic dictionary가 선택된 edge function을 포함하고 support recovery에 성공해야 함 | Misspecified dictionary와 finite-sample support-recovery regime |
| Typed structure가 sample 또는 search complexity를 줄인다 | 미검 | Confirmatory evidence 없음 | Representational equivalence 결과에서 함의되지 않음 | Research A1: source-frozen matched sample-size 연구; A2 width 방향은 기록했지만 numeric freeze 전 |
| Richer five-layer state가 correct minimal support를 넘어 predictive value를 더한다 | 미검 | Confirmatory evidence 없음 | 현재 attribution은 graph information과 minimal support를 식별 | Information-symmetric arm을 사용하는 turnover 및 model-audit 연구 |
| Raw observation에서 entity discovery와 cross-time identity maintenance | 현재 estimand 밖 | Generator가 entity set과 identity를 제공 | Visual 또는 perceptual population을 표집하지 않음 | 선행 gate 이후 별도로 동결한 perception 연구 |

현재 원고는 release candidate다. 2026년 8월 22일 기준 arXiv 제출의 유일한
blocker는 endorsement이며 과학적·패키징 결함은 없다. Research A와 model-audit
연구는 별도 후속 산출물이며, 현재 Paper 01·03·04의 최초 공개판에 합치지 않는다.

## 공개 원고 세트

- Paper 01은 동반 이론 및 구조 명세 기술보고서다.
- Paper 03은 OOD 일반화, rollout, learned-routing criterion의 봉인 평가 기록이다.
- Paper 04는 graph 정보와 head factorization의 prospective 귀속 연구다.

각 원고는 구조가 일치하는 영문판과 한국어판으로 제공한다. 개발 과정에서 작성한
Paper 02 구성 논문들과 submission/review snapshot은 로컬에 보존하지만 이 공개
릴리스에는 포함하지 않는다. 정본 목록과 각 원고의 상태는
[papers/README_KO.md](papers/README_KO.md)에 정리했다.

```text
papers/          공개 정본인 영문·한국어 TeX 원고 세 쌍
src/tsi/         참조 구현과 실험·분석 모듈
tests/           단위·정합성 테스트
tools/           재현·검증 명령
experiments/     공개 가능한 소규모 결과와 확증 audit
artifacts/       Zenodo release manifest
benchmarks/      Versioned public audit/development benchmark
research/        현재 확증 cohort와 분리된 후속 연구 설계
reproduction/    동결 source와 clean-room 구현
```

Live module의 분류와 보존 기준은 [src/tsi/README_KO.md](src/tsi/README_KO.md),
frozen source와 clean-room 자산의 취급 규칙은
[reproduction/README_KO.md](reproduction/README_KO.md)에 있다. Benchmark의 범위와
non-blind 한계는 [benchmarks/README_KO.md](benchmarks/README_KO.md)에 명시했다.
후속 연구의 지위와 분리 원칙은 [research/README_KO.md](research/README_KO.md)에
명시했다.

## 엄밀성 원칙

직관은 최종 결과가 아니다. 각 주장은 적용 영역과 상태를 부여하고, 정의·정리·증명
또는 경험 가설로 명확히 분류한다. 증명되지 않은 주장은 열린 문제로 표시하고,
counterexample과 경험 검증 의무를 분리한다.

## 이중언어 원칙

모든 논문은 동일한 절 순서, formal environment 순서, citation key와 byte-identical
bibliography를 갖는 영문·한국어 TeX 쌍으로 유지한다. Checker는 bibliography에 없는
citation과 미인용 bibliography entry도 거부한다.

```bash
PYTHONPATH=src python3 tools/check_bilingual_parity.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 주장 범위

경험 증거는 선언된 합성 world family에 한정된다. Real-world validity,
unconstrained neural structure discovery, universal model superiority, 외부 독립 재현을
확립하지 않는다. Paper 03과 04가 공유하는 endpoint는 하나의 동결 분석에 속하며 독립
증거가 아니다.

## 라이선스와 인용

소스 코드는 MIT, 데이터와 문서는 CC BY 4.0이다. `LICENSE`, `LICENSE-DATA.md`,
`CITATION.cff`를 참조한다. Paper 03/04 증거를 정확히 재사용할 때는 버전 DOI
`10.5281/zenodo.22004526`을 인용한다.
