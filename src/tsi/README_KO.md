# TSI Python 모듈

이 디렉터리는 공개 이론 audit와 Paper 03/04 증거 workflow에 사용되는 live Python
package다. 하나의 production model만 담은 폴더가 아니다. 수학적 참조 객체, 동결 또는
prospective 연구 실행 코드, 개발 과정과 부정 결과를 보존하는 코드가 함께 있다.

## 모듈 분류

| 분류 | 파일 | 역할 |
|---|---|---|
| 구조 이론 참조 구현 | `structured_space.py`, `topological.py`, `geometric.py`, `geometric_validation.py`, `relational.py`, `dynamical.py`, `coherent.py`, `order_topology.py`, `labeled_topology.py`, `metric_graph.py`, `attribute_geometry.py`, `coherence_spectrum.py`, `bridge_repair.py` | 이론 기술보고서와 관련된 유한 구성 및 실행 가능한 audit |
| Paper 03 역사적 경로 | 최종 `paper34_*` 계열을 제외한 `paper3_*` | preregistration, 봉인 OOD/rollout/validity workflow, 보조 benchmark, ablation 및 개발 경로 보존 |
| Paper 03/04 prospective resolution | `paper34_*` | 최종 graph-information/head-factorization cohort, multiplicity audit, noise sensitivity, retrospective power, world-derivation audit |
| Paper 04 귀속 및 stress 연구 | `paper4_*` | comparator, capacity matching, misspecification, diagnostic, statistical analysis |
| 출판 정합성 검사 | `paper_parity.py` | 영문·한국어 TeX 구조와 display-math parity |
| 공개 benchmark 지원 | `structural_attribution_benchmark.py` | Artifact 무결성, leakage-reduced participant view 및 portable submission 검사 |
| 후속 Research A | `research_a_*` | 충분 envelope, selector audit, development-only calibration, freeze-candidate contract, 외부 custodian seed 검증, confirmatory 구성 및 analysis |

`paper3_learned_*`에는 탐색 단계에 머물거나 실패한 neural/routing 설계도 포함된다.
이 파일들의 존재는 해당 설계를 확증 증거로 승격하지 않으며 unconstrained neural
structure discovery 주장도 뒷받침하지 않는다. 테스트와 기록된 결과가 최종 주장 경계에
도달한 과정을 문서화하므로 보존한다.

## 실행 방법

사용자용 명령은 `tools/`에 있다. `src/tsi`의 대부분은 command-line entry point가
아니라 import 가능한 모듈이다. 저장소 루트에서 다음과 같이 실행한다.

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Paper 03/04의 정확한 절차는 `REPRODUCE_KO.md`를 따른다. 설치 확인만을 목적으로
one-shot 또는 sealed-access tool을 다시 실행해서는 안 된다. 기존 lock과 ledger는
증거 artifact다.

## Live source와 frozen source

이 폴더의 파일은 유지보수되는 live 구현이다. Prospective confirmatory run에 실제로
사용된 byte-preserved source는
`reproduction/frozen_source/paper34_resolution_v1`에 있다. Live 모듈에 reporting 또는
audit 변경이 생겨도 frozen copy는 그대로 유지한다.

## 보존 기준

다른 모듈, 공개 tool, test, artifact/freeze manifest 또는 문서화된 재현 경로가 요구하는
모듈은 공개 tree에 남긴다. Test에서만 사용하는 개발 모듈은 관련 test와 provenance
artifact를 함께 검토하여 어떤 원고나 보존 증거 주장도 의존하지 않음을 확인한 뒤에만
제거할 수 있다. 이러한 제거는 묵시적으로 수행하지 않는다.
