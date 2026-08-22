# Research A: Structured Support-Recovery Efficiency

Research A는 두 arm이 동일 observable transition을 받고 동일 graph를 조건으로 사용할
때 typed support restriction이 finite-sample support recovery와 held-out composition
prediction을 개선하는지 검정한다.

이 directory는 design workspace이며 frozen preregistration이 아니다. Theory,
estimand, SESOI, multiplicity family, power calculation과 stopping rule이 review를
통과하기 전에는 confirmatory seed를 생성하거나 sealed experiment를 실행할 수 없다.

- `theory.md`: finite-sample envelope의 영문 증명
- `preregistration_draft.json`: machine-readable design draft
- `theory_KO.md`: 동일 구조의 한국어 증명
- `DESIGN_HISTORY_KO.md`: development grid 수정 및 power 근거
- `ROBUSTNESS_PLAN_KO.md`: A1 confirmatory seed 전에 기록한 A2 방향
- `RESULTS_KO.md`: 봉인 A1 판정, simultaneous interval 및 범위 경계
- `EXTERNAL_REPLAY_KO.md`: package 무결성, 실행 및 독립성 규칙

Development pilot은 `experiments/research_a_v1/development_report.json`에
기록한다. 이 자료는 power와 decision threshold를 보정할 수 있지만 confirmatory
cohort에 들어갈 수 없다.

주 비교는 notation 대 notation이 아니다. Generator-compatible edge-family support 9개를
대상으로 하는 structured search와 55개 output-feature position을 대상으로 하는
unstructured seven-step search의 비교다. Typed arm과 동률이어야 하는 isomorphic generic
nine-support control을 포함해 notation과 search restriction을 분리한다.
