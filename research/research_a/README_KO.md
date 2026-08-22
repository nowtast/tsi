# Research A: Structured Support-Recovery Efficiency

Research A는 두 arm이 동일 observable transition을 받고 동일 graph를 조건으로 사용할
때 typed support restriction이 finite-sample support recovery와 held-out composition
prediction을 개선하는지 검정한다.

Research A1은 완료되었고 external replay도 마쳤다. Research A2는 numeric draft다.
Development와 power 단계는 완료했지만 frozen 상태가 아니며 A2 confirmatory seed는
생성하지 않았다.

- `theory.md`: finite-sample envelope의 영문 증명
- `preregistration_draft.json`: machine-readable design draft
- `theory_KO.md`: 동일 구조의 한국어 증명
- `DESIGN_HISTORY_KO.md`: development grid 수정 및 power 근거
- `ROBUSTNESS_PLAN_KO.md`: A1 confirmatory seed 전에 기록한 A2 방향
- `RESULTS_KO.md`: 봉인 A1 판정, simultaneous interval 및 범위 경계
- `EXTERNAL_REPLAY_KO.md`: package 무결성, 실행 및 독립성 규칙
- `A2_THEORY_KO.md`: width, noise boundary 및 양방향 separation 증명
- `A2_PREREGISTRATION_DRAFT_KO.md`: review-ready A2 numeric design
- `research_a2_preregistration_draft.json`: machine-readable A2 draft
- `A2_CLEANROOM_REPLAY_SPEC_KO.md`: independent-software replay 필수 동작

Development pilot은 `experiments/research_a_v1/development_report.json`에
기록한다. 이 자료는 power와 decision threshold를 보정할 수 있지만 confirmatory
cohort에 들어갈 수 없다.

A2 development 및 power artifact는 `experiments/research_a2_v1/` 아래에 있다. 이들은
nonconfirmatory이며 이후 A2 cohort와 합칠 수 없다.

주 비교는 notation 대 notation이 아니다. Generator-compatible edge-family support 9개를
대상으로 하는 structured search와 55개 output-feature position을 대상으로 하는
unstructured seven-step search의 비교다. Typed arm과 동률이어야 하는 isomorphic generic
nine-support control을 포함해 notation과 search restriction을 분리한다.
