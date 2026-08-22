# Research A2 사전등록 초안

## 1. 상태

이 문서는 review를 위한 numeric draft이며 frozen 상태가 아니다. Confirmatory seed는
생성하지 않았다. Executable contract digest는
`c3c07798ef01406974640b743603b8e49174879b5e466c386fae249c141a967b`이다.

Research A2의 방향은 A1 confirmatory seed 전에 기록했다. Development label
`TSI-RESEARCH-A2-DEVELOPMENT-v1`과 power label
`TSI-RESEARCH-A2-PROSPECTIVE-POWER-v1`은 deterministic하며 nonconfirmatory다.

## 2. 독립 Population

각 efficiency axis는 135개 independent world를 포함한다. 각 scope condition도
135개 world를 포함한다. 이 수는 pre-A1 minimum 126 이상 후보 중 9개 matched
family-pair stratum과 5개 special-family-pair stratum 모두로 나누어지는 최소값이다.
Matched stratum당 15개, special stratum당 27개 world가 배정된다.

각 world는 두 arm에 동일 graph와 raw row를 제공한다. Training은 primitive action을
사용한다. Test는 noise probability 0.12인 1,200개 held-out two-action composition
case를 사용한다. Held-out set은 fitting이나 selection에 사용하지 않는다.

## 3. Width 축

Generic output-feature position count는 55, 100, 300이고 train prefix는 10, 15, 20,
25, 30, 40이다. 모든 dictionary는 true seven-term support를 포함하며 모든 generic
fit은 7번 greedy move를 수행한다. Typed search는 바뀌지 않는다. Training noise는
0.08이다. 11개 A1 feature 뒤의 nuisance descriptor는 state coordinate 오름차순,
degree 1 다음 2, action coordinate 오름차순으로 정렬한다. Width 100은 첫 9개를
사용한다. Width 300은 가능한 descriptor 50개 중 49개를 사용하며 \(a_4x_4^2\) 하나만
제외한다.

각 width-by-prefix cell에서 paired outcome은 generic-minus-typed composition NLL과
typed-minus-generic exact recovery다. 이름을 명시한 36개 outcome은 familywise alpha
0.05의 하나의 Bonferroni family를 이룬다. 두 simultaneous lower bound가 모두 0보다
크고 두 point estimate가 endpoint SESOI, 즉 NLL 0.01과 recovery 0.10 이상일 때만 한
cell에 joint advantage가 있다. Width gate는 모든 width에서 이런 cell이 하나 이상일
것을 요구한다.

## 4. Noise 축

Training noise probability는 0.08, 0.30, 0.60, 0.80이고 train prefix는 15, 20, 30,
40, 80, 160이다. Generic arm은 55 position과 7 move를 사용한다. 한 world 안에서 네
training stream은 clean row와 nested corruption mask를 공유한다.

동일한 두 outcome으로 이름을 명시한 48개 endpoint를 만들고 familywise alpha 0.05의
별도 Bonferroni family를 구성한다. Noise gate는 0.08, 0.30, 0.60 각각에서
joint SESOI-qualified advantage가 하나 이상일 것을 요구한다. 0.80의 여섯 sample-size
summary와 simultaneous interval은 모두 필수 boundary-stress output이지만 advantage
gate에는 들어가지 않는다. 따라서 gate 통과는 선언 population에서 0.60까지의
robustness만 지지하며 0.80 또는 임의 noise level의 robustness를 뜻하지 않는다. Gate와
무관하게 전체 degradation curve를 보고한다.

## 5. 양방향 Scope 축

Confirmatory prefix는 320이다. Matched condition은 둘 다 representable한 catalog를
비교한다. Typed misspecification의 모든 generator pair는 cubic을 포함하고 generic
catalog만 cubic을 포함한다. Generic misspecification에서는 정렬된 cubic occurrence를
quadratic으로 바꾸며 typed catalog만 quadratic을 유지한다. 각 catalog는 55 position을
가진다.

각 condition의 NLL 및 center-accuracy difference, 총 6개 outcome이 세 번째 Bonferroni
family를 이룬다. Scope gate는 다음을 모두 요구한다.

1. Matched simultaneous interval이 NLL margin 0.01과 accuracy margin 0.025 안에 있다.
2. Typed misspecification에서 두 interval upper bound가 0보다 작고 두 point estimate가
   generic을 최소 0.10만큼 지지한다.
3. Generic misspecification에서 두 interval lower bound가 0보다 크고 두 point estimate가
   typed를 최소 0.10만큼 지지한다.

Scope gate는 efficiency decision을 바꿀 수 없다.

## 6. Development-only Power

Stratified bootstrap은 36개 matched development world와 45개 scope world를 사용해
20,000 iteration을 수행했다. 135 worlds에서 all-width gate의 estimated power는 1.0,
through-0.60 conjunctive noise gate는 0.9994, conjunctive scope gate는 1.0이었다.
Noise 0.80 group power는 0.0이었으며 이를 명시적으로 보고한다. 이 결과를 숨기거나
required advantage로 바꾸지 않는다.

Development 및 power report는
`experiments/research_a2_v1/development_report.json`과
`experiments/research_a2_v1/prospective_power.json`이다. 이 자료는 confirmatory cohort에
들어갈 수 없다.

## 7. Freeze 및 실행 순서

필수 순서는 이 초안의 external review, 전체 test 및 clean-room dry run, 외부 seed
custodian 한 명을 지정한 source freeze, freeze digest의 commit과 공개 기록, 그 이후
custodian의 32-byte 단일 draw, custodian attestation 검증, 생성된 seed commitment의
commit과 push, one-shot execution이다. 저자는 confirmatory seed를 생성, 선별, 선택,
재추출 또는 교체할 수 없다. Seed commitment 이후 threshold, endpoint 또는 cohort
repair는 허용하지 않는다. 잔여 신뢰 경계는 이름을 고정한 custodian의 single-draw
attestation이 진실이라는 점이며, 그 digest, freeze 결박 및 실행 후 seed 공개는 감사할 수
있다.
