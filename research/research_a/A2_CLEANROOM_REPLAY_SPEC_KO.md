# Research A2 Clean-Room Replay 명세

## 1. 필요한 독립성

Replay 구현은 Python project module을 import하거나 Python selector source를 복사할 수
없다. Portable JSON artifact, 이 명세 및 numeric preregistration만 입력으로 사용할 수
있다. 같은 machine에서 실행하는 것은 software replay이며 independent external
replication이 아니다.

## 2. 입력과 연산

Portable artifact는 world specification, training row, held-out row 및 audit용 noiseless
center를 포함한다. 모든 연산은 modulo seven integer arithmetic이다. NLL은 matched
coordinate에 probability 0.88, 6개 mismatch 각각에 0.02를 사용한다. Exact support는
순서를 무시한 output-feature-coefficient triple을 비교한다.

## 3. Selector

Typed selector는 5개 nonzero direct coefficient를 독립적으로 추정하고 supplied edge마다
ordered declared family 중 하나와 nonzero coefficient 하나를 선택한다. Tie는 error
count, family order, coefficient order 순으로 해결한다.

Generic selector는 zero increment에서 시작한다. 정확히 7번의 각 move에서 아직 사용하지
않은 모든 output-feature position과 coefficient 1부터 6까지를 평가한다. Output,
feature, coefficient order에서 lexicographically first minimum을 선택하며 같은
output-feature position을 재사용하지 않는다.

Width feature order, nuisance descriptor order, typed catalog 및 alternative catalog는
`research_a2_features.py`와 `research_a2_preregistration_draft.json`에 선언된 순서와
정확히 같아야 한다.

## 4. 필수 Replay 출력

Replay는 모든 world-level endpoint를 absolute tolerance `1e-12` 이내에서, 모든
exact-recovery Boolean을 정확히, 세 simultaneous-interval analysis를 `1e-12` 이내에서,
모든 gate decision을 정확히 재현해야 한다. Row count, family-pair balance,
composition-stratum cycle 및 paired misspecification metadata도 검증해야 한다.

## 5. Freeze 전 Dry Run

Source freeze 전에 public fixed test seed로 만든 fixture를 end-to-end replay해야 한다.
Fixture와 audit은 developmental software test이며 confirmation에서 제외한다.
Confirmatory portable artifact는 one-shot execution 전에는 생성하지 않는다.
