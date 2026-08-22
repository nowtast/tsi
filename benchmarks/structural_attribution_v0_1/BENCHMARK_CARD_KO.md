# Benchmark Card

## Population

Attribution cohort는 독립 finite world 120개로 구성된다. 각 world는 modulo 7인 다섯
coordinate, 30개 two-edge motif 중 하나, 선언한 세 family에서 선택한 edge head 두
개를 갖는다. Training은 primitive-action transition 800개, selection은 분리된
transition 400개, evaluation은 held-out two-coordinate-action transition 1,200개를
사용한다.

## Task와 estimand

1. Train과 selection data만으로 common-target graph와 head family 두 개를 회수한다.
2. Held-out transition target을 예측하고 world-level composition NLL을 보고한다.

Reference graph effect는 wrong-routing minus correct-routing NLL이며 universal model
ranking이 아니다. Stress cohort는 training에서 dormant이고 지정 test composition에서
활성화되는 nonadditive term을 가진 별도 120-world family다.

## 정보 대칭

Attribution을 위해 비교하는 method는 동일 observable state, action, partition과
corruption process를 받아야 한다. 서로 다른 inductive bias를 가질 수 있지만 한 arm에만
identity, graph answer, typed relation, test target 또는 reference output을 제공해서는
안 된다.

## 평가 단위

독립 단위는 world다. Transition은 nested observation이며 inferential interval에서
독립 sample로 취급하지 않는다. Hyperparameter 및 model selection은 test-target access
전에 끝나야 한다.

## Reference 범위

보관한 reference arm은 correct/wrong graph 아래 factorized typed, matched
seven-parameter generic sparse와 55-parameter generic dense head다. 이 값은 integrity
anchor이며 미래 method가 특정 ranking을 재현해야 한다는 뜻이 아니다.

## 한계

- Answer가 공개되어 v0.1로 blind leaderboard를 주장할 수 없다.
- Entity discovery, cross-time identity, visual input, continuous state와 real-world
  validity가 없다.
- Reference cohort는 source family를 공유하며 외부 또는 연구실 간 replication이 아니다.
- Submission policy 선언은 형식만 machine-check하며 진실성을 audit하지 않는다. 독립
  실행이 별도로 필요하다.
