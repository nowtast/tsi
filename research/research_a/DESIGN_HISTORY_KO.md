# Research A1 설계 이력

## 최초 grid

첫 development run은 36개 world와 50부터 12,800까지의 training size를 사용했다.
모든 size와 모든 world에서 typed, isomorphic 및 unstructured arm이 exact support를
회수했다. 따라서 최초 grid에는 ceiling defect가 있었고 finite-sample efficiency를
식별할 수 없었다.

## 수정 grid

같은 공개 development seed label을 사용하는 두 번째 development-only run에서 size
5, 8, 10, 12, 15, 20, 25, 30, 40, 50을 평가했다. 전이 구간이 관측되었다. Size
15에서 typed exact recovery는 0.556, generic exact recovery는 0.194였고,
generic-minus-typed composition NLL은 0.25888이었다. Size 40에서는 두 exact-recovery
rate가 0.972였으며 paired outcome이 같았다.

Confirmatory grid는 5, 10, 15, 20, 25, 30, 40, 50으로 고정했다. 이는 투명하게
기록한 development-driven design choice이며 confirmatory result가 아니다.

## Power와 해석

Prospective power calculation은 9개 ordered family pair별 stratified bootstrap을
사용하며 36개 low-grid development world만 resample한다. Confirmatory world 126개는
family-pair stratum마다 정확히 14개를 배정한다. 하나 이상의 joint advantage 뒤에
더 큰 size의 joint equivalence가 나타나는 사건의 추정 power는 1.0이었다. 작은
development cohort 때문에 이 추정치가 낙관적일 수 있으므로 더 작은 90-world 후보 대신
126개를 유지했다.

이론값 263, 861, 10,163은 보수적인 sufficient upper envelope다. Empirical
transition 예측값이 아니므로 development transition이 훨씬 앞에서 나타난 것과 모순되지
않는다.
