# 1단계 Proof Audit

## 현재 범위

통합 초안은 structural equivalence, survivor-level layer preservation의
composition closure, quotient-level action-conditioned rollout bound의 세 가지
formal result를 포함한다.

## 완료한 검사

- 정의에서 사용한 symbol을 preamble 또는 local section에서 선언했다.
- Partial transition의 tagged extension을 사용 전에 명시했다.
- `\Sigma'`의 target simplicial complex를 `L`로 일관되게 표기했다.
- Structural discrepancy를 equivalence class 위의 metric으로 정의했다.
- Composition proof에서 각 layer equality를 restriction하기 전에 domain, intermediate
  survivor, image를 정의했다.
- Rollout proof는 triangle inequality, one-step error bound, true-update Lipschitz
  assumption만 사용한다.
- One-step counterexample은 rollout theorem의 assumption과 분리했다.

## 남은 audit

다음 pass에서는 Stage 2에서 가져올 theorem을 하나씩 검증한 뒤 통합 원고에 추가한다.
Persistent topology, metric-measure discrepancy, categorical descent, causal
intervention, turnover case는 정확한 assumption과 counterexample을 함께 확인한
경우에만 추가한다.

## Stage 2A 지속적 위상 감사

- 유한 담체, 체, 단조 여과, 하위수준 복합체, persistence diagram, 절단 서명을 사용 전에 정의했다.
- 여과 안정성 결과는 외부에서 도입한 정리임을 명시했다. 가정은 고정 담체, 공통 체, 단조 실수값 여과, 공통 단위이다.
- 증명에서는 두 하위수준 포함을 직접 확인하고 interleaving을 구성했으며, 외부 의존성은 algebraic stability 정리 하나로 제한했다.
- 여과를 보존하는 단순 복합체 동형 명제는 각 수준의 호몰로지 동형과 가환하는 지속 모듈 도식으로 증명했다.
- Betti 수 반례는 torus와 쐐기합의 기본군을 사용해 호몰로지 일치와 호모토피 동형을 분리한다.
- 완전성, 대응 없는 담체 간 비교, 신경망의 담체 발견은 증명하지 않는다는 경계를 명시했다.

게이트 결과: Stage 2A 위상을 외부 정리 의존성과 함께 영문·한국어 통합 원고에 편입했다. metric-measure, category, dynamics 확장은 아직 게이트 전이다.

## Stage 2B metric-measure 감사

- 유한 metric-measure 상태, 확률 정규화, full support, label map을 coupling objective보다 먼저 정의했다.
- coupling의 marginal 조건과 hard label compatibility를 명시했으며, label mass profile이 맞지 않으면 discrepancy를 무한대로 둔다.
- zero discrepancy 증명에서 유한 coupling polytope의 compactness, 연속성, full-support에 의한 support coverage, 비음수 항, support를 bijection graph로 만드는 대각 논리를 각각 확인했다.
- 결과 bijection이 거리, label, 질량을 보존함을 별도로 보였다.
- label mass equality 필요조건은 coupling marginal의 합으로 증명했다.
- sampling robustness, empirical convergence rate, 임의 담체에서 coupling 존재, coordinate loss 우월성은 주장하지 않는다고 명시했다.

게이트 결과: Stage 2B metric-measure 확장을 영문·한국어 통합 원고에 편입했다. 다음 게이트 대상은 categorical descent이다.

## Stage 2C categorical descent 감사

- finite quiver, free path category, path congruence, presented schema, generator realization을 descent보다 먼저 정의했다.
- finite-relation category 명제에서 existential relational composition으로 associativity와 두 identity law를 직접 확인했다.
- descent 정리는 양방향으로 증명했다. factorization은 equation preservation을 함의하고, equation preservation은 smallest path congruence를 통해 quotient functor를 만든다.
- 모든 quotient morphism이 path representative를 가지므로 uniqueness가 성립한다.
- natural-equivalence generator criterion은 relation witness의 transport와 path-length induction으로 증명했다.
- equation을 확인하지 않은 generator agreement는 categorical validity가 아니며, schema discovery도 주장하지 않는다고 명시했다.

게이트 결과: Stage 2C categorical descent를 영문·한국어 통합 원고에 편입했다. 다음 게이트 대상은 dynamics이다.

## Stage 2D dynamics 감사

- finite action-prefix category와 tracked action-history functor를 trajectory 주장보다 먼저 정의했다.
- one-step action edge로부터 existence, functoriality, uniqueness를 주는 unique-extension theorem을 증명했다.
- trajectory preservation은 앞서 증명한 composition-closure theorem을 반복 적용해 도출했다.
- turnover와 topology, geometry, relation defect를 유한 survivor structure 위에서 정의했다.
- 네 zero condition과 그 결합이 full integrated structural isomorphism과 동치임을 별도로 증명했다.
- empty tracking proposition으로 survivor-only preservation의 vacuity를 기록하고 turnover를 별도 양으로 둘 근거를 제시했다.
- endpoint만으로 temporal identity를 추론하거나 causal counterfactual 및 unconstrained mechanism discovery를 주장하지 않는다고 명시했다.

게이트 결과: Stage 2D dynamics를 편입했다. Stage 2A~2D 수학적 확장이 완료되었으며, 다음 작업은 empirical-validation integration이다.
## 리뷰어 수정 감사 (2026-08-11)

- Schema는 generator보다 먼저 finite quiver와 path equation으로 표시된다.
- State에 filtration f, full-support mass mu, relation-to-simplex 양립성을 명시했다.
- Structural isomorphism은 label, simplex, distance, filtration, mass, generator relation을 보존한다.
- Tracked transition은 label-preserving partial injection이며 category composition을 정의했다.
- Zero criterion은 topology, geometry, relation, turnover, filtration, mass의 여섯 양이다.
- Rollout discrepancy는 extended pseudometric이며 Lipschitz와 quotient-equivariance를 별도 명제로 분리했다.
- Running example에서 비공허성, 양립성, 불변성, survivor defect와 turnover의 차이를 계산했다.
- 표준 imported mathematics와 framework-specific construction을 구분했다.

빌드 상태: 영문 PDF 15쪽, 한국어 PDF 14쪽(XeLaTeX). 남은 작업은 proof-by-proof audit과 표기 정리다.