# 1단계 Proof Audit

## 현재 범위

통합 원고는 structural equivalence, survivor-level composition closure, 여섯
defect criterion, topology·metric-measure·category·dynamics 확장 결과와
quotient-level rollout bound를 포함한다.

## 완료한 검사

- 정의에서 사용한 symbol을 preamble 또는 local section에서 선언했다.
- Partial transition의 tagged extension을 사용 전에 명시했다.
- `\Sigma'`의 target simplicial complex를 `L`로 일관되게 표기했다.
- Structural discrepancy를 equivalence class 위의 extended pseudometric으로
  정의하고 구체적인 discrete metric instance를 제시했다.
- Composition proof에서 각 layer equality를 restriction하기 전에 domain, intermediate
  survivor, image를 정의했다.
- Rollout proof는 triangle inequality, one-step error bound, true-update Lipschitz
  assumption만 사용한다.
- One-step counterexample은 rollout theorem의 assumption과 분리했다.

## Stage 2 감사 원칙

외부에서 가져온 각 Stage 2 theorem은 통합 원고에 편입하기 전에 검증했다.
Persistent topology, metric-measure discrepancy, categorical descent,
action-conditioned dynamics와 turnover는 정확한 assumption, proof obligation,
경계와 counterexample을 아래에 기록한 뒤에만 편입했다.

## Stage 2A 지속적 위상 감사

- 유한 담체, 체, 단조 여과, 하위수준 복합체, persistence diagram, 절단 서명을 사용 전에 정의했다.
- 여과 안정성 결과는 외부에서 도입한 정리임을 명시했다. 가정은 고정 담체, 공통 체, 단조 실수값 여과, 공통 단위이다.
- 증명에서는 두 하위수준 포함을 직접 확인하고 interleaving을 구성했으며, 외부 의존성은 algebraic stability 정리 하나로 제한했다.
- 여과를 보존하는 단순 복합체 동형 명제는 각 수준의 호몰로지 동형과 가환하는 지속 모듈 도식으로 증명했다.
- 두 isolated vertex의 label은 고정한 채 filtration value를 교환하면 persistence
  diagram은 같지만 filtration value를 보존하는 label-preserving bijection은
  존재하지 않음을 반례로 확인했다.
- 완전성, 대응 없는 담체 간 비교, 신경망의 담체 발견은 증명하지 않는다는 경계를 명시했다.

게이트 결과: 외부 정리 의존성과 labeled-carrier 불완전성 경계를 기록하고
Stage 2A topology를 영문·국문 통합 원고에 편입했다.

## Stage 2B metric-measure 감사

- 유한 metric-measure 상태, 확률 정규화, full support, label map을 coupling objective보다 먼저 정의했다.
- coupling의 marginal 조건과 hard label compatibility를 명시했으며, label mass profile이 맞지 않으면 discrepancy를 무한대로 둔다.
- zero discrepancy 증명에서 유한 coupling polytope의 compactness, 연속성, full-support에 의한 support coverage, 비음수 항, support를 bijection graph로 만드는 대각 논리를 각각 확인했다.
- 결과 bijection이 거리, label, 질량을 보존함을 별도로 보였다.
- label mass equality 필요조건은 coupling marginal의 합으로 증명했다.
- sampling robustness, empirical convergence rate, 임의 담체에서 coupling 존재, coordinate loss 우월성은 주장하지 않는다고 명시했다.

게이트 결과: full-support와 hard-label assumption을 기록하고 Stage 2B
metric-measure 확장을 영문·국문 통합 원고에 편입했다.

## Stage 2C categorical descent 감사

- finite quiver, free path category, path congruence, presented schema, generator realization을 descent보다 먼저 정의했다.
- finite-relation category 명제에서 existential relational composition으로 associativity와 두 identity law를 직접 확인했다.
- descent 정리는 양방향으로 증명했다. factorization은 equation preservation을 함의하고, equation preservation은 smallest path congruence를 통해 quotient functor를 만든다.
- 모든 quotient morphism이 path representative를 가지므로 uniqueness가 성립한다.
- natural-equivalence generator criterion은 relation witness의 transport와 path-length induction으로 증명했다.
- equation을 확인하지 않은 generator agreement는 categorical validity가 아니며, schema discovery도 주장하지 않는다고 명시했다.

게이트 결과: equation-preservation 경계를 기록하고 Stage 2C categorical
descent를 영문·국문 통합 원고에 편입했다.

## Stage 2D dynamics 감사

- finite action-prefix category와 tracked action-history functor를 trajectory 주장보다 먼저 정의했다.
- one-step action edge로부터 existence, functoriality, uniqueness를 주는 unique-extension theorem을 증명했다.
- trajectory preservation은 앞서 증명한 composition-closure theorem을 반복 적용해 도출했다.
- topology, geometry, relation, turnover, filtration, mass defect를 유한
  survivor structure 위에서 정의했다.
- 여섯 zero condition과 그 결합이 full integrated structural isomorphism과
  동치임을 별도로 증명했다.
- empty tracking proposition으로 survivor-only preservation의 vacuity를 기록하고 turnover를 별도 양으로 둘 근거를 제시했다.
- endpoint만으로 temporal identity를 추론하거나 causal counterfactual 및 unconstrained mechanism discovery를 주장하지 않는다고 명시했다.

게이트 결과: Stage 2D dynamics를 편입하여 Stage 2A~2D 수학적 확장을
완료했다. Paper 03과 04는 제한된 경험적 조작화와 형식 state 중 검정하지 않은
부분을 기록한다.

## 리뷰어 수정 감사 (2026-08-11)

- Schema는 generator보다 먼저 finite quiver와 path equation으로 표시된다.
- State에 filtration f, full-support mass mu, relation-to-simplex 양립성을 명시했다.
- Structural isomorphism은 label, simplex, distance, filtration, mass, generator relation을 보존한다.
- Tracked transition은 label-preserving partial injection이며 category composition을 정의했다.
- Zero criterion은 topology, geometry, relation, turnover, filtration, mass의 여섯 양이다.
- Rollout discrepancy는 extended pseudometric이며 Lipschitz와 quotient-equivariance를 별도 명제로 분리했다.
- Running example에서 비공허성, 양립성, 불변성, survivor defect와 turnover의 차이를 계산했다.
- 표준 imported mathematics와 framework-specific construction을 구분했다.

참고문헌 보정 전 빌드 상태(2026-08-11): 영문 PDF 15쪽, 한국어 PDF
14쪽(XeLaTeX). 이 과거 page count는 아래 감사 결과로 대체된다.

## 참고문헌 귀속 감사 (2026-08-20)

- 원고에 related-work 및 positioning 절과 실제 인용된 참고문헌 20개를 추가했다.
  미인용 bibliography entry와 bibliography에 없는 citation key는 없다.
- Finite-relation 및 presented-schema 재료를 relation calculus와 functorial data
  문헌에 귀속했다. TSI의 `FinRel`-valued carrier 및 추가 structural layer와
  set-valued categorical database instance의 차이를 명시했다.
- Filtered-stability 증명에서 외부 의존성을 실제 사용 지점에 표시했다. Algebraic
  stability를 적용하기 전에 유한 담체로부터 pointwise finite-dimensionality와
  q-tameness가 성립함을 확인했다.
- Hard-label discrepancy를 constrained Gromov--Wasserstein-type objective로
  위치시켰다. Exactness 결과는 원고 내부에서 증명하고 coupling 및 gluing 배경은
  원전에 귀속했다.
- Quotient rollout 결과를 reward 및 transition으로 정의되는 bisimulation metric과
  policy-performance claim에서 명시적으로 분리했다.
- 영문·국문의 citation key를 구조적으로 검사하고 두 `references.bib`가 byte-identical
  이어야 통과하도록 했다. 누락, 불일치, 미인용 key가 있으면 checker가 실패한다.

게이트 결과: 참고문헌 부재 blocker를 해소했다. Clean build 결과는 영문 18쪽,
국문 17쪽이다.
