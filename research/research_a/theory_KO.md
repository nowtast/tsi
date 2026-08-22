# Research A 이론: 유한 표본 Support Recovery

## 1. 지위와 범위

이 문서는 Research A 설계의 충분 finite-sample envelope를 유도한다. 동결 연구 결과,
lower bound 또는 empirical crossing point를 보고하지 않는다. 결과는 모든 arm에 graph를
제공하는, 선언된 modulo-7 primitive-intervention population에만 적용된다.

증명은 다음 세 문제를 분리한다.

1. seven-term support가 알려진 경우의 coefficient estimation
2. 9개 typed family support 안에서의 support-and-coefficient recovery
3. 55-position generic dictionary에서의 seven-step greedy recovery

## 2. Population과 selector

\(\mathbb F_7\)을 원소가 7개인 field라 하자. Training row의 state
\(S\in\mathbb F_7^5\)는 uniform하게 표집한다. Primitive action
\(A\in\mathbb F_7^5\)의 active coordinate \(J\)는
\(\{1,\ldots,5\}\)에서 uniform하고, active magnitude는 \(\{1,2\}\)에서
uniform하며, 나머지 action coordinate는 0이다. State와 action은 독립이다.

제공된 graph의 target을 \(t\), 서로 다른 두 source를 \(r_1,r_2\)라 하자. Source
\(r\)에 대해 다음을 정의한다.

\[
\begin{aligned}
h_{r,L}(S,A)&=A_r(1+S_t),\\
h_{r,Q}(S,A)&=A_r(1+S_t)^2,\\
h_{r,S}(S,A)&=A_r(1+S_r+S_t).
\end{aligned}
\]

모든 연산은 \(\mathbb F_7\)에서 수행한다. 참 transition increment는 각 output의
direct term \(m_jA_j\) 5개와 output \(t\)의 edge term
\(c_kh_{r_k,F_k}\) 2개를 포함한다. 따라서 true support size는 \(s=7\)이다. 모든
true coefficient는 nonzero다. Generic dictionary는 direct feature 5개와 제공된 각
source의 head feature 3개를 가지므로 feature는 11개, output-feature position은
\(p=5\times11=55\)개다.

각 output coordinate의 observation noise \(E\)는 독립이고 symmetric하다.

\[
\Pr(E=0)=1-\eta,\qquad
\Pr(E=e)=\frac{\eta}{6}\quad(e\ne0),\qquad 0\leq\eta<\frac67.
\]

Typed selector는 각 direct cell에서 nonzero coefficient 6개, 각 edge에서 18개
family-coefficient 선택지에 대한 coordinatewise empirical zero-one loss를 최소화한다.
이 factorization은 9개 ordered head-family support를 검색하는 것과 같지만
\(9\cdot6^7=2{,}519{,}424\)개 joint parameterization을 직접 열거하지 않는다.
Isomorphic generic control은 generic feature coordinate에서 정확히 같은 검색을 수행한다.
Unstructured selector는 55개 position에서 nonzero-coefficient move를 greedy하게 7번
추가한다.

## 3. 식별 가정

이 finite modular 설계에서는 실수값 restricted-eigenvalue condition의 역할을 다음과
같이 명시한다.

- **A1 (독립 row):** training row는 위 population에서 독립적으로 표집한다.
- **A2 (공통 정보):** 모든 arm은 동일 row와 동일하게 제공된 graph를 받는다. 어느
  arm도 true head family 또는 coefficient를 관측하지 않는다.
- **A3 (algebraic beta-min):** 7개 true coefficient는 모두 \(\mathbb F_7\)에서
  nonzero다. 따라서 서로 다른 coefficient는 공통 feature가 nonzero일 때 불일치한다.
- **A4 (cell excitation):** direct feature는 확률 \(1/5\)로 active하며, 모든 true
  head feature는 확률 \((1/5)(6/7)=6/35\)로 nonzero다.
- **A5 (family separation):** 임의의 nonzero coefficient를 허용해도 서로 다른 두
  head family는 source coordinate가 active라는 조건 아래 state의 최소 \(5/7\)에서
  불일치한다.
- **A6 (positive noise gap):** \(\eta<6/7\)이므로 noiseless center가 유일한 mode다.

이는 population-specific identifiability condition이다. 임의의 correlated dictionary나
continuous-state regression에 관한 주장이 아니다.

## 4. Noise-risk 항등식

다음을 정의하자.

\[
\gamma(\eta)=(1-\eta)-\frac{\eta}{6}=1-\frac{7\eta}{6}>0.
\]

**보조정리 1 (risk gap).** Deterministic candidate value \(g(X)\)와 true center
\(f(X)\)에 대해 coordinatewise zero-one risk는 다음을 만족한다.

\[
R(g)-R(f)=\gamma(\eta)\Pr\{g(X)\ne f(X)\}.
\]

**증명.** \(g(X)=f(X)\)이면 두 loss가 같다. 두 값이 다르면 true center의 loss는
\(\eta\), 고정된 wrong value의 loss는 \(1-\eta/6\)이다. 조건부 차이는
\(\gamma(\eta)\)이며, \(X\)에 대해 expectation을 취하면 항등식을 얻는다.
\(\square\)

## 5. Known-support coefficient 경계

Direct coefficient의 informative row 확률은 \(1/5\)이다. Edge coefficient의 경우
해당 source가 선택되고 head value가 nonzero여야 한다. 세 head family는 uniform한 관련
state의 정확히 \(1/7\)에서 0이므로 최소 informative probability는 다음과 같다.

\[
q_{\min}=\frac15\frac67=\frac6{35}.
\]

**정리 1 (known-support modal recovery).** 알려진 feature에서 empirical
zero-one-loss minimizer로 7개 coefficient를 각각 추정하자. 독립 row가 \(n\)개일 때,

\[
\Pr\{\widehat\theta\ne\theta\}
\leq
7e^{-nq_{\min}/8}
+42e^{-nq_{\min}\gamma(\eta)^2/4}.
\]

**증명.** 한 coefficient의 informative row 수를 \(M\)이라 하자. Multiplicative
Chernoff bound에 의해
\(\Pr\{M<nq_{\min}/2\}\leq e^{-nq_{\min}/8}\)이다. 이제
\(M\geq nq_{\min}/2\)를 조건으로 true coefficient와 고정된 wrong nonzero value를
비교한다. Row별 wrong-minus-true loss는 \([-1,1]\)에 속하고 보조정리 1에 의해 평균은
\(\gamma(\eta)\)이다. Hoeffding inequality로 empirical order가 뒤집힐 확률은
\(e^{-M\gamma(\eta)^2/2}\leq e^{-nq_{\min}\gamma(\eta)^2/4}\)이다.
Coefficient 7개와 더 큰 후보 집합 \(\mathbb F_7\) 안의 wrong value 6개에 union bound를
적용하면 표시한 경계를 얻는다. \(\square\)

구현은 confirmatory selector가 zero를 제외하더라도 의도적으로 six-value union을
사용한다. 따라서 이는 보수적인 envelope다.

## 6. Typed support-selection 경계

**보조정리 2 (family separation).** Nonzero \(a,b\in\mathbb F_7\)에 대해 서로 다른
두 scaled head family는 관련 state의 최대 \(2/7\)에서만 일치한다.

**증명.** \(u=1+S_t\)라 하자. Scaled linear와 quadratic pair의 등식
\(au=bu^2\)은 \(u=0\)과 최대 하나의 nonzero value에서만 성립한다. Target-only
family와 source-target family는 고정된 각 \(S_t\)에 대해 최대 하나의 \(S_r\)에서만
일치하므로 state pair의 최대 \(1/7\)에서 일치한다. 같은 family의 서로 다른
coefficient는 그 feature가 0인 곳에서만 일치하며 이 역시 \(1/7\) event다. 최악의
agreement는 \(2/7\)이므로 disagreement는 최소 \(5/7\)이다. \(\square\)

**정리 2 (factorized typed ERM).** Coordinatewise typed selector가 direct
coefficient 전체, 두 head family 및 두 edge coefficient를 모두 회수하지 못할 확률은
다음 이하이다.

\[
25e^{-n\gamma(\eta)^2/50}
+34e^{-n\gamma(\eta)^2/98}.
\]

**증명.** Wrong direct coefficient는 해당 action coordinate가 선택될 때마다 truth와
다르므로 disagreement probability는 \(1/5\)다. Direct cell 5개마다 wrong nonzero
choice가 5개이므로 비교는 25개다. Edge에서는 보조정리 2와 source activation에 의해
최소 disagreement가 \((1/5)(5/7)=1/7\)이다. 각 edge의 18개
family-coefficient candidate 중 17개가 wrong이므로 비교는 34개다. Local comparison의
row별 wrong-minus-true loss는 \([-1,1]\)에 속한다. 보조정리 1과 Hoeffding inequality로
exponent는 각각 \(-n\gamma^2(1/5)^2/2=-n\gamma^2/50\)과
\(-n\gamma^2(1/7)^2/2=-n\gamma^2/98\)이다. Union bound를 적용하면 결과를 얻는다.
\(\square\)

Typed control과 isomorphic generic control은 동일 function을 동일 loss와 tie-breaking으로
열거하므로 fitted function도 row별로 같아야 한다. 이는 empirical hypothesis가 아니라
implementation invariant다.

## 7. Unstructured greedy 경계

7개 true move의 임의의 proper subset에서 primitive action은 남은 mechanism cell을
서로 분리한다. 빠진 direct term을 추가하면 population loss는
\((7/35)\gamma\), 빠진 edge term을 추가하면 \((6/35)\gamma\)만큼 개선된다.

Incorrect dictionary move는 같은 action coordinate를 사용해 빠진 term 하나를 부분적으로
모방할 수 있다. 유용한 agreement는 해당 action을 조건으로 최대 \(2/7\)이며,
quadratic-versus-direct 비교에서 이 최대 agreement count가 나온다. 추가 harm을 무시하면
incorrect move의 improvement에 대한 보수적 상한은 \((2/35)\gamma\)이다. 따라서 최소
correct-versus-incorrect improvement margin은 다음과 같다.

\[
\mu(\eta)=\frac4{35}\gamma(\eta).
\]

**정리 3 (seven-step generic greedy recovery).** 선언한 generic dictionary와
deterministic tie-breaking에 대해 7개 true move 회수 실패 확률은 다음 이하이다.

\[
(2^7-1)(55)(6)
\exp\left(-\frac{n\mu(\eta)^2}{8}\right)
=41910\exp\left(-\frac{n\mu(\eta)^2}{8}\right).
\]

**증명.** True move의 임의의 proper subset과 임의의 candidate
output-feature-coefficient move를 생각하자. 그 empirical improvement를 남아 있는 true
move와 비교한다. 두 move가 서로 다른 output에 영향을 줄 수 있으므로 row별 두
improvement의 차이는 \([-2,2]\)에 속하고 population mean은 최소 \(\mu(\eta)\)다.
Hoeffding inequality는 order reversal을 \(e^{-n\mu^2/8}\)로 제한한다. Proper true
subset은 \(2^7-1=127\)개이며, 각 subset의 move는 최대 \(55\times6=330\)개다. Union
bound에 의해 모든 greedy choice가 동시에 correct하다. 7단계 induction으로 exact
recovery를 얻는다. \(\square\)

## 8. 수치 audit와 해석

Draft training noise \(\eta=0.08\)과 failure level \(\delta=0.05\)에서 executable
sufficient envelope는 다음과 같다.

| 절차 | 충분한 \(n\) | 해당 \(n\)에서 계산한 bound |
|---|---:|---:|
| Known-support coefficient estimation | 263 | 0.02895 |
| Typed support-and-coefficient ERM | 861 | 0.02484 |
| Unstructured seven-step greedy search | 10,163 | 0.04996 |

이 값은 보수적인 sufficient upper envelope다. 두 값의 차이는 minimax lower bound가
아니며 generic selector가 더 많은 sample을 필요로 한다는 것을 증명하지 않는다. 특히
\(\log\binom{55}{7}\)을 point crossing 예측에 사용하지 않는다. Envelope는 candidate
grid를 12,800 row까지 확장할 근거만 제공한다. Sealed experiment에서는 동결한 SESOI,
multiplicity 및 stopping rule 아래 advantage-to-equivalence transition을 추정해야 한다.

이 well-specified population에서는 exact support recovery가 held-out composition의
deterministic center 일치를 함의한다. 그러나 selector failure, 유한 test sample 및
robustness population에서는 그 함의가 성립하지 않을 수 있으므로 composition NLL은 별도
empirical estimand로 유지한다.

## 9. 증명 경계

정리는 wrong supplied graph, dictionary에 없는 family, continuous state, 임의 action
policy, perceptual entity discovery 또는 dependent row를 다루지 않는다. Dictionary
misspecification과 candidate-width expansion은 별도 robustness study여야 한다.
`preregistration_draft.json`의 unresolved design field를 동결하기 전에는 confirmatory
seed를 생성할 수 없다.
