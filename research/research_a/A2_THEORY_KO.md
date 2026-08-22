# Research A2 이론: Search Width, Noise 및 양방향 범위

## 1. 목적과 주장 경계

Research A2는 A1에서 발견한 finite-sample advantage가 generic search width와
training noise에 대해 견고한지, 그리고 두 model class 사이에서 representability를
의도적으로 교환하면 그 advantage의 방향이 반전되는지 묻는다. A1 estimand는 변경하지
않는다. Misspecification 실험은 범위 및 반증 audit이며 실패한 efficiency gate를 구제할
수 없다.

모든 연산은 field \(\mathbb F_7\)에서 수행한다. State는
\(x=(x_0,\ldots,x_4)\), action은 \(a=(a_0,\ldots,a_4)\)이고, 모든 world에서
두 arm에 동일한 two-source graph를 제공한다. Held-out outcome은 two-action
composition transition이다.

## 2. Candidate-width 구성

A1 generic dictionary는 11개 observable feature, 즉 5개 direct action coordinate와
두 graph edge 각각의 3개 head feature를 가진다. Research A2는 이 11개 feature를 앞
11개 position에 그대로 유지한다. 그 뒤 다음 modular monomial을 lexicographic order로
추가한다.

\[
  \nu_{rjd}(x,a)=a_r x_j^d,
  \qquad r,j\in\{0,\ldots,4\},\quad d\in\{1,2\}.
\]

따라서 55, 100, 300-position search는 각각 11, 20, 60개 feature를 5개 output
coordinate와 교차한다. 모든 search는 정확히 7번의 nonzero-coefficient greedy move를
수행한다. Typed class와 fitting rule은 바뀌지 않는다.

### 명제 1: Representability 유지

A1에서 표현 가능한 모든 generating function은 각 A2 width dictionary에서도 동일한
7개 output-feature-coefficient term으로 표현 가능하다.

**증명.** 모든 width의 첫 11개 feature는 A1 dictionary와 항별로 동일하다. 7개 true
term은 이 position만 사용한다. Feature 추가는 기존 term을 제거하거나 변경하지 않는다.
따라서 동일한 7-term coefficient vector가 세 width 모두에서 유지된다. \(\square\)

### 명제 2: Projective duplicate nuisance feature 부재

각 선언 width와 30개 graph 각각에 대해 포함된 feature는 nonzero function이며, 동일한
action support를 가진 어떤 두 feature도 모든 \(7^5\) state에서 nonzero scalar
multiple 관계가 아니다.

**증명 및 실행 가능한 전수검사.** 서로 다른 action support는 action coordinate로
분리된다. 고정 support에서 nuisance state factor는 \(x_j\)와 \(x_j^2\)이다. 서로 다른
coordinate는 독립적으로 변한다. 같은 coordinate에서 \(x\)와 \(x^2\)가
\(\mathbb F_7\) 위의 함수로 scalar multiple일 수 없다. \(x=1\)의 등식은 scalar를
1로 고정하지만 \(x=2\)에서는 \(2=4\)를 요구하기 때문이다. Direct constant와 선언된
세 head function도 서로 다르다. `audit_width_feature_libraries()`는 각 graph에서
16,807개 state를 모두 소진하여 전체 projective signature set을 검증한다. 따라서
analytic argument와 구현 feature order의 일치를 sampling 없이 확인한다. \(\square\)

따라서 width 축은 search multiplicity를 측정하며 불균등한 representational truth
class를 비교하지 않는다.

## 3. Training-noise 경계

Deterministic center를 probability \(1-p\)로 정확히 관측하고, probability \(p\)로
나머지 6개 field value 중 하나를 균등하게 관측한다고 하자. Center가 coordinate-wise
unique mode인 필요충분조건은 다음과 같다.

\[
  1-p > \frac{p}{6}
  \quad\Longleftrightarrow\quad
  p < \frac{6}{7}.
\]

선언 probability \(0.08,0.30,0.60,0.80\)은 모두 \(6/7\)보다 작다. Mode gap
\((1-p)-p/6\)은 단조 감소하고 \(p=0.80\)에서 \(1/15\)가 된다. 한 world 안의 모든
level은 동일 clean row, uniform random variable 및 nonzero shift를 사용한다. 같은
uniform을 threshold하므로 corruption mask가 중첩된다. 따라서 level 간 차이는 서로
다른 action coverage 때문에 발생하지 않는다.

이 명제는 population boundary를 식별하지만 finite-sample guarantee는 아니다.
\(6/7\) 근처에서는 가장 큰 선언 prefix에서도 어느 selector도 center를 복원한다고
보장할 수 없다.

## 4. Fourth-family 분리

\(u=1+x_t\), \(v=x_s\)라 쓰고 active source action을 생략한다. 세 typed edge
factor와 네 번째 factor는 다음과 같다.

\[
  L=u,\qquad Q=u^2,\qquad S=u+v,\qquad C=u^3.
\]

### 명제 3: Cubic은 typed family union 밖에 있다

\(\mathbb F_7\)의 nonzero coefficient에 대해 \(C\)는 \(L,Q,S\)의 모든 scalar
multiple과 49개 ordered \((u,v)\) pair 중 각각 최소 28, 35, 42개에서 다르다.

**증명.** \(a,b\ne0\)에서 \(au^3=bu\)는 \(u(au^2-b)=0\)이다. 가능한 \(u\)는 최대
3개이므로 일치 pair는 최대 21개다. \(au^3=bu^2\)는 \(u^2(au-b)=0\)이고 가능한
\(u\)는 최대 2개이므로 일치는 최대 14개다. \(au^3=b(u+v)\)에서는 각 고정 \(u\)마다
정확히 하나의 \(v\)가 있으므로 정확히 7개가 일치한다. 따라서 disagreement lower
bound는 28, 35, 42다. 모든 nonzero coefficient pair의 전수검사는 각 bound가
달성됨을 확인한다. \(\square\)

### 명제 4: Quadratic은 alternative generic span 밖에 있다

\(Q=u^2\)는 direct action factor, \(L\), \(C\), \(S\)의
\(\mathbb F_7\)-linear span에 속하지 않는다.

**증명.** \(S=u+v\)의 nonzero coefficient는 \(v\) dependence를 만들지만 \(Q\)에는
그 dependence가 없으므로 exact equality에서는 그 coefficient가 0이어야 한다. 남은
등식은 모든 \(u\in\mathbb F_7\)에 대해
\(u^2=d+\alpha u+\beta u^3\)이다. 양변 차이는 degree가 최대 3인데 root가 7개이므로
zero polynomial이어야 한다. 그러나 \(u^2\) coefficient가 \(-1\)이므로 모순이다.
모든 \(7^4\) coefficient vector와 49개 state pair의 전수검사는 최소 28개
disagreement를 찾아 구현 catalog가 증명과 일치함을 확인한다. \(\square\)

## 5. 대칭 Misspecification estimand

Typed catalog는 \((L,Q,S)\), alternative generic catalog는 \((L,C,S)\)다. 둘 다
11개 feature와 55개 output-feature position을 가진다.

- Typed-misspecified condition의 모든 family pair는 \(C\)를 포함한다.
- Generic-misspecified condition에서는 정렬된 pair의 모든 \(C\)를 \(Q\)로 바꾼다.
- Paired world는 graph, multiplier, edge coefficient, state, action, corruption uniform,
  shift를 공유한다.
- Matched condition은 두 arm 모두에 \((L,Q,S)\)를 준다.

따라서 유효한 scope 결과는 matched equivalence, typed misspecification에서 generic
superiority, generic misspecification에서 typed superiority를 모두 요구한다. 한 방향
결과만 얻으면 scope gate는 실패한다.

## 6. 이 이론이 증명하지 않는 것

위 명제들은 retained support, feature separation, noise-mode boundary 및
representability reversal을 증명한다. Finite-sample advantage, typed structure의
universal superiority 또는 natural image에서의 validity는 증명하지 않는다. 첫 번째는
A2의 선언 synthetic population에서 empirical하게 검정하며 뒤의 두 항목은 주장 범위
밖에 둔다.
