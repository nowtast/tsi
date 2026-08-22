# Research A2 Theory: Search Width, Noise, and Bidirectional Scope

## 1. Purpose and claim boundary

Research A2 asks whether the finite-sample advantage found in A1 is robust to
generic search width and training noise, and whether that advantage reverses
when representability is deliberately exchanged between the two model classes.
It does not alter the A1 estimand. The misspecification experiment is a scope
and falsification audit and cannot rescue a failed efficiency gate.

All arithmetic is in the field \(\mathbb F_7\). A state is
\(x=(x_0,\ldots,x_4)\), an action is \(a=(a_0,\ldots,a_4)\), and every world
supplies the same two-source graph to both arms. The held-out outcome is a
two-action composition transition.

## 2. Candidate-width construction

The A1 generic dictionary has eleven observable features: five direct action
coordinates and three head features for each of two graph edges. Research A2
retains these eleven features in the first eleven positions. It then appends,
in lexicographic order, the modular monomials

\[
  \nu_{rjd}(x,a)=a_r x_j^d,
  \qquad r,j\in\{0,\ldots,4\},\quad d\in\{1,2\}.
\]

The 55-, 100-, and 300-position searches therefore use 11, 20, and 60
features, respectively, crossed with five output coordinates. Every search
takes exactly seven greedy nonzero-coefficient moves. The typed class and its
fitting rule do not change.

### Proposition 1: retained representability

Every generating function representable in A1 is representable in each A2
width dictionary by exactly the same seven output-feature-coefficient terms.

**Proof.** The first eleven features at every width are term-for-term identical
to the A1 dictionary. The seven true terms use only those positions. Appending
features does not remove or alter any of them. Therefore the same seven-term
coefficient vector remains available at all three widths. \(\square\)

### Proposition 2: no projective duplicate nuisance feature

For each declared width and each of the 30 graphs, the included features are
nonzero functions and no two features with the same action support are
nonzero scalar multiples over all \(7^5\) states.

**Proof and executable exhaustion.** Distinct action supports are separated by
their action coordinate. For a fixed support, the nuisance state factors are
\(x_j\) and \(x_j^2\). Distinct coordinates vary independently. On one
coordinate, \(x\) and \(x^2\) cannot be scalar multiples as functions on
\(\mathbb F_7\): equality at \(x=1\) fixes the scalar to one, while equality at
\(x=2\) would require \(2=4\). The direct constant and the three declared head
functions are likewise distinct. `audit_width_feature_libraries()` exhausts
all \(16,807\) states for every graph and verifies the complete projective
signature set. Thus the analytic argument is checked against the implemented
feature order without sampling. \(\square\)

The width axis consequently measures search multiplicity. It does not compare
unequal representational truth classes.

## 3. Training-noise boundary

Let the deterministic center be observed correctly with probability \(1-p\).
With probability \(p\), one of the other six field values is selected uniformly.
The center is the unique coordinate-wise mode exactly when

\[
  1-p > \frac{p}{6}
  \quad\Longleftrightarrow\quad
  p < \frac{6}{7}.
\]

The declared probabilities \(0.08,0.30,0.60,0.80\) are all below \(6/7\).
Their mode gaps \((1-p)-p/6\) decrease monotonically, reaching \(1/15\) at
\(p=0.80\). Within a world, all levels use the same clean rows, uniform random
variables, and nonzero shifts. Thresholding the same uniforms makes corruption
masks nested. Thus variation between levels is not caused by different action
coverage.

This proposition identifies a population boundary, not a finite-sample
guarantee. Near \(6/7\), neither selector is guaranteed to recover the center
at the largest declared prefix.

## 4. Fourth-family separation

Write \(u=1+x_t\) and \(v=x_s\). The three typed edge factors, with an active
source action suppressed, are

\[
  L=u,\qquad Q=u^2,\qquad S=u+v,
\]

and the fourth factor is \(C=u^3\).

### Proposition 3: cubic is outside the typed family union

For nonzero coefficients in \(\mathbb F_7\), \(C\) disagrees with every scalar
multiple of \(L,Q,S\) on at least 28, 35, and 42 of the 49 ordered \((u,v)\)
pairs, respectively.

**Proof.** For \(a,b\ne0\), equality \(au^3=bu\) is
\(u(au^2-b)=0\). It has at most three values of \(u\), hence at most 21 of 49
pairs. Equality \(au^3=bu^2\) is \(u^2(au-b)=0\), with at most two values of
\(u\), hence at most 14 pairs. For \(au^3=b(u+v)\), each fixed \(u\) admits
exactly one \(v\), hence exactly seven pairs. The disagreement lower bounds are
therefore 28, 35, and 42. Exhaustion over all nonzero coefficient pairs verifies
that each bound is attained. \(\square\)

### Proposition 4: quadratic is outside the alternative generic span

The function \(Q=u^2\) is not in the \(\mathbb F_7\)-linear span of a direct
action factor, \(L\), \(C\), and \(S\).

**Proof.** Any nonzero coefficient on \(S=u+v\) creates dependence on \(v\),
while \(Q\) has none, so that coefficient must be zero for exact equality.
The remaining equality would be
\(u^2=d+\alpha u+\beta u^3\) for every \(u\in\mathbb F_7\). Its difference is a
polynomial of degree at most three with seven roots. It must be the zero
polynomial, contradicting the coefficient \(-1\) on \(u^2\). Exhaustion over
all \(7^4\) coefficient vectors and all 49 state pairs finds a minimum of 28
disagreements, so the implemented catalog obeys the proof. \(\square\)

## 5. Symmetric misspecification estimand

The typed catalog is \((L,Q,S)\); the alternative generic catalog is
\((L,C,S)\). Both have eleven features and 55 output-feature positions.

- In the typed-misspecified condition, every family pair contains \(C\).
- In the generic-misspecified condition, the aligned pair replaces every
  occurrence of \(C\) by \(Q\).
- Paired worlds share graph, multipliers, edge coefficients, states, actions,
  corruption uniforms, and shifts.
- The matched condition gives both arms \((L,Q,S)\).

Accordingly, a valid scope result requires matched equivalence, generic
superiority when typed is misspecified, and typed superiority when generic is
misspecified. A one-directional result fails the scope gate.

## 6. What the theory does not prove

The propositions prove retained support, feature separation, the noise-mode
boundary, and representability reversal. They do not prove a finite-sample
advantage, a universal superiority of typed structure, or validity on natural
images. Those are empirical questions. A2 tests the first under its declared
synthetic population; the latter two remain outside its claim scope.
