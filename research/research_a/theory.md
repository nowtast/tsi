# Research A Theory: Finite-Sample Support Recovery

## 1. Status and scope

This document derives sufficient finite-sample envelopes for the Research A
design. It does not report a frozen study, a lower bound, or an empirical
crossing point. The result applies only to the declared modulo-7 primitive-
intervention population with the graph supplied to every arm.

The proof separates three questions:

1. coefficient estimation when the seven-term support is known;
2. support-and-coefficient recovery within the nine typed family supports; and
3. seven-step greedy recovery in the 55-position generic dictionary.

## 2. Population and selectors

Let \(\mathbb F_7\) be the field with seven elements. A training row contains a
state \(S\in\mathbb F_7^5\), sampled uniformly, and a primitive action
\(A\in\mathbb F_7^5\). Its active coordinate \(J\) is uniform on
\(\{1,\ldots,5\}\), its active magnitude is uniform on \(\{1,2\}\), and all
other action coordinates are zero. State and action are independent.

A supplied graph has target \(t\) and two distinct sources \(r_1,r_2\). For a
source \(r\), define

\[
\begin{aligned}
h_{r,L}(S,A)&=A_r(1+S_t),\\
h_{r,Q}(S,A)&=A_r(1+S_t)^2,\\
h_{r,S}(S,A)&=A_r(1+S_r+S_t).
\end{aligned}
\]

All arithmetic is in \(\mathbb F_7\). The true transition increment contains
five direct terms \(m_jA_j\), one in each output, and two edge terms
\(c_kh_{r_k,F_k}\) in output \(t\). Thus the true support size is \(s=7\).
Every true coefficient is nonzero. The generic dictionary has the five direct
features and the three head features for each supplied source, hence 11
features and \(p=5\times11=55\) output-feature positions.

For each output coordinate, observation noise \(E\) is independent and
symmetric:

\[
\Pr(E=0)=1-\eta,\qquad
\Pr(E=e)=\frac{\eta}{6}\quad(e\ne0),qquad 0\leq\eta<\frac67.
\]

The typed selector minimizes coordinatewise empirical zero-one loss over six
nonzero coefficients for each direct cell and over 18 family-coefficient
choices for each edge. This factorization is equivalent to searching the nine
ordered head-family supports, but avoids enumerating all
\(9\cdot6^7=2{,}519{,}424\) joint parameterizations. The isomorphic generic
control runs exactly the same search in generic feature coordinates. The
unstructured selector greedily adds seven nonzero-coefficient moves from the
55 positions.

## 3. Identification assumptions

The assumptions replacing a real-valued restricted-eigenvalue condition are
explicit for this finite modular design.

- **A1 (independent rows):** training rows are independent draws from the
  population above.
- **A2 (common information):** every arm receives the same rows and the same
  supplied graph; no arm observes the true head families or coefficients.
- **A3 (algebraic beta-min):** all seven true coefficients are nonzero in
  \(\mathbb F_7\). Distinct coefficients therefore disagree whenever their
  common feature is nonzero.
- **A4 (cell excitation):** a direct feature is active with probability
  \(1/5\); every true head feature is nonzero with probability
  \((1/5)(6/7)=6/35\).
- **A5 (family separation):** after allowing arbitrary nonzero coefficients,
  two distinct head families disagree on at least \(5/7\) of the states
  conditional on activation of their source coordinate.
- **A6 (positive noise gap):** \(\eta<6/7\), so the noiseless center remains the
  unique modal observation.

These are population-specific identifiability conditions. They are not claims
about arbitrary correlated dictionaries or continuous-state regression.

## 4. Noise-risk identity

Define

\[
\gamma(\eta)=(1-\eta)-\frac{\eta}{6}=1-\frac{7\eta}{6}>0.
\]

**Lemma 1 (risk gap).** For a deterministic candidate value \(g(X)\) and true
center \(f(X)\), coordinatewise zero-one risk satisfies

\[
R(g)-R(f)=\gamma(\eta)\Pr\{g(X)\ne f(X)\}.
\]

**Proof.** If \(g(X)=f(X)\), the two losses coincide. If they differ, the true
center has loss \(\eta\), whereas the fixed wrong value has loss
\(1-\eta/6\). Their conditional difference is \(\gamma(\eta)\). Taking the
expectation over \(X\) proves the identity. \(\square\)

## 5. Known-support coefficient bound

For a direct coefficient, an informative row occurs with probability \(1/5\).
For an edge coefficient, its source must be selected and its head value must be
nonzero. Each of the three head families is zero on exactly \(1/7\) of the
uniform relevant states, so the minimum informative probability is

\[
q_{\min}=\frac15\frac67=\frac6{35}.
\]

**Theorem 1 (known-support modal recovery).** Estimate each of the seven
coefficients by the empirical zero-one-loss minimizer on its known feature. For
\(n\) independent rows,

\[
\Pr\{\widehat\theta\ne\theta\}
\leq
7e^{-nq_{\min}/8}
+42e^{-nq_{\min}\gamma(\eta)^2/4}.
\]

**Proof.** For any coefficient, let \(M\) be its number of informative rows.
The multiplicative Chernoff bound gives
\(\Pr\{M<nq_{\min}/2\}\leq e^{-nq_{\min}/8}\). Conditional on
\(M\geq nq_{\min}/2\), compare the true coefficient with one fixed wrong
nonzero value. The per-row wrong-minus-true loss lies in \([-1,1]\) and has
mean \(\gamma(\eta)\) by Lemma 1. Hoeffding's inequality bounds reversal of
their empirical order by
\(e^{-M\gamma(\eta)^2/2}\leq e^{-nq_{\min}\gamma(\eta)^2/4}\).
Union bounds over seven coefficients, and over six possible wrong values in the
larger candidate set \(\mathbb F_7\), give the displayed bound. \(\square\)

The implementation deliberately uses the six-value union although the
confirmatory selector will exclude zero; this is a conservative envelope.

## 6. Typed support-selection bound

**Lemma 2 (family separation).** For nonzero \(a,b\in\mathbb F_7\), two
different scaled head families agree on at most \(2/7\) of the relevant states.

**Proof.** Put \(u=1+S_t\). A scaled linear and quadratic pair obeys
\(au=bu^2\) only at \(u=0\) and at at most one nonzero value. A target-only
family and a source-target family can agree for at most one value of \(S_r\)
for each fixed \(S_t\), hence on at most \(1/7\) of state pairs. Two unequal
coefficients in the same family agree only where that feature is zero, also a
\(1/7\) event. The worst agreement is therefore \(2/7\), proving at least
\(5/7\) disagreement. \(\square\)

**Theorem 2 (factorized typed ERM).** The coordinatewise typed selector fails
to recover all direct coefficients, both head families, and both edge
coefficients with probability at most

\[
25e^{-n\gamma(\eta)^2/50}
+34e^{-n\gamma(\eta)^2/98}.
\]

**Proof.** A wrong direct coefficient differs from the truth whenever its
action coordinate is selected, with probability \(1/5\). There are five wrong
nonzero choices in each of five direct cells, yielding 25 comparisons. For an
edge, Lemma 2 and source activation give minimum disagreement
\((1/5)(5/7)=1/7\). Each edge has 18 family-coefficient candidates, 17 of them
wrong, yielding 34 comparisons. In a local comparison the per-row
wrong-minus-true loss lies in \([-1,1]\). Lemma 1 and Hoeffding give exponents
\(-n\gamma^2(1/5)^2/2=-n\gamma^2/50\) and
\(-n\gamma^2(1/7)^2/2=-n\gamma^2/98\). A union bound proves the result.
\(\square\)

Because the typed and isomorphic generic controls enumerate identical
functions with identical losses and tie-breaking, their fitted functions must
be identical row by row. This is an implementation invariant, not an empirical
hypothesis.

## 7. Unstructured greedy bound

At any proper subset of the seven true moves, primitive actions make the
remaining mechanism cells disjoint. Adding a missing direct term improves
population loss by \((7/35)\gamma\); adding a missing edge term improves it by
\((6/35)\gamma\).

An incorrect dictionary move can use the same action coordinate and partially
imitate one missing term. Its useful agreement is at most \(2/7\) conditional
on that action; the quadratic-versus-direct comparison attains this largest
agreement count. Ignoring any additional harm gives the conservative upper
bound \((2/35)\gamma\) on its improvement. Therefore the minimum
correct-versus-incorrect improvement margin is

\[
\mu(\eta)=\frac4{35}\gamma(\eta).
\]

**Theorem 3 (seven-step generic greedy recovery).** For the declared generic
dictionary and deterministic tie-breaking, failure to recover the seven true
moves is bounded by

\[
(2^7-1)(55)(6)
\exp\left(-\frac{n\mu(\eta)^2}{8}\right)
=41910\exp\left(-\frac{n\mu(\eta)^2}{8}\right).
\]

**Proof.** Consider any proper subset of the true moves and any candidate
output-feature-coefficient move. Compare its empirical improvement with a
remaining true move. The difference of two per-row improvements lies in
\([-2,2]\), because the moves may affect different outputs, and its population
mean is at least \(\mu(\eta)\). Hoeffding bounds an order reversal by
\(e^{-n\mu^2/8}\). There are \(2^7-1=127\) proper true subsets and at most
\(55\times6=330\) moves at each subset. A union bound makes every greedy choice
correct simultaneously. Induction over seven steps then gives exact recovery.
\(\square\)

## 8. Numerical audit and interpretation

At the draft training noise \(\eta=0.08\) and failure level \(\delta=0.05\),
the executable sufficient envelopes are:

| Procedure | Sufficient \(n\) | Bound evaluated at that \(n\) |
|---|---:|---:|
| Known-support coefficient estimation | 263 | 0.02895 |
| Typed support-and-coefficient ERM | 861 | 0.02484 |
| Unstructured seven-step greedy search | 10,163 | 0.04996 |

These values are conservative sufficient upper envelopes. Their difference is
not a minimax lower bound and does not prove that the generic selector requires
more samples. In particular, \(\log\binom{55}{7}\) is not used to predict a
point crossing. The envelopes justify extending the candidate grid through
12,800 rows; the sealed experiment must estimate any advantage-to-equivalence
transition under frozen SESOI, multiplicity, and stopping rules.

Exact support recovery implies equality of the recovered deterministic center
on held-out compositions in this well-specified population. Composition NLL is
nevertheless retained as a separate empirical estimand because selector
failure, finite test samples, and robustness populations need not obey that
implication.

## 9. Proof boundary

The theorems do not cover a wrong supplied graph, missing dictionary families,
continuous states, arbitrary action policies, perceptual entity discovery, or
dependent rows. Dictionary misspecification and candidate-width expansion must
be separate robustness studies. No confirmatory seed may be generated until
the unresolved design fields in `preregistration_draft.json` are frozen.
