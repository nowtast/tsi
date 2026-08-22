# Research A1 Confirmatory 결과

## 판정

Prospective one-shot A1 cohort는 선언한 matched-dictionary population에서 typed
support restriction의 finite-sample efficiency advantage를 지지한다. Universal
representational superiority 또는 point-valued crossing은 지지하지 않는다.

Training size 10, 15, 20, 25에서 joint advantage가 검출됐고 size 50에서 처음으로
joint equivalence가 검출됐다. Size 30과 40은 동결한 simultaneous rule 아래에서
indeterminate였으므로 보고하는 transition band는 point estimate \(n^*\)가 아니라
\([25,50]\)이다.

## Primary 결과

모든 interval은 명시한 world-level endpoint 16개에 대한 two-sided
Bonferroni-simultaneous interval이다. Positive NLL은 unstructured generic selector의
composition prediction이 더 나쁨을 뜻한다. Positive recovery difference는 typed
selector의 exact recovery가 더 많음을 뜻한다.

| Train size | Generic minus typed NLL | Typed minus generic exact recovery | Typed rate | Generic rate | 판정 |
|---:|---:|---:|---:|---:|---|
| 5 | 0.53189 [0.44609, 0.61769] | 0.01587 [-0.01716, 0.04891] | 0.016 | 0.000 | Neither |
| 10 | 0.32427 [0.25547, 0.39308] | 0.16667 [0.06816, 0.26517] | 0.222 | 0.056 | Joint advantage |
| 15 | 0.17773 [0.10812, 0.24733] | 0.16667 [0.06816, 0.26517] | 0.492 | 0.325 | Joint advantage |
| 20 | 0.09687 [0.04406, 0.14968] | 0.15079 [0.05621, 0.24538] | 0.714 | 0.563 | Joint advantage |
| 25 | 0.05263 [0.01584, 0.08943] | 0.11905 [0.03345, 0.20465] | 0.849 | 0.730 | Joint advantage |
| 30 | 0.02047 [-0.00397, 0.04491] | 0.04762 [-0.00867, 0.10391] | 0.897 | 0.849 | Indeterminate |
| 40 | 0.00660 [-0.00469, 0.01789] | 0.02381 [-0.01649, 0.06411] | 0.984 | 0.960 | Indeterminate |
| 50 | -0.00002 [-0.00008, 0.00004] | 0.00000 [0.00000, 0.00000] | 0.992 | 0.992 | Joint equivalence |

Typed control과 isomorphic generic control은 모든 world와 모든 size에서 prediction과
composition NLL이 정확히 같았다. 따라서 관측한 advantage는 notation이 아니라
support-search restriction에 귀속된다.

## Provenance

- 독립 단위: 126 worlds
- Family balance: 9개 ordered family-pair stratum마다 14 worlds
- Graph coverage: graph motif 30개 전체, graph마다 4 또는 5 worlds
- Contract digest: `9a37d3f2d9e424dd9e6a00f1235b4f3fa07b2dfc8d26633f06dd632cba04cdee`
- Freeze digest: `9ea1b9b75cea3d51dcf401f62fee812cea4c9e374569df5161433c389b234339`
- Seed commitment: `49956b58a7cb4e129304cbb621596946c0175c165607758cbe30a33c9053da0e`
- Public commitment commit: `cc99dfc05614084ecd89460d24d5f5a958f3e8c0`
- Confirmatory analysis SHA-256: `0cca53e69c63ffacaccbf7a66064eedc84bb6bf6479c0a9eada4e4cf2b4c1e5d`

Source freeze와 seed commitment는 실행 전에 push했고 root seed는 one-shot 실행 후
공개했다. Zero-project-import Node.js replay가 endpoint 16개를 failure 0건으로
재계산했다. 그러나 두 구현은 같은 저자 workflow에서 만들었고 외부 인원이 independent
replay를 완료하지 않았다. 따라서 이 결과를 independently replicated라고 기술하면 안
된다.

## 경계와 다음 gate

결과는 supplied correct graph, 선언한 modulo-7 primitive-intervention population,
matched dictionary 및 특정 factorized와 seven-step greedy selector에 한정된다. A2는 새
contract와 seed 아래에서 candidate width, training-noise degradation 및 양방향
misspecification을 검정한다. A2는 A1을 변경하거나 보수할 수 없다.
