# Structural Attribution Benchmark v0.1

이 directory는 기존 Paper 03/04 attribution cohort를 versioned하고 machine-checkable한
audit benchmark로 포장한다. 새 실험을 추가하지 않으며 frozen reference result를 독립
evidence로 바꾸지 않는다.

## 사용 목적

Benchmark는 graph/head recovery와 held-out two-mechanism composition prediction을
분리한다. 독립 단위는 world다. Fitting은 `train`, graph/head selection은
`selection`만 읽을 수 있고 어느 단계도 test target이나 보관된 answer를 읽을 수 없다.

`portable_inputs.json`을 직접 열지 말고 participant loader를 사용한다.

```python
from pathlib import Path
from tsi.structural_attribution_benchmark import load_participant_worlds

worlds = load_participant_worlds(Path.cwd())
```

Loader는 `graph`, `families`, `expected_row`를 제거하고 test case에서 target state를
제거한다. 원래 portable file은 frozen reproduction을 위해 공개되어 있으므로 v0.1은
blind leaderboard가 아니라 audit 및 development benchmark다. 정보 정책 준수 선언은
그 자체로 독립 validation이 아니다.

## 파일

- `benchmark.json`: benchmark contract와 source artifact hash
- `reference_results.json`: frozen analysis에 연결된 exact value
- `submission_schema.json`: portable report 형식
- `examples/minimal_submission.json`: 형식만 검사하는 one-world smoke example
- `BENCHMARK_CARD_KO.md`: estimand, 정보 정책, 한계와 재사용 조건

검사 명령은 다음과 같다.

```bash
PYTHONPATH=src python3 tools/check_structural_attribution_benchmark.py
PYTHONPATH=src python3 tools/check_structural_attribution_benchmark.py \
  benchmarks/structural_attribution_v0_1/examples/minimal_submission.json
```

공개 evidence payload DOI는 `10.5281/zenodo.22004526`이다. 정확히 재사용할 때는
`benchmark.json`의 contract digest와 artifact hash를 보존해야 한다.
