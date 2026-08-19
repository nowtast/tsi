# Stage 2-I0 Validation

Run:

```bash
PYTHONPATH=src python3 experiments/stage2_i0/run_validation.py
```

The script has two distinct roles:

- exhaustive oracle checks on a six-state finite family;
- a fixed-seed learned feasibility pilot comparing a coherent decoder with an
  independently predicted relation baseline under increased observation noise.

The learned pilot does not establish identifiability, consistency, or theorem
correctness.
