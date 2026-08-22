# Research A1 Design History

## Initial grid

The first development run used 36 worlds and training sizes from 50 through
12,800. Every typed, isomorphic, and unstructured arm recovered the exact
support in every world at every size. The initial grid therefore had a ceiling
defect and could not identify finite-sample efficiency.

## Corrected grid

A second development-only run, under the same public development seed label,
evaluated sizes 5, 8, 10, 12, 15, 20, 25, 30, 40, and 50. The transition was
observable: at size 15, typed exact recovery was 0.556 and generic exact
recovery was 0.194; generic-minus-typed composition NLL was 0.25888. At size
40, both exact-recovery rates were 0.972 and their paired outcomes were equal.

The confirmatory grid was fixed to 5, 10, 15, 20, 25, 30, 40, and 50. This is
a documented development-driven design choice, not a confirmatory result.

## Power and interpretation

The prospective power calculation stratified bootstrap resamples by the nine
ordered family pairs and uses only the 36 low-grid development worlds. A count
of 126 confirmatory worlds gives exactly 14 worlds per family-pair stratum. Its
estimated power for at least one joint advantage followed by a later joint
equivalence region was 1.0. That estimate may be optimistic because the
development cohort is small, so 126 was retained instead of the smaller
90-world candidate.

The theory values 263, 861, and 10,163 are conservative sufficient upper
envelopes. They are not predictions of the empirical transition and are not
contradicted by the much earlier development transition.
