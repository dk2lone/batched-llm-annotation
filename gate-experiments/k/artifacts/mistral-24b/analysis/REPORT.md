# Gate K report

Generated 2026-09-11T22:46:39.167489+00:00.
Backend ['transformers'], model ['mistralai/Mistral-Small-3.1-24B-Instruct-2503'], 48 results, primary representation distance `cosine`.

## Verdict: `model_capability_failure`

the model did not clear the 2D accuracy and parse-quality floor, so any high-dimensional failure is uninterpretable

| condition | value | threshold | passed |
|---|---|---|---|
| model_solves_sparse_2d_anchor | 0.979 | 0.650 | True |
| anchor_parse_quality | 0.979 | 0.020 | False |
| repr_contrast_sparse_over_dense | n/a | 1.200 | False |
| repr_contrast_sparse_over_nuisance | n/a | 1.200 | False |
| accuracy_sparse_minus_dense | n/a | 0.050 | False |
| accuracy_sparse_minus_nuisance | n/a | 0.050 | False |
| nearest_removal_selectivity_sparse_minus_dense | n/a | 0.050 | False |

Threshold sensitivity:

- thresholds loosened by factor 0.80: `model_capability_failure`
- thresholds tightened by factor 1.20: `model_capability_failure`

## Construction

Invariants passed: True. Failed: none.

## Paired arms at fixed ambient dimension (`order_random` jobs)

### sparse_zero_pad vs dense_redundant_projection

| dim | n | acc A | acc B | acc delta | repr RC A | repr RC B | repr ratio | raw RC A | raw RC B | B vs A (repr) |
|---|---|---|---|---|---|---|---|---|---|---|

### sparse_zero_pad vs independent_nuisance

| dim | n | acc A | acc B | acc delta | repr RC A | repr RC B | repr ratio | raw RC A | raw RC B | B vs A (repr) |
|---|---|---|---|---|---|---|---|---|---|---|

## Causal row ablation

| arm | dim | n | drop nearest | drop random | difference | flip nearest | flip random |
|---|---|---|---|---|---|---|---|

## Ordering control (behavioural, not a metric mechanism)

| arm | dim | n | nearest-first minus random |
|---|---|---|---|

## Per-cell detail

| cell | n | accuracy | margin | raw RC | repr RC | repr NF ratio | repr CoV | rho(latent, repr) | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sparse_zero_pad|d2|order_random | 48 | 0.979 | -0.117 | 30.924 | 0.746 | 0.576 | 0.126 | 0.518 | 2178 |
