# Gate K report

Generated 2026-09-11T23:24:25.047338+00:00.
Backend ['transformers'], model ['mistralai/Mistral-Small-3.1-24B-Base-2503'], 1266 results, primary representation distance `cosine`.

## Verdict: `mechanism_unsupported`

sparse padding does no better than dense projection and no representation contrast exists

| condition | value | threshold | passed |
|---|---|---|---|
| model_solves_sparse_2d_anchor | 0.896 | 0.650 | True |
| anchor_parse_quality | 0.000 | 0.020 | True |
| repr_contrast_sparse_over_dense | 0.305 | 1.200 | False |
| repr_contrast_sparse_over_nuisance | 0.578 | 1.200 | False |
| accuracy_sparse_minus_dense | -0.125 | 0.050 | False |
| accuracy_sparse_minus_nuisance | 0.375 | 0.050 | True |
| nearest_removal_selectivity_sparse_minus_dense | 0.188 | 0.050 | True |

Threshold sensitivity:

- thresholds loosened by factor 0.80: `mechanism_unsupported`
- thresholds tightened by factor 1.20: `mechanism_unsupported`

## Construction

Invariants passed: True. Failed: none.

## Paired arms at fixed ambient dimension (`order_random` jobs)

### sparse_zero_pad vs dense_redundant_projection

| dim | n | acc A | acc B | acc delta | repr RC A | repr RC B | repr ratio | raw RC A | raw RC B | B vs A (repr) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 48 | 0.896 | 0.958 | -0.062 | 1.029 | 1.131 | 0.909 | 30.924 | 92.350 | distance_deconcentrated |
| 8 | 20 | 1.000 | 0.950 | 0.050 | 0.865 | 2.991 | 0.289 | 32.786 | 38.172 | distance_deconcentrated |
| 32 | 48 | 0.979 | 1.000 | -0.021 | 0.611 | 3.175 | 0.192 | 30.924 | 31.454 | distance_deconcentrated |
| 64 | 48 | 0.750 | 0.979 | -0.229 | 1.665 | 3.984 | 0.418 | 30.924 | 31.475 | distance_deconcentrated |

### sparse_zero_pad vs independent_nuisance

| dim | n | acc A | acc B | acc delta | repr RC A | repr RC B | repr ratio | raw RC A | raw RC B | B vs A (repr) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 48 | 0.896 | 0.750 | 0.146 | 1.029 | 0.969 | 1.062 | 30.924 | 30.924 | distance_concentrated |
| 8 | 20 | 1.000 | 0.600 | 0.400 | 0.865 | 1.950 | 0.443 | 32.786 | 1.854 | distance_deconcentrated |
| 32 | 48 | 0.979 | 0.479 | 0.500 | 0.611 | 1.607 | 0.380 | 30.924 | 0.697 | distance_deconcentrated |
| 64 | 48 | 0.750 | 0.500 | 0.250 | 1.665 | 2.145 | 0.777 | 30.924 | 0.435 | distance_deconcentrated |

## Causal row ablation

| arm | dim | n | drop nearest | drop random | difference | flip nearest | flip random |
|---|---|---|---|---|---|---|---|
| sparse_zero_pad | 32 | 48 | 0.333 | 0.021 | 0.312 | 0.333 | 0.021 |
| sparse_zero_pad | 64 | 47 | 0.064 | -0.021 | 0.085 | 0.106 | 0.106 |
| dense_redundant_projection | 32 | 48 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dense_redundant_projection | 64 | 48 | 0.021 | 0.000 | 0.021 | 0.021 | 0.000 |
| independent_nuisance | 32 | 48 | 0.000 | -0.021 | 0.021 | 0.042 | 0.062 |
| independent_nuisance | 64 | 48 | 0.000 | -0.042 | 0.042 | 0.042 | 0.042 |

## Ordering control (behavioural, not a metric mechanism)

| arm | dim | n | nearest-first minus random |
|---|---|---|---|
| sparse_zero_pad | 64 | 47 | 0.213 |
| dense_redundant_projection | 64 | 48 | 0.021 |
| independent_nuisance | 64 | 48 | 0.312 |

## Per-cell detail

| cell | n | accuracy | margin | raw RC | repr RC | repr NF ratio | repr CoV | rho(latent, repr) | tokens |
|---|---|---|---|---|---|---|---|---|---|
| dense_redundant_projection|d2|order_random | 48 | 0.958 | -0.869 | 92.350 | 1.131 | 0.475 | 0.162 | 0.328 | 2002 |
| dense_redundant_projection|d8|order_random | 48 | 0.979 | -0.931 | 33.868 | 2.781 | 0.270 | 0.267 | 0.348 | 6208 |
| dense_redundant_projection|d32|ablate_nearest | 48 | 1.000 | -0.922 | 12.962 | 2.635 | 0.285 | 0.242 | 0.277 | 23190 |
| dense_redundant_projection|d32|ablate_random | 48 | 1.000 | -0.901 | 31.288 | 3.230 | 0.245 | 0.263 | 0.349 | 23190 |
| dense_redundant_projection|d32|order_random | 48 | 1.000 | -0.898 | 31.454 | 3.175 | 0.248 | 0.258 | 0.337 | 24387 |
| dense_redundant_projection|d64|ablate_nearest | 48 | 0.958 | -0.925 | 13.090 | 3.301 | 0.250 | 0.269 | 0.230 | 46383 |
| dense_redundant_projection|d64|ablate_random | 48 | 0.979 | -0.918 | 31.261 | 4.043 | 0.208 | 0.302 | 0.299 | 46382 |
| dense_redundant_projection|d64|order_nearest_first | 48 | 1.000 | -0.922 | 31.475 | 1.265 | 0.446 | 0.163 | -0.397 | 48779 |
| dense_redundant_projection|d64|order_random | 48 | 0.979 | -0.919 | 31.475 | 3.984 | 0.210 | 0.300 | 0.277 | 48779 |
| independent_nuisance|d2|order_random | 48 | 0.750 | -0.944 | 30.924 | 0.969 | 0.512 | 0.150 | 0.363 | 2001 |
| independent_nuisance|d8|order_random | 48 | 0.625 | -0.959 | 1.895 | 2.076 | 0.332 | 0.233 | 0.318 | 6212 |
| independent_nuisance|d32|ablate_nearest | 48 | 0.479 | -0.957 | 0.691 | 1.527 | 0.405 | 0.158 | 0.155 | 23174 |
| independent_nuisance|d32|ablate_random | 48 | 0.500 | -0.957 | 0.683 | 1.517 | 0.408 | 0.158 | 0.224 | 23175 |
| independent_nuisance|d32|order_random | 48 | 0.479 | -0.957 | 0.697 | 1.607 | 0.394 | 0.160 | 0.215 | 24370 |
| independent_nuisance|d64|ablate_nearest | 48 | 0.500 | -0.941 | 0.428 | 2.092 | 0.329 | 0.236 | 0.029 | 46356 |
| independent_nuisance|d64|ablate_random | 48 | 0.542 | -0.939 | 0.430 | 2.133 | 0.325 | 0.237 | 0.055 | 46357 |
| independent_nuisance|d64|order_nearest_first | 48 | 0.812 | -0.938 | 0.435 | 1.759 | 0.368 | 0.220 | -0.918 | 48751 |
| independent_nuisance|d64|order_random | 48 | 0.500 | -0.944 | 0.435 | 2.145 | 0.323 | 0.246 | 0.052 | 48751 |
| sparse_zero_pad|d2|order_random | 48 | 0.896 | -0.944 | 30.924 | 1.029 | 0.496 | 0.165 | 0.340 | 2001 |
| sparse_zero_pad|d8|order_random | 20 | 1.000 | -0.939 | 32.786 | 0.865 | 0.540 | 0.106 | 0.135 | 6393 |
| sparse_zero_pad|d32|ablate_nearest | 48 | 0.646 | -0.938 | 12.586 | 0.636 | 0.613 | 0.083 | 0.084 | 24063 |
| sparse_zero_pad|d32|ablate_random | 48 | 0.958 | -0.935 | 30.923 | 0.639 | 0.611 | 0.083 | 0.109 | 24063 |
| sparse_zero_pad|d32|order_random | 48 | 0.979 | -0.935 | 30.924 | 0.611 | 0.623 | 0.078 | 0.127 | 25305 |
| sparse_zero_pad|d64|ablate_nearest | 48 | 0.688 | -0.931 | 12.586 | 1.518 | 0.398 | 0.210 | 0.008 | 48191 |
| sparse_zero_pad|d64|ablate_random | 47 | 0.766 | -0.930 | 30.769 | 1.481 | 0.405 | 0.207 | 0.021 | 48191 |
| sparse_zero_pad|d64|order_nearest_first | 47 | 0.957 | -0.943 | 30.839 | 1.470 | 0.407 | 0.223 | -0.962 | 50681 |
| sparse_zero_pad|d64|order_random | 48 | 0.750 | -0.931 | 30.924 | 1.665 | 0.377 | 0.233 | 0.016 | 50681 |
