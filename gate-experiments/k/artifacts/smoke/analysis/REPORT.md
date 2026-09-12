# Gate K report

Generated 2026-09-11T20:56:31.180310+00:00.
Backend ['mock'], model ['mock'], 72 results, primary representation distance `cosine`.

**These numbers come from the deterministic mock backend. They test the code path only. They are not evidence about any language model.**

## Verdict: `ambiguous`

mixed pattern across the pre-registered conditions

| condition | value | threshold | passed |
|---|---|---|---|
| model_solves_sparse_2d_anchor | 1.000 | 0.500 | True |
| anchor_parse_quality | 0.000 | 0.020 | True |
| repr_contrast_sparse_over_dense | 0.447 | 1.100 | False |
| repr_contrast_sparse_over_nuisance | 207.971 | 1.100 | True |
| accuracy_sparse_minus_dense | 0.000 | 0.050 | False |
| accuracy_sparse_minus_nuisance | 0.250 | 0.050 | True |
| nearest_removal_selectivity_sparse_minus_dense | 0.250 | 0.050 | True |

Threshold sensitivity:

- thresholds loosened by factor 0.80: `ambiguous`
- thresholds tightened by factor 1.20: `ambiguous`

## Construction

Invariants passed: True. Failed: none.

## Paired arms at fixed ambient dimension (`order_random` jobs)

### sparse_zero_pad vs dense_redundant_projection

| dim | n | acc A | acc B | acc delta | repr RC A | repr RC B | repr ratio | raw RC A | raw RC B | B vs A (repr) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 1.000 | 1.000 | 0.000 | 768.159 | 982.643 | 0.782 | 15.159 | 14.545 | distance_deconcentrated |
| 8 | 4 | 1.000 | 1.000 | 0.000 | 768.159 | 1718.106 | 0.447 | 15.159 | 23.041 | distance_deconcentrated |

### sparse_zero_pad vs independent_nuisance

| dim | n | acc A | acc B | acc delta | repr RC A | repr RC B | repr ratio | raw RC A | raw RC B | B vs A (repr) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 1.000 | 1.000 | 0.000 | 768.159 | 768.159 | 1.000 | 15.159 | 15.159 | distance_deconcentrated |
| 8 | 4 | 1.000 | 0.750 | 0.250 | 768.159 | 3.694 | 207.971 | 15.159 | 1.072 | distance_concentrated |

## Causal row ablation

| arm | dim | n | drop nearest | drop random | difference | flip nearest | flip random |
|---|---|---|---|---|---|---|---|
| sparse_zero_pad | 8 | 4 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| dense_redundant_projection | 8 | 4 | 0.750 | 0.000 | 0.750 | 0.750 | 0.000 |
| independent_nuisance | 8 | 4 | 0.500 | 0.000 | 0.500 | 0.500 | 0.000 |

## Ordering control (behavioural, not a metric mechanism)

| arm | dim | n | nearest-first minus random |
|---|---|---|---|
| sparse_zero_pad | 8 | 4 | 0.000 |
| dense_redundant_projection | 8 | 4 | 0.000 |
| independent_nuisance | 8 | 4 | 0.000 |

## Per-cell detail

| cell | n | accuracy | margin | raw RC | repr RC | repr NF ratio | repr CoV | rho(latent, repr) | tokens |
|---|---|---|---|---|---|---|---|---|---|
| dense_redundant_projection|d2|order_random | 4 | 1.000 | -0.005 | 14.545 | 982.643 | 0.002 | 0.865 | 0.923 | 147 |
| dense_redundant_projection|d8|ablate_nearest | 4 | 0.250 | 0.022 | 1.059 | 1.575 | 0.404 | 0.378 | 0.829 | 263 |
| dense_redundant_projection|d8|ablate_random | 4 | 1.000 | -0.050 | 22.436 | 1718.105 | 0.015 | 0.816 | 0.957 | 263 |
| dense_redundant_projection|d8|attention_capture | 4 | 1.000 | -0.050 | 23.041 | 1718.106 | 0.015 | 0.707 | 0.923 | 322 |
| dense_redundant_projection|d8|order_nearest_first | 4 | 1.000 | -0.050 | 23.041 | 1718.106 | 0.015 | 0.707 | 0.923 | 322 |
| dense_redundant_projection|d8|order_random | 4 | 1.000 | -0.050 | 23.041 | 1718.106 | 0.015 | 0.707 | 0.923 | 322 |
| independent_nuisance|d2|order_random | 4 | 1.000 | -0.028 | 15.159 | 768.159 | 0.006 | 0.748 | 0.940 | 147 |
| independent_nuisance|d8|ablate_nearest | 4 | 0.250 | 0.043 | 0.677 | 1.646 | 0.386 | 0.302 | 0.386 | 263 |
| independent_nuisance|d8|ablate_random | 4 | 0.750 | -0.282 | 0.982 | 3.458 | 0.291 | 0.389 | 0.286 | 263 |
| independent_nuisance|d8|attention_capture | 4 | 0.750 | -0.268 | 1.072 | 3.694 | 0.261 | 0.377 | 0.351 | 322 |
| independent_nuisance|d8|order_nearest_first | 4 | 0.750 | -0.268 | 1.072 | 3.694 | 0.261 | 0.377 | 0.351 | 322 |
| independent_nuisance|d8|order_random | 4 | 0.750 | -0.268 | 1.072 | 3.694 | 0.261 | 0.377 | 0.351 | 322 |
| sparse_zero_pad|d2|order_random | 4 | 1.000 | -0.028 | 15.159 | 768.159 | 0.006 | 0.748 | 0.940 | 147 |
| sparse_zero_pad|d8|ablate_nearest | 4 | 0.000 | -0.056 | 0.758 | 2.359 | 0.307 | 0.422 | 0.857 | 263 |
| sparse_zero_pad|d8|ablate_random | 4 | 1.000 | -0.028 | 14.198 | 690.085 | 0.006 | 0.891 | 0.929 | 263 |
| sparse_zero_pad|d8|attention_capture | 4 | 1.000 | -0.028 | 15.159 | 768.159 | 0.006 | 0.748 | 0.940 | 322 |
| sparse_zero_pad|d8|order_nearest_first | 4 | 1.000 | -0.028 | 15.159 | 768.159 | 0.006 | 0.748 | 0.940 | 322 |
| sparse_zero_pad|d8|order_random | 4 | 1.000 | -0.028 | 15.159 | 768.159 | 0.006 | 0.748 | 0.940 | 322 |

## Attention (descriptive only)

{
  "mean_spearman_row_mass_vs_latent_distance_by_dim": {
    "8": -0.7023809523809524
  },
  "n_jobs": 12,
  "note": "weak evidence; reported for description only"
}
