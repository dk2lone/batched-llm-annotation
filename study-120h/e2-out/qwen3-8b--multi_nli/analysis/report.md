# Freeze B E2: Qwen3-8B / MultiNLI

## Cell assessment

This cell is positive and practically meaningful under its interim 16-slot Holm test.

This is one fresh E2 cell. It cannot establish panel recurrence or measured interval coverage.

## Raw assignment results

| assignment | joint accuracy | exact accuracy | joint covariance | exact covariance | delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.735840 | 0.642090 | 0.010558 | -0.001472 | 0.012030 |
| 2 | 0.734131 | 0.637939 | 0.009685 | 0.002343 | 0.007342 |
| 3 | 0.748291 | 0.636963 | 0.003960 | -0.000546 | 0.004506 |

## Primary result

| quantity | value |
|---|---:|
| joint covariance | 0.008068 |
| exact-context covariance | 0.000108 |
| joint minus exact, errors as correct | 0.007959 |
| joint minus exact, errors as errors | 0.007929 |
| joint minus exact, unresolved dropped | 0.007948 |
| joint minus exact, whole calls discarded | 0.007911 |
| replay p, one-sided | 0.000200 |
| Holm p, 16 primary slots | 0.003200 |

The replay-centred 95% shift interval is [0.006273, 0.009670].

## Accuracy and falsification

Mean joint accuracy is 0.739421; mean exact-context accuracy is 0.638997; their signed difference is 0.100423. Solo accuracy is 0.742850.

The observed exact-control covariance is 0.000108. The exact-control regrouping is reported as a falsification check, separate from the primary paired replay test.

## Boundaries

All 17,152 sealed first calls and their assignment metadata were verified before calculation. 3 exact-context target rows and 0 joint target rows were unresolved; 5 solo calls were unresolved. 0 retries were used.

Only one primary cell is complete. The admission lock contains ten primary cells from two model families and already marks the recurrence scope unsupported, so this result cannot be promoted to an across-model or across-task claim.

No confidence-interval coverage was measured. That requires the independently generated E5 repeated-sampling stream.
