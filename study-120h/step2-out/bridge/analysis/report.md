# Historical Qwen3-8B / Civil Comments bridge analysis

## Verdict: REPLICATION_SUPPORTED

The historical cell replicates excess same-call error covariance under the frozen Step 0 rules. This result is limited to Qwen3-8B on Civil Comments and is separate from the fresh Freeze B panel.

## Primary result

| quantity | value |
|---|---:|
| joint covariance | 0.005031 |
| exact-context covariance | -0.000421 |
| joint minus exact, errors as correct | 0.005451 |
| joint minus exact, errors as errors | 0.005442 |
| joint minus exact, unresolved dropped | 0.005451 |
| joint minus exact, whole calls discarded | 0.005455 |
| replay p, errors as correct | 0.000500 |
| replay p, errors as errors | 0.000500 |

The errors-as-correct shift interval is [0.004562, 0.006298].

## Falsification and context control

Fake-batch covariance is -0.000421, inside the frozen absolute 0.001 band. Its contrast with joint covariance is 0.005451, with one-sided p = 0.000500. Filler covariance is -0.000211.

## Data quality

The analysis verified 18,432 target rows, 12,672 new first calls, the sealed assignments, and 2,048 reused solo rows. Seven exact-context and ten filler responses were unresolved. No joint response was unresolved. The seven affected exact-context groups are removed only in the whole-call sensitivity.

No confidence-interval coverage result is available from this bridge.
