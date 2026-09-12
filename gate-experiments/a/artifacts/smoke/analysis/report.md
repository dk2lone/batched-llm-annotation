# Gate A result: within-batch error correlation

> **MOCK DATA. NOT A RESULT.** These rows come from the deterministic mock backend, which synthesises errors from the gold labels. They check the pipeline, never the model. No pilot has run.

- Config: `smoke` (fingerprint `2012dce5eb75ffa1`)
- Backend: mock; model: mock/none
- Rows: 672 over 402 jobs and 96 documents

## Verdict: UNDETERMINED

- too few joint-condition errors to estimate a stable within-batch correlation: counts by assignment are [20, 17, 9], requirement is 20 each

Aggregate ICC 0.0577, decision interval [-0.0572, 0.2222], largest fake-batch p 0.3035.

## Accuracy and parsing by condition

| condition | rows | docs | accuracy | error rate CI | parse failure rate |
|---|---|---|---|---|---|
| filler | 280 | 96 | 0.8854 | [0.0816, 0.1493] | 0.0278 |
| joint | 287 | 96 | 0.8403 | [0.1180, 0.2083] | 0.0035 |
| solo | 93 | 93 | 0.9032 | [0.0430, 0.1613] | 0.0312 |

## Paired error deltas, per document

| contrast | docs | mean error delta | CI | excludes zero |
|---|---|---|---|---|
| filler_minus_solo | 93 | 0.0143 | [-0.0503, 0.0735] | False |
| joint_minus_filler | 96 | 0.0451 | [-0.0069, 0.0903] | False |
| joint_minus_solo | 93 | 0.0645 | [-0.0037, 0.1291] | False |

## Within-real-batch ICC, joint condition

| assignment | ICC | CI | batches | DEFF | n_eff | fake-batch p |
|---|---|---|---|---|---|---|
| 1 | 0.0974 | [-0.0275, 0.2222] | 6 | 2.462 | 39.0 | 0.0199 |
| 2 | 0.0504 | [-0.0465, 0.1501] | 6 | 1.756 | 54.1 | 0.0846 |
| 3 | 0.0252 | [-0.0572, 0.0828] | 6 | 1.378 | 69.7 | 0.3035 |
| aggregate | 0.0577 | [-0.0087, 0.1178] | - | 1.865 | 50.9 | - |

Aggregate strategy: Every target document appears once per assignment, so the three per-assignment ICCs are computed on disjoint rows but on the same documents. Pooling all assignments into one ANOVA would count each document three times and understate the standard error. The aggregate point estimate is therefore the unweighted mean of the per-assignment ICCs. Its bootstrap resamples batches inside each assignment independently, which ignores the dependence between assignments induced by the shared document set, so the aggregate interval is optimistic. The reported decision interval is the conservative envelope: the lowest lower bound and the highest upper bound across the aggregate interval and the three per-assignment intervals.

## Placebo ICC, filler condition

corresponding joint batch id; the 16 targets were labelled in 16 separate prompts, so any positive value is not a batching effect

| assignment | ICC | CI | fake-batch p |
|---|---|---|---|
| 1 | -0.0547 | [-0.0688, -0.0506] | 0.9403 |
| 2 | -0.0272 | [-0.0667, 0.0404] | 0.8010 |
| 3 | -0.0370 | [-0.0635, -0.0106] | 0.7761 |

## Placebo ICC, solo condition

corresponding joint batch id; the 16 targets were labelled in 16 separate prompts, so any positive value is not a batching effect

| assignment | ICC | CI | fake-batch p |
|---|---|---|---|
| 1 | -0.0139 | [-0.0601, 0.0407] | 0.6269 |
| 2 | -0.0550 | [-0.0712, -0.0512] | 0.9701 |
| 3 | -0.0311 | [-0.0705, 0.0219] | 0.7214 |

## Output position

| condition | p any slot difference | slot-error correlation | p trend |
|---|---|---|---|
| filler | 0.5821 | -0.0040 | 0.9502 |
| joint | 0.6269 | 0.0238 | 0.7015 |

## Caveats carried by this report

- Parse failures are dropped from the ICC. If they cluster by batch, that deletion is not ignorable. See `icc.sensitivity_failures_as_errors`.
- The aggregate interval treats the three assignments as independent. The decision interval widens it to the conservative envelope.
- A single gate on one model and one dataset bounds nothing beyond that cell.
