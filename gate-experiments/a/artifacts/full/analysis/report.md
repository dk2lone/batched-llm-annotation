# Gate A result: within-batch error correlation

- Config: `full` (fingerprint `3f8f95dd5d297e92`)
- Backend: vllm; model: Qwen/Qwen3-8B
- Rows: 14000 over 8375 jobs and 2000 documents

## Verdict: PROCEED

- point estimate 0.0754 >= 0.02
- conservative CI lower bound 0.0182 excludes zero
- largest fake-batch p-value 0.0010 < 0.05

Aggregate ICC 0.0754, decision interval [0.0182, 0.1314], largest fake-batch p 0.0010.

## Accuracy and parsing by condition

| condition | rows | docs | accuracy | error rate CI | parse failure rate |
|---|---|---|---|---|---|
| filler | 5993 | 2000 | 0.9153 | [0.0734, 0.0953] | 0.0012 |
| joint | 5998 | 2000 | 0.8871 | [0.1023, 0.1240] | 0.0003 |
| solo | 1982 | 1982 | 0.8325 | [0.1509, 0.1847] | 0.0090 |

## Paired error deltas, per document

| contrast | docs | mean error delta | CI | excludes zero |
|---|---|---|---|---|
| filler_minus_solo | 1982 | -0.0823 | [-0.0976, -0.0666] | True |
| joint_minus_filler | 2000 | 0.0282 | [0.0202, 0.0362] | True |
| joint_minus_solo | 1982 | -0.0542 | [-0.0696, -0.0391] | True |

## Within-real-batch ICC, joint condition

| assignment | ICC | CI | batches | DEFF | n_eff | fake-batch p |
|---|---|---|---|---|---|---|
| 1 | 0.0803 | [0.0434, 0.1153] | 125 | 2.205 | 906.5 | 0.0010 |
| 2 | 0.0887 | [0.0456, 0.1314] | 125 | 2.330 | 858.0 | 0.0010 |
| 3 | 0.0572 | [0.0182, 0.1034] | 125 | 1.858 | 1076.2 | 0.0010 |
| aggregate | 0.0754 | [0.0499, 0.0980] | - | 2.131 | 938.0 | - |

Aggregate strategy: Every target document appears once per assignment, so the three per-assignment ICCs are computed on disjoint rows but on the same documents. Pooling all assignments into one ANOVA would count each document three times and understate the standard error. The aggregate point estimate is therefore the unweighted mean of the per-assignment ICCs. Its bootstrap resamples batches inside each assignment independently, which ignores the dependence between assignments induced by the shared document set, so the aggregate interval is optimistic. The reported decision interval is the conservative envelope: the lowest lower bound and the highest upper bound across the aggregate interval and the three per-assignment intervals.

## Placebo ICC, filler condition

corresponding joint batch id; the 16 targets were labelled in 16 separate prompts, so any positive value is not a batching effect

| assignment | ICC | CI | fake-batch p |
|---|---|---|---|
| 1 | 0.0009 | [-0.0155, 0.0174] | 0.4426 |
| 2 | 0.0071 | [-0.0083, 0.0230] | 0.1888 |
| 3 | 0.0064 | [-0.0085, 0.0218] | 0.2098 |

## Placebo ICC, solo condition

corresponding joint batch id; the 16 targets were labelled in 16 separate prompts, so any positive value is not a batching effect

| assignment | ICC | CI | fake-batch p |
|---|---|---|---|
| 1 | -0.0011 | [-0.0159, 0.0131] | 0.5265 |
| 2 | 0.0010 | [-0.0153, 0.0181] | 0.4136 |
| 3 | -0.0053 | [-0.0178, 0.0071] | 0.7353 |

## Output position

| condition | p any slot difference | slot-error correlation | p trend |
|---|---|---|---|
| filler | 0.0629 | -0.0298 | 0.0210 |
| joint | 0.0030 | 0.0313 | 0.0180 |

## Caveats carried by this report

- Parse failures are dropped from the ICC. If they cluster by batch, that deletion is not ignorable. See `icc.sensitivity_failures_as_errors`.
- The aggregate interval treats the three assignments as independent. The decision interval widens it to the conservative envelope.
- A single gate on one model and one dataset bounds nothing beyond that cell.
