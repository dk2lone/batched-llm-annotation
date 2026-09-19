# B5 interim analysis

Interim selected-cell analysis; this cannot replace the final four-cell result.

## Primary result

The equal-weight selected-cell mean slope of the standardized joint-minus-exact contrast on log2(batch size) is 0.004107 (95% cluster-bootstrap interval [0.0010218356311477582, 0.006974437467465448]; two-sided bootstrap p = 0.0070).

The interim selected-cell test against zero rejects: **True**.

## Contrast by size

| size | standardized contrast | 95% interval | raw contrast |
|---:|---:|---|---:|
| 2 | 0.053304 | [0.04155255119341113, 0.06530324076927937] | 0.008122 |
| 4 | 0.100356 | [0.09129870799221783, 0.10912285062186498] | 0.014348 |
| 8 | 0.104747 | [0.09672509556772718, 0.11322909039041189] | 0.014818 |
| 16 | 0.099071 | [0.09071622493508669, 0.10703944180815814] | 0.014190 |
| 32 | 0.074481 | [0.06684255215257387, 0.08179802239152607] | 0.012107 |

## Cells

| cell | slope | 95% interval | two-sided p |
|---|---:|---|---:|
| Qwen/Qwen3-8B::civil_comments | -0.015833 | [-0.020239761951136918, -0.011909124803591657] | 0.0000 |
| ibm-granite/granite-3.3-8b-instruct::imdb | 0.024047 | [0.019648049323213807, 0.028423184658852168] | 0.0000 |

## Interim threshold reading

Matched interpretations: non_monotonic.

Practical contrast 0.016667; practical slope 0.004167. Success is not defined as a positive slope, and a wide interval is not evidence of equivalence.

## Historical comparison

The sealed A9 size-16 raw covariance contrasts are listed beside this result for reference only. They used the earlier output contract and are never pooled into this experiment.

- Qwen/Qwen3-8B::civil_comments: 0.007119
- ibm-granite/granite-3.3-8b-instruct::imdb: 0.018779
