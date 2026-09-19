# B5 batch-size dose-response analysis

Complete four-cell confirmatory analysis.

## Primary result

The equal-weight cross-family slope of the standardized joint-minus-exact contrast on log2(batch size) is -0.009095 (95% cluster-bootstrap interval [-0.011181794781879095, -0.007181566258109577]; two-sided bootstrap p = 0.0000).

The prespecified test against zero rejects: **True**.

## Contrast by size

| size | standardized contrast | 95% interval | raw contrast |
|---:|---:|---|---:|
| 2 | 0.079479 | [0.0712508880987406, 0.08804782204277052] | 0.014486 |
| 4 | 0.080658 | [0.07481656344744364, 0.08625642703077122] | 0.013729 |
| 8 | 0.071110 | [0.06629237874301927, 0.07618578494977069] | 0.011592 |
| 16 | 0.059559 | [0.05481319336357254, 0.06433577943058887] | 0.009321 |
| 32 | 0.044553 | [0.04026137423599505, 0.04852355115720319] | 0.007675 |

## Cells

| cell | slope | 95% interval | two-sided p |
|---|---:|---|---:|
| Qwen/Qwen3-8B::civil_comments | -0.015833 | [-0.020239761951136918, -0.011909124803591657] | 0.0000 |
| ibm-granite/granite-3.3-8b-instruct::imdb | 0.024047 | [0.019648049323213807, 0.028423184658852168] | 0.0000 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | -0.035284 | [-0.038958422969711075, -0.03171761639534349] | 0.0000 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | -0.009311 | [-0.013340578874797032, -0.005551888420436207] | 0.0000 |

## Prespecified reading

Matched interpretations: non_monotonic.

Practical contrast 0.016667; practical slope 0.004167. Success is not defined as a positive slope, and a wide interval is not evidence of equivalence.

## Historical comparison

The sealed A9 size-16 raw covariance contrasts are listed beside this result for reference only. They used the earlier output contract and are never pooled into this experiment.

- Qwen/Qwen3-8B::civil_comments: 0.007119
- ibm-granite/granite-3.3-8b-instruct::imdb: 0.018779
- meta-llama/Llama-3.1-8B-Instruct::civil_comments: 0.001564
- mistralai/Mistral-7B-Instruct-v0.3::multi_nli: 0.012427
