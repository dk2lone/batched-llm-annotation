# Repaired B3 recurrence sensitivity

This is a post-execution sensitivity analysis. The sealed A9 analysis remains primary.

Exact linked repairs used: 3,002
Repaired recurrence gate: passes
Global joint-minus-exact covariance: 0.007899 (one-sided replay p = 0.000200; 95% shift interval [0.007648, 0.008146]).
Holm-positive cells at the 0.002 floor: 13 of 16.

## Comparison to sealed A9

| quantity | sealed A9 | repaired sensitivity |
|---|---:|---:|
| recurrence gate | True | True |
| global delta | 0.009610 | 0.007899 |
| Holm-positive cells | 13 | 13 |

## Repaired cells

| family | task | repairs | delta | raw p | Holm p | positive |
|---|---|---:|---:|---:|---:|---:|
| qwen | ag_news | 14 | 0.003273 | 0.000200 | 0.003200 | yes |
| granite | civil_comments | 545 | 0.002739 | 0.000200 | 0.003200 | yes |
| llama | multi_nli | 31 | 0.005825 | 0.000200 | 0.003200 | yes |
| mistral | imdb | 158 | 0.013883 | 0.000200 | 0.003200 | yes |
| qwen | imdb | 9 | 0.011308 | 0.000200 | 0.003200 | yes |
| granite | multi_nli | 158 | 0.001632 | 0.000400 | 0.003200 | no |
| llama | civil_comments | 12 | 0.001551 | 0.000200 | 0.003200 | no |
| mistral | ag_news | 294 | 0.008327 | 0.000200 | 0.003200 | yes |
| qwen | multi_nli | 14 | 0.007030 | 0.000200 | 0.003200 | yes |
| granite | imdb | 416 | 0.018193 | 0.000200 | 0.003200 | yes |
| llama | ag_news | 21 | 0.006262 | 0.000200 | 0.003200 | yes |
| mistral | civil_comments | 533 | 0.007178 | 0.000200 | 0.003200 | yes |
| qwen | civil_comments | 93 | 0.007076 | 0.000200 | 0.003200 | yes |
| granite | ag_news | 248 | 0.021369 | 0.000200 | 0.003200 | yes |
| llama | imdb | 59 | 0.001338 | 0.000200 | 0.003200 | no |
| mistral | multi_nli | 397 | 0.009398 | 0.000200 | 0.003200 | yes |

## Gate checks

| criterion | passed |
|---|---:|
| complete_16_cell_grid | yes |
| positive_global_effect | yes |
| global_replay_p_at_most_0.05 | yes |
| at_least_three_positive_model_families | yes |
| at_least_three_positive_tasks | yes |
| all_leave_one_model_out_means_positive | yes |
| all_leave_one_task_out_means_positive | yes |
| all_four_global_malformed_treatments_at_least_0.002 | yes |

All repaired vectors are complete. The four former malformed-output treatments therefore collapse to the same result.

This analysis measures reference-relative disagreement dependence, not semantic accuracy or confidence-interval coverage.
