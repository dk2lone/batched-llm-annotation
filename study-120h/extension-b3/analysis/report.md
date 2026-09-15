# X1R4 A9 full-grid analysis

## Result

The frozen cross-family recurrence gate **passes**.

The equal-cell joint-minus-exact covariance effect is 0.009610 (one-sided replay p = 0.000200; 95% shift interval [0.009360, 0.009857]).

13 of 16 cells are Holm-positive at the 0.002 practical floor, covering 4 model families and 4 tasks.

This analysis measures reference-relative disagreement dependence. It does not measure confidence-interval coverage; that requires the separate B4 repeated-sampling stage.

## Cells

| family | task | joint cov. | exact cov. | delta | raw p | Holm p | interaction |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen | ag_news | 0.003061 | -0.000249 | 0.003311 | 0.000200 | 0.003200 | -0.009426 |
| granite | civil_comments | 0.001441 | -0.000178 | 0.001620 | 0.000200 | 0.003200 | -0.005623 |
| llama | multi_nli | 0.008467 | 0.001994 | 0.006473 | 0.000200 | 0.003200 | 0.004703 |
| mistral | imdb | 0.018311 | 0.000868 | 0.017444 | 0.000200 | 0.003200 | 0.000751 |
| qwen | imdb | 0.010795 | -0.000514 | 0.011309 | 0.000200 | 0.003200 | 0.001563 |
| granite | multi_nli | 0.003561 | -0.000571 | 0.004132 | 0.000200 | 0.003200 | -0.007020 |
| llama | civil_comments | -0.000326 | -0.001891 | 0.001564 | 0.000200 | 0.003200 | 0.003703 |
| mistral | ag_news | 0.023940 | 0.001384 | 0.022556 | 0.000200 | 0.003200 | 0.002874 |
| qwen | multi_nli | 0.006568 | -0.000568 | 0.007136 | 0.000200 | 0.003200 | 0.001986 |
| granite | imdb | 0.019491 | 0.000712 | 0.018779 | 0.000200 | 0.003200 | 0.003031 |
| llama | ag_news | 0.010281 | 0.003985 | 0.006296 | 0.000200 | 0.003200 | -0.003061 |
| mistral | civil_comments | 0.004289 | 0.000058 | 0.004231 | 0.000200 | 0.003200 | -0.003956 |
| qwen | civil_comments | 0.007148 | 0.000029 | 0.007119 | 0.000200 | 0.003200 | 0.005877 |
| granite | ag_news | 0.030653 | 0.002303 | 0.028350 | 0.000200 | 0.003200 | 0.009612 |
| llama | imdb | 0.000930 | -0.000092 | 0.001022 | 0.000200 | 0.003200 | -0.005345 |
| mistral | multi_nli | 0.011977 | -0.000450 | 0.012427 | 0.000200 | 0.003200 | 0.000331 |

## Malformed-output sensitivity

| treatment | global delta | at least 0.002 |
|---|---:|---:|
| unresolved_as_correct | 0.009610 | yes |
| unresolved_as_error | 0.015495 | yes |
| unresolved_dropped | 0.007566 | yes |
| whole_call_discarded | 0.007522 | yes |

## Model and task margins

| margin | delta |
|---|---:|
| model: qwen | 0.007219 |
| model: granite | 0.013220 |
| model: llama | 0.003839 |
| model: mistral | 0.014164 |
| task: civil_comments | 0.003633 |
| task: multi_nli | 0.007542 |
| task: imdb | 0.012138 |
| task: ag_news | 0.015128 |

## Leave-one-out checks

| omitted margin | remaining-grid delta |
|---|---:|
| model: qwen | 0.010408 |
| model: granite | 0.008407 |
| model: llama | 0.011534 |
| model: mistral | 0.008092 |
| task: civil_comments | 0.011603 |
| task: multi_nli | 0.010300 |
| task: imdb | 0.008768 |
| task: ag_news | 0.007771 |

## Accuracy and parse moderators

| family | task | joint acc. | exact acc. | solo acc. | joint parse | exact parse | solo parse | terminal malformed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | ag_news | 0.785487 | 0.638750 | 0.699255 | 0.998698 | 0.999878 | 0.998901 | 0 |
| granite | civil_comments | 0.900716 | 0.873606 | 0.826650 | 0.988932 | 0.985107 | 0.980225 | 61 |
| llama | multi_nli | 0.485273 | 0.412790 | 0.410203 | 0.997396 | 0.999634 | 0.997803 | 1 |
| mistral | imdb | 0.594606 | 0.616247 | 0.671260 | 0.974609 | 0.997762 | 0.992188 | 0 |
| qwen | imdb | 0.896349 | 0.901595 | 0.944071 | 0.998698 | 0.999837 | 0.999634 | 0 |
| granite | multi_nli | 0.334670 | 0.383007 | 0.336270 | 0.994792 | 0.994710 | 0.997559 | 7 |
| llama | civil_comments | 0.713379 | 0.784299 | 0.836652 | 1.000000 | 0.999797 | 0.999146 | 0 |
| mistral | ag_news | 0.342063 | 0.445999 | 0.562646 | 0.964844 | 0.991984 | 0.994751 | 3 |
| qwen | multi_nli | 0.731869 | 0.633067 | 0.731850 | 0.998698 | 0.999919 | 0.998779 | 0 |
| granite | imdb | 0.811575 | 0.847960 | 0.909833 | 0.968099 | 0.989421 | 0.986938 | 25 |
| llama | ag_news | 0.669992 | 0.578337 | 0.609950 | 0.999349 | 0.999634 | 0.998657 | 0 |
| mistral | civil_comments | 0.878015 | 0.872464 | 0.798580 | 0.904297 | 0.990967 | 0.979980 | 0 |
| qwen | civil_comments | 0.876061 | 0.895215 | 0.756431 | 0.996745 | 0.999146 | 0.991821 | 0 |
| granite | ag_news | 0.366415 | 0.501762 | 0.523003 | 0.981120 | 0.992757 | 0.994995 | 6 |
| llama | imdb | 0.912821 | 0.889681 | 0.909424 | 0.987630 | 0.998820 | 0.998657 | 24 |
| mistral | multi_nli | 0.385704 | 0.584137 | 0.679631 | 0.992188 | 0.987061 | 0.991821 | 0 |

## Fake-batch controls

| family | task | exact-context fake covariance | solo fake covariance |
|---|---|---:|---:|
| qwen | ag_news | -0.000249 | -0.000286 |
| granite | civil_comments | -0.000178 | -0.000108 |
| llama | multi_nli | 0.001994 | 0.000042 |
| mistral | imdb | 0.000868 | -0.000860 |
| qwen | imdb | -0.000514 | -0.000104 |
| granite | multi_nli | -0.000571 | -0.000672 |
| llama | civil_comments | -0.001891 | -0.000476 |
| mistral | ag_news | 0.001384 | -0.000045 |
| qwen | multi_nli | -0.000568 | -0.000461 |
| granite | imdb | 0.000712 | -0.000085 |
| llama | ag_news | 0.003985 | -0.000185 |
| mistral | civil_comments | 0.000058 | 0.000817 |
| qwen | civil_comments | 0.000029 | -0.000514 |
| granite | ag_news | 0.002303 | 0.000033 |
| llama | imdb | -0.000092 | -0.000212 |
| mistral | multi_nli | -0.000450 | -0.000161 |

## Frozen gate checks

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

Every clean event ledger, retry link, call identity, prompt hash, assignment, frame binding, execution plan, and sealed schedule was verified before analysis. The excluded interrupted Llama run was not read as scientific evidence.
