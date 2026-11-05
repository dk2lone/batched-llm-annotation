# CPU follow-up analyses E2-E5

All analyses use existing B3/B4/B5 outputs. No model inference or GPU work was launched.

## E2: interval and recovery robustness

Whole-call bootstrap rows are a fresh seeded 2,000-draw recomputation for each replicate. Small differences from the sealed B4 bootstrap summary are Monte Carlo variation; the manuscript retains the prespecified B4 values.

| cell | method | coverage | 95% MC interval | width | vs Wald |
|---|---|---|---|---|---|
| ibm-granite/granite-3.3-8b-instruct::imdb | iid_wald | 0.7775 | [0.7335, 0.8173] | 0.0683 | reference |
| ibm-granite/granite-3.3-8b-instruct::imdb | iid_wilson | 0.7825 | [0.7388, 0.8220] | 0.0682 | +0.0050 |
| ibm-granite/granite-3.3-8b-instruct::imdb | call_t | 0.9300 | [0.9004, 0.9530] | 0.1216 | +0.1525 |
| ibm-granite/granite-3.3-8b-instruct::imdb | iid_wilson_fpc | 0.7600 | [0.7151, 0.8010] | 0.0660 | -0.0175 |
| ibm-granite/granite-3.3-8b-instruct::imdb | call_t_fpc | 0.9175 | [0.8861, 0.9425] | 0.1178 | +0.1400 |
| ibm-granite/granite-3.3-8b-instruct::imdb | whole_call_bootstrap | 0.9150 | [0.8832, 0.9404] | 0.1144 | +0.1375 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | iid_wald | 0.9675 | [0.9451, 0.9826] | 0.0785 | reference |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | iid_wilson | 0.9700 | [0.9482, 0.9844] | 0.0783 | +0.0025 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | call_t | 0.9650 | [0.9420, 0.9807] | 0.0802 | -0.0025 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | iid_wilson_fpc | 0.9675 | [0.9451, 0.9826] | 0.0758 | +0.0000 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | call_t_fpc | 0.9625 | [0.9389, 0.9789] | 0.0776 | -0.0050 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | whole_call_bootstrap | 0.9500 | [0.9238, 0.9692] | 0.0754 | -0.0175 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | iid_wald | 0.9075 | [0.8748, 0.9340] | 0.0842 | reference |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | iid_wilson | 0.9075 | [0.8748, 0.9340] | 0.0839 | +0.0000 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | call_t | 0.9525 | [0.9268, 0.9712] | 0.1079 | +0.0450 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | iid_wilson_fpc | 0.9025 | [0.8691, 0.9297] | 0.0813 | -0.0050 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | call_t_fpc | 0.9500 | [0.9238, 0.9692] | 0.1044 | +0.0425 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | whole_call_bootstrap | 0.9450 | [0.9179, 0.9652] | 0.1018 | +0.0375 |
| Qwen/Qwen3-8B::civil_comments | iid_wald | 0.8800 | [0.8441, 0.9102] | 0.0573 | reference |
| Qwen/Qwen3-8B::civil_comments | iid_wilson | 0.8750 | [0.8385, 0.9058] | 0.0574 | -0.0050 |
| Qwen/Qwen3-8B::civil_comments | call_t | 0.9600 | [0.9359, 0.9770] | 0.0837 | +0.0800 |
| Qwen/Qwen3-8B::civil_comments | iid_wilson_fpc | 0.8750 | [0.8385, 0.9058] | 0.0555 | -0.0050 |
| Qwen/Qwen3-8B::civil_comments | call_t_fpc | 0.9550 | [0.9298, 0.9731] | 0.0811 | +0.0750 |
| Qwen/Qwen3-8B::civil_comments | whole_call_bootstrap | 0.9550 | [0.9298, 0.9731] | 0.0789 | +0.0750 |

Repair-free complete-replicate subsets are descriptive only:

| cell | reference reps | evaluation reps | target | IID coverage |
|---|---|---|---|---|
| ibm-granite/granite-3.3-8b-instruct::imdb | 82 | 133 | 0.192931 | 0.7895 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | 229 | 373 | 0.290188 | 0.9651 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | 180 | 277 | 0.615028 | 0.9061 |
| Qwen/Qwen3-8B::civil_comments | 220 | 354 | 0.124902 | 0.8842 |

## E3: direct practical-floor sensitivity

| cell | delta | Holm p vs 0.002 | shift CI low | passes |
|---|---|---|---|---|
| Qwen/Qwen3-8B::ag_news | 0.003311 | 0.0416 | 0.002206 | yes |
| ibm-granite/granite-3.3-8b-instruct::civil_comments | 0.001620 | 1.0000 | 0.001111 | no |
| meta-llama/Llama-3.1-8B-Instruct::multi_nli | 0.006473 | 0.0032 | 0.005377 | yes |
| mistralai/Mistral-7B-Instruct-v0.3::imdb | 0.017444 | 0.0032 | 0.016336 | yes |
| Qwen/Qwen3-8B::imdb | 0.011309 | 0.0032 | 0.010775 | yes |
| ibm-granite/granite-3.3-8b-instruct::multi_nli | 0.004132 | 0.0032 | 0.003264 | yes |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | 0.001564 | 1.0000 | 0.000667 | no |
| mistralai/Mistral-7B-Instruct-v0.3::ag_news | 0.022556 | 0.0032 | 0.021170 | yes |
| Qwen/Qwen3-8B::multi_nli | 0.007136 | 0.0032 | 0.005933 | yes |
| ibm-granite/granite-3.3-8b-instruct::imdb | 0.018779 | 0.0032 | 0.017993 | yes |
| meta-llama/Llama-3.1-8B-Instruct::ag_news | 0.006296 | 0.0032 | 0.005060 | yes |
| mistralai/Mistral-7B-Instruct-v0.3::civil_comments | 0.004231 | 0.0032 | 0.003632 | yes |
| Qwen/Qwen3-8B::civil_comments | 0.007119 | 0.0032 | 0.006560 | yes |
| ibm-granite/granite-3.3-8b-instruct::ag_news | 0.028350 | 0.0032 | 0.026944 | yes |
| meta-llama/Llama-3.1-8B-Instruct::imdb | 0.001022 | 1.0000 | 0.000507 | no |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | 0.012427 | 0.0032 | 0.011122 | yes |

This inverts the existing mean-centred replay distribution as a location-shift sensitivity. It is not an exact randomization test of the composite causal null delta <= 0.002.

## E4: assignment and scale robustness

Global delta by assignment: 0.010133, 0.008993, 0.009705.

Leave-one-assignment-out global delta: 0.009349, 0.009919, 0.009563.

Conditional paired-call bootstrap 95% interval: [0.009073, 0.010109].

| cell | raw delta | standardized delta | excess DE | effective N |
|---|---|---|---|---|
| Qwen/Qwen3-8B::ag_news | 0.003311 | 0.0193 | 1.289 | 6353 |
| ibm-granite/granite-3.3-8b-instruct::civil_comments | 0.001620 | 0.0178 | 1.266 | 6469 |
| meta-llama/Llama-3.1-8B-Instruct::multi_nli | 0.006473 | 0.0257 | 1.385 | 5914 |
| mistralai/Mistral-7B-Instruct-v0.3::imdb | 0.017444 | 0.0729 | 2.094 | 3912 |
| Qwen/Qwen3-8B::imdb | 0.011309 | 0.1217 | 2.826 | 2899 |
| ibm-granite/granite-3.3-8b-instruct::multi_nli | 0.004132 | 0.0183 | 1.274 | 6429 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | 0.001564 | 0.0096 | 1.144 | 7162 |
| mistralai/Mistral-7B-Instruct-v0.3::ag_news | 0.022556 | 0.0977 | 2.465 | 3323 |
| Qwen/Qwen3-8B::multi_nli | 0.007136 | 0.0359 | 1.539 | 5323 |
| ibm-granite/granite-3.3-8b-instruct::imdb | 0.018779 | 0.1250 | 2.875 | 2850 |
| meta-llama/Llama-3.1-8B-Instruct::ag_news | 0.006296 | 0.0301 | 1.452 | 5642 |
| mistralai/Mistral-7B-Instruct-v0.3::civil_comments | 0.004231 | 0.0431 | 1.646 | 4976 |
| Qwen/Qwen3-8B::civil_comments | 0.007119 | 0.0658 | 1.987 | 4123 |
| ibm-granite/granite-3.3-8b-instruct::ag_news | 0.028350 | 0.1211 | 2.817 | 2909 |
| meta-llama/Llama-3.1-8B-Instruct::imdb | 0.001022 | 0.0127 | 1.191 | 6881 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | 0.012427 | 0.0522 | 1.783 | 4595 |

## E5: exploratory output-position patterns

Joint-minus-exact error slope per output position: -0.000242 (descriptive p=0.1106).

Excess covariance slope per pair-distance step: -0.000139 (descriptive p=0.0001).

Post hoc descriptive analysis of fixed outputs. Position and distance were not randomized as separate interventions, so slopes do not identify a mechanism.

| distance | joint cov. | exact cov. | delta |
|---|---|---|---|
| 1 | 0.010776 | 0.000651 | 0.010125 |
| 2 | 0.010954 | 0.000543 | 0.010412 |
| 3 | 0.010211 | 0.000731 | 0.009480 |
| 4 | 0.010293 | 0.000788 | 0.009505 |
| 5 | 0.010320 | 0.000451 | 0.009869 |
| 6 | 0.010015 | 0.000880 | 0.009136 |
| 7 | 0.009844 | -0.000013 | 0.009857 |
| 8 | 0.010211 | 0.000223 | 0.009988 |
| 9 | 0.009864 | 0.000527 | 0.009337 |
| 10 | 0.008923 | -0.000756 | 0.009679 |
| 11 | 0.008409 | 0.000023 | 0.008386 |
| 12 | 0.009031 | 0.000513 | 0.008517 |
| 13 | 0.008257 | -0.000324 | 0.008581 |
| 14 | 0.008101 | -0.000081 | 0.008182 |
| 15 | 0.007431 | -0.000791 | 0.008222 |

## Input integrity

- `/Users/dz/bigdata/study-120h/extension-b4/revised-analysis-20260918T034636Z/REVISED_ANALYSIS.json`: `f7b5e9ac5bbeace107765a702fe2a0bdfcec34ee4c9ee51cb0c1913e22326443`
- `/Users/dz/bigdata/study-120h/extension-b3/analysis/analysis.json`: `b540b8f41edc93013ca2d6027384b22f48cfe489dc3ec93d3f288ae3a89206da`
- `/Users/dz/bigdata/study-120h/extension-b5/run-01/analysis-final-a/ANALYSIS.json`: `ea6d286029be4399e8c33da65fe71ad8fd24672991ced5de3c24fe0b0951523b`
