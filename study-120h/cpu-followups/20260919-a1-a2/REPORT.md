# CPU follow-ups A1-A2

All calculations use existing B3/B4/B5 outputs. No model inference or GPU work was launched.

## A1: batch size and effective information

| batch | excess corr. | excess DE [95%] | effective N [95%] | total joint DE |
|---|---|---|---|---|
| 2 | 0.0795 | 1.079 [1.071, 1.088] | 7589 [7529, 7647] | 0.996 |
| 4 | 0.0807 | 1.242 [1.224, 1.259] | 6596 [6508, 6690] | 1.147 |
| 8 | 0.0711 | 1.498 [1.464, 1.533] | 5469 [5343, 5595] | 1.438 |
| 16 | 0.0596 | 1.893 [1.822, 1.965] | 4327 [4169, 4496] | 1.894 |
| 32 | 0.0446 | 2.381 [2.248, 2.504] | 3440 [3271, 3644] | 2.444 |

Pairwise excess dependence decreases after batch size 4, but the excess design effect increases monotonically because each label shares a call with more labels. The variance-equivalent effective N falls from 7,589 at batch size 2 to 3,440 at size 32.

Post-hoc descriptive transformation of the prespecified B5 standardized joint-minus-exact contrast. The 95% intervals monotonically transform the stored B5 parent-group cluster-bootstrap percentile endpoints. The excess design effect is 1 + (b-1) times the standardized contrast; it is not the total joint-arm design effect. Effective N is the corresponding variance-equivalent interpretation for 8,192 rows under exchangeable within-call dependence.

## A2: covariance-to-variance coherence

| cell | B3 implied DE | B4 empirical ratio [95%] | difference |
|---|---|---|---|
| ibm-granite/granite-3.3-8b-instruct::imdb | 2.960 | 2.711 [2.328, 3.078] | -0.249 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | 0.976 | 0.899 [0.783, 1.009] | -0.077 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | 1.755 | 1.346 [1.156, 1.553] | -0.409 |
| Qwen/Qwen3-8B::civil_comments | 1.990 | 1.702 [1.456, 1.949] | -0.288 |

Descriptive Pearson correlation: 0.9867. Descriptive Spearman correlation: 1.0000. Mean absolute difference: 0.256.

Post-hoc four-cell coherence check only. B3 converts observed joint covariance to a design effect under exchangeable within-call dependence; B4 measures empirical variance inflation in fresh repeated samples. Four cells do not support a formal calibration or generalization claim, and the disclosed B4 technical recovery boundary remains unchanged.

## Input integrity

- `/Users/dz/bigdata/study-120h/extension-b3/analysis/analysis.json`: `b540b8f41edc93013ca2d6027384b22f48cfe489dc3ec93d3f288ae3a89206da`
- `/Users/dz/bigdata/study-120h/extension-b4/revised-analysis-20260918T034636Z/REVISED_ANALYSIS.json`: `f7b5e9ac5bbeace107765a702fe2a0bdfcec34ee4c9ee51cb0c1913e22326443`
- `/Users/dz/bigdata/study-120h/extension-b5/run-01/analysis-final-a/ANALYSIS.json`: `ea6d286029be4399e8c33da65fe71ad8fd24672991ced5de3c24fe0b0951523b`
