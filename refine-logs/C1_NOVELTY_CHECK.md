# C1 Novelty Check Report

Search date: 2026-09-08. This is a bounded literature check, not a systematic review.

## Proposed method

Freeze a historical recommender task and all per-user predictions, then vary only the dataset snapshot cutoff used to compute k-core eligibility. Trace newly admitted historical users by whether they were directly active in the future suffix or entered without any future interaction of their own, and quantify each group's contribution to the pairwise model-gap shift.

## Core claims

1. **Dynamic k-core updates propagate beyond changed-edge endpoints — LOW.** This is established in dynamic core-maintenance research and is background, not a contribution.
2. **Future-only interactions can admit historical recommender-evaluation users who have no future interaction themselves — MEDIUM.** No exact empirical study was located, but the possibility follows from known k-core propagation. The defensible novelty is measurement and provenance in recommender benchmarks.
3. **Nonincident entrants materially change a fixed model comparison — MEDIUM-HIGH if demonstrated.** This remains an empirical hypothesis. It is valuable only if the effect is practically meaningful, exceeds the historical-balance reference, and replicates.
4. **A frozen-prediction, provenance-resolved audit isolates this effect — MEDIUM.** The controls are a useful combination of standard techniques, not a new statistical method.

## Closest prior work

| Work | Overlap | Remaining difference |
|---|---|---|
| Li and Yu, *Efficient Core Maintenance in Large Dynamic Graphs* (2012) | Edge updates can alter a propagated set of vertices | No temporal recommender snapshots, evaluation cohorts, or model comparisons |
| Guo and Sekerinski, *Parallel Order-Based Core Maintenance in Dynamic Graphs* (ICPP 2023; revised 2024) | Core maintenance under bursts of edge updates and affected vertices | No future-incident versus nonincident evaluation provenance |
| Doerfel et al., *The Role of Cores in Recommender Benchmarking* (TIST 2016) | Core construction changes benchmark populations and algorithm comparisons | No future-only intervention or frozen predictions |
| Winter, *Dataset Pruning in RecSys and ML* (2025 thesis) | Pruning levels strongly select users and affect algorithm performance | Varies pruning and retrains models; no snapshot-only eligibility intervention |
| Parajuli et al., *On the Convergent Validity of Offline Evaluation Designs for Recommender Systems* (RecSys 2026) | Evaluation design, filtering thresholds, and candidate construction alter fixed-model conclusions | No future snapshot cutoff, entrant provenance, or historical-balance reference |
| Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation* (TOIS 2023) | Future information changes accuracy and model order | Future data enters model training; C1 changes eligibility only |

## Search coverage

Claims were searched with direct recommender/k-core terms, dynamic-graph propagation terms, and adjacent benchmark-selection terms. Searches included arXiv, general scholarly web discovery, Semantic Scholar results retained from the earlier run, ICLR 2025/2026, NeurIPS 2025, ICML 2025/2026, and March–September 2026 arXiv work. At least three formulations were used for each claim. The recent search found Parajuli et al. but no paper explicitly tracing future-suffix-induced nonincident historical entrants and their contribution to a frozen recommender comparison.

Failed or rate-limited searches were treated as unknown evidence. Absence of a located match is not proof that none exists.

## Assessment

- **Overall novelty:** 6/10 for the provenance-resolved variant; 4/10 in Claude Opus 5's broader skeptical assessment of the original C1 method.
- **Recommendation:** PROCEED WITH CAUTION.
- **Method novelty:** modest. Frozen predictions, snapshot-only eligibility, balance matching, and subgroup decomposition are individually standard.
- **Potential empirical novelty:** meaningful if nonincident entrants are common and explain a replicated comparison shift beyond matched historical variation.
- **Main risk:** reviewers may see the result as an obvious consequence of known k-core propagation and known filtering sensitivity.

## Defensible positioning

> We introduce a provenance-resolved audit of future-conditioned k-core selection in recommender evaluation. Holding historical targets and per-user predictions fixed, the audit distinguishes users directly touched by post-test activity from historical users admitted only through recursive k-core propagation, and quantifies each group's contribution to shifts in pairwise model comparisons.

Do not claim discovery of nonlocal k-core propagation, temporal leakage in general, or the fact that filtering changes benchmark outcomes. Treat the nonincident-entrant analysis as a diagnostic extension unless the pilot shows a substantial replicated effect.

