# Round 3 Refinement

## Problem Anchor

- **Bottom-line problem**: Determine whether conclusions drawn from a fixed historical recommender benchmark depend on interactions recorded only after its test period because full-timeline k-core filtering lets the future decide which historical users and items are eligible.
- **Core technical bottleneck**: Existing work shows that temporal leakage, filtering, and evaluation design can change recommender results, but it does not isolate future-conditioned eligibility from changes to training data, labels, candidates, or predictions.
- **Non-goals**: This project does not invent temporal splitting, discover k-core sensitivity, propose a new recommender, train on future records, or interpret changing-population averages as causal effects on a fixed population.
- **Hard constraints**: Use public datasets, CPU baselines, deterministic preprocessing, no paid APIs, and a two-day pilot before expanding. No empirical effect or ranking reversal is assumed.
- **Success criteria**: The clean pipeline remains bitwise invariant as the observation horizon grows, while a realistic post-test suffix produces a reproducible eligibility change and an evaluation consequence that survives fixed-prediction attribution, common-cohort analysis, and a size-matched historical-only cohort null. A stable near-zero effect across datasets is an informative boundary result, not a positive headline result.

## Anchor Check

- **Original bottleneck**: Future records can retroactively select the scoring population for a fixed historical recommender comparison.
- **Why the revised method still addresses it**: The revision changes only endpoint and null pre-registration details.
- **Reviewer suggestions rejected as drift**: None.

## Simplicity Check

- **Dominant contribution after revision**: One fixed-history horizon-sensitivity audit for a frozen pairwise model gap.
- **Components removed or merged**: Six possible pilot gates become one primary test.
- **Reviewer suggestions rejected as unnecessary complexity**: The familywise maximum-null alternative was not used because one primary endpoint is simpler and more interpretable.
- **Why the remaining mechanism is the smallest adequate route**: All other horizons and NDCG remain diagnostic rather than claim-generating.

## Changes Made

### 1. Froze the primary endpoint

- **Reviewer said**: Six available metric-horizon tests permit opportunistic selection.
- **Action**: The sole pilot continuation test is the popularity-versus-item-kNN pairwise Recall@20 gap at the final observation horizon. Intermediate horizons and NDCG@10 are descriptive.
- **Reasoning**: One endpoint avoids a familywise testing problem and directly tests the maximum realistic suffix exposure.
- **Impact on core method**: The pilot has one falsifiable gate.

### 2. Completed sparse-stratum handling

- **Reviewer said**: Stratum construction and insufficient donors need deterministic rules.
- **Action**: Compute all quantile boundaries once on \(U^*\) using data available by \(t_0\). If a donor stratum is short, merge first across the nearest target-popularity bin, then recent-activity bin, then training-degree bin; ties merge downward before upward. If the final donor pool is still too small or balance exceeds 0.1, mark the null unsupported.
- **Reasoning**: The matching procedure cannot adapt to \(d_u\) or observed shifts.
- **Impact on core method**: Null construction is fully prespecified.

## Revised Proposal

### Title

**The Future Chooses the Test Set: A Fixed-History Audit of Recommender Model Comparisons**

### Contribution statement

C1 contributes a controlled audit that freezes a historical recommendation task and every per-user prediction, then varies only the future observation horizon used to compute k-core eligibility. The primary estimand tests whether future-conditioned admission shifts a frozen pairwise model gap beyond shifts produced by historically comparable cohorts. The method contribution is the controlled audit; whether real datasets exhibit a practically meaningful excess shift remains an empirical hypothesis.

### Fixed task and intervention

Freeze a training cutoff and test window. For every user with training history, select the chronologically last test-window interaction as one target, breaking timestamp ties by stable item ID. Define \(U^*\) before k-core filtering as users whose target is in the training-item catalog. Train popularity and sparse item-kNN once on suffix-independent historical rows and save each user's Recall@20 difference \(d_u\).

At each horizon \(h\), compute the same iterative k-core on records observed through \(h\). Let \(E_h(u)=1\) only when user \(u\) and the frozen target item lie in that core. Future records may change only \(E_h\). The primary estimand is the change from \(t_0\) to the final observation horizon in the mean \(d_u\) among eligible users.

### Matched null

Real entrants are users outside the baseline eligible set who become eligible at the final horizon. Draw equally sized pseudo-entrant cohorts from the remaining baseline risk set, matching on training degree, activity during the 30 days before the test window, and target-item training popularity. Compute quantile boundaries once on \(U^*\) using data available by \(t_0\), then run 2,000 stratified draws.

When a stratum lacks donors, merge in this fixed order: nearest target-popularity bin, nearest recent-activity bin, then nearest training-degree bin, breaking equal-distance ties downward before upward. Require standardized mean differences below 0.1. If donors or balance remain inadequate, the matched-null test is unsupported rather than repaired after inspecting outcomes.

### Primary pilot gate

The only continuation test is the absolute change in the popularity-versus-item-kNN Recall@20 gap at the final horizon. It must exceed both the matched-null 95th percentile and an absolute 0.002 Recall@20 threshold. NDCG@10 and intermediate horizons are descriptive trajectories; ranking reversal is secondary evidence.

### Controls and paper-stage bridge

Hash historical rows, target labels, candidate catalog, model inputs, and per-user predictions; all must remain invariant across horizons. The warm-universe construction must produce zero missing-user and missing-item fallbacks. If the primary gate passes, run the previously specified 2×2 training-mask/evaluation-cohort attribution on Amazon and a second dataset with a maintained contemporary recommender pair. This connects the isolated selection diagnostic to filtered benchmark pipelines in practice.

### Claim boundary

The proposal does not claim that changing cohorts changes averages is new. It claims a new controlled audit for testing whether future-conditioned benchmark eligibility selects model-relative difficulty beyond a prespecified historical null. A systematic real-world effect is claimable only after the empirical gate and replication pass.

