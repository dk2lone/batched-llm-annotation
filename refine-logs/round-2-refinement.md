# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem**: Determine whether conclusions drawn from a fixed historical recommender benchmark depend on interactions recorded only after its test period because full-timeline k-core filtering lets the future decide which historical users and items are eligible.
- **Core technical bottleneck**: Existing work shows that temporal leakage, filtering, and evaluation design can change recommender results, but it does not isolate future-conditioned eligibility from changes to training data, labels, candidates, or predictions.
- **Non-goals**: This project does not invent temporal splitting, discover k-core sensitivity, propose a new recommender, train on future records, or interpret changing-population averages as causal effects on a fixed population.
- **Hard constraints**: Use public datasets, CPU baselines, deterministic preprocessing, no paid APIs, and a two-day pilot before expanding. No empirical effect or ranking reversal is assumed.
- **Success criteria**: The clean pipeline remains bitwise invariant as the observation horizon grows, while a realistic post-test suffix produces a reproducible eligibility change and an evaluation consequence that survives fixed-prediction attribution, common-cohort analysis, and a size-matched historical-only cohort null. A stable near-zero effect across datasets is an informative boundary result, not a positive headline result.

## Anchor Check

- **Original bottleneck**: Future records can retroactively select the scoring population for a fixed historical recommender comparison.
- **Why the revised method still addresses it**: The historical rows, one target per user, model predictions, and practical thresholds are frozen before horizon-specific k-core eligibility is computed.
- **Reviewer suggestions rejected as drift**: None.

## Simplicity Check

- **Dominant contribution after revision**: One fixed-history horizon-sensitivity audit for a pairwise recommender gap.
- **Components removed or merged**: The entrant equation is interpretation, not a separate contribution; the 2×2 study is a paper-stage bridge, not part of the initial pilot.
- **Reviewer suggestions rejected as unnecessary complexity**: Foundation-model methods remain excluded.
- **Why the remaining mechanism is the smallest adequate route**: A deterministic target rule and fixed stratified resampling close the two remaining identification gaps without adding a model or learning component.

## Changes Made

### 1. Froze one evaluation target per user

- **Reviewer said**: Horizon-dependent target membership could silently change the labels.
- **Action**: For each user, choose the chronologically last eligible interaction in the frozen historical test window, breaking timestamp ties by stable item ID. Build \(U^*\) only from users with earlier training history and whose chosen target is in the frozen training-item catalog.
- **Reasoning**: Each user has one immutable target, so \(E_h(u)\) can only include or exclude a fixed scoring unit.
- **Impact on core method**: Labels and per-user predictions cannot change with the suffix.

### 2. Prespecified the matched null

- **Reviewer said**: The risk set, matching algorithm, balance checks, and repetitions were unspecified.
- **Action**: Use \(R=U^*\setminus E_{t_0}\). For each horizon, exclude actual entrants from the donor pool, place users in prespecified quantile strata based on training degree, 30-day pre-test activity, and target-item training popularity, and sample the observed entrant count within strata 2,000 times. Require standardized mean differences below 0.1; otherwise the null is unsupported for that horizon.
- **Reasoning**: Every matching variable is historical, the sample size and composition are controlled, and lack of overlap becomes an explicit failure rather than a hidden modeling choice.
- **Impact on core method**: An excess shift cannot be explained only by entrant count or the matched historical covariates.

### 3. Made thresholds meaningful near a zero baseline gap

- **Reviewer said**: A fraction of a near-zero baseline gap is not useful.
- **Action**: Freeze both a null threshold and metric-specific absolute practical thresholds before scoring: 0.002 Recall@20 and 0.001 NDCG@10. A pilot signal must exceed the matched-null 95th percentile and the relevant absolute threshold.
- **Reasoning**: The gate remains defined when the baseline pairwise gap is zero.
- **Impact on core method**: The continuation decision is reproducible and cannot be chosen after viewing results.

## Revised Proposal

### Title

**The Future Chooses the Test Set: A Fixed-History Audit of Recommender Model Comparisons**

### Contribution statement

C1 contributes a controlled audit that holds a historical recommendation task and every per-user prediction fixed, then varies only how far into the future the interaction graph is observed when computing k-core eligibility. Its primary estimand measures whether future-conditioned admission shifts a pairwise model gap beyond changes produced by historically comparable cohorts. The audit separates a new method contribution from its unverified empirical question: whether the excess shift is practically meaningful on real benchmarks.

### Fixed historical task

Freeze a training cutoff and historical test window. For every user with training history, select the chronologically last interaction in the test window as one target, breaking timestamp ties by stable item ID. Restrict the pre-k-core universe \(U^*\) to users whose frozen target is in the training-item catalog. Train two recommenders once on all suffix-independent historical training rows and save each user's metric difference \(d_u=m_A(u)-m_B(u)\).

### Horizon intervention and estimand

At each observation horizon \(h\), compute the same iterative k-core using records observed through \(h\). Let \(E_h(u)=1\) when user \(u\) and that user's frozen target item are in the resulting core. The suffix can change only \(E_h\); historical rows, targets, catalog, models, and predictions remain fixed.

Measure

\[
\Delta_{\mathrm{sel}}(h)=
\operatorname{mean}\{d_u:E_h(u)=1\}-
\operatorname{mean}\{d_u:E_{t_0}(u)=1\}.
\]

Interpret the shift using the share of newly admitted users and the difference between entrants' and incumbents' mean \(d_u\). This is an explanation of the estimand, not another claimed contribution.

### Historical-covariate-matched null

Define the baseline risk set \(R=U^*\setminus E_{t_0}\). For horizon \(h\), actual entrants are users in \(R\) with \(E_h(u)=1\). Excluding those entrants, form a donor pool from the remaining risk set. Before computing any outcomes, freeze quantile strata for historical training degree, activity in the 30 days before the test window, and target-item training popularity. Draw the same number of pseudo-entrants within those strata 2,000 times and recompute the pairwise-gap shift after adding each pseudo-cohort to the incumbents. Require standardized mean differences below 0.1 for each covariate; if overlap is inadequate, report that the matched-null comparison is unsupported at that horizon.

### Pilot gate

Use raw Amazon 2014 Beauty ratings, three observation horizons, popularity and sparse item-item nearest-neighbor models, Recall@20, NDCG@10, and paired user-bootstrap intervals. Continue only if the observed absolute shift exceeds both the matched-null 95th percentile and a frozen absolute threshold: 0.002 Recall@20 or 0.001 NDCG@10 for the corresponding analysis. A ranking reversal is secondary evidence, not a requirement.

### Connection to benchmark practice

The unfiltered frozen models isolate selection. If the pilot passes, run the 2×2 training-mask/evaluation-cohort study on Amazon and one second dataset, including one maintained contemporary recommender pair. That paper-stage analysis tests whether the isolated selection effect persists when using the filtered training pipelines found in practice. A one-dataset null ends the positive pilot; a broad null claim requires replication and tight uncertainty.

### Claim boundary

The proposed novelty is the fixed-history, fixed-prediction horizon-sensitivity audit for pairwise model comparisons. The project does not claim that changing cohorts changes averages is new. It can claim systematic future-conditioned selection only if observed shifts exceed the matched historical null and practical threshold. Temporal leakage, k-core sensitivity, provenance hashing, and train-only filtering remain prior concepts or controls.

