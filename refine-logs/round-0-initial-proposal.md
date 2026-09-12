# Round 0 Initial Proposal

## Problem Anchor

- **Bottom-line problem**: Determine whether conclusions drawn from a fixed historical recommender benchmark depend on interactions recorded only after its test period because full-timeline k-core filtering lets the future decide which historical users and items are eligible.
- **Core technical bottleneck**: Existing work shows that temporal leakage, filtering, and evaluation design can change recommender results, but it does not isolate future-conditioned eligibility from changes to training data, labels, candidates, or predictions.
- **Non-goals**: This project does not invent temporal splitting, discover k-core sensitivity, propose a new recommender, train on future records, or interpret changing-population averages as causal effects on a fixed population.
- **Hard constraints**: Use public datasets, CPU baselines, deterministic preprocessing, no paid APIs, and a two-day pilot before expanding. No empirical effect or ranking reversal is assumed.
- **Success criteria**: The clean pipeline remains bitwise invariant as the observation horizon grows, while a realistic post-test suffix produces a reproducible eligibility change and an evaluation consequence that survives fixed-prediction attribution, common-cohort analysis, and a size-matched historical-only cohort null. A stable near-zero effect across datasets is an informative boundary result, not a positive headline result.

## Failure Analysis

The benchmark construction order creates the failure. A pipeline first observes the complete user-item graph, applies iterative k-core filtering, and only afterward creates chronological training and test windows. An interaction after the historical test cutoff can therefore raise a user's or item's degree above the threshold and retroactively admit earlier records. The benchmark's historical evaluation cohort then depends on how long the dataset happened to be observed.

The obvious part is that changing a cohort can change its mean score. The unresolved question is whether realistic future suffixes induce informative selection that materially changes historical model comparisons after all other inputs are fixed. The experiment must distinguish that effect from generic sample-size variation, fallback handling for newly eligible entities, and filtered training-row changes.

## Dominant Contribution

**A controlled horizon-dependence audit for recommender benchmarks:** hold the historical train/validation/test windows, candidate catalog, labels, fitted predictions, and scoring code fixed; reveal progressively later interactions only to recompute eligibility; then attribute any historical evaluation change to the evaluation cohort, the filtered historical training rows, or their interaction.

This is a measurement and attribution contribution. The novelty claim is the controlled intervention and decomposition, not temporal leakage, k-core filtering, provenance hashes, or train-only filtering by themselves.

## Supporting Contribution

A falsifiable reporting protocol that combines clean-pipeline invariance checks, eligibility trajectories over observation horizons, common-cohort estimates, and a size-matched historical-only cohort null. The protocol identifies whether a reported effect is specifically future-conditioned or merely the expected variability from evaluating a different-sized historical cohort.

## Method

### 1. Freeze the historical benchmark

Choose a training cutoff, validation cutoff, test cutoff, and historical item catalog using timestamps no later than the test cutoff. Never split identical timestamps. Define all labels and candidate items from this frozen historical record.

### 2. Reveal only the observation horizon

Construct suffix endpoints at the test cutoff and progressively later timestamps. For each endpoint, recompute the same iterative k-core mask on the graph observed through that endpoint. Future records may affect only eligibility. They never enter training, validation, test labels, hyperparameter selection, candidate construction, or feature computation.

### 3. Estimate the eligibility trajectory

For every horizon, record retained users, items, historical interactions, test targets, and overlap with the clean train-only cohort. Characterize retained and excluded entities using historical covariates only. This establishes whether future eligibility is associated with prior activity or item popularity rather than merely changing sample size.

### 4. Separate selection from training effects

Run a 2-by-2 design:

| Factor | Clean | Future-conditioned |
|---|---|---|
| Historical training-row mask | Train-only eligibility | Horizon-dependent eligibility applied only to historical rows |
| Historical evaluation cohort | Train-only eligible cohort | Horizon-dependent eligible cohort |

The primary selection estimate freezes a trained model and every user's predictions, then changes only which eligible users contribute to the metric. The training estimate fixes a common evaluation cohort and changes only which historical training rows are retained. Report the interaction descriptively without a causal-population claim.

### 5. Rule out trivial explanations

- Compare the observed cohort effect with repeated size-matched cohorts selected using historical information only.
- Report common-cohort results so changing target populations are never confused with same-population performance changes.
- Freeze the candidate catalog and add an intersection-catalog sensitivity check so unseen-item fallback rules cannot drive the result.
- Hash training rows, targets, catalog, model inputs, and predictions. Any clean-protocol change across suffixes is an implementation failure.

## Minimal Validation

Use raw Amazon 2014 Beauty ratings for the pilot. Run popularity and sparse item-item nearest-neighbor recommendation with deterministic ties and full-catalog evaluation. Report Recall@20, NDCG@10, pairwise model gaps, cohort retention, and paired user-bootstrap intervals across three observation horizons. Add MovieLens 1M and stronger CPU baselines only if the pilot passes the prespecified continuation rule.

## Claim Boundaries

The paper may claim that a common preprocessing order makes a historical benchmark horizon-dependent only if the invariant controls pass. It may claim practical evaluation consequences only if they are reproducible under realistic suffixes and exceed the historical-only size-matched null. It may not claim that all recommender benchmarks are invalid, that ranking reversals are guaranteed, or that k-core filtering itself is newly identified as risky.

