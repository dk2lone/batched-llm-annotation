# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**: Determine whether conclusions drawn from a fixed historical recommender benchmark depend on interactions recorded only after its test period because full-timeline k-core filtering lets the future decide which historical users and items are eligible.
- **Core technical bottleneck**: Existing work shows that temporal leakage, filtering, and evaluation design can change recommender results, but it does not isolate future-conditioned eligibility from changes to training data, labels, candidates, or predictions.
- **Non-goals**: This project does not invent temporal splitting, discover k-core sensitivity, propose a new recommender, train on future records, or interpret changing-population averages as causal effects on a fixed population.
- **Hard constraints**: Use public datasets, CPU baselines, deterministic preprocessing, no paid APIs, and a two-day pilot before expanding. No empirical effect or ranking reversal is assumed.
- **Success criteria**: The clean pipeline remains bitwise invariant as the observation horizon grows, while a realistic post-test suffix produces a reproducible eligibility change and an evaluation consequence that survives fixed-prediction attribution, common-cohort analysis, and a size-matched historical-only cohort null. A stable near-zero effect across datasets is an informative boundary result, not a positive headline result.

## Anchor Check

- **Original bottleneck**: Future records can retroactively select the users and items included in a historical recommender evaluation.
- **Why the revised method still addresses it**: The revision formalizes the fixed historical universe, the horizon-specific eligibility indicator, and a fixed per-user model difference. Only the eligibility indicator can depend on the future suffix.
- **Reviewer suggestions rejected as drift**: None. The requested formalization removes ambiguity without changing the research question.

## Simplicity Check

- **Dominant contribution after revision**: A formal horizon-sensitivity estimand that measures how future-conditioned eligibility shifts a frozen pairwise model comparison, together with an entrant/incumbent decomposition and a historical-covariate-matched null.
- **Components removed or merged**: The audit protocol is no longer called a supporting contribution. The 2×2 retraining analysis becomes a gated attribution check after the primary selection effect passes.
- **Reviewer suggestions rejected as unnecessary complexity**: No foundation-model component is added because it would not address benchmark validity.
- **Why the remaining mechanism is the smallest adequate route**: One fixed historical universe, one horizon-dependent eligibility predicate, one frozen per-user model difference, and one matched null directly test the non-obvious claim.

## Changes Made

### 1. Fixed the evaluation universe and scoring interface

- **Reviewer said**: A baseline k-core universe cannot gain members as the horizon grows, while a broader universe can introduce users the clean model cannot score.
- **Action**: Define a pre-k-core warm universe and train the primary frozen models on all suffix-independent historical training rows needed to score it.
- **Reasoning**: This lets the horizon-specific mask change membership while predictions remain valid and unchanged for every potentially admitted user.
- **Impact on core method**: The primary selection effect is now identifiable without cold-user fallback.

### 2. Replaced the changing-mean claim with a model-gap estimand

- **Reviewer said**: A changed cohort mean is obvious and weak.
- **Action**: Make the pairwise per-user model difference the outcome, decompose each shift into entrant share and entrant-versus-incumbent model-relative difficulty, and test it against a historical-covariate-matched null.
- **Reasoning**: The paper now asks whether future eligibility selects a cohort with unexplained model-relative difficulty, not whether arbitrary resampling changes an average.
- **Impact on core method**: The novelty claim becomes narrower, falsifiable, and less vulnerable to the obviousness objection.

### 3. Gated the secondary attribution analysis

- **Reviewer said**: The 2×2 design and invariance checks should be controls, not parallel contributions.
- **Action**: Run the fixed-prediction selection analysis first. Run filtered-training 2×2 attribution only if the primary shift clears the prespecified continuation rule.
- **Reasoning**: This reduces pilot work while preserving a way to explain a detected effect.
- **Impact on core method**: The pilot has one primary claim and fewer moving parts.

## Revised Proposal

### Title

**The Future Chooses the Test Set: Measuring Horizon-Dependent Model Comparisons in Recommender Benchmarks**

### Thesis

When a recommender benchmark applies k-core filtering before a chronological split, later interactions can retroactively decide whose historical outcomes are scored. C1 introduces a fixed-history horizon-sensitivity estimand that tests whether this future-conditioned admission shifts a frozen pairwise model gap beyond variation explained by historically observable cohort composition.

### Formal setup

Let \(t_0\) be the frozen historical test cutoff. Construct \(U^*\) before k-core filtering as every user with at least one historical training interaction and at least one test target in the frozen training-item catalog. Train both pilot recommenders on the same unfiltered, suffix-independent historical training rows, and compute a fixed per-user metric difference \(d_u=m_A(u)-m_B(u)\) for every \(u\in U^*\).

For observation horizon \(h\ge t_0\), compute the iterative k-core on records observed through \(h\). Let \(E_h(u)=1\) only when user \(u\) and the user's frozen test target satisfy the resulting user/item eligibility rule. Future records determine \(E_h\) and nothing else.

The primary estimand is

\[
\Delta_{\mathrm{sel}}(h)=
\frac{\sum_{u\in U^*}E_h(u)d_u}{\sum_{u\in U^*}E_h(u)}-
\frac{\sum_{u\in U^*}E_{t_0}(u)d_u}{\sum_{u\in U^*}E_{t_0}(u)}.
\]

Because k-core membership grows with the observation horizon, decompose \(\Delta_{\mathrm{sel}}(h)\) into the proportion of newly admitted users and the difference between entrants' and incumbents' mean \(d_u\). This identifies whether future eligibility preferentially admits users for whom the two models perform differently.

### Null comparison

For each horizon, construct repeated pseudo-entrant cohorts from users available at \(t_0\), matching the real entrants on cohort size, pre-test user degree, recent activity, test-target count, and test-target popularity. The primary empirical signal is an absolute model-gap shift beyond the 95th percentile of this matched null. Before outcomes are inspected, also freeze a practical threshold expressed as a fraction of the baseline pairwise model gap.

### Controls and gated attribution

- Hash all historical rows, labels, candidates, model inputs, and per-user predictions; these must remain unchanged across horizons.
- Report whether any missing-user or missing-item policy is activated. The primary warm-universe construction should make both counts zero.
- Report cohort overlap and common-cohort scores, but treat them as validity checks.
- Only if the primary selection signal clears both continuation thresholds, run the 2×2 historical-training-mask/evaluation-cohort experiment to separate selection, retraining, and interaction effects.

### Pilot

Use raw Amazon 2014 Beauty ratings, three observation horizons, popularity and sparse item-item nearest-neighbor recommenders, Recall@20 as the utility metric, NDCG@10 as the ranking metric, and paired user bootstrap intervals. Continue to a paper-scale study only if the model-gap shift exceeds both the matched-null threshold and the prespecified practical threshold. A one-dataset null ends the positive pilot; a paper-level null claim requires a second dataset with tight uncertainty.

### Claim boundary

The method contribution is the fixed-history, fixed-prediction horizon-sensitivity estimand and its entrant decomposition. The audit may support an empirical claim that future-conditioned eligibility systematically selects model-relative difficulty only after the pilot passes. K-core sensitivity, temporal leakage in general, train-only filtering, provenance hashes, and cohort means are not claimed as new.

