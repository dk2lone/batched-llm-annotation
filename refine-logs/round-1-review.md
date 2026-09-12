# Round 1 Review

<details>
<summary>Full reviewer response</summary>

# C1 Research-Refine Review

**CALIBRATION: none**

## Problem Anchor

- **Bottom-line problem**: Determine whether conclusions drawn from a fixed historical recommender benchmark depend on interactions recorded only after its test period because full-timeline k-core filtering lets the future decide which historical users and items are eligible.
- **Core technical bottleneck**: Existing work shows that temporal leakage, filtering, and evaluation design can change recommender results, but it does not isolate future-conditioned eligibility from changes to training data, labels, candidates, or predictions.
- **Non-goals**: This project does not invent temporal splitting, discover k-core sensitivity, propose a new recommender, train on future records, or interpret changing-population averages as causal effects on a fixed population.
- **Hard constraints**: Use public datasets, CPU baselines, deterministic preprocessing, no paid APIs, and a two-day pilot before expanding. No empirical effect or ranking reversal is assumed.
- **Success criteria**: The clean pipeline remains bitwise invariant as the observation horizon grows, while a realistic post-test suffix produces a reproducible eligibility change and an evaluation consequence that survives fixed-prediction attribution, common-cohort analysis, and a size-matched historical-only cohort null. A stable near-zero effect across datasets is an informative boundary result, not a positive headline result.

The anchor is preserved. The proposal consistently studies future-conditioned benchmark membership, not future-data training or causal performance changes.

## Scores

| Dimension | Score | Weighted contribution |
|---|---:|---:|
| Problem Fidelity | 9/10 | 1.35 |
| Method Specificity | 6/10 | 1.50 |
| Contribution Quality | 5/10 | 1.25 |
| Frontier Leverage | 7/10 | 1.05 |
| Feasibility | 7/10 | 0.70 |
| Validation Focus | 6/10 | 0.30 |
| Venue Readiness | 5/10 | 0.25 |
| **Weighted composite** | **6.40/10** | |

## GAP

The proposal defines a clean observation-horizon intervention, but it has not yet turned the result into something stronger than “changing the evaluated cohort can change its mean.” Its defensible gap is narrower: future-conditioned admission may systematically alter a fixed pairwise model gap beyond variation produced by historically defined cohorts of comparable size and composition. The current design also leaves the scoreable evaluation universe ambiguous. Because k-core membership is monotone as records are added, a universe restricted to baseline k-core-eligible users cannot change under later suffixes; a broader universe admits users whom the clean filtered model may not be able to score without fallback. Until this is resolved, the primary effect can be either identically zero or driven by cold-user behavior.

## Dimension findings

### Problem Fidelity: 9/10

The proposal directly targets the fixed problem and states the relevant non-goals. It correctly avoids treating changing-population averages as causal effects.

### Method Specificity: 6/10

**Weakness:** The eligibility predicate, fixed universe, and scoring interface are not jointly coherent. “Fixed warm historical evaluation universe using only pre-test eligibility” could make the cohort invariant because later k-core sets contain the earlier core. If the universe includes later-admitted users, a model trained after the clean k-core mask may have discarded all their rows. The intersection-catalog check addresses unseen items, not unscoreable users.

**Concrete fix:** Define three objects formally:

1. \(U^*\): every user with a pre-cutoff interaction history and a frozen historical test target, constructed before k-core filtering.
2. \(E_h(u)\): the exact user-and-target eligibility indicator produced by the graph observed through horizon \(h\).
3. \(d_u\): a fixed per-user difference between two models trained on a suffix-independent historical row set broad enough to score every user in \(U^*\).

Make the primary estimand

\[
\Delta_{\mathrm{sel}}(h)
=
\frac{\sum_{u\in U^*}E_h(u)d_u}{\sum_{u\in U^*}E_h(u)}
-
\frac{\sum_{u\in U^*}E_{h_0}(u)d_u}{\sum_{u\in U^*}E_{h_0}(u)}.
\]

Use the 2×2 filtered-training analysis only as secondary attribution. Specify separate deterministic policies for missing users and missing items, and report whether either is activated.

**Priority: CRITICAL**

### Contribution Quality: 5/10

**Weakness:** Horizon dependence follows almost mechanically from applying k-core before a temporal split. A changed mean is also algebraically inevitable whenever admitted users differ from incumbents. Hashing, common-cohort reporting, and factorial decomposition are useful controls but not individually strong innovations.

**Concrete fix:** Make the sole contribution a structured model-comparison selection effect. Decompose the change in the pairwise model gap into the admitted cohort’s weight and its mean advantage relative to incumbents. Compare the observed effect against a null that preserves cohort size and pre-test degree, activity, target popularity, and target count. The claim becomes that future-conditioned eligibility selects users with model-relative difficulty not explained by ordinary historically observable cohort variation.

Treat invariance checks and the 2×2 analysis as controls, not additional contributions.

**Priority: CRITICAL**

### Frontier Leverage: 7/10

The deliberately simple design is appropriate. LLM, VLM, RL, or other foundation-model components would not solve the evaluation-validity bottleneck. The recent leverage comes from benchmark auditing and reproducible counterfactual evaluation, not model architecture.

### Feasibility: 7/10

The two-day pilot is plausible with deterministic popularity and sparse item-item models. Full-catalog scoring on roughly two million interactions may still require candidate batching and memory limits, but no new methodological component is needed.

### Validation Focus: 6/10

**Weakness:** The validation suite is mostly focused, but the decisive threshold remains unspecified and the proposed size-matched null is too weak. A result can exceed a size-only null simply because future admissions concentrate low-degree users, which is predictable from historical data. A near-zero result on one dataset is not a credible boundary finding.

**Concrete fix:** Before observing outcomes, define continuation as both:

- the absolute pairwise-gap shift exceeds the 95th percentile of a historical-covariate-matched permutation null; and
- the shift exceeds a fixed fraction of the original model gap, with ranking reversal reported only as a stronger secondary outcome.

Do not claim a null boundary from the Amazon pilot. Require replication with tight uncertainty on a second dataset before making that claim.

**Priority: IMPORTANT**

### Venue Readiness: 5/10

**Weakness:** The project is suitable for a high-school symposium pilot, but its paper-level contribution remains conditional on an unverified empirical result. Without structured selection beyond matched cohort variation, reviewers can dismiss it as a controlled demonstration of known filtering sensitivity.

**Concrete fix:** Present the method as a formally specified horizon-sensitivity estimand with an entrant/incumbent decomposition. Advance to a paper only if the pilot shows either a reproducible excess shift in pairwise model gaps or a replicated, tightly bounded null result. Do not lead with the graph-theoretic fact that future edges alter k-core membership.

**Priority: IMPORTANT**

## Method novelty versus empirical novelty

- **Method novelty:** Moderate. The fixed-prediction horizon intervention and training-mask/cohort attribution are a new application of standard experimental and reproducibility tools.
- **Unverified empirical novelty:** Potentially strong if realistic suffixes systematically change model gaps beyond covariate-matched cohort variation.
- **Already established:** k-core sensitivity, preprocessing leakage, evaluation-design sensitivity, and the fact that cohort composition affects aggregate scores.
- **Not yet supportable:** Any assertion that rankings materially change, that the effect generalizes across datasets, or that a near-zero result defines broad safe conditions.

## Simplification Opportunities

1. Remove the “supporting contribution” label. Hashes, invariance checks, common-cohort estimates, and null cohorts are validity controls.
2. Make the fixed-prediction selection estimand the only primary analysis. Run the full 2×2 retraining design only if that effect clears the continuation rule.
3. Keep one ranking metric and one utility metric for the pilot. Additional models, cutoffs, and k values belong only in the expanded study.

## Modernization Opportunities

Do not add foundation-model components. If the pilot passes, add one maintained contemporary recommender for external validity instead of expanding into several legacy baselines. This is validation, not method novelty.

## Drift Warning

**NONE.** The proposal remains centered on whether post-test observations determine historical benchmark eligibility. The required revisions clarify the estimand and scoreability without changing that problem.

## Top fixes

1. **CRITICAL:** Resolve the monotone-k-core versus fixed-universe contradiction and guarantee fixed predictions for every potentially admitted user.
2. **CRITICAL:** Replace “the cohort mean changed” with a pairwise-gap estimand, entrant decomposition, and historically covariate-matched null.
3. **IMPORTANT:** Predefine a practical continuation threshold and require two-dataset evidence before interpreting a near-zero result.
4. **IMPORTANT:** Treat the audit protocol and 2×2 design as controls around one contribution, not parallel contributions.

## Verdict

**REVISE — 6.40/10**

The direction is viable, but it is not READY. The proposal preserves the problem and is feasible, yet its main estimand is currently ambiguous and its contribution remains vulnerable to the central obviousness objection.

</details>

