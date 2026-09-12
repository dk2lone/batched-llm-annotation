# Review Summary

**Problem**: Future records can retroactively select the users and items scored in a fixed historical recommender benchmark.
**Initial Approach**: Append future suffixes only for k-core eligibility and use a 2×2 training-mask/cohort comparison.
**Date**: 2026-09-08
**Rounds**: 4 / 5
**Final Score**: 9.05 / 10
**Final Verdict**: READY

## Problem Anchor

- **Bottom-line problem**: Determine whether conclusions drawn from a fixed historical recommender benchmark depend on interactions recorded only after its test period because full-timeline k-core filtering lets the future decide which historical users and items are eligible.
- **Core technical bottleneck**: Existing work shows that temporal leakage, filtering, and evaluation design can change recommender results, but it does not isolate future-conditioned eligibility from changes to training data, labels, candidates, or predictions.
- **Non-goals**: This project does not invent temporal splitting, discover k-core sensitivity, propose a new recommender, train on future records, or interpret changing-population averages as causal effects on a fixed population.
- **Hard constraints**: Use public datasets, CPU baselines, deterministic preprocessing, no paid APIs, and a two-day pilot before expanding. No empirical effect or ranking reversal is assumed.
- **Success criteria**: The clean pipeline remains bitwise invariant as the observation horizon grows, while a realistic post-test suffix produces a reproducible eligibility change and an evaluation consequence that survives fixed-prediction attribution, common-cohort analysis, and a size-matched historical-only cohort null. A stable near-zero effect across datasets is an informative boundary result, not a positive headline result.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What Changed | Solved? | Remaining Risk |
|---|---|---|---|---|
| 1 | Fixed cohort could not gain members; changed mean was obvious | Defined a pre-k-core warm universe, frozen pairwise gap, entrant interpretation, and matched null | Yes | Null underspecified |
| 2 | Target labels and matching procedure ambiguous | Froze one target per user and specified risk set, strata, repetitions, balance, and thresholds | Yes | Multiple endpoints |
| 3 | Six possible pilot tests allowed opportunistic selection | Froze Recall@20 at the final horizon as the only gate and fixed sparse-stratum merging | Yes | Empirical effect unverified |
| 4 | No proposal-design blocker | No further method change | Yes | Pilot may be negative |

## Final Status

- Anchor status: preserved
- Focus status: tight
- Modernity status: intentionally conservative; foundation-model components do not address the problem
- Strongest part: the suffix changes only eligibility while every scoring input and prediction stays fixed
- Remaining weakness: publishable empirical novelty depends on passing the pilot and replication

