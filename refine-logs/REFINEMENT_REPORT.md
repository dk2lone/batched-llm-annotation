# Refinement Report

**Problem**: Future records can retroactively select the users and items scored in a fixed historical recommender benchmark.
**Initial Approach**: Append future suffixes only for k-core eligibility and use a 2×2 training-mask/cohort comparison.
**Date**: 2026-09-08
**Internal rounds**: 4 / 5
**Internal final score**: 9.05 / 10
**Internal verdict**: READY for pilot design
**Independent Claude novelty score**: 4 / 10
**Independent Claude verdict**: PROCEED WITH CAUTION
**Provenance-variant novelty score**: 6 / 10, same-family provisional

## Problem Anchor

- **Bottom-line problem**: Determine whether conclusions drawn from a fixed historical recommender benchmark depend on interactions recorded only after its test period because full-timeline k-core filtering lets the future decide which historical users and items are eligible.
- **Core technical bottleneck**: Existing work shows that temporal leakage, filtering, and evaluation design can change recommender results, but it does not isolate future-conditioned eligibility from changes to training data, labels, candidates, or predictions.
- **Non-goals**: This project does not invent temporal splitting, discover k-core sensitivity, propose a new recommender, train on future records, or interpret changing-population averages as causal effects on a fixed population.

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9 | 6 | 5 | 7 | 7 | 6 | 5 | 6.40 | REVISE |
| 2 | 9 | 8 | 6 | 7 | 8 | 7 | 6 | 7.35 | REVISE |
| 3 | 10 | 9 | 8 | 8 | 9 | 8 | 9 | 8.70 | REVISE |
| 4 | 10 | 10 | 8 | 8 | 9 | 10 | 9 | 9.05 | READY |

## Final proposal snapshot

- Freeze one historical task, one target per user, two models, and every prediction.
- Change only the dataset snapshot cutoff used for k-core eligibility.
- Measure the shift in a pairwise Recall@20 gap.
- Compare real entrants with a preregistered historical-balance null.
- Separate future-incident entrants from historical users admitted without any future interaction of their own, and decompose their signed contributions to the model-gap shift.
- Expand only after a positive, powered pilot; do not infer a paper contribution from design quality alone.

## Remaining weaknesses

The closest RecSys work already fixes models while varying evaluation filters, so C1's method delta is an application-specific assembly rather than a new statistical primitive. Dynamic k-core propagation is also established and cannot be claimed as new. The practical threshold still needs a pre-result justification. A publishable empirical claim requires a nontrivial effect, stronger models, a second dataset, and the gated 2×2 attribution.

## Output files

- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Internal review summary: `refine-logs/REVIEW_SUMMARY.md`
- Claude independent review: `refine-logs/claude3-independent-review.md`
- Claude feedback decisions: `refine-logs/CLAUDE_FEEDBACK_DECISIONS.md`
- Claim-level novelty check: `refine-logs/C1_NOVELTY_CHECK.md`
- Executable pilot plan: `c1-pilot-plan.md`
