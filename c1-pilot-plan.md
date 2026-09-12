# Two-day pilot: future-dependent recommendation cohorts

Status: ready to implement. Project: `/Users/dz/bigdata` (not a git repository, so there is no branch or PR).

## Question

When models, predictions, targets, and historical records are fixed, can adding only post-test observations to k-core eligibility change the measured popularity-versus-item-kNN Recall@20 gap? If so, is the shift driven by users active in the future suffix or by historical users admitted indirectly through the new k-core fixed point?

## Frozen setup

Use raw [Amazon 2014 Beauty ratings](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html), not its prefiltered 5-core derivative. Validate the file, timestamp range, duplicates, and license first.

Choose training, test, and future windows from timestamp quantiles without splitting ties. Freeze the choices before model scoring. The remaining timeline supplies nested snapshot cutoffs at the test boundary, an intermediate future point, and the complete release.

Define a pre-k-core warm universe from historical information. Each user must have training history and exactly one deterministic test target: the last test-window interaction, with stable item-ID tie-breaking. The target must appear in the fixed training-item catalog.

Train popularity and sparse item-kNN once on the same historical rows. Save full-catalog top-20 predictions and each user's Recall@20 difference. No future record may enter training, targets, candidates, or scoring.

## Eligibility intervention

At each snapshot cutoff, compute an iterative user-item 5-core. A user is eligible only if both the user and frozen target item survive. Also compute k=0 as an implementation control.

Hash the historical rows, targets, catalog, model inputs, and predictions. Every hash must be identical across cutoffs. Any mismatch is an implementation failure.

## Primary endpoint and balance reference

The only primary endpoint is the change from the test-boundary cutoff to the complete-release cutoff in the mean popularity-versus-item-kNN Recall@20 gap among eligible users.

Build 2,000 pseudo-entrant cohorts from the baseline risk set. Match on training degree, activity in the 30 days before testing, and target-item training popularity, using bins fixed before the outcome analysis. Require standardized mean differences below 0.1 and report donor support. Call this a historical-balance reference, not a causal test.

Before looking at model comparisons, use the eligible-user count and per-user gap variance to document the minimum detectable effect and set the practical effect floor.

## Mechanism readout

For users absent from the baseline core but present at the final cutoff, report:

- future-incident entrants, who have a post-test interaction;
- nonincident entrants, who have no post-test interaction but become eligible after their graph neighborhood changes;
- each group's count, share, covariate balance, mean pairwise gap, and signed contribution to the total shift.

This classification is deterministic and avoids claiming a unique causal path through a batch k-core update.

## Two-day execution

Day 1: validate data, freeze cutoffs and targets, build eligibility masks, classify entrants, and pass every invariant check. Produce cohort counts before inspecting model differences.

Day 2: train the two CPU baselines once, compute the primary endpoint and historical-balance reference, document the power-based practical floor, and produce one retention/provenance table plus one cutoff trajectory figure.

Use sparse item-item computation. The pilot should fit on a CPU machine with about 16 GB RAM; this is a planning estimate, not a measured runtime.

## Decision rule

Continue if the absolute final-cutoff shift exceeds both the historical-balance 95th percentile and the prespecified practical floor, with adequate entrant and donor support. A substantial nonincident-entrant contribution strengthens the mechanism story but is not required.

Stop or redesign if invariants fail, the matched reference is unsupported, the effect appears only at an extreme cutoff, or the result reduces to a negligible number of future-active users. If the pilot passes, add a second raw temporal dataset, a maintained contemporary model pair, and the 2-by-2 training-mask/evaluation-cohort attribution.
