# Research Proposal: The Future Chooses the Test Set

## Contribution

C1 is a controlled audit of **future-dependent benchmark eligibility**. It freezes a historical recommendation task, candidate catalog, fitted models, targets, and every per-user prediction, then varies only the dataset snapshot cutoff used to compute k-core eligibility. The primary test asks whether that future-conditioned admission changes a frozen pairwise model gap more than historically comparable cohort changes do.

The contribution is narrower than temporal leakage or k-core sensitivity, which are established. The unresolved part is isolating eligibility selection caused only by post-test observations and tracing whether future-incident or nonincident entrants account for the resulting benchmark change.

## Frozen historical task

Use raw Amazon 2014 Beauty ratings. Fix training and test cutoffs without splitting tied timestamps. Define a pre-k-core universe \(U^*\) containing users with training history whose frozen target is in the training-item catalog. For each user, select the chronologically last test-window interaction as the sole target, breaking timestamp ties by stable item ID.

Train popularity and sparse item-kNN once on suffix-independent historical rows. Freeze the candidate catalog, model inputs, predictions, and each user's pairwise Recall@20 gap \(d_u\).

## Snapshot-cutoff intervention

Construct nested dataset snapshot cutoffs from the same immutable release. At each cutoff \(h\), recompute the same iterative k-core over records observed through \(h\). Let \(E_h(u)=1\) only when user \(u\) and the user's frozen target item are in that core. Future records may change only \(E_h\).

The sole primary endpoint is

\[
\Delta_{\mathrm{sel}}=
\operatorname{mean}\{d_u:E_{h_{final}}(u)=1\}-
\operatorname{mean}\{d_u:E_{t_0}(u)=1\}.
\]

Intermediate cutoffs, NDCG@10, and ranking reversals are descriptive.

## Entrant provenance

Partition newly eligible historical users into two prespecified groups:

1. **Future-incident entrants:** users with at least one interaction in the added post-test suffix.
2. **Nonincident entrants:** users with no interaction in that suffix who nevertheless become eligible because the iterative k-core fixed point changes around them.

Report each group's share, historical covariates, mean \(d_u\), and signed contribution to \(\Delta_{\mathrm{sel}}\). This tests whether the observed shift is merely direct admission of more-active users or includes an indirect eligibility cascade. Dynamic k-core propagation itself is prior work; the proposed contribution is its benchmark-specific provenance and evaluation consequence.

## Historical-balance null

Compare real entrants with equally sized pseudo-entrant cohorts drawn from the baseline risk set. Match using only information available by \(t_0\): training degree, activity during the 30 days before the test window, and target-item training popularity. Compute quantile boundaries once on \(U^*\), then use 2,000 stratified draws.

When a stratum lacks donors, merge in a fixed order: nearest target-popularity bin, nearest recent-activity bin, then nearest training-degree bin, breaking equal-distance ties downward before upward. Require standardized mean differences below 0.1 and report donor-pool support. If balance or donor support is inadequate, mark the null unsupported. This is a balance reference, not a causal null or a p-value.

## Primary pilot gate

Before observing model results, estimate uncertainty and minimum detectable effect from the eligible-user count and per-user \(d_u\) distribution. Set the practical Recall@20 floor from that power analysis and report its basis rather than using an arbitrary constant.

Continue only if the final-cutoff absolute \(\Delta_{\mathrm{sel}}\):

1. exceeds the historical-balance null's 95th percentile;
2. exceeds the prespecified practical floor; and
3. is not explained only by a tiny, poorly supported entrant group.

## Invariants and paper-stage bridge

Hash historical rows, target labels, candidate catalog, model inputs, and per-user predictions. All hashes must remain unchanged across cutoffs. The warm-universe construction must produce zero missing-user and missing-item fallbacks.

If the pilot passes, run a 2-by-2 historical-training-mask/evaluation-cohort attribution, add a maintained contemporary model pair, and replicate on a second raw temporal dataset. Those are paper-stage checks, not extra pilot endpoints.

## Claim boundary

Do not claim that cohort changes alter averages, that k-core updates propagate, or that filtering choices matter. Claim only that the audit isolates whether post-test observations retroactively select a historically fixed evaluation population, identifies which entrant provenance drives the shift, and measures whether the effect exceeds a prespecified historical-balance reference. A real-world effect is claimable only after the gate and replication pass.
