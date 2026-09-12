Independently evaluate the research idea below. Use your own novelty-check process and search the scholarly web through September 8, 2026. Do not read or rely on local novelty reports, handoff files, or reviewer logs as evidence. Verify candidate prior work from primary paper pages, publisher pages, arXiv, OpenReview, DBLP, Semantic Scholar, or OpenAlex. Failed searches are unknown, not evidence of absence.

Tasks:

1. Restate the single contribution in plain language and say whether it is clear.
2. Search direct terms, adjacent terminology, and the domain-free mechanism for exact or near-exact prior work.
3. Separate what is already known from the remaining method novelty and the still-unverified empirical novelty.
4. Identify the five closest papers, with title, year, link, exact overlap, and exact difference.
5. Give a skeptical novelty score from 1-10 and one verdict: PROCEED, PROCEED WITH CAUTION, or STOP.
6. Attack the idea as a reviewer. Focus on triviality, identification, evaluation-unit validity, matched-null validity, threshold choice, dataset/model dependence, and whether the contribution is enough for an IEEE BigData-style paper.
7. Suggest concrete changes that improve the contribution without adding unrelated modules.
8. End with a revised 3-5 sentence contribution pitch and a minimal go/no-go pilot.

Do not claim that an empirical effect exists. Do not use “first” or “no prior work” unless the search evidence is unusually strong. Cite every literature claim with a working URL.

IDEA

Title: The Future Chooses the Test Set: A Fixed-History Audit of Recommender Model Comparisons

C1 contributes a controlled audit that freezes a historical recommendation task and every per-user prediction, then varies only the future observation horizon used to compute k-core eligibility. Its primary estimand tests whether future-conditioned admission shifts a frozen pairwise model gap beyond shifts produced by historically comparable cohorts. The method contribution is the controlled audit; whether real datasets exhibit a practically meaningful excess shift remains an empirical hypothesis.

Freeze a training cutoff and test window. For every user with training history, select the chronologically last test-window interaction as one target, breaking timestamp ties by stable item ID. Define the pre-k-core universe U* as users whose frozen target is in the training-item catalog. Train popularity and sparse item-kNN once on suffix-independent historical rows and save each user's Recall@20 difference d_u.

At each horizon h, compute the same iterative k-core on records observed through h. Let E_h(u)=1 only when user u and the frozen target item lie in that core. Future records may change only E_h. Historical rows, target labels, candidate catalog, model inputs, and predictions remain fixed. The primary estimand is the change from the test cutoff to the final observation horizon in the mean d_u among eligible users. Interpret the shift through the share of entrants and their mean d_u relative to incumbents.

Compare real entrants with 2,000 equally sized pseudo-entrant cohorts drawn from the remaining baseline risk set, matching on historical training degree, recent activity, and target-item training popularity. All strata and deterministic sparse-stratum merging rules are frozen using information available by the test cutoff, and balance requires standardized mean differences below 0.1.

The only pilot continuation test is the absolute change in the popularity-versus-item-kNN Recall@20 gap at the final horizon. It must exceed both the matched-null 95th percentile and 0.002 absolute Recall@20. Intermediate horizons and NDCG@10 are descriptive. If the gate passes, run a 2x2 training-mask/evaluation-cohort attribution on Amazon and a second dataset with a maintained contemporary recommender pair.

The proposal does not claim that changing cohorts changes averages is new. It claims a controlled fixed-history, fixed-prediction audit for testing whether future-conditioned benchmark eligibility selects model-relative difficulty beyond a prespecified historical null.
