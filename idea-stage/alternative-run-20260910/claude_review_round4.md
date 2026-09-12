# Independent round 4: produce three defensible survivors

You are the final adversarial reviewer for research ideas aimed at the IEEE BigData 2026 High School Symposium. Use WebSearch and WebFetch yourself. Do not trust the candidate descriptions, their citations, or Codex's novelty judgments. Search primary papers and return exact links.

The student prefers topics 2, 5, and 7: machine learning/statistics for big data, personalization/recommendation, and data mining with LLMs. A 5-page paper must still contain a master/PhD-level research object: a precise estimand, algorithm or theorem, a falsifiable experiment, and honest limitations. GPU access is available. Kill routine domain transfers and combinations of known tools.

## What round 3 found

You killed the original three and proposed P1-P3. A fresh OpenAlex/arXiv/Semantic Scholar search found a direct collision for P3:

- Zhao et al., *Membership Inference Attacks with False Discovery Rate Control*, 2025: https://arxiv.org/abs/2508.07066

P3 also claimed train and holdout membership statistics are exchangeable after a model is trained on the train group. That is not generally true: training status changes the distributions. Treat P3 as killed unless you can prove a distinct valid object beyond the paper above.

## P1 under challenge: difference in discontinuities at a model cutoff

Question: how much of an LLM recommender's advantage comes from possible pretraining exposure to an item?

Proposed design: item release date is the running variable; documented model data cutoff is threshold. Estimate the jump in per-item ranking success for an LLM and subtract the jump for a collaborative baseline trained on identical interaction data. Repeat across three open models with different cutoffs; each estimated jump should move with its model's cutoff.

Concerns you must resolve:

1. Release before cutoff only creates eligibility for exposure, not actual exposure. This is a fuzzy/encouragement estimand, not a direct memorization effect.
2. Documented cutoff dates may be coarse or unreliable, and continued pretraining/fine-tuning may move effective cutoffs.
3. Explain exactly what outcome and dataset can include enough items/users on both sides of each cutoff. MovieLens is too old. Amazon Reviews 2023 may work only for older model cutoffs; search for better temporally timestamped recommendation data.
4. A collaborative baseline may not difference out a product-platform discontinuity coinciding with the cutoff.

Search for exact recommendation collisions and relevant temporal-contamination/cutoff-discontinuity work. Repair the estimand and design or kill it.

## P2 under challenge: exact history-identity test and lower bound on personalized users

Goal: quantify how many users an LLM ranker actually personalizes for.

Possible repaired design: for each focal user, define a matched set containing the user's own history plus donor histories matched on length, activity, category composition, and time. Randomly permute which history identity is supplied to a fixed candidate set and score the held-out true item. The per-user null must be stated narrowly as invariance of the outcome to the identity of the history within this matched set. Combine valid tests with permutation closed testing to obtain a 1-alpha lower confidence bound on the number of users whose outcome depends on history identity.

Concerns you must resolve:

1. A single own history is structurally special because its held-out item was generated from that same user's process; matched donor histories are not exchangeable under a broad no-personalization null. Define a sharp design-based null under which randomization inference is actually valid, or kill it.
2. If every computational counterfactual can be queried, clarify why a p-value is needed and what randomness supports it.
3. Search whether LLM personalization/recommendation papers already use user-history shuffling with per-user significance, multiple testing, true-discovery lower bounds, or randomization inference.
4. Keep the contribution distinct from a routine application of Vesely/Finos/Goeman closed testing.

Repair or kill P2.

## Required replacement search

We need exactly three survivors overall. Propose replacements for every killed candidate. You may leave recommendation/LLM evaluation if a stronger idea fits topics 2, 5, or 7. Do not propose:

- FDR-controlled or count-certified membership inference;
- prompt variance decomposition;
- simple title/alias perturbation or Caliper transferred to recommendation;
- generic profile hallucination detection/context ablation;
- LLM-judge correction with PPI/PRECISE;
- metric-only robustness audits;
- dataset-version replay, which is the student's existing idea.

For each candidate, independently search at least three query formulations. Report:

1. exact research question and contribution;
2. closest three to five papers and exact URLs;
3. whether any paper has the same estimand + guarantee/algorithm + experimental setting;
4. validity audit of the theorem/test/identification claim;
5. smallest decisive experiment, public data, compute estimate, and strongest falsification;
6. most likely fatal assumption;
7. scores 1-10 for novelty, technical depth, feasibility, and venue fit.

Hard-pass only if novelty >= 6 and technical depth >= 6 after the search. Return a verdict table. Do not ask the user questions. If you cannot find three, say so and continue generating/searching in the same response until you do.
