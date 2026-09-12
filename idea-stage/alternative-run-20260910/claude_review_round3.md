# Final adversarial review of three research ideas

Independently verify three candidate papers for the IEEE BigData 2026 High School Symposium. Do your own WebSearch/WebFetch literature search through 9 September 2026. Do not trust the citations or novelty conclusions in this prompt. Read `idea.md` only to ensure these ideas are distinct from the existing repository-provenance replay project. You may read `claude_review_round2.txt` for the prior critique, but independently check every decisive paper.

The venue allows five pages including references and judges technical quality, novelty, relevance, and clarity. The student wants three additional master/PhD-level ideas at the intersection of recommendation/personalization, statistical ML, and LLM data mining. Public data and one GPU are available.

A candidate passes only if novelty and technical depth are both at least 6/10 after all stated repairs. Kill routine domain transfers, metrics-only audits, or claims that depend on untestable assumptions. If you kill one, supply and independently search a stronger replacement.

## F1: Catalogue-specific title advantage by a two-arm randomized audit

For each user and fixed 20-item candidate list, compare two randomized title perturbations while holding non-title attributes and positions fixed:

- Arm A permutes real titles among items from the benchmark catalogue within matched blocks.
- Arm B substitutes real titles from a disjoint catalogue, matched on item type, genre/category, release era, token length, and a pretraining-familiarity proxy such as Wikipedia page views.

The list is the randomization unit because ranks interfere. The estimand is the difference in average target-item rank displacement between arms, called catalogue-specific title advantage. Do not call it memorization. Use an exact randomization test for the unit-level sharp null that assignment to Arm A versus B changes no list outcome; use randomization confidence intervals for the average contrast rather than falsely calling `tau=0` a Fisher sharp null. Independently randomize display positions.

Validate size with a title-blind oracle and power with a synthetic-catalogue Qwen-1.5B LoRA model trained on known item-successor pairs. A real-data run uses MovieLens 1M; a second domain is optional if it weakens matching. Closest known threats include *Do LLMs Memorize Recommendation Datasets?* (SIGIR 2025), *On the Memorization Behavior of LLMs in Generative Recommendation* (2026), *Benchmark Leakage Trap* (2026), and *Caliper: Probing Lexical Anchors versus Causal Structure in LLMs* (2026).

## F2: False-discovery-controlled evidence filter for preference profiles

An open LLM converts Amazon Reviews 2023 histories into about eight atomic preference claims per user. It also cites all history spans supporting each claim. For each claim, a matched permutation test replaces every support-like span above a semantic threshold with within-user, same-category, sentiment-matched control spans. The statistic is the teacher-forced log probability of the fixed claim text. Benjamini-Yekutieli filters claims at a target FDR.

The formal claim is explicitly conditional: if matched span replacements are exchangeable under the null, randomization p-values are super-uniform and BY controls the expected fraction of retained claims that do not depend on any support-like history span. This is prompt-level dependence, not truth about the person. Report p-value/FDR sensitivity to matching radius and 300 resamples. Validate realized FDR on planted histories; test profile faithfulness, profile-edit responsiveness, and non-inferiority of held-out recommendation accuracy. Strong falsifications are claim-user shuffling and a fluency-only statistic.

Closest known threats include *Transparent and Scrutable Recommendations Using Natural Language User Profiles* (ACL 2024), *GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification* (2026), *The Personalization Mirage* (2026), generic conformal factuality, and conditional-randomization feature-inference work.

## F3: Stratified audit certificates for LLM-completed recommendation evaluation

Two recommenders produce top-10 lists. An LLM assigns binary relevance to the pooled user-item pairs not covered by a small human audit sample. Items are stratified by an exposure proxy such as popularity decile. Within every stratum, the audit items are sampled uniformly without replacement. An exact finite-population upper confidence bound gives a simultaneous budget `b_s` on the number of remaining LLM label errors in stratum `s`.

For a pair of recommenders, write their average DCG@10 difference as a linear function of pooled binary relevance labels. Under the per-stratum error budgets, obtain the exact minimum and maximum difference by flipping, within each stratum, the labels with the largest signed changes first. The theorem is conditional: if each true error count is at most `b_s`, the true DCG difference lies in the computed interval; the greedy endpoints are sharp. The winner is certified only if the interval excludes zero. Also report the minimum error budget at which the winner can flip. Do not make the same claim for nDCG: its label-dependent normalization breaks the linear proof, and prior work argues DCG is the sounder off-policy metric.

Validate on KuaiRec's 99.6%-observed small matrix. Use its captions/categories for an open LLM judge, simulate uniform and popularity-biased missingness, sample small audit sets, and test simultaneous coverage plus certification rate. Compare against the raw LLM winner and PRECISE/prediction-powered inference. Closest known threats include *On the Reliability of Sampling Strategies in Offline Recommender Evaluation* (RecSys 2025), *Do LLM-judges Align with Human Relevance in Cranfield-style Recommender Evaluation?* (2025), *LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation* (2026), Otero and Parapar (2026), PRECISE (AAAI 2026), *Demystifying Prediction Powered Inference* (2026), and *On (Normalised) Discounted Cumulative Gain as an Off-Policy Evaluation Metric for Top-n Recommendation* (2023).

For each idea:

1. Give the closest prior work with exact titles and links.
2. Explain whether the central contribution is new or a routine combination of known tools.
3. Audit the estimand and claimed finite-sample property for mathematical correctness.
4. Identify the most likely fatal assumption and the strongest falsification.
5. State the smallest defensible experiment and realistic compute.
6. Score novelty, technical depth, feasibility, and venue fit from 1 to 10.

End with a hard pass/kill verdict for each. A pass requires novelty >= 6 and technical depth >= 6. Do not edit files and do not ask the user a question.
