# Adversarial research-idea review, round 2

You are the hostile second reviewer for proposed IEEE BigData 2026 High School Symposium papers. The venue allows five pages including references and scores technical quality, novelty, relevance, and clarity. Read `idea.md` so you do not duplicate the existing provenance-replay idea. Inspect `claude_review_round1.md` and the JSONL files matching `literature/stage3_nameaudit*`, `literature/stage3_profile_fdr*`, and `literature/stage3_circular*`. Use WebSearch/WebFetch to check literature through 9 September 2026.

The student wants three genuinely defensible master/PhD-level ideas spanning LLM data mining, recommendation/personalization, and statistical ML. Public data and open-weight models are preferred. One GPU is available, and the experiment must be executable by one strong high-school student before 20 September 2026.

Round 1 killed simulator-error rank bounds, generic serialization certification, and prompt-variance decomposition. It conditionally retained the item-name intervention. Since then, two close 2026 papers changed the landscape:

- *The Personalization Mirage: How LLMs Fabricate User Profiles, and Why Self-Monitoring Misleads* already defines evidence-to-claim faithfulness and benchmarks unsupported personal claims, but explicitly leaves mitigation and downstream recommendation impact untested.
- *Eval-Pair Matrix: Answer-Paired Meta-Evaluation of LLM Judges for Grounded RAG* already uses a crossed generator-by-judge matrix and exact-answer pairing outside recommendation.

Review these revised candidates:

1. **Matched-name randomization audit for memorized recommendation.** For every fixed user and fixed 20-item candidate list, randomly permute real item titles among items matched on genre/category, popularity band, and token length while holding all non-name attributes fixed. Treat the list as the randomization unit because ranks interfere. Use an exact Fisher randomization test for the sharp null that recognizable names have no effect, report an alias-averaged title-free ranking score, and validate size/power with a title-blind null oracle and an injected name-memorization oracle. Run open Qwen/Llama models on MovieLens 1M and Amazon Beauty. Closest work includes Di Palma et al., *Do LLMs Memorize Recommendation Datasets?* (SIGIR 2025), Kim et al., *On Memorization Behavior of LLMs in Generative Recommendation* (2026), generic popularity-bias studies, and *Caliper: Probing Lexical Anchors versus Causal Structure in LLMs* (2026, outside recommendation).

2. **False-discovery-controlled preference profiles.** An LLM converts a user's review history into atomic preference claims. For each claim, a black-box conditional randomization test repeatedly replaces its purported supporting history spans with within-user, same-category sentiment-matched control spans. The statistic is the change in claim probability or downstream item score. Benjamini-Yekutieli filters the profile to control the expected fraction of retained claims that are not causally tied to their cited history under arbitrary dependence. Validate finite-sample FDR on planted histories with known claims, then test whether filtering improves claim faithfulness, profile-edit response, and held-out recommendation accuracy on public Amazon review data. Closest work is *Transparent and Scrutable Recommendations Using Natural Language User Profiles* (ACL 2024), *The Personalization Mirage* (2026), PersonaMem/dynamic profiling, and generic conformal or attribution work. The claimed contribution is a black-box mitigation with formal error control and downstream tests, not another hallucination benchmark.

3. **Crossed correction for circular LLM-expanded relevance judgments.** Otero and Parapar (2026) use an LLM first to induce a textual user profile and then to judge unobserved user-item relevance. Cross profile-generator families G and relevance-judge families J in a 2x2 or 3x3 matrix. On deliberately masked observed ratings, estimate the same-family G=J interaction using answer/item-paired contrasts while controlling generator and judge main effects. Model the interaction as a function of profile-item lexical similarity and popularity, then calibrate a bias correction on masked labels and apply it to genuinely unjudged pooled recommendations. Report bias-corrected NDCG differences with cluster-bootstrap intervals and whether algorithm rankings reverse. Falsify with profile paraphrases, profile swapping, and a deterministic ground-truth judge. Closest work includes Otero and Parapar, generic LLM self-preference papers, Dietz et al.'s circularity guidance for IR evaluation, and the 2026 Eval-Pair Matrix paper. The claimed contribution is a recommendation-specific identification and correction procedure for the two-role profile/judge loop.

For each candidate:

- Find the closest prior work and give exact titles and links.
- Decide whether the central contribution is new or a routine application.
- State the exact estimand, theorem or validated property that could honestly be claimed.
- Identify fatal assumptions, especially whether the randomization/exchangeability needed for a causal or FDR statement actually holds.
- Give the smallest defensible experiment and strongest falsification test.
- Score novelty, technical depth, feasibility, and venue fit from 1 to 10.

Kill any candidate that does not deserve a paper. A survivor needs novelty and technical depth of at least 6/10 after revision. If fewer than three survive, propose replacements grounded in explicit limitations of specific current papers and develop them enough to be reviewed in round 3. End with a ranked verdict. Do not edit files.
