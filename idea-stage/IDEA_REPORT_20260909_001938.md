# Research Idea Report

**Direction**: A novel, feasible recommender-system evaluation question for the IEEE BigData 2026 High School Symposium
**Generated**: 2026-09-09
**Ideas evaluated**: 14 generated topics, 13 after merging timestamp variants, 0 piloted, 3 ranked finalists

## Decision

Choose **The Reviews from Tomorrow** for a 48-hour feasibility gate. Keep C1, **The Future Chooses the Test Set**, as the fallback.

The novelty assessment is provisional. It combines targeted literature searches, an authenticated Semantic Scholar query, and a fresh same-family reviewer. It is not an exhaustive systematic review, and no empirical pilot has been run.

## Recommended Ideas

### 1. The Reviews from Tomorrow: Auditing Future-Text Leakage in Recommender Benchmarks

- **Method (what we actually do)**: Pick two historical cutoffs in Amazon Reviews and Yelp. Train one rating-only control and one review-aware recommender. Build user and item text profiles three ways: from past reviews only, from the full corpus with held-out target reviews removed, and from the full corpus including target reviews. Keep interactions, test targets, candidate items, and model settings fixed, then split the effect into future text from the same user and future text about the same item.
- **Research question**: How much of a review-aware recommender's reported accuracy comes from review text written after the prediction time?
- **Hypothesis**: Post-cutoff reviews improve measured performance even after the held-out target review is removed, showing indirect future-text leakage rather than only trivial target leakage.
- **Minimum experiment**: One Amazon category first, then Yelp replication; two cutoffs; one rating-only control; one established lightweight review-aware model; past-only, future-without-target, and future-with-target conditions; user-side and item-side ablations.
- **Expected outcome**: Report performance inflation, model-ranking changes, confidence intervals, and leakage dose. The idea passes only if the effect survives removing target reviews and the aggregation practice is found in at least one credible public implementation or documented protocol.
- **Novelty**: **7/10, provisional**. Closest work studies general temporal leakage, temporal review modeling, or review removal/distortion. The proposed delta is a controlled timestamp intervention that holds the recommendation task fixed and decomposes target, user-side, and item-side future text.
- **Feasibility**: Medium. Public timestamped data; CPU or a small GPU; approximately 7–10 days after the gate.
- **Risk**: Medium. A reviewer may call it an obvious implementation mistake or show that established pipelines already restrict text to training data.
- **Contribution type**: Diagnostic measurement study plus evaluation guidance.
- **Pilot result**: Skipped. The user has not authorized implementation.
- **Reviewer's likely objection**: The experiment attacks a deliberately leaky pipeline rather than a practice used in real work.
- **Why we should do this**: The question is simple, relevant to modern text-aware recommenders, and more distinct from known filtering-sensitivity work than C1.

### 2. C1: The Future Chooses the Test Set

- **Method (what we actually do)**: Freeze historical test targets and model predictions. Extend only the future interaction suffix used to compute k-core eligibility. Track direct future-incident entrants and indirect historical entrants admitted through k-core propagation, then measure their effect on past model comparisons.
- **Research question**: Can future data change who qualifies for a historical recommender benchmark and thereby change its leaderboard?
- **Novelty**: **5–6/10, provisional**. Filtering sensitivity, temporal leakage, and dynamic k-core propagation are known. The remaining delta is the controlled future-suffix intervention and entrant-provenance decomposition.
- **Feasibility**: High. CPU baselines and a detailed two-day pilot plan already exist.
- **Risk**: Medium. The effect may be small, or reviewers may see it as a narrow preprocessing corner case.
- **Contribution type**: Benchmark audit.
- **Pilot result**: Skipped.
- **Why keep it**: It is the strongest fallback because the causal isolation and implementation plan are already mature.

### 3. Broken Profiles: User-ID Fragmentation

- **Method (what we actually do)**: Split selected users' histories into two pseudo-identities using chronological, random, and device-like rules. Keep original test targets fixed and compare the effect with matched edge deletion and history truncation.
- **Research question**: How much can identity fragmentation distort individual recommendations and model rankings?
- **Novelty**: **6/10, provisional**. The proposed node-splitting stress test differs from ordinary interaction deletion, but user linkage, cold start, shared accounts, and graph perturbation are established areas.
- **Feasibility**: Medium. Two public datasets and lightweight models can run on CPU.
- **Risk**: Medium-high. Without real identity ground truth, conclusions depend on synthetic assumptions.
- **Contribution type**: Data-veracity robustness benchmark.
- **Pilot result**: Skipped.
- **Reviewer's likely objection**: This is synthetic cold-start under another name.

## Other Ideas Reviewed

| Idea | Provisional disposition |
|---|---|
| File-row-order dependence | Caution; clear and cheap, but may be only a framework bug report. |
| Interaction deduplication sensitivity | Reserve; repeated behavior is not necessarily erroneous duplication. |
| Repeated events versus unique pairs | Reserve; useful mainly in repeat-consumption domains. |
| Right-censored label maturation | Reserve; important but delayed-feedback literature is extensive. |
| Selective missingness | Drop; substantial overlap with selection and exposure-bias research. |
| Rating timestamp as delayed label | Drop; true consumption time is unavailable, making the result assumption-driven. |
| Global versus user-relative relevance | Drop; threshold sensitivity is expected and personal thresholds are hard to justify. |
| Timestamp blocks and timestamp resolution | Drop as a standalone paper; pseudo-sequences and splitting sensitivity are already studied. |
| Macro versus event-weighted aggregation | Abandon; Simpson-style effects in recommender evaluation are published. |
| Sampled-negative seed instability | Abandon; directly covered by sampled-evaluation studies. |
| Score tie-breaking | Abandon; a 2026 paper directly studies tie-breaking-induced ranking changes. |

## Closest Prior Work for the Recommended Idea

| Work | Overlap | Remaining difference |
|---|---|---|
| [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060) | Future information and model-ranking effects | Does not isolate future textual side information while freezing the task. |
| [Review-based Recommender Systems: A Survey](https://doi.org/10.1145/3742421) | Maps review-aware methods and evaluation practice | Survey, not a causal future-text leakage audit. |
| [RAREval: Do Reviews Matter for Recommendations in the Era of Large Language Models?](https://arxiv.org/abs/2512.12978) | Tests review availability and perturbations | Does not align review profiles to historical prediction cutoffs or decompose future sources. |
| [Reducing contextual noise in review-based recommendation](https://www.sciencedirect.com/science/article/pii/S0020025526000095) | Describes common aggregation of user/item reviews | Does not establish the magnitude of post-cutoff text leakage in historical evaluation. |

## 48-Hour Gate

1. Reproduce one lightweight review-aware baseline on one Amazon Reviews category.
2. Compare past-only text with full-corpus text while excluding every held-out target review.
3. Audit several credible public implementations or protocols for whether post-cutoff aggregation can occur.

Proceed only if the metric effect is nontrivial, survives target-review removal, and the practice is real. Otherwise return to C1.

## Venue Fit

The topic fits the symposium's recommender-systems, text-data, temporal-data, and data-quality scope. Eligibility remains separate: the first author must be a high-school student and the primary contributor under the current symposium rules.

## Suggested Execution Order

1. Run the 48-hour gate for The Reviews from Tomorrow.
2. If any gate condition fails, execute the existing C1 pilot.
3. Do not switch to the lower-ranked ideas without new literature or pilot evidence.
