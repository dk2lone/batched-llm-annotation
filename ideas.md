# Research ideas

Updated: 2026-09-11

This is the main index for the IEEE BigData 2026 idea search. Novelty scores are
coverage-complete literature judgments through 2026-09-10, not proof that no unindexed paper
exists. Gates K and A have now run on real open-weight models. Gate K failed its mechanism
test; Gate A returned the pre-registered `PROCEED` verdict.

## Current decision

**A: batched LLM annotation dependence** is now the recommended direction to develop. Its Gate A
run on Qwen3-8B and Civil Comments found aggregate within-prompt error ICC 0.0754 with a
conservative 95% interval of [0.0182, 0.1314]. All three fake-batch permutation tests returned
`p = 0.0010`, while the matched-length filler placebo ICC was 0.0048. The frozen verdict is
`PROCEED`; this is one model-dataset cell, not yet a complete paper result.

**K: distance concentration and generic LLM tabular failure** remains the most novel literature
direction, but its mechanism gate failed on Mistral Small 3.1 24B Base. The model passed the
capability anchor, then sparse padding had lower representation contrast than both matched
alternatives and was 12.5 percentage points less accurate than dense projection at high
dimensions. Stop K rather than expanding it to external datasets. The decisive audit is in
`gate-experiments/k/artifacts/mistral-24b-base/analysis/COMPLETION_AUDIT.md`.

| Rank | ID | Idea | Status | Novelty | Main reason |
|---:|---|---|---|---:|---|
| 1 | A | Batched LLM annotation dependence | Gate passed; develop | 7/10 | Real-model gate found dependence beyond matched-length placebo |
| 2 | A2 | Factorial content versus co-occurrence exposure | Gate | 7/10 | Randomized exposure decomposition is unclaimed |
| 3 | A3 | Controlled validation exposure and agentic Type-I error | Repair, then gate | 6.5/10 | Exact intervention is new; adaptive-holdout phenomenon is established |
| 4 | A1 | Repository self-disagreement via provenance replay | Narrowed fallback | 5/10 | Broad finding already exists; provenance attribution may remain |
| 5 | I1 | The Reviews from Tomorrow | Narrowed fallback | 5/10 | Target and temporal leakage are established; only a residual effect remains |
| — | K | Distance concentration and generic LLM tabular failure | Gate failed; stop | 7.5/10 | Novel question, but the proposed mechanism failed its frozen causal gate |
| — | T | Within-row redundancy and tabular likelihood pathology | Drop in current form | 4/10 | Construct overlaps feature correlation and the proposed measurement fails |

The completed seven-idea audit, claim-level verdicts, closest papers, and raw search coverage are in
`idea-stage/novelty-check-20260910/NOVELTY_REPORT.md`. The earlier five-idea audit in
`idea-stage/novelty-recheck-20260910/NOVELTY_RECHECK_20260910.md` and the comparison in
`idea-stage/FIVE_WAY_COMPARISON_20260910.md` are retained as history; their rankings are superseded.

## A2: Factorial pretraining exposure

**Simple question:** When pretraining improves an LLM recommender, how much of the gain comes
from learning item meaning, and how much comes from learning which items occur together?

**Brief plan:**

1. Generate a synthetic catalogue with unfamiliar item IDs, content attributes, and a separate
   latent preference graph that produces item co-occurrences.
2. Confirm that content and co-occurrence carry useful but distinguishable information about the
   latent preferences. Stop if the two signals cannot be separated.
3. Randomize items into a 2 x 2 design: neither exposure, content only, co-occurrence only, or
   both, with multiple exposure doses.
4. Continue-pretrain a small open model and verify with closed-book probes that it learned both
   document families. Stop if either probe fails.
5. Fine-tune the exposed model on interaction sequences that exclude held-out target transitions,
   then estimate both main effects and their interaction with randomization inference.

**First gate:** Build the synthetic catalogue and measure the three signal relationships before
training. If content and co-occurrence are not sufficiently distinct, the factorial experiment
does not identify two mechanisms.

## A3: Validation exposure and agentic Type-I error

**Simple question:** How much can training exposure to hidden validation outcomes inflate the
false-positive rate of an LLM agent that selects its own statistical tests?

**Brief plan:**

1. Generate many tabular dataset instances under a known global null, each with discovery and
   hidden validation splits.
2. First give the validation split directly to the agent as a positive control. Stop if access
   does not raise rejection above the nominal Type-I error rate.
3. Repair the original confounded design by randomizing instance-level exposure inside a shared
   model, or by training multiple independently seeded models per exposure arm.
4. Compare full-validation exposure, discovery-only exposure, and no exposure while keeping the
   agent interface and reachable test family fixed.
5. Verify validation-noise recall, run the agentic test loop, and measure realized Type-I error;
   include a fixed-selector negative control and planted alternatives for power.

**First gate:** Run only the in-context positive control. Weight-level exposure is not worth
training unless the testing pipeline can express the predicted false-positive inflation.

## A1: Repository provenance replay

**Simple question:** Can different repository versions or preprocessing paths for the same public
dataset change which model wins?

**Brief plan:**

1. Download the canonical Adult source files and OpenML entries `did=179` and `did=1590`; record
   checksums and versions.
2. Reconstruct each repository entry from the canonical rows with an explicit transformation
   graph and verify exact row- and column-level agreement.
3. Attribute every difference to a specific transformation, beginning with the reported binning
   of `age` and `hours-per-week`.
4. Run paired, fixed-fold comparisons of random forest and depth-8 decision tree on both lineages.
5. Expand to other datasets only if the replay is exact and the transformation changes the model
   comparison materially.

**First gate:** Reproduce the Adult transformation and reported rank reversal from saved scripts
and artifacts. Stop if either the exact reconstruction or the rank reversal fails.

## I1: The Reviews from Tomorrow

**Status:** Narrowed fallback. The broad claim is not novel; only the residual non-target
future-text effect remains open, still subject to its 48-hour gate.

**Simple question:** How much of a review-based recommender's reported accuracy comes from reviews written after the recommendation date?

**Brief plan:**

1. Use timestamped Amazon Reviews and Yelp data with historical cutoffs.
2. Train a rating-only baseline and at least one lightweight review-aware recommender.
3. Build user and product text profiles using only past reviews, then rebuild them with post-cutoff reviews.
4. Keep the interactions, test items, candidate items, and model settings fixed.
5. Separate leakage from the held-out target review, later reviews by the same user, and later reviews about the same product.

**Potential contribution:** After removing every held-out target review, a controlled
measurement of leakage from later reviews by the same user and later reviews about the same
item, plus a timestamp-safe evaluation protocol.

**Closest work:** TransNets (RecSys 2017) already shows that much of review text's predictive
value comes from the unavailable target review. General temporal-leakage studies and RAREval
cover adjacent effects. The remaining difference is changing only non-target text availability
while freezing the recommendation task and decomposing later-user from later-item reviews.

**Main risk:** Reviewers may call this an obvious bad implementation. We must show that at least one credible public implementation or published protocol can aggregate post-cutoff reviews.

**48-hour gate:**

- Find a credible affected implementation or protocol.
- Show a measurable past-only versus future-text difference on one dataset.
- Show that the effect remains after removing all held-out target reviews.

If any of these fails, stop this direction.

## I2: The Future Chooses the Test Set (C1)

**Status:** Earlier fallback retained for reference. Fully refined, but not piloted and no
longer in the top-five comparison.

**Simple question:** Can interactions that happen in the future change which users and products qualify for an old recommender-system test and therefore change which model appears best?

**What we would do:**

1. Choose a historical training and test period.
2. Train the recommenders once and freeze their predictions and historical test targets.
3. Gradually reveal later interactions, allowing them to affect only the dataset's k-core eligibility filter.
4. Measure which historical users and items enter the benchmark and whether model scores or rankings change.
5. Separate direct entrants with future activity from indirect entrants admitted through k-core graph propagation.

**Potential contribution:** A controlled future-suffix intervention showing whether future-dependent filtering alters historical benchmark conclusions.

**Known overlap:** Temporal leakage, preprocessing sensitivity, filtering-induced ranking changes, and dynamic k-core propagation already exist. Only the frozen-task intervention and entrant-provenance analysis may be new.

**Main risk:** The indirect entrants may be rare or may not materially change model comparisons.

**Next test:** Follow `c1-pilot-plan.md` using raw Amazon 2014 Beauty ratings, then replicate on MovieLens 1M if the pilot passes.

## Backup ideas

### I3: Broken Profiles

**Question:** What happens when one person's history is mistakenly split across multiple user IDs?

**Method:** Create controlled pseudo-identities, compare chronological, random, and device-like splits, and use matched deletion controls so the result is not merely caused by shorter histories.

**Why it is not first:** There is no public ground truth for real identity fragmentation. A reviewer may dismiss it as synthetic cold-start or ordinary missing-data sensitivity.

### I4: Same Data, Different File Order

**Question:** Can shuffling an unchanged interaction file alter the split, sampled negatives, metrics, or model ranking?

**Method:** Run many row permutations, hash every intermediate artifact, locate the first divergence, then test whether canonical sorting removes all variation.

**Why it is not first:** If only one implementation fails, this is a narrow framework bug report. If all tested systems are deterministic, there is no paper result.

## Compared independent candidate set

An independent idea-generation run from 2026-09-10 produced four candidates relevant to
IEEE BigData topics 2, 5, and 7. They are preserved under
`idea-stage/alternative-run-20260910/` and were compared in the five-idea audit. A1 through A3
remain active; A4 was dropped after the completed novelty check and is preserved below.

The canonical IDs below prevent collisions with the IDs used inside the archived run.

| Canonical ID | Source ID | Idea | Source-run status |
|---|---|---|---|
| A1 | G2 | Repository self-disagreement via provenance replay | Rank 3; reproduce the missing pilot first |
| A2 | I1 | Factorial pretraining exposure: item content versus co-occurrence | Rank 2; recommender-focused alternative |
| A3 | I2 | Controlled validation-data exposure and agentic Type-I error | Rank 1; recommended gate |
| A4 | I3 | Contamination-immune agentic tests and the price of open-endedness | Dropped; direct theory collision |

Treat the archived run's novelty scores and claimed experimental results as unverified within
this main workspace until its evidence and scripts are audited. The archive does not contain
the `scratchpad/` scripts referenced by its `idea.md`.

## Rejected candidates

### A4: Contamination-immune agentic tests

**Why rejected:** Its constant-threshold guarantee follows from established post-selection and
multiple-testing theory. Max-statistic calibration is also not uniquely optimal because step-down
and closed-testing procedures already exist.

**Do not revive without:** A materially different theorem or empirical question that is not an
application of known simultaneous-inference results to an LLM selector.

### F1: Silent Cache Hits (C2)

**Question:** Do incorrect accepted cache matches in LLM log parsers delay the discovery of new event families?

**Why rejected:** Recent work already demonstrates persistent demo-pool poisoning and silent trie absorption. A narrower study of naturally occurring accepted-hit failures remained possible, but natural errors may be too rare and the experiment is more complex than the recommender audits.

**Do not claim:** Cache-error persistence as a new mechanism.

### F2: Local Colors, Global Dependencies (C3)

**Question:** Can remote point movements change distortion colors on triangles whose local geometry did not change?

**Why rejected:** The effect follows directly from global normalization in the Gap Index definition. Without evidence that it causes a material interpretation failure, it is an observation about established mathematics rather than a research contribution.

## Other rejected recommender questions

| ID | Question | Why rejected or deprioritized |
|---|---|---|
| F3 | Do tied or coarse timestamps change sequential-recommender results? | A 2021 pseudo-sequence paper and RecSys 2025 splitting work already cover much of it. |
| F4 | Does the random negative-sampling seed change model rankings? | Directly studied in sampled-evaluation literature. |
| F5 | Does score tie-breaking change recommender rankings? | Directly matched by a 2026 tie-breaking paper. |
| F6 | Can macro versus event-weighted metrics reverse conclusions? | Simpson-style aggregation effects have already been published. |
| F7 | Do alternative relevance thresholds change rankings? | Expected sensitivity with difficult-to-justify personal thresholds. |
| F8 | Does selective missingness change rankings? | Strong overlap with selection bias, exposure bias, and missing-not-at-random research. |
| F9 | Are dataset endpoints affected by right-censored labels? | Important, but delayed-feedback and censoring literature is extensive; the clean experiment is harder. |
| F10 | Does deduplicating repeated interactions change conclusions? | Repeated interactions may be legitimate behavior rather than duplicate records. |
| F11 | Should repeated test events count once or many times? | Potentially useful, but limited mainly to music and other repeat-consumption domains. |
| F12 | Are rating timestamps delayed versions of consumption time? | Actual consumption time is unavailable, so results depend on simulated delay assumptions. |

## Where the supporting material lives

| File | Contents |
|---|---|
| `idea-stage/FIVE_WAY_COMPARISON_20260910.md` | Current comparison, ranking, risks, and decisive gates for the five candidates |
| `idea-stage/novelty-recheck-20260910/NOVELTY_RECHECK_20260910.md` | Completed three-source novelty verdicts and updated five-way ranking |
| `idea-stage/novelty-recheck-20260910/QUERY_PLAN.tsv` | Three-round query definitions for the five candidates |
| `idea-stage/novelty-recheck-20260910/*.jsonl` | Raw arXiv, OpenAlex, and Semantic Scholar search evidence |
| `idea-stage/IDEA_REPORT.md` | Full comparison and latest ranking of recommender questions |
| `idea-stage/alternative-run-20260910/idea.md` | Preserved independent run containing A1 and summaries of A2–A4 |
| `idea-stage/alternative-run-20260910/idea-generation-report.md` | Detailed designs, audits, and literature comparisons for A2–A4 |
| `HANDOFF.md` | Overall project history, boundaries, and earlier finalists |
| `refine-logs/FINAL_PROPOSAL.md` | Detailed current C1 design |
| `refine-logs/C1_NOVELTY_CHECK.md` | Claim-level C1 novelty assessment |
| `c1-pilot-plan.md` | Two-day C1 pilot and stopping criteria |
| `refine-logs/claude3-independent-review.md` | Claude Opus 5 review of C1 |
| `novelty-check-report.md` | Earlier novelty verification for C1 and C2 |
| `verification-update.md` | Literature-recovery results and corrections |
| `research-audit.md` | Search queries, evidence, rejected ideas, and datasets |

## Venue constraint

The recommender ideas fit the IEEE BigData 2026 High School Symposium scope. The first author must be a high-school student and the primary contributor under the currently posted rules.
