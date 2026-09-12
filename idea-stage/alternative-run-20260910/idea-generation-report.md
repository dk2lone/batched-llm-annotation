# Three additional research directions for IEEE BigData 2026

**Status:** literature and adversarial novelty audit completed 2026-09-10  
**Target:** [IEEE BigData 2026 High School Symposium](https://bigdataieee.org/BigData2026/high-school-symposium/)  
**Relevant topics:** 2 (machine learning/statistics), 5 (personalization and recommendation), 7 (data mining with LLMs)

These are alternatives to the existing G2 provenance-replay project in `idea.md`. They do
not reuse G2's repository-version intervention. Novelty means that no same-design paper was
found in the searches below; it is not a guarantee that no unpublished or poorly indexed
work exists.

## Final ranking

| Rank | Code | Direction | Novelty | Depth | Feasibility | Venue fit | Status |
|---|---|---|---:|---:|---:|---:|---|
| 1 | I1 | Factorial pretraining exposure: item content versus co-occurrence | 7 | 7 | 5 | 8 | **Pass** |
| 2 | I2 | Controlled validation-data exposure and agentic Type-I error | 7 | 7 | 5 | 8 | **Pass** |
| 3 | I3 | Contamination-immune agentic tests and the price of open-endedness | 6 | 7 | 8 | 8 | **Pass, highest collision risk** |

I1 is the strongest match to topics 5 and 7 and has the cleanest causal question. I2 is the
strongest statistics/LLM idea. I3 is the cheapest decisive project, but its novelty rests on
a necessary-and-sufficient result for memorizing agents rather than on the standard
max-statistic correction itself.

---

## I1 — Factorial pretraining exposure: item content versus co-occurrence

### Research question

When pretraining improves an LLM recommender, how much of the gain comes from learning what
items *are*, and how much comes from learning which items occur together?

Prior papers show that recommendation benchmarks can leak into LLMs, that generative
recommenders memorize transitions, and that entity exposure affects LLM judgments. None of
the located papers randomizes semantic-content and item-co-occurrence exposure separately
and measures their causal effects on recommendation.

### Precise contribution

Create a synthetic catalogue whose identifier tokens are absent from the base model's
knowledge. Generate two partially independent information sources:

- content attributes for each item;
- a latent user-preference graph that induces item-item co-occurrence.

Independently randomize each item to a 2 x 2 factorial assignment:

1. no content documents, no co-occurrence documents;
2. content only;
3. co-occurrence only;
4. both.

Randomize two or three exposure doses inside each cell. Continue pretraining a 1B-class open
model, then fine-tune it on identical user-item sequences whose held-out target transitions
are absent from fine-tuning.

Because all items share one model, ordinary no-interference language is false. Define each
estimand as an **assignment-policy effect**, averaged over the randomized assignments of all
other items:

\[
\tau_{content}=E[Y_i(1,C_i)-Y_i(0,C_i)],\qquad
\tau_{cooc}=E[Y_i(C_i,1)-Y_i(C_i,0)],
\]

plus the factorial interaction. `Y_i` is pre-registered as item-level target log-rank, with
Hit@10 as a secondary outcome. Use design-based randomization inference at the item level.

### What makes it a paper

- A randomized decomposition of two proposed sources of LLM recommendation ability.
- An interference-aware estimand instead of an observational correlation with corpus
  frequency.
- A dose-response curve and an item-level randomization test.
- A pretraining probe that separates “the documents were not learned” from “the information
  was learned but did not improve ranking.”

The single-run controlled-pretraining method in [Train Once, Answer All](https://arxiv.org/abs/2509.23383)
can reduce cost, but it must be paired with its dependence test and one twin-model contrast.
If that check fails, train separate models for the factorial contrasts.

### Smallest decisive experiment

1. Generate 400 synthetic items, 100 per factorial cell, with two exposure doses.
2. Make content and co-occurrence only partially informative about the same latent preference;
   report their mutual information so the two signals are demonstrably distinct.
3. Continue pretrain a 1B model with paraphrased documents.
4. Run closed-book probes for both document families. Stop if either probe fails.
5. Fine-tune on sequences excluding the held-out target transitions.
6. Estimate both main effects and the interaction by randomization inference.
7. Replicate only the direction of effect on real post-cutoff Amazon items; make no causal
   claim for that replication.

Expected compute is roughly 18–30 H100-hours including the twin-model check. The strongest
falsification is: both probes succeed, yet both ranking effects remain near zero at every
dose. The main failure mode is that recommendation fine-tuning relearns the held-out
co-occurrence signal, so the target transitions must be excluded by construction.

### Closest work and remaining gap

- [Benchmark Leakage Trap](https://arxiv.org/abs/2602.13626) injects blended real corpora and
  studies recommendation leakage, but has no item-level factorial randomization or separation
  of content from co-occurrence.
- [On the Memorization Behavior of LLMs in Generative Recommendation](https://arxiv.org/abs/2606.17276)
  studies memorized recommendation behavior rather than randomized pretraining exposure.
- [Pretraining Exposure Explains Popularity Judgments](https://arxiv.org/abs/2605.12382)
  relates exposure to entity judgments observationally.
- [Impact of Co-occurrence on Factual Knowledge of LLMs](https://arxiv.org/abs/2310.08256)
  studies factual knowledge, not recommendation or a content/co-occurrence factorial design.
- [Estimating Average Causal Effects under General Interference](https://doi.org/10.1214/16-AOAS1005)
  supplies the correct estimand language; it is a method dependency, not the application result.

**Claim boundary:** causal only for the specified generator, base model, corpus construction,
and exposure doses.

---

## I2 — Controlled validation-data exposure and agentic Type-I error

### Research question

How much can pretraining exposure to hidden validation outcomes inflate the Type-I error of
an LLM agent that selects its own falsification tests?

[POPPER](https://arxiv.org/abs/2502.09858) proves Type-I error control for agentic sequential
falsification under a condition that includes independence between agent training and the
datasets used by the agent. Interface controls can hide unused validation data from the
prompt, but cannot enforce that training-data independence for a pretrained model.

### Precise contribution

Generate many independent tabular dataset instances under a known global null. Each instance
has discovery and hidden validation splits. Randomize whole instances to three continued-
pretraining arms:

- full exposure: discovery and validation rows/outcomes;
- discovery-only exposure;
- no exposure.

After training, every agent receives only the discovery split through the same interface.
The validation pipeline uses the hidden split. Give the agent a large, fixed reachable family
of columns, subgroup definitions, and outcome transformations; otherwise a fixed test remains
valid and exposure cannot inflate selection error.

For arm `a`, let `Y=1` when the pipeline rejects the global null at nominal level `alpha`.
The primary estimand is

\[
\Delta=P(Y=1\mid A=full)-P(Y=1\mid A=none).
\]

Random assignment of dataset instances identifies this contrast for the experimental training
regime. It does not estimate contamination in a hosted model.

### What makes it a paper

- A controlled causal violation of the train-independence condition in an agentic testing
  theorem.
- Direct measurement of realized Type-I error, rather than benchmark accuracy or memory recall.
- A decomposition of identifier-mediated recall: repeat validation after random re-keying of
  columns, categories, and row identifiers.
- Negative and positive controls that show whether the pipeline can express the failure.

Re-keying is a diagnostic, not a guaranteed repair. It is a bijection and cannot remove
numeric-value memorization or change the realized validation noise.

### Smallest decisive experiment

1. Generate 300 null dataset instances with discovery and validation splits.
2. Randomize 100 instances to each exposure arm.
3. Continue pretrain or LoRA-adapt one 1B-class model per arm at several exposure doses.
4. Verify recall of validation noise before running the agent.
5. Run a reduced POPPER-style loop at `alpha=0.05` with a broad declared test family.
6. Add a planted-alternative condition for power, a fixed non-LLM selector as a negative
   control, and an agent receiving validation data in context as a positive control.

Expected compute is about 20–30 GPU-hours. The decisive falsification is: the in-context leak
inflates rejection, the model recalls the exposed validation noise, but the full-exposure arm
still rejects at the nominal rate. That would show that weight-level exposure does not transfer
into test selection in this regime.

### Closest work and remaining gap

- [POPPER](https://arxiv.org/abs/2502.09858) supplies the theorem and assumption being stress-tested;
  it does not randomize pretraining exposure.
- [Investigating Data Contamination for Pre-training Language Models](https://arxiv.org/abs/2401.06059)
  studies contamination effects on model performance, not downstream hypothesis-test size.
- [How Much Can We Forget about Data Contamination?](https://arxiv.org/abs/2410.03249)
  studies persistence of contamination, not agentic Type-I error.
- [Elephants Never Forget](https://arxiv.org/abs/2404.06209) demonstrates tabular memorization
  but does not connect it to adaptive statistical validity.
- [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629)
  handles analysts adaptively querying a holdout, not an analyst exposed to the holdout before
  the protocol begins.

**Claim boundary:** exposure causes inflation only under the tested generator, model, dose,
and reachable hypothesis family. A null result remains interpretable only if the memorization
probe and positive control succeed.

---

## I3 — Contamination-immune agentic tests and the price of open-endedness

### Research question

What is the smallest correction that guarantees Type-I error control when an LLM agent may
have memorized the validation set, and what scientific flexibility must be surrendered to get
that guarantee?

I2 measures a failure mode. I3 is a separate theoretical procedure paper. Do not develop both
for the same submission.

### Precise contribution

Let `F` be a finite, pre-declared family of falsification tests with a computable joint null
distribution. Assume an adversarial agent may know the realized validation data and may select
any `f` in `F`.

Prove the following necessary-and-sufficient result:

1. **Sufficiency:** calibrating to the `1-alpha` quantile of
   `max_{f in F} T_f` controls Type-I error for every selection rule, including a memorizing
   LLM agent.
2. **Necessity:** any universally valid threshold must be at least that max-statistic quantile,
   because an adversarial agent can choose `argmax_{f in F} T_f`.
3. **Fresh-data alternative:** if data are drawn only after the agent commits its test function,
   the ordinary pre-specified test remains valid without the family-wide penalty.

The result formalizes an exact tradeoff: validation-set contamination immunity requires either
fresh post-commitment data or correction over the entire test family reachable by the agent.
Free-form hypothesis generation has no finite declared family, so this guarantee deliberately
removes the open-endedness that motivates agentic discovery.

The standard max-statistic correction is not novel. The candidate survives only if the paper
centers the universal-agent necessity result, its tightness, and the empirical cost for an
agentic falsification system.

### Smallest decisive experiment

1. Generate 500 null tabular datasets and matched planted alternatives.
2. Compare a normal agent, a fixed non-LLM selector, and an oracle-leak agent that receives
   the validation split.
3. Compare the naive split test, declared-family max correction, and fresh-data validation.
4. Measure Type-I error, power, and the growth of the critical value as `|F|` and dependence
   inside `F` change.
5. Repeat with several natural-language surface forms mapping to the same declared tests to
   separate language flexibility from statistical reachability.

This should require under two GPU-hours plus CPU simulation. The strongest falsification is
that the naive procedure retains nominal size against the oracle-leak agent while the agent can
select across the full family; that would eliminate the motivating failure.

### Closest work and remaining gap

- [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629)
  and [The Generic Holdout](https://arxiv.org/abs/1809.05596) control adaptive reuse through
  restricted interaction with the holdout. They do not model an agent that already knows its
  realization through training.
- [POPPER](https://arxiv.org/abs/2502.09858) assumes training independence and provides the
  agentic setting, but no contamination-immune threshold.
- [Valid Feature-Level Inference for Tabular Foundation Models via the CRT](https://arxiv.org/abs/2603.06609)
  addresses feature-level inference. Conditional randomization alone does not solve this
  problem when the agent selects the statistic using the realized resampled variable.
- [Fisher-R1](https://arxiv.org/abs/2608.07437) trains hypothesis-testing agents but does not
  prove validation-data contamination immunity.
- [Decoy-Calibrated Failure Audits for Language Models](https://arxiv.org/abs/2606.09046)
  uses decoys and a holdout for failure discovery, a different target and guarantee.

**Claim boundary:** the theorem requires a finite reachable family with a computable joint
null. For unrestricted natural-language tests, the result is an impossibility statement rather
than a usable correction.

---

## Ideas killed during the audit

| Candidate | Reason for rejection |
|---|---|
| Prompt uncertainty decomposition | A 2025 WWW paper and 2026 variance-component work already cover it. |
| Simulator-error rank bounds | Prediction-powered ranking and best-arm identification with LLM judges occupy the core object. |
| Serialization-invariance certificate | Stability does not imply accuracy; Caliper and robustness work cover the perturbation design. |
| Single-arm item-title placebo | Caliper transferred to recommendation; familiarity and topical relatedness remain confounded. |
| Two-arm catalogue-title advantage | Same domain-transfer problem, infeasible matching on MovieLens, and average-effect interval was misstated. |
| FDR evidence filter for LLM profiles | ContextCite plus BY is a routine combination; the matched-replacement null is non-exchangeable. |
| LLM-judge error-budget DCG certificate | Correct worst-case mathematics, but realistic error rates make every interval contain zero. |
| LLM-judge correction with PPI | PRECISE and related prediction-powered evaluation papers already cover it. |
| Circular profile generator/judge correction | Eval-Pair Matrix, circularity work, and transport non-identification occupy the design. |
| Cutoff difference-in-discontinuities | Direct 2026 method collision and non-identification of leakage from passive temporal scores. |
| Per-user history-swap discovery count | The useful null is observationally non-exchangeable; the sharp invariance null is degenerate. |
| Certified count of memorized users | Direct collision with [FDR-controlled membership inference](https://arxiv.org/abs/2508.07066); train/holdout exchangeability also fails after training. |
| Budget-optimal full-catalog item sampling | [Towards Reliable Item Sampling](https://arxiv.org/abs/2211.15743) already derives an error-optimized estimator and adaptive sampler. |
| Anytime-valid full-catalog rank certification | [Top-k on a Budget](https://arxiv.org/abs/2601.20989) solves the stronger weak/strong-oracle problem with PAC guarantees and lower bounds. |

## Search and review record

Searches used the local `lit-research` client against OpenAlex, the official arXiv API, and
Semantic Scholar. Semantic Scholar requests were serialized with at least 1.2 seconds between
calls and usually 3 seconds. The default arXiv export mirror repeatedly timed out or returned
429, so the identical client was pointed at the official `arxiv.org/api/query` endpoint. That
fallback returned successfully.

Every final candidate received at least three distinct query formulations and successful
results from all three literature ecosystems. For I3, exact late-stage OpenAlex retries were
rate-limited; its Popper/contamination literature was already covered by the successful I2
OpenAlex searches, while three I3-specific arXiv searches and two Semantic Scholar searches
completed. Failed API calls were never treated as zero-result novelty evidence.

Raw records are in `literature/`:

- `final_s1*` — I2 / controlled Type-I exposure;
- `final_s3*` — I1 / factorial pretraining exposure;
- `final_c2star*` — I3 / contamination-immune testing;
- `repl_*` and earlier `final_title*`, `final_profile*`, `final_bounds*` — killed candidates.

Independent Claude Opus 5 reviews:

- `claude_review_round3.txt` killed the first three finalists and proposed replacements;
- `claude_review_round4.txt` killed those replacements and proposed the next set;
- `claude_review_round5.txt` killed the adaptive rank certificate, corrected I1/I2, and
  hard-passed I1, I2, and I3.

The first `claude` account hit its monthly spend limit during round 4. The completed round-4
review used the explicitly authorized `claude2` wrapper; round 5 used `claude3`. Both ran the
verified `claude-opus-5` model with independent WebSearch/WebFetch access.

## Recommendation

Build **I1** if the goal is the strongest match to personalization/recommendation and LLM data
mining. Build **I2** if the goal is the sharper statistical critique. Use **I3** when compute or
experimental turnaround matters most, while accepting its higher novelty risk. Do not combine
I2 and I3 into one five-page submission.
