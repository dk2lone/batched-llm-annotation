# Independent Claude novelty audit

## Search coverage and limitations

**Sources that worked**

1. arXiv API. It responded over HTTPS with a user agent. I used it for keyword search and for ID verification.
2. OpenAlex API. I used the `title_and_abstract.search` filter. Full-text search returned too much noise.
3. Web search. It gave access to OpenReview, ICLR proceedings, ACM, and publisher pages.

**Sources that failed**

1. **Semantic Scholar Graph API. Blocked for the whole session.** Every call returned HTTP 429 "Too Many Requests". I retried three times with delays of 4 s and 8 s. No Semantic Scholar result contributed to any verdict. Treat this as missing coverage, not as zero hits.
2. **OpenReview forum pages. Blocked by a browser verification page.** The fetch of `openreview.net/forum?id=ecbEfXdqrV` returned a challenge page. I recovered the same paper through its arXiv mirror.
3. **ICLR 2026 proceedings PDF. Not parsable.** The fetch of the `proceedings.iclr.cc` PDF returned binary data. I recovered the same paper through its arXiv mirror.
4. **Cross-model verification. Not available.** The skill specifies a second opinion from `gpt-5.6-sol` through the Codex MCP server. That server reported `CONNECTION_CLOSED` at session start. All verdicts below are single-model verdicts.

**Verification method.** I resolved every cited arXiv paper through `export.arxiv.org/api/query?id_list=<id>`. That call returns the canonical title and submission date. All IDs below passed. One candidate for idea A, arXiv 2607.20528 "PromptPack", did **not** resolve through `id_list`. I removed it from all verdicts.

**Coverage limits.** ICML 2026 and NeurIPS 2026 proceedings are not published as of 2026-09-11. I covered those venues only through arXiv preprints. arXiv coverage runs through 2026-09-11 and includes hits from September 2026.

---

## Ranking

The rank order follows defensible novel contribution. Feasibility broke the tie between A and A2.

| Rank | Idea | Novelty | Contribution | Verdict |
|---|---|---|---|---|
| 1 | **A** Batched LLM annotation dependence | 7/10 | 8/10 | PROCEED |
| 2 | **A2** Factorial content versus co-occurrence exposure | 7/10 | 7/10 | PROCEED WITH CAUTION |
| 3 | **K** Distance concentration and tabular failure | 6/10 | 7/10 | PROCEED WITH CAUTION |
| 4 | **A3** Validation exposure and agentic Type-I error | 6/10 | 6/10 | PROCEED WITH CAUTION |
| 5 | **I1** Non-target future-review leakage | 5/10 | 5/10 | PROCEED WITH CAUTION |
| 6 | **T** Within-row redundancy and likelihood pathology | 3/10 | 3/10 | REPAIR THEN PROCEED |
| 7 | **A1** Repository provenance replay | 2/10 | 4/10 | ABANDON |

---

## K

### Core claim verdicts

| # | Claim | Novelty | Basis |
|---|---|---|---|
| K1 | Generic decoder-only LLM accuracy on in-context tabular prediction drops as dimension rises | **NONE** | Established 5 weeks ago by arXiv 2608.02412 |
| K2 | Dimension itself is not the cause. Concentration of distances in the model's row representations is the cause | **HIGH** | 2608.02412 states the mechanism is open. No paper tests this |
| K3 | Decisive test: high-dimension datasets with low representation-space concentration against matched concentrating datasets | **HIGH** | No prior controlled decoupling of dimension from concentration in an LLM |
| K4 | Relative contrast measured inside the model's own row representations | **MEDIUM** | Relative contrast is a standard metric. Its use on LLM row states is new |
| K5 | Causal removal of nearest rows plus permutation and ordering controls | **MEDIUM** | Demonstration-order controls are common. The nearest-row removal test is not |

### Closest prior work

| Paper | Year | Venue | URL | Exact overlap | Key difference |
|---|---|---|---|---|---|
| Garnelo & Czarnecki, *Why Large Language Models Fail at Tabular Prediction* | 2026 | arXiv preprint | https://arxiv.org/abs/2608.02412 | The whole phenomenon. Random linear projections over 31 datasets. LLM is the only method of nine whose accuracy falls with dimension. LLM matches a local distance-based method in 2D | The authors write: "We do not claim to have identified the internal mechanism". No representation analysis. No concentration measurement. No concentration intervention |
| Kirichenko, Izmailov & Wilson, *Why Normalizing Flows Fail to Detect Out-of-Distribution Data* | 2020 | arXiv preprint | https://arxiv.org/abs/2006.08545 | Same style of argument: a geometric property of the input, not capacity, explains a failure | Different model class and different task |
| *A Mechanistic Study of Tabular Foundation Models* | 2026 | arXiv preprint | https://arxiv.org/abs/2605.21288 | Internal analysis of tabular in-context models | Studies TabPFN-class models, not generic LLMs. No concentration claim |

### Score reasoning

Method novelty is **low**. Relative contrast and distance concentration come from the hubness literature. The proposal applies them, it does not invent them.

Experimental-design novelty is **high**. Garnelo & Czarnecki swept dimension. They could not separate dimension from concentration, because a random projection changes both together. The matched-concentration construction breaks that confound. That is the exact control their paper lacks.

Empirical-finding novelty is **high** if the intervention succeeds. The finding would close a question that a recent, visible preprint left open in print.

**Novelty 6/10. Contribution 7/10.**

### Recommendation

**PROCEED WITH CAUTION.**

The phenomenon is gone. Do not write a paper that re-establishes it. Write the mechanism paper and cite 2608.02412 as the motivating result in the first paragraph.

Move fast. Any group that reads 2608.02412 will reach for the same mechanism. The open-mechanism sentence in that abstract is an invitation.

### Defensible positioning

State the contribution as one sentence: a recent result shows LLM tabular accuracy dissolves with dimension and leaves the cause open, and this work identifies the cause by decoupling dimension from representation-space concentration.

Keep attention maps descriptive. Do not build a claim on them. Reviewers reject attention-as-explanation arguments.

### Kill criterion

Build the matched dataset pair first. Measure relative contrast in the model's row representations. If you cannot build two datasets at the same dimension with a large and stable gap in relative contrast, stop. If accuracy is equal across that gap, the hypothesis is false and you must report that as a negative result or stop.

---

## A

### Core claim verdicts

| # | Claim | Novelty | Basis |
|---|---|---|---|
| A-1 | Batch prompting degrades per-item annotation quality | **NONE** | Known. See arXiv 2503.15551 and 2608.02681 |
| A-2 | Batching induces within-batch covariance among error indicators, with model, prompt, decoding, and item population fixed | **HIGH** | No paper measures this quantity. OpenAlex returns 0 |
| A-3 | That covariance breaks confidence-interval coverage and Type-I error for downstream estimators of LLM labels | **HIGH** | Coverage failure is studied for other causes, not for batch structure |
| A-4 | Randomized re-assignment of the same items into many batches, across a batch-size sweep | **HIGH** | This is an identification design. Prior work compares batch sizes on fixed splits |
| A-5 | Fake-batch permutation control plus length-matched filler separates batch structure from context length | **HIGH** | No prior use found |

### Closest prior work

| Paper | Year | Venue | URL | Exact overlap | Key difference |
|---|---|---|---|---|---|
| Cheng et al., *Batch Prompting: Efficient Inference with LLM APIs* | 2023 | EMNLP 2023 Industry Track | https://arxiv.org/abs/2301.08721 | Introduces the practice. Reports accuracy against batch size | Treats accuracy as the only outcome. No dependence structure. No inference validity |
| *Efficient but Vulnerable: Benchmarking and Defending LLM Batch Prompting Attack* | 2025 | arXiv preprint | https://arxiv.org/abs/2503.15551 | Shows one item in a batch changes answers for other items | Adversarial framing. Measures mean accuracy loss, not error correlation |
| *Safety in Batches? Understanding and Mitigating Safety Failures in Batch Prompting* | 2026 | arXiv preprint | https://arxiv.org/abs/2608.02681 | Same cross-item contamination effect | Safety outcome. No statistical estimand |
| Egami et al., *Using Imperfect Surrogates for Downstream Inference (DSL)* | 2023 | NeurIPS 2023 | https://arxiv.org/abs/2306.04746 | Corrects downstream inference from LLM labels | Assumes independent labeling errors. Batching breaks that assumption. This is the gap |
| Messing, *Hidden Measurement Error in LLM Pipelines* | 2026 | arXiv preprint | https://arxiv.org/abs/2604.11581 | Shows naive intervals undercover and get worse as n grows | Variance comes from judge choice, temperature, and prompt phrasing. Not from batch structure |
| Camuffo et al., *Variance-Aware LLM Annotation for Strategy Research* | 2026 | arXiv preprint | https://arxiv.org/abs/2601.02370 | Five variance sources for LLM annotation. Notes correlated errors bias estimates | Lists "interface effects" but does not isolate batching or estimate intra-batch correlation |
| *Large Language Model Hacking* | 2025 | arXiv preprint | https://arxiv.org/abs/2509.08825 | Type-I error inflation from LLM annotation choices | Inflation comes from researcher configuration search, not from a design-induced dependence |

### Score reasoning

Method novelty is **low to medium**. The intra-batch correlation coefficient and cluster-robust variance are textbook tools. The dossier correctly treats the correction as a baseline, not as the contribution.

Experimental-design novelty is **high**. Randomizing the same item into many batch assignments makes the batch an exogenous treatment. That design identifies the effect. Nothing found does this.

Empirical-finding novelty is **high** and the reach is wide. Large parts of applied NLP, computational social science, and evaluation practice batch items to cut cost. If batching induces dependence, a large body of reported intervals is too narrow.

**Novelty 7/10. Contribution 8/10.**

### Recommendation

**PROCEED.** This is the cleanest gap of the seven. It is also cheap. It needs API calls, not training runs.

### Defensible positioning

Name the estimand in the title or the abstract. Do not sell "batching hurts accuracy". Sell "batching creates a cluster structure that the field's estimators assume away".

Report the design effect, not only the correlation coefficient. Applied readers act on an effective sample size.

State the cluster-robust correction as the obvious repair and show its residual failure modes. That removes the most common objection before a reviewer raises it.

### Kill criterion

Run the batch-size sweep on one item population first. Estimate the intra-batch correlation of the error indicator. If the estimate is near zero and its interval excludes a practically relevant value, stop. Without measurable correlation, there is no coverage damage and no paper.

---

## T

### Core claim verdicts

| # | Claim | Novelty | Basis |
|---|---|---|---|
| T1 | The likelihood pathology is rare in native tabular data relative to image and text embeddings | **NONE. Full collision** | arXiv 2602.09593 established this in February 2026 |
| T2 | Reproduce the pathology rate on native tabular and embedding subsets | **NONE** | Same paper. 47 tabular datasets and 10 embedding datasets in ADBench |
| T3 | The cause is a lack of within-sample redundancy, not dimension and not ordinary feature correlation | **MEDIUM** | The same paper already tests dimension and feature correlation. Redundancy is the residual |
| T4 | A redundancy measure with enough resolution for short mixed-type rows | **MEDIUM, and unproven** | No such measure found. It is also the hardest part of the project |
| T5 | Controls that separate redundancy from correlation | **MEDIUM** | Depends entirely on T4 |

### Closest prior work

| Paper | Year | Venue | URL | Exact overlap | Key difference |
|---|---|---|---|---|---|
| Kim, Phee & Yoon, *Why the Counterintuitive Phenomenon of Likelihood Rarely Appears in Tabular Anomaly Detection with Deep Generative Models?* | 2026 | arXiv preprint, under OpenReview review | https://arxiv.org/abs/2602.09593 | The research question, the empirical claim, the tabular-against-embedding contrast, the dataset scale, and a domain-agnostic definition of the phenomenon. It also studies dimensionality and feature correlation | It does not name redundancy. That is the only gap left |
| Nalisnick et al., *Do Deep Generative Models Know What They Don't Know?* | 2019 | ICLR 2019 | https://arxiv.org/abs/1810.09136 | Original phenomenon | Images only |
| Kirichenko, Izmailov & Wilson, *Why Normalizing Flows Fail to Detect OOD Data* | 2020 | arXiv preprint | https://arxiv.org/abs/2006.08545 | Blames "local pixel correlations". This is the redundancy story on the image side | Images only. No tabular test. No general redundancy measure |
| Serrà et al., *Input complexity and out-of-distribution detection* | 2020 | ICLR 2020 | https://arxiv.org/abs/1909.11480 | Corrects likelihood by a compression-based complexity term. This is a redundancy proxy | The dossier already excludes compression at row scale |

### Score reasoning

Method novelty is **low**. The only candidate method is the redundancy measure, and it does not exist yet.

Experimental-design novelty is **very low**. Kim et al. already ran the reproduction that T lists as a deliverable.

Empirical-finding novelty is **low**. The headline finding is published. What remains is a correction to one competing explanation inside somebody else's paper.

**Novelty 3/10. Contribution 3/10.**

### Recommendation

**REPAIR THEN PROCEED.** Understand the repair is itself a hard research problem.

The only survivable paper is narrow. It must refute the dimensionality and feature-correlation explanation in Kim et al. and replace it with redundancy. That paper cannot be written until the redundancy measure works.

### Defensible positioning

Do not present the rarity finding as your result. Cite 2602.09593 as the established fact in the first sentence.

Frame the paper as an explanation contest with a decisive control. Two datasets must match on dimension and on pairwise feature correlation but differ in redundancy.

### Kill criterion

Build the redundancy measure first, before any experiment. Apply it to synthetic rows with known redundancy. If the measure cannot separate known-redundant from known-independent rows at row widths typical of tabular data, stop the project. The dossier already closed the two easy routes. There is no fallback.

---

## A2

### Core claim verdicts

| # | Claim | Novelty | Basis |
|---|---|---|---|
| A2-1 | Synthetic catalogue with unfamiliar identifiers, content attributes, and a separately generated latent preference graph | **HIGH** | No such construct found for recommendation |
| A2-2 | 2 by 2 randomized continued-pretraining exposure design with multiple doses | **HIGH** | Factorial pretraining designs exist. None targets this question |
| A2-3 | Closed-book probes verify acquisition before fine-tuning | **MEDIUM** | Probe-then-test is standard practice in knowledge-injection work |
| A2-4 | Estimate both main effects and their interaction by randomized assignment | **HIGH** | The field asserts this decomposition. No work identifies it causally |

### Closest prior work

| Paper | Year | Venue | URL | Exact overlap | Key difference |
|---|---|---|---|---|---|
| *Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M* | 2025 | arXiv preprint | https://arxiv.org/abs/2505.10212 | Asks whether pretraining already contains the evaluation data | Observational detection. No randomization. No content-against-co-occurrence split |
| *Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?* | 2026 | arXiv preprint | https://arxiv.org/abs/2602.13626 | Injects leaked benchmark data and measures the performance change. Separates domain-relevant from domain-irrelevant leakage | Real benchmarks, not a synthetic catalogue. No factorial design. No latent preference graph |
| *On the Memorization Behavior of LLMs in Generative Recommendation* | 2026 | arXiv preprint | https://arxiv.org/abs/2606.17276 | Memorization during recommendation training | Descriptive. No exposure randomization |
| *Catalog-Native LLM: Speaking Item-ID Dialect with Less Entanglement for Recommendation* | 2025 | ICLR 2026 | https://arxiv.org/abs/2510.05125 | Names the entanglement of text knowledge and catalog signal. Continued pretraining | A method paper. It separates the two signals by architecture, not by measurement |

### Score reasoning

Method novelty is **low**. Continued pretraining, synthetic corpora, and closed-book probes are all standard.

Experimental-design novelty is **high**. The 2 by 2 randomized exposure with dose levels turns an untestable narrative into an estimable effect. The interaction term is the real target. It answers whether content knowledge and co-occurrence knowledge substitute or complement.

Empirical-finding novelty is **high**. The field repeats the claim that LLMs help recommendation because of world knowledge. That claim has no causal evidence.

**Novelty 7/10. Contribution 7/10.**

### Recommendation

**PROCEED WITH CAUTION.**

Two reasons for caution. Compute cost is real. Four arms times several doses times several seeds means many continued-pretraining runs. External validity is the other risk.

### Defensible positioning

Present the interaction term as the headline. A main-effects-only result is weaker and predictable.

Put one real-catalogue check in the paper. Choose a held-out real domain and show the direction of the effect agrees. That answers the external-validity objection.

State the estimand precisely. You estimate the effect of exposure on a base model under a fixed fine-tuning procedure. You do not estimate the effect of web-scale pretraining.

### Kill criterion

Run the closed-book probes after the smallest dose. If the model does not acquire the injected content and co-occurrence knowledge at a measurable rate, the design has no treatment and the project stops. Check this before any recommendation fine-tuning.

---

## A3

### Core claim verdicts

| # | Claim | Novelty | Basis |
|---|---|---|---|
| A3-1 | An LLM agent that adaptively chooses statistical tests inflates the false-positive rate | **LOW** | Widely asserted. Related evidence in arXiv 2509.08825 and 2608.07437 |
| A3-2 | Training exposure to hidden validation outcomes specifically inflates that rate | **HIGH** | No paper measures this |
| A3-3 | Many instances under a known global null, each with discovery and hidden validation splits | **MEDIUM** | Global-null generators are standard in statistics. They are new for agent evaluation |
| A3-4 | Instance-level randomized exposure inside a shared model, or several seeds per arm, to remove the one-model-per-arm confound | **HIGH** | This is a real methodological point. No prior use found |
| A3-5 | Measure recall of validation noise as the mediator | **HIGH** | A direct test of the leakage channel. No prior use found |

### Closest prior work

| Paper | Year | Venue | URL | Exact overlap | Key difference |
|---|---|---|---|---|---|
| Huang et al., *Automated Hypothesis Validation with Agentic Sequential Falsifications* (Popper) | 2025 | ICML 2025 | https://arxiv.org/abs/2502.09858 | LLM agents plus Type-I error control for hypothesis validation | Builds a procedure that controls error by construction. Does not measure inflation caused by exposure |
| Miao et al., *Fisher-R1: Training LLM Agents for Reliable Hypothesis Testing* | 2026 | arXiv preprint | https://arxiv.org/abs/2608.07437 | 425 hypothesis-testing tasks. Shows agents make invalid inferential choices | Measures task success, not realized Type-I error under a global null. No exposure arms |
| Bertran, Roth & Wu, *What Fits (Into Few Tokens) Doesn't Overfit* | 2026 | arXiv preprint | https://arxiv.org/abs/2606.11045 | LLM research agents that adaptively reuse a validation set. Directly studies adaptive overfitting | Outcome is model performance, not Type-I error. Exposure is in-context, not weight-level |
| *Large Language Model Hacking* | 2025 | arXiv preprint | https://arxiv.org/abs/2509.08825 | Manufactured false positives in 94.4% of tested null hypotheses | The source is researcher search over annotation configurations. Not agent exposure to validation outcomes |

### Score reasoning

Method novelty is **low**. Global-null simulation and false-positive counting are standard statistics.

Experimental-design novelty is **high**. The instance-level randomization inside one model is the strongest part of the proposal. One model per arm confounds the treatment with every other difference between two training runs. Solving that confound is a contribution that generalizes past this paper.

Empirical-finding novelty is **medium**. A reviewer may call the direction obvious. The magnitude, the mediation through recall of validation noise, and the fixed-selector control carry the paper.

**Novelty 6/10. Contribution 6/10.**

### Recommendation

**PROCEED WITH CAUTION.** The risk is not a collision. The risk is that the result reads as expected.

### Defensible positioning

Make the identification the contribution. Say that the paper measures a leakage effect on agent error rates without the one-model-per-arm confound, and that prior work on contamination cannot do this.

Report the mediator. Show that the agent recovers validation-split noise, and show that this recovery predicts the excess false positives. The mechanism converts an expected direction into evidence.

Report power on planted alternatives in the same table as Type-I error. A result that shows inflated false positives with no gain in true positives is much harder to dismiss.

### Kill criterion

Run the in-context positive control first. It is cheap. If direct validation access in the prompt does not inflate the false-positive rate above the nominal level, your agent, task family, or null generator is wrong. Fix it before you spend money on weight-level exposure.

---

## A1

### Core claim verdicts

| # | Claim | Novelty | Basis |
|---|---|---|---|
| A1-1 | Active repository entries from the same canonical dataset contain undocumented preprocessing differences | **NONE** | Documented by arXiv 2503.09159 |
| A1-2 | Those differences change model comparisons | **NONE** | Shown by arXiv 2503.09159 and by arXiv 2407.02112 at NeurIPS 2024 |
| A1-3 | Exact reconstruction from checksummed canonical files with an explicit transformation graph | **MEDIUM** | An engineering contribution, not a scientific claim |
| A1-4 | Intervene on one named transformation at a time and attribute the change in performance or winner | **NONE** | arXiv 2503.09159 already does exactly this |

### Closest prior work

| Paper | Year | Venue | URL | Exact overlap | Key difference |
|---|---|---|---|---|---|
| Tschalzev et al., *Unreflected Use of Tabular Data Repositories Can Undermine Research Quality* | 2025 | arXiv preprint | https://arxiv.org/abs/2503.09159 | Almost total. It documents that OpenML versioning misled a major benchmark into a preprocessed version of `seattlecrime6`. It changes one transformation at a time. Treating a date feature as categorical benefits all models on `electricity`. Adding a date difference feature on `nyc-taxi` "reverses the performance tendencies of tree-based models versus neural networks" | It does not reconstruct from canonical sources with checksums. That is the only gap |
| Tschalzev et al., *A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data* | 2024 | NeurIPS 2024 | https://arxiv.org/abs/2407.02112 | Dataset-specific preprocessing changes model rankings considerably and shrinks performance differences | Uses expert pipelines on 10 Kaggle datasets. Not a provenance audit |

### Score reasoning

Method novelty is **very low**. Per-transformation intervention under fixed folds and models is exactly what Tschalzev et al. 2025 did.

Experimental-design novelty is **very low** for the same reason.

Empirical-finding novelty is **very low**. The finding that preprocessing reverses winners is published, with named datasets.

The checksummed reconstruction is genuine work and it is genuinely useful. It is an artifact contribution. It is not a novel scientific claim, and the dossier presents it as one.

**Novelty 2/10. Contribution 4/10.** The contribution score is above the novelty score because the artifact has real value.

### Recommendation

**ABANDON** the proposal as stated. The stated contribution is "exact provenance reconstruction plus causal per-transformation attribution". The second half is taken.

One alternative survives. Rebuild it as a datasets-and-benchmarks artifact submission. Then the claim is a verified, reproducible provenance layer for a named repository, and the scientific finding belongs to Tschalzev et al. That is a different paper with a different claim. Do not attempt it under a novelty argument.

### Defensible positioning

If you take the artifact route, make coverage the headline number. Report how many entries reconstruct to exact row and column agreement, and how many do not. An entry that cannot be reconstructed is your most interesting result.

### Kill criterion

Pick five repository entries. Try to reconstruct them exactly from canonical files. If four or five reconstruct without difficulty, there is no problem to report. If none reconstruct, the transformation search space is too large and the project does not scale. You need a middle outcome.

---

## I1

### Core claim verdicts

| # | Claim | Novelty | Basis |
|---|---|---|---|
| I1-1 | Temporal leakage inflates recommender accuracy | **NONE** | Established by TOIS 2023 and RecSys 2025 work |
| I1-2 | The target review leaks the answer | **NONE** | The dossier already concedes this |
| I1-3 | Non-target reviews written after the cutoff still inflate accuracy after the target review is removed | **MEDIUM to HIGH** | No prior measurement found. OpenAlex returns no relevant hit |
| I1-4 | Separate later same-user reviews from later same-item reviews | **MEDIUM** | A useful decomposition. Not found in prior work |
| I1-5 | A timestamp-safe evaluation protocol for text profiles | **LOW** | Timestamp-safe protocols exist for interactions. Extending them to text is routine |

### Closest prior work

| Paper | Year | Venue | URL | Exact overlap | Key difference |
|---|---|---|---|---|---|
| Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation* | 2023 | ACM TOIS 41(3):75 | https://arxiv.org/abs/2010.11060 | Shows that off-timeline splits inflate accuracy and make relative model order unpredictable | Interaction-level leakage only. No review text |
| *Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders* | 2025 | ACM RecSys 2025 | https://arxiv.org/abs/2507.16289 | Quantifies the drop from removing temporal leakage. Reports 21.7% to 73.4% lower sampled nDCG@10 | Splits interactions, not text profiles |
| Au & Lin, *PeReGrINE* | 2026 | arXiv preprint | https://arxiv.org/abs/2604.07788 | Applies explicit temporal cutoffs so retrieved evidence precedes the target review | A review-generation benchmark. It enforces the cutoff. It never measures the cost of violating it |
| *SplitLight: An Exploratory Toolkit for Recommender Systems Datasets and Splits* | 2026 | arXiv preprint | https://arxiv.org/abs/2602.19339 | Tooling for splits and leakage inspection | Interaction-level. No text-profile condition |

### Score reasoning

Method novelty is **very low**. The proposal is a controlled ablation over text conditions.

Experimental-design novelty is **medium**. Freezing interactions, targets, candidates, split, and model, then varying only the text condition, is the right design and nobody has run it.

Empirical-finding novelty is **medium**. The direction is predictable. The magnitude is not. If a large part of reported review-aware gains comes from post-cutoff text, that changes how the sub-field reports results.

**Novelty 5/10. Contribution 5/10.**

### Recommendation

**PROCEED WITH CAUTION.** This is the cheapest project of the seven and the least likely to be scooped. It is also the smallest claim.

### Defensible positioning

Lead with the magnitude, not the existence. "Removing the target review is not sufficient" is the sentence that carries this paper.

Publish the corrected numbers for several published review-aware models. A protocol paper with no re-measured baselines will not persuade a reviewer.

Keep the two decompositions separate in every table. Later same-user text and later same-item text have different causes and different fixes.

### Kill criterion

Run one model on one dataset with the target review removed in every condition. Compare past-only text against later non-target text. If the accuracy gap is inside the seed-to-seed variation, stop. The whole paper rests on that residual gap.

---

## Cross-idea conclusion

I formed this ranking from live searches only. I did not read any earlier ranking, score, or literature note in this project, and no prior ordering was available to me.

**I rank A first: batched LLM annotation dependence.**

Three reasons put it above the others.

1. No collision. Two specific OpenAlex queries on the core estimand returned zero results. The batch-prompting literature measures mean accuracy. The LLM-labels-for-inference literature assumes independent errors. Neither side has looked at the join.
2. The design settles the question. Randomizing the same items into different batch assignments makes the batch an exogenous treatment. Most novelty-rich proposals lack an identification argument this clean.
3. The cost is low and the reach is wide. The work needs API calls, not training runs. The finding applies to any pipeline that batches items to save tokens.

A2 is a close second and loses only on cost and external validity. K is third. Its mechanism delta is real, but it depends on a preprint that is five weeks old, and the same opening is visible to everybody who reads that abstract.

**The exact result A must establish.**

A single number decides the paper. Under randomized batch assignment, with model, prompt, decoding, and item population fixed, the intra-batch correlation of the error indicator must be **clearly greater than zero, and must survive the fake-batch permutation control and the length-matched filler control**. That combination rules out context length and prompt content as the cause and leaves batch structure.

That correlation must then translate into measured damage. Nominal 95% confidence intervals for a realistic downstream estimator must undercover by a practically relevant margin, or realized Type-I error must exceed the nominal level. Undercoverage to roughly 90% or below, at a batch size that practitioners actually use, is the result that justifies the paper.

If the correlation is near zero, or if it appears only at batch sizes nobody uses, there is no paper. Run that measurement before anything else.

**The one thing waiting on you.** Decide whether to start A. Nothing else in this report needs your input.

Sources: [2608.02412](https://arxiv.org/abs/2608.02412), [2602.09593](https://arxiv.org/abs/2602.09593), [2503.09159](https://arxiv.org/abs/2503.09159), [2407.02112](https://arxiv.org/abs/2407.02112), [2010.11060](https://arxiv.org/abs/2010.11060), [2507.16289](https://arxiv.org/abs/2507.16289), [2502.09858](https://arxiv.org/abs/2502.09858), [2606.11045](https://arxiv.org/abs/2606.11045), [2608.07437](https://arxiv.org/abs/2608.07437), [2509.08825](https://arxiv.org/abs/2509.08825), [2604.11581](https://arxiv.org/abs/2604.11581), [2601.02370](https://arxiv.org/abs/2601.02370), [2306.04746](https://arxiv.org/abs/2306.04746), [2301.08721](https://arxiv.org/abs/2301.08721), [2503.15551](https://arxiv.org/abs/2503.15551), [2608.02681](https://arxiv.org/abs/2608.02681), [2006.08545](https://arxiv.org/abs/2006.08545), [1810.09136](https://arxiv.org/abs/1810.09136), [1909.11480](https://arxiv.org/abs/1909.11480), [2505.10212](https://arxiv.org/abs/2505.10212), [2602.13626](https://arxiv.org/abs/2602.13626), [2606.17276](https://arxiv.org/abs/2606.17276), [2510.05125](https://arxiv.org/abs/2510.05125), [2605.21288](https://arxiv.org/abs/2605.21288), [2604.07788](https://arxiv.org/abs/2604.07788), [2602.19339](https://arxiv.org/abs/2602.19339)

