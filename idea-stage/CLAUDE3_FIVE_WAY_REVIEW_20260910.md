# Independent five-way research-idea audit

Reviewer: Claude Opus 5 (`claude-opus-5`), independent pass
Date: 2026-09-10
Scope: I1, A1, A2, A3, A4 for the IEEE BigData 2026 High School Symposium

This review is literature-based and design-based. No experiment was run. No experimental result
is claimed anywhere in this document.

---

## 1. Independence protocol and how it was followed

Read before scoring:

- `AGENTS.md`.
- `ideas.md`, the I1 section.
- `idea-stage/IDEA_REPORT.md`.
- `idea-stage/alternative-run-20260910/idea.md` for A1.
- `idea-stage/alternative-run-20260910/idea-generation-report.md` for A2, A3, and A4.

Not read before scoring: `idea-stage/FIVE_WAY_COMPARISON_20260910.md` and every
`claude_review_round*` file. Both stayed closed until the scores in section 3 were final.

**One protocol deviation.** The read of the I1 section of `ideas.md` started at line 1 and
included the "Current decision" table at lines 7 to 26. That table shows the existing five-way
ranking. The exposure is recorded here so the reader can discount it. The scores below were
derived from the primary literature and from the designs, and they do not reproduce that table.

### Search-source status

Record every source failure, because a failed request is not evidence of absence.

| Source | Status | Effect on this review |
|---|---|---|
| Web search | Worked for all 16 queries | Primary discovery source |
| arXiv official API (`arxiv.org/api/query`) | Worked | Primary verification source; 26 identifiers checked |
| arXiv export mirror (`export.arxiv.org`) | Connection failure on every call | Not used |
| Page fetch (HTML) | Worked | Used to read POPPER assumptions and venue rules |
| Page fetch (PDF) | Returned undecoded binary for 2 arXiv PDFs | Worked around with the HTML versions |
| OpenAlex | 3 queries returned; 2 returned HTTP 429; returned results had low precision | Used only as a weak negative check |
| Semantic Scholar | 1 query returned; later queries returned HTTP 429, then timed out | Treated as unavailable, not as a zero-result |

Because OpenAlex and Semantic Scholar coverage is partial, **every novelty score here is
provisional**. None is a definitive "no prior work" verdict.

### Citation verification

All 26 arXiv identifiers cited in the source idea documents and in this review were checked
against the live arXiv API. Every one resolves, and every title matches the work attributed to
it. The source documents do not cite a paper that does not exist.

---

## 2. Verified venue constraints

From the official symposium page:

- Page limit: papers "should not exceed 5 pages, including all figures, tables, and references."
- Submission deadline: **September 20, 2026**. That is 10 days from today.
- Eligibility: the primary author must be a high school student during submission, and the
  student must be the main contributor.
- Relevant topics: 1 (foundations and theory of data mining), 2 (machine learning and
  statistical methods), 3 (mining heterogeneous sources including text), 5 (personalization and
  recommendation), 7 (data mining with large language models).
- Judging criteria: "technical quality, novelty, relevance, and presentation clarity."

Source: <https://bigdataieee.org/BigData2026/high-school-symposium/>

The audit brief states that GPU access and time are adequate. The feasibility score in
section 3 honors that assumption. The deadline is a separate, verified fact that the
assumption cannot remove, so section 3 reports it in its own column.

---

## 3. Scores

Every score is 1 to 10. Higher is better.

| ID | Idea | Novelty | Clarity | Venue fit | Feasibility | Ships by 2026-09-20 | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| A1 | Repository self-disagreement via provenance replay | 5 | 8 | 7 | 9 | Yes | Conditional pass |
| I1 | The Reviews from Tomorrow | 5 | 7 | 8 | 8 | Probably | Conditional pass |
| A2 | Factorial pretraining exposure, content versus co-occurrence | 7 | 6 | 8 | 6 | No | Conditional pass, later venue |
| A3 | Controlled validation exposure and agentic Type-I error | 7 | 7 | 6 | 5 | No | Conditional pass, later venue |
| A4 | Contamination-immune agentic tests | 3 | 4 | 5 | 8 | Yes | Reject |

Feasibility assumes adequate GPUs and adequate calendar time, as the brief instructs. It scores
technical executability only. It does not penalize A2 or A3 for needing model training.

---

## 4. I1: The Reviews from Tomorrow

### Research question

How much of a review-aware recommender's reported accuracy comes from review text written after
the historical prediction date, after the held-out target review is removed?

### Strongest defensible contribution

A timestamp-aligned protocol for building user and item text profiles in review-aware
recommenders, plus a decomposition of the residual metric inflation into two sources: later
reviews by the same user, and later reviews about the same item.

### Closest published work

| Work | Link | Same-design comparison |
|---|---|---|
| Catherine and Cohen, *TransNets: Learning to Transform for Recommendation*, RecSys 2017 | <https://arxiv.org/abs/1704.02298> | **Closest.** Shows that DeepCoNN's accuracy depends on the target user's review of the target item being present at test time, and builds a model that works without it. This is the same mechanism as the largest term I1 proposes to measure, published in 2017. |
| Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation*, TOIS 2023 | <https://arxiv.org/abs/2010.11060> | Global-timeline leakage changes measured utility and reorders models. Uses interactions, not text. |
| Shevchenko et al., *Time to Split*, RecSys 2025 | <https://arxiv.org/abs/2507.16289> | Splitting strategy drives temporal leakage and distorts model rankings. No text intervention. |
| *Don't Get Ahead of Yourself*, RecSys 2025 | <https://doi.org/10.1145/3705328.3759329> | Leakage in offline evaluation of sequential recommenders. No text intervention. |
| Tan et al., *Do Reviews Matter for Recommendations in the Era of Large Language Models?* (RAREval) | <https://arxiv.org/abs/2512.12978> | Removes and distorts reviews and reports that accuracy often does not fall. Does not align profiles to a cutoff. |
| Srifi et al., *Review-based Recommender Systems: A Survey* | <https://arxiv.org/abs/2405.05562> | Survey of the method family. Not an audit. |

### Novelty: 5

No located paper holds the recommendation task fixed and varies only the timestamp window of
the text profile. That is a real difference. It does not reach 7, for one reason: TransNets
published the target-review leakage result in 2017, and that is the dominant term. What is
left for I1 is the residual, non-target, post-cutoff text. That is a refinement of a settled
finding, not a new finding.

**The project's own closest-work table omits TransNets.** `idea-stage/IDEA_REPORT.md` lists four
closest works and none is TransNets. This is the most consequential literature gap found in this
audit. A reviewer who knows the review-based recommendation literature will raise TransNets in
the first paragraph of their report.

### Contribution clarity: 7

The estimand is a clean difference in one metric under one manipulated input. The deliverable
is a protocol. Both are easy to state in five pages.

### Venue fit: 8

Topic 5 and topic 3. An evaluation-protocol paper is well matched to the symposium's level.

### Feasibility: 8

Public timestamped data (Amazon Reviews, Yelp). CPU or one small GPU. The real cost is
reproducing one review-aware baseline, which is more work than a rating-only control.

### Main fatal risk

RAREval already reports that removing review text often does not reduce accuracy. If present
review text contributes little, then *future* review text contributes less still, and the
residual effect after target removal is near zero. There is then no effect to report. The
second risk is the one the project already names: a reviewer calls the leaky pipeline a
deliberate error rather than an observed practice.

### Smallest decisive feasibility gate

Take one Amazon category and one temporal cutoff. Build item-side text profiles two ways:
past-only, and full-corpus with every held-out target review deleted. Train the same
review-aware model on each. Compare the metric. If the gap sits inside the seed-to-seed noise
band, stop the direction. This is one pair of training runs and one text rebuild.

### Verdict: conditional pass

---

## 5. A1: Repository self-disagreement via provenance replay

### Research question

Do public repositories publish entries for the same source dataset that differ by undocumented
preprocessing, and does that difference change which model wins?

### Strongest defensible contribution

An exact, checksum-anchored reconstruction of published repository entries from one canonical
source file, which converts an observational difference between entries into a controlled
per-transformation effect, together with a demonstrated change in which model wins.

### Closest published work

| Work | Link | Same-design comparison |
|---|---|---|
| Tschalzev et al., *Unreflected Use of Tabular Data Repositories Can Undermine Research Quality*, 2025 | <https://arxiv.org/abs/2503.09159> | **Closest.** Documents that OpenML's version numbering led researchers to use preprocessed instead of original versions (it names `seattlecrime6`), that wrong preprocessing understates performance, and that the measured winner reverses between models on `colic` and `profb`. It does **not** reconstruct entries from a canonical source and does **not** run a factorial transformation intervention. |
| Tschalzev et al., *A Data-Centric Perspective on Evaluating ML Models for Tabular Data*, NeurIPS 2024 D&B | <https://arxiv.org/abs/2407.02112> | Dataset-specific preprocessing changes rankings. Uses expert pipelines on Kaggle data, not lineage-linked variants of one source. |
| Longjohn et al., *Benchmark Data Repositories for Better Benchmarking*, 2024 | <https://arxiv.org/abs/2410.24100> | Repository design and reproducibility. No row-linked performance discrepancy. |

### Novelty: 5

The replay-and-intervene *method* is new. The *finding* is not. Paper 2503.09159 already
establishes "repository variants and preprocessing choices change which model looks better",
which is A1's headline. A1 supplies attribution where that paper supplies demonstration.

A second discount applies to the specific effect claimed in `idea.md`. Random forest loses
about 1 point of accuracy when `age` and `hours-per-week` are pre-binned into five levels, while
histogram gradient boosting does not. That is the expected consequence of discarding
within-bin resolution for a model that splits on continuous values, next to a model that bins
internally. It is a correct result and a poor surprise.

The project's own closest-work list for A1 does not cite 2503.09159.

### Contribution clarity: 8

Highest of the five. Every claim is a checkable number: a checksum, a percentage of rows
reproduced, a paired difference, a p-value.

### Venue fit: 7

Topics 1 and 8. A data-quality audit is squarely in scope. It is weaker against the stated
"novelty" criterion than against "technical quality".

### Feasibility: 9

Highest of the five. CPU only, public data, hours of runtime.

### Claimed pilot versus reproducible evidence

**The claimed pilot is not reproducible in this workspace.** Findings:

1. `idea.md` names five scripts under `scratchpad/provenance/` and `scratchpad/`:
   `replay_effect.py`, `rank_flip.py`, `check_n1.py`, `check_n3.py`, and `codex_out.txt`.
   A filesystem search of `/Users/dz/bigdata` and of `/Users/dz` to depth 4 returns **none** of
   them. No `scratchpad/` or `provenance/` directory exists under the project.
2. No input or output artifact is cached. There is no `adult.data`, no `adult.test`, no OpenML
   download, and no CSV anywhere in the project tree.
3. The "Plan and full evidence" pointer is a `claude.ai/code/artifact/...` URL. That is a
   session artifact, not a repository, and it is not retrievable as project evidence.
4. The only files in the project that mention `replay_effect` or `rank_flip` are `idea.md`
   itself and `FIVE_WAY_COMPARISON_20260910.md`.

Therefore the four-model effect table, the recovered `age` and `hours-per-week` cut points, the
claimed 100.0000% reproduction over 48,842 rows, and the 10-fold rank flip
(p = 4.84e-06) must all be treated as **unverified claims**. They are a lead. They are not
evidence. The project's own comparison file records the same absence, which is to its credit.

### Main fatal risk

Not novelty and not compute. The risk is that the paper's central evidence currently exists
only as prose. If the replay does not reproduce the OpenML columns exactly on re-run, the
contribution collapses, because exactness is what separates A1 from 2503.09159.

### Smallest decisive feasibility gate

Re-run the reconstruction. Download `adult.data` and `adult.test`, apply the recovered five-level
cut points, and confirm the column match against OpenML `did=179` on all 48,842 rows. Then
re-run the 10-fold paired random forest against depth-8 decision tree on `did=179` and
`did=1590`. Commit the script, the checksums, and an environment lock. This is one script and
under an hour. It converts the whole claim into evidence.

### Verdict: conditional pass, conditional only on the cheap re-run

---

## 6. A2: Factorial pretraining exposure, item content versus co-occurrence

### Research question

When pretraining improves an LLM recommender, how much of the gain comes from learning what
items are, and how much from learning which items occur together?

### Strongest defensible contribution

An item-level randomized 2x2 factorial over injected pretraining documents, with an
interference-aware assignment-policy estimand and design-based randomization inference at the
item level.

### Closest published work

| Work | Link | Same-design comparison |
|---|---|---|
| *Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?* | <https://arxiv.org/abs/2602.13626> | **Closest on intervention.** Injects a blended corpus (10% in-domain, 60% out-of-domain) into Vicuna-7B by LoRA and measures the clean-versus-contaminated gap in AUC and UAUC. Checked directly: it uses **real** benchmark items, **no** item-level randomization, and it serializes content and interaction signals into the same prompts rather than crossing them as separate factors. |
| *On the Memorization Behavior of LLMs in Generative Recommendation* | <https://arxiv.org/abs/2606.17276> | Studies memorized transitions. Observational, not a randomized exposure design. |
| *Pretraining Exposure Explains Popularity Judgments in LLMs*, SIGIR 2026 | <https://arxiv.org/abs/2605.12382>, <https://doi.org/10.1145/3805712.3809958> | Relates corpus exposure to entity judgments. Observational. |
| Kang et al., *Impact of Co-occurrence on Factual Knowledge of LLMs* | <https://arxiv.org/abs/2310.08256> | The co-occurrence mechanism, applied to factual recall rather than ranking. No content factor. |
| Allen-Zhu and Li, *Physics of Language Models, Part 3.1 / 3.2* | <https://arxiv.org/abs/2309.14316>, <https://arxiv.org/abs/2309.14402> | The canonical controlled synthetic-pretraining methodology A2 transfers. Not applied to recommendation. |
| *DiscRec: Disentangled Semantic-Collaborative Modeling for Generative Recommendation*, and the wider semantic-versus-collaborative line | <https://arxiv.org/abs/2506.15576> | **Owns the question.** A whole subfield already asks how much comes from item meaning versus co-occurrence. It answers architecturally, through dual branches and gating, not causally through randomized exposure. |

### Novelty: 7

Stating precisely what remains different, as the brief requires: no located work randomizes
semantic-content documents and item-co-occurrence documents as **separate crossed factors at the
item level** inside a single pretraining corpus, and no located work defines an assignment-policy
estimand under interference with a recommendation ranking outcome. The design is new.

The *question* is not new. The disentanglement literature owns it. So the novelty is
methodological, and the paper must be framed that way.

### Contribution clarity: 6

The estimand is well specified, and the interference-aware framing is correct: with one shared
model, no-interference assumptions are false, so an assignment-policy contrast averaged over the
random assignment of all other items is the right object. The claim boundary is the weakness. The
result is causal only for the synthetic generator, the base model, the corpus construction, and
the chosen doses. The report's own plan says the real-data replication makes no causal claim. A
reviewer will ask what the synthetic number means for a real catalogue, and the honest answer is
"directionally, nothing more".

### Venue fit: 8

Topics 5 and 7, directly.

### Feasibility: 6

The compute figure (18 to 30 H100-hours) is the smallest part of the cost and is not the
constraint. The pipeline is: build a generator with controlled mutual information between two
signal families, produce paraphrased document families, continue pretrain a 1B model, run
closed-book probes for both families, fine-tune on sequences with held-out target transitions
excluded by construction, run item-level randomization inference, and run the twin-model
dependence check that `Train Once, Answer All` (<https://arxiv.org/abs/2509.23383>) requires
before its single-run shortcut is valid. Each stage can fail silently. Six is generous.

Against the 2026-09-20 deadline this is not deliverable, and the five-page limit cannot hold the
generator description, the estimand, the probes, the factorial, and the dose-response curve.

### Main fatal risk

**Construct validity of the co-occurrence factor.** To make items co-occur in a text corpus you
must write documents that assert the co-occurrence, and such a document is itself content about
the items. The two factors are not cleanly separable in the medium. Reporting the mutual
information between each family and the latent preference is a good control and the plan includes
it, but it does not by itself show that the two families are distinct from each other. If a
reviewer rejects the separation, the entire decomposition, and therefore the entire paper,
collapses.

### Smallest decisive feasibility gate

Run no model at all. Generate the synthetic catalogue and both document families, then compute
three empirical mutual-information quantities: content family against the latent preference,
co-occurrence family against the latent preference, and the two families against each other. If
the third is not small relative to the first two, the factorial separates nothing and the design
is dead. Pure CPU, about one hour, no training.

### Verdict: conditional pass for a later venue; not for 2026-09-20

---

## 7. A3: Controlled validation-data exposure and agentic Type-I error

### Research question

How much can pretraining exposure to hidden validation outcomes inflate the Type-I error of an
LLM agent that selects its own falsification tests?

### Strongest defensible contribution

A controlled, randomized violation of the training-independence assumption in POPPER, with the
size of a statistical test as the measured outcome.

### Closest published work

| Work | Link | Same-design comparison |
|---|---|---|
| Huang et al., *Automated Hypothesis Validation with Agentic Sequential Falsifications* (POPPER), ICML 2025 | <https://arxiv.org/abs/2502.09858> | Supplies the theorem under stress. Does not randomize pretraining exposure. |
| *Investigating Data Contamination for Pre-training Language Models* | <https://arxiv.org/abs/2401.06059> | Contamination against model performance, not test size. |
| *How Much Can We Forget about Data Contamination?* | <https://arxiv.org/abs/2410.03249> | Persistence of contamination, not test size. |
| Bordt et al., *Elephants Never Forget* | <https://arxiv.org/abs/2404.06209> | Tabular memorization. Does not connect it to adaptive statistical validity. |
| Dwork et al., *Generalization in Adaptive Data Analysis and Holdout Reuse* | <https://arxiv.org/abs/1506.02629> | An analyst querying a holdout adaptively, not an analyst pre-exposed to it. |
| Rogers et al., *Max-Information, Differential Privacy, and Post-Selection Hypothesis Testing* | <https://arxiv.org/abs/1604.03924> | The general theory of how much post-selection inflates error. A method dependency. |
| Cuccarese, *Epistemic Blinding* | <https://arxiv.org/abs/2604.06013> | Checked as a possible collision. It is inference-time only, does not randomize exposure, and does not measure Type-I error. Not a collision. |

### Novelty: 7

Stating precisely what remains different: no located work randomizes pretraining exposure to a
hidden validation split and measures the **realized size** of an agentic falsification loop. The
outcome variable is a test's Type-I error, not benchmark accuracy and not a memorization score.
That is a genuine gap.

### Mathematical validity audit

**Verified premise.** POPPER's Assumption 2 was read directly from the paper. It states: "The
training process of the agents is independent of D. Let D_{i-1} := {D_s}_{s<=i-1} be the datasets
used by the agents before iteration i. The e-values obey E[e_i | D_{i-1}] <= 1 under h_i^0."
Theorem 4 then gives P(y = 1) <= alpha under H_0 from Assumptions 1, 2, and 3. A3 describes this
assumption correctly.

**Selection-induced Type-I error: mechanism is sound.** POPPER aggregates e-values
multiplicatively, E_i = product of e_s, and validity rests on the conditional bound
E[e_i | D_{i-1}] <= 1. If the agent chooses the sub-hypothesis using knowledge of the realized
D_i, that conditional bound fails, the supermartingale argument breaks, and Ville's inequality no
longer bounds the rejection probability. The proposed failure route is real.

**Training-data independence: correctly targeted.** The assumption is explicit in the source
theorem and cannot be enforced by interface controls on a pretrained model. A3's framing holds.

**Reachable hypothesis family: correctly required.** With a single fixed test, exposure cannot
inflate size, because there is nothing to select. A3 requires a large declared family of columns,
subgroups, and outcome transformations. That is necessary, and the report states it.

**Design flaw found: the arm effect is aliased with the training run.** The plan says "continue
pretrain or LoRA-adapt one 1B-class model per arm". With three arms and three models, arm and
model are perfectly confounded. Randomizing the 300 dataset instances to arms does not repair
this, because every instance inside an arm is evaluated by the same single model. The
model-training variance component has n = 1 per arm, so any reported interval understates
uncertainty, and the estimated `delta` conflates "exposure" with "this particular training run".
Two repairs exist:

1. Train several seeds per arm and treat the model, not the instance, as the unit of analysis.
   This multiplies compute.
2. Better, and cheaper: borrow A2's own device. Randomize instance-level exposure **inside one
   pretraining corpus and one model**, so that exposed and unexposed instances share the model.
   The contrast then becomes an assignment-policy effect identified by within-model
   randomization, exactly as A2 defines it. This removes the aliasing and cuts training cost.

That A2 and A3 come from the same run, and that only A2 handles the shared-model problem, makes
this omission easy to fix and hard to defend if left in.

**Power is thin.** At 100 instances per arm and alpha = 0.05, the binomial standard error on the
no-exposure arm is about 0.022. Only a large inflation is detectable at conventional power,
roughly 5% rising to 17%. That calculation already ignores the model-level variance in the
previous paragraph, so true power is lower than it looks. Either raise the instance count
substantially or adopt repair 2.

### Contribution clarity: 7

The estimand is stated correctly as a randomized contrast for the experimental training regime,
and the report says plainly that it does not estimate contamination in a hosted model. That
honesty is a strength.

### Venue fit: 6

Topics 2 and 7 admit it. But this is a statistics paper about the size of a test in an agentic
falsification loop. It is not a Big Data paper and it is not about recommendation. It is the
weakest venue match of the four serious candidates, not the strongest.

### Feasibility: 5

More moving parts than A2, all of which must work at once: a null-data generator, several
continued-pretraining runs after the aliasing repair, a memorization probe, a reduced POPPER
loop, a planted-alternative power condition, a non-LLM negative control, and an in-context
positive control. Five is a fair score for technical executability even with adequate GPUs and
time. Against the 2026-09-20 deadline it is not deliverable.

### Main fatal risk

**Instance recognition, not memorization.** To exploit weight-level exposure, the agent must
first identify *which* synthetic instance it is looking at from the discovery split alone, and
then recall that instance's validation outcomes. With independent synthetic null tabular data,
the discovery split carries almost no identifying signal, and a 1B model has little capacity for
noise-level associative retrieval. A closed-book probe that cues the instance explicitly does not
show the agent can do this implicitly while choosing a test. The most likely result is
`delta` near zero. The report calls that outcome a clean falsification, and it is interpretable,
but a null result plus a five-page limit is a hard paper to place.

### Smallest decisive feasibility gate

Skip pretraining entirely. Run the positive control alone: give the agent the validation split
**in context**, then measure the rejection rate under the global null with the full declared
family. If in-context access does not drive the rejection rate well above alpha, the pipeline
cannot express the failure at all, and no amount of weight-level exposure will reveal anything.
This costs a few hours of inference and no training, and it is the single cheapest way to kill or
confirm the direction.

### Verdict: conditional pass for a later venue, after the aliasing repair; not for 2026-09-20

---

## 8. A4: Contamination-immune agentic tests and the price of open-endedness

### Research question

What is the smallest correction that guarantees Type-I error control when an LLM agent may have
memorized the validation set, and what flexibility must be surrendered to get it?

### Strongest defensible contribution

Not the theorem. At most, the empirical cost curve: how the critical value grows and how power
falls as the declared family grows and as dependence inside the family changes, together with
the surface-form experiment that separates language flexibility from statistical reachability.

### Closest published work

| Work | Link | Same-design comparison |
|---|---|---|
| Berk, Brown, Buja, Zhang, Zhao, *Valid Post-Selection Inference*, Annals of Statistics 41(2):802-837, 2013 | <https://projecteuclid.org/journals/annals-of-statistics/volume-41/issue-2/Valid-post-selection-inference/10.1214/12-AOS1077.full> | **Direct collision.** PoSI reduces post-selection inference to simultaneous inference over the reachable family, and states that the result is "universally valid under all possible model selection procedures", explicitly covering informal and post-hoc selection. That is A4's claim, for an arbitrary selector rather than an LLM one. |
| Goeman, Hemerik, Solari, *Only Closed Testing Procedures are Admissible for Controlling False Discovery Proportions*, Annals of Statistics 49(2), 2021 | <https://arxiv.org/abs/1901.04885> | An admissibility result for inference valid under arbitrary post-hoc selection. A necessity-flavoured theorem already exists, and it is sharper than A4's. |
| Westfall and Young maxT; Romano and Wolf step-down, JASA 2005 | <https://www.tandfonline.com/doi/abs/10.1198/016214504000000539> | The max-statistic calibration itself, with dependence handled by resampling. A4 concedes this is not novel. |
| Rogers et al., *Max-Information, DP, and Post-Selection Hypothesis Testing* | <https://arxiv.org/abs/1604.03924> | Quantifies exactly how much adaptive selection can inflate error. |
| POPPER | <https://arxiv.org/abs/2502.09858> | The agentic setting. |
| *Fisher-R1* | <https://arxiv.org/abs/2608.07437> | Trains hypothesis-testing agents. No contamination immunity. |

### Is the necessity result more than a direct `argmax` corollary?

**No.** Here is the whole proof for a constant threshold `c` applied to `T_fhat`, where `fhat` is
chosen by the agent. Universal validity means `P(T_S > c) <= alpha` for every selection rule `S`.
The set of selection rules contains `S = argmax_f T_f`. For that rule,
`T_S = max_f T_f`. Substituting gives `P(max_f T_f > c) <= alpha`, hence
`c >= q_{1-alpha}(max_f T_f)`. That is two lines. The supremum over selection rules is attained
by a member of the family, so nothing has to be proved about the supremum. It is a corollary.

There is a place where a sharper statement lives, and A4 does not make it. Allow the threshold to
depend on the test, a vector `c_f` instead of a constant `c`. Universal validity then reads
`P(exists f : T_f > c_f) <= alpha`, which is exactly strong familywise error control. The set of
admissible threshold vectors is then the whole class of FWER-controlling critical regions. The
single maxT constant is one member of that class, and generally not the best one, because
step-down and closed-testing procedures dominate it. So A4's stated necessity is **trivial read as
a lower bound** and **false read as a uniqueness claim**. Repairing it lands the paper on PoSI
(2013) and on Goeman et al. (2021).

The sufficiency clause is textbook maxT and FWER control. The fresh-data clause is the definition
of pre-registration. All three parts are prior work, in three different places.

### Novelty: 3

### Contribution clarity: 4

The source report concedes that the standard max-statistic correction is not novel and that the
candidate survives only if the universal-agent necessity result carries the paper. It does not
carry it.

### Venue fit: 5

The symposium judges on novelty. A theory paper whose central theorem is a two-line corollary of
a 2013 Annals of Statistics result is a poor match. The empirical cost curve would fit, but that
is a much smaller paper than the one proposed.

### Feasibility: 8

Cheapest of the five: under two GPU-hours plus CPU simulation.

### Main fatal risk

A reviewer who knows PoSI, or Westfall-Young, or closed testing, rejects the central claim in one
sentence. This is the most likely single-sentence rejection among the five candidates.

### Smallest decisive feasibility gate

Write the necessity proof out in full, then read Berk et al. (2013) section 4 and Goeman et al.
(2021). If the proof reduces to the `argmax` substitution above, drop the theoretical framing and
keep only the empirical cost curve as a short measurement paper. One afternoon of reading, no
compute.

### Verdict: reject

---

## 9. Final ranked table

| Rank | ID | Idea | Novelty | Clarity | Venue fit | Feasibility | Total | Ships by 2026-09-20 | Verdict |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | A1 | Repository self-disagreement via provenance replay | 5 | 8 | 7 | 9 | 29 | Yes | Conditional pass |
| 2 | I1 | The Reviews from Tomorrow | 5 | 7 | 8 | 8 | 28 | Probably | Conditional pass |
| 3 | A2 | Factorial pretraining exposure, content versus co-occurrence | 7 | 6 | 8 | 6 | 27 | No | Conditional pass, later venue |
| 4 | A3 | Controlled validation exposure and agentic Type-I error | 7 | 7 | 6 | 5 | 25 | No | Conditional pass, later venue |
| 5 | A4 | Contamination-immune agentic tests | 3 | 4 | 5 | 8 | 20 | Yes | Reject |

This order holds under both readings. It holds under the brief's assumption that GPUs and time
are adequate, because A1 and I1 lead on the unweighted total. It also holds under the verified
2026-09-20 deadline, which removes A2 and A3 from contention for this submission.

### Pursue: A1

A1 is the only candidate that can become a finished, verified, five-page paper by 2026-09-20. Its
claims are checkable, its compute is a laptop, and its remaining work is a re-run rather than a
build. Its novelty is the lowest of the top three, and the paper must be framed honestly as an
attribution method rather than as the discovery that repository variants matter, because
arXiv 2503.09159 already showed that they do.

Do the re-run in section 5 first. Until the replay reproduces the OpenML columns exactly, A1 has
no evidence, only a claim in a markdown file.

### Best recommender-specific alternative: I1

A2 is the better recommender idea on novelty and it is the better long-term project. It cannot
ship in 10 days. I1 can, it targets topic 5 directly, and its gate is one pair of training runs.
If I1 is chosen, cite TransNets in the first paragraph and scope the contribution to the residual
non-target future text. Do not present target-review leakage as a finding.

### Evidence that would reverse the top choice

| Code | Evidence | Effect |
|---|---|---|
| R1 | The A1 replay fails to reproduce the OpenML `did=179` columns exactly, or the forest-versus-tree flip does not survive a fresh 10-fold paired run | A1 falls out; I1 takes first place |
| R2 | A published paper is found that reconstructs repository entries from a canonical source and runs a factorial preprocessing intervention | A1's novelty drops to 3; I1 takes first place |
| R3 | I1's gate shows a clear residual effect after target-review removal, **and** a credible public implementation or documented protocol is found that aggregates post-cutoff text | I1 ties or passes A1 |
| R4 | The submission target moves to a venue six or more weeks out | A2 takes first place on novelty and venue fit; A3 becomes viable after the aliasing repair |
| R5 | The A3 in-context positive control shows large inflation, **and** a probe shows a 1B model can identify a synthetic instance from its discovery split alone | A3's main risk clears; it moves ahead of A2 for a later venue |

---

## 10. Comparison with `idea-stage/FIVE_WAY_COMPARISON_20260910.md`

Read after the scores above were fixed.

### Agreements

1. **A4 ranks last, for the same reason.** Both reviews identify the `argmax` corollary problem.
   This review goes further and names the specific prior work: PoSI (Berk et al. 2013) and the
   closed-testing admissibility result (Goeman et al. 2021). It also shows the necessity claim is
   false when read as a uniqueness statement.
2. **A1's pilot is not reproducible.** The existing comparison already records that no
   `replay_effect.py`, `rank_flip.py`, `check_n1.py`, or `check_n3.py` was found. This independent
   search confirms it, and adds that no input or output artifact is cached either. Recording this
   honestly was the right call.
3. **A2's shared-model interference is a genuine risk**, and both reviews name it.
4. **A3's main risk is that weight-level exposure may not steer test selection.** Both reviews
   name it. Both propose an in-context positive control as part of the gate.
5. **Novelty scores stay provisional** because OpenAlex and Semantic Scholar coverage is partial.
   Both reviews reach the same conclusion, from independent failures on the same two services.

### Disagreements

1. **The ranking.** The existing file ranks A3, A2, A1, I1, A4. This review ranks A1, I1, A2, A3,
   A4. The difference does not come from the deadline. It survives the brief's own assumption that
   GPUs and time are adequate.
2. **A3's venue fit of 9 is too high.** A paper about the realized size of a statistical test
   inside an agentic falsification loop maps to topics 2 and 7, but it is not a Big Data paper and
   it is not about recommendation. This review gives 6.
3. **A3 has an identification flaw the existing comparison does not name.** Training one model per
   arm aliases the arm effect with the training run. Instance-level randomization does not repair
   it, because all instances in an arm share one model. This is the single most important
   technical defect in A3, and the fix, within-model randomized exposure, is already used by A2 in
   the same source document.
4. **I1's novelty of 7 is too high, and the closest-work table has a gap.** TransNets
   (<https://arxiv.org/abs/1704.02298>, RecSys 2017) published the target-review leakage result
   nine years ago. Neither `IDEA_REPORT.md` nor the comparison file cites it. This review gives 5.
5. **A1's novelty of 6 to 7 is too high.** Tschalzev et al.
   (<https://arxiv.org/abs/2503.09159>, March 2025) already documents OpenML version confusion,
   preprocessing-driven performance change, and reversal of which model wins. Neither source
   document cites it. This review gives 5.
6. **A1's `Evidence` score of 4 out of 10 is too generous.** The evidence is prose in a markdown
   file plus a dead session-artifact link, with no scripts and no cached data. That is a 1 or a 2.
   A1 still ranks first here, because the gap closes in under an hour of work.
7. **The existing decision sets aside the deadline by assumption.** The comparison states that
   compute cost is not a reason to reject. Fair. But the verified submission deadline is
   2026-09-20, which is 10 days away, and neither A2 nor A3 can produce a complete five-page paper
   in that window. A ranking that drives the next 10 days must say so.

### Minor accuracy note

Three entries in the comparison file's "Primary sources" list carry paraphrased titles that do
not match the live arXiv records:

| arXiv ID | Title in the comparison file | Actual title |
|---|---|---|
| 2512.12978 | "Do Reviews Matter: Evaluating the Role of Reviews in Recommender Systems" | "Do Reviews Matter for Recommendations in the Era of Large Language Models?" |
| 2602.13626 | "The Benchmark Leakage Trap: A Unified Contamination Framework for LLM-Based Recommendation" | "Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?" |
| 2509.23383 | "Train Once, Answer All: Universal Pretraining Data Influence Estimation" | "Train Once, Answer All: Many Pretraining Experiments for the Cost of One" |

All three identifiers resolve and all three papers are the intended ones. Fix the titles before
any of them reaches a bibliography.

---

## 11. Status of claims in this document

- No experiment was run. No pilot was run. No experimental result is claimed.
- Every number in section 3 and section 9 is a bounded literature and design judgment.
- Every novelty score is provisional, because OpenAlex returned HTTP 429 on 2 of 5 queries and
  Semantic Scholar returned HTTP 429 or timed out on 4 of 5 queries.
- Every arXiv identifier cited here was verified against the live arXiv API on 2026-09-10.
- The A1 pilot numbers in `idea-stage/alternative-run-20260910/idea.md` are recorded here as
  unverified claims, because the scripts and artifacts needed to reproduce them are not present.
