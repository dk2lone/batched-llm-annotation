# Five-way idea comparison

Date: 2026-09-10

This report compares the current recommender idea with the four candidates preserved from
the independent idea-generation run. The ranking assumes that GPU access and project time
are available, so compute cost is a feasibility concern but not a reason to reject an idea.
No experiment or pilot was run for this comparison.

## Decision

Choose **A3: Controlled validation-data exposure and agentic Type-I error** for the next
feasibility gate. It has the strongest combination of a causal intervention, a meaningful
statistical endpoint, and a direct challenge to an assumption used by an existing agentic
testing method.

Choose **A2: Factorial pretraining exposure** instead if the paper must remain directly about
recommender systems. A2 has a comparably strong experiment, but its shared-model interference
and synthetic-catalogue design make interpretation harder.

## Ranking

Scores are comparative planning judgments from 1 to 10. `Evidence` measures how ready the
idea is to implement from material currently preserved in this workspace. Novelty remains
provisional.

| Rank | ID | Short title | Novelty | Depth | Feasibility with GPU | Venue fit | Evidence | Decision |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | A3 | Validation exposure and Type-I error | 7 | 9 | 7 | 9 | 3 | Recommended gate |
| 2 | A2 | Content versus co-occurrence exposure | 7 | 8 | 7 | 9 | 3 | Recommender-focused alternative |
| 3 | A1 | Repository provenance replay | 6–7 | 7 | 8 | 8 | 4 | Evidence-first fallback |
| 4 | I1 | Reviews from Tomorrow | 7 | 7 | 9 | 9 | 3 | Secondary fallback |
| 5 | A4 | Contamination-immune agentic tests | 5–6 | 7 | 8 | 8 | 3 | Deprioritize without a sharper theorem |

The ranks do not imply that A3 or A2 is experimentally proven. They prioritize potential
contribution now that compute and elapsed time are not binding constraints.

## 1. A3: Controlled validation-data exposure and agentic Type-I error

**Question:** How much can pretraining exposure to hidden validation outcomes inflate the
Type-I error of an LLM agent that selects its own falsification tests?

**Why it ranks first:** Dataset instances can be randomized to full exposure,
discovery-only exposure, or no exposure. The endpoint is realized false rejection under a
known global null, rather than a memory score or benchmark-accuracy change. This directly
tests the consequence of violating the training-data independence assumption in POPPER.

**Closest overlap:** POPPER supplies the agentic testing framework and its independence
condition. Data-contamination and tabular-memorization papers establish that models can retain
training examples, but the located work does not randomize exposure and measure downstream
Type-I error.

**Decisive gate:** Before building the complete agent loop, expose a small open model to
validation instances and verify both of the following:

1. A memorization probe succeeds at the planned exposure dose.
2. An in-context positive-control agent can exploit a sufficiently wide reachable hypothesis
   family to inflate false rejections.

**Fatal risk:** Weight-level exposure may not survive strongly enough to steer test selection.
If the memorization probe succeeds but the full-exposure arm remains at nominal Type-I error
while the in-context positive control inflates, the proposed causal result fails.

## 2. A2: Factorial pretraining exposure to item content and co-occurrence

**Question:** When pretraining improves an LLM recommender, how much of the gain comes from
learning item meaning, and how much comes from learning which items occur together?

**Why it ranks second:** The 2 x 2 randomized exposure design separates two mechanisms that
observational recommendation benchmarks blend together. Synthetic identifiers prevent prior
knowledge of the catalogue, and exposure dose can be randomized within each cell.

**Closest overlap:** Recent work studies benchmark leakage and memorized transitions in
generative recommendation. Controlled-pretraining and factual-knowledge studies manipulate
training exposure in other settings. The located work does not independently randomize item
content and co-occurrence exposure and estimate their effects on recommendation.

**Decisive gate:** Train a small model on a reduced synthetic catalogue and require a monotonic
memorization response to exposure dose before scaling the factorial study.

**Fatal risk:** All items share one model, so treatments can interfere. Fine-tuning may also
relearn the co-occurrence structure and erase the intended contrast. The claims must remain
limited to the generator, model, corpus construction, and exposure doses used.

## 3. A1: Repository self-disagreement through provenance replay

**Question:** Can different versions or preprocessing paths for the nominally same public
dataset reverse model rankings?

**Why it ranks third:** It is concrete, easy to explain, and can produce an immediately useful
provenance audit. Its source report describes a rank flip under transformations of the OpenML
Adult dataset.

**Closest overlap:** Data-centric benchmark studies already show that preprocessing choices
change measured performance and model rankings. Benchmark-repository work also treats
dataset documentation and reproducibility. A1 needs a stronger contribution than another
preprocessing-sensitivity demonstration, such as a same-lineage replay method that attributes
the disagreement to exact transformations.

**Decisive gate:** Rebuild the claimed Adult replay from raw sources, commit the transformation
graph and environment lock, and reproduce the rank flip before expanding to other datasets.

**Fatal risk:** The archived run does not contain the referenced `scratchpad/` scripts or their
input/output artifacts. The claimed preliminary result is therefore a lead, not reproduced
evidence in this workspace.

## 4. I1: The Reviews from Tomorrow

**Question:** How much of a review-aware recommender's reported accuracy comes from review text
written after the historical prediction date?

**Why it remains viable:** The intervention is clean: freeze interactions and predictions,
then vary only which review text is available. It is strongly aligned with recommender-system
evaluation and needs less infrastructure than A2 or A3.

**Closest overlap:** Prior work covers temporal leakage in offline recommender evaluation,
review removal and distortion, and review-aware recommendation broadly. The proposed
difference is the isolated past-only versus post-cutoff text contrast plus decomposition by
target review, later user reviews, and later item reviews.

**Decisive gate:** Find a credible affected implementation or protocol, demonstrate a
past-only versus post-cutoff-text difference, and show that the effect survives removal of
held-out target reviews.

**Fatal risk:** Without a credible affected protocol, reviewers can dismiss the setup as a
deliberately invalid pipeline rather than an observed evaluation problem.

## 5. A4: Contamination-immune agentic tests

**Question:** What correction guarantees Type-I error control when an agent may know the
validation set, and how much scientific flexibility must be surrendered?

**Why it ranks fifth:** The proposed answer uses a declared test family and the null quantile
of its maximum statistic, or fresh data collected after the test is committed. This is elegant
and cheap to demonstrate, but max-statistic correction and adaptive-holdout protection are
mature ideas.

**Closest overlap:** Adaptive-data-analysis work already formalizes safe holdout reuse, and
multiple-testing theory already supplies family-wide maximum-statistic control. The candidate
depends on making the universal memorizing-agent necessity result more than a direct
`argmax` corollary.

**Decisive gate:** Obtain an independent theorem audit showing a nontrivial result beyond the
standard maximum-statistic argument, with a clear empirical quantity that matters for an
agentic discovery system.

**Fatal risk:** If the proof remains a one-line reduction to an adversarial selector, the work
is an application note rather than a full research contribution.

## Literature-check status

The current comparison ran three query formulations per idea. arXiv returned results for all
15 rounds. OpenAlex returned results for 10 of 15 rounds; the remaining requests received HTTP
429 responses. Semantic Scholar coverage was partial. The preserved independent run contains
additional searches, but this pass did not independently reproduce all of them.

Under the idea-generation review rule, the incomplete source coverage means none of the five
novelty values is a definitive `NOVEL` verdict. The scores are suitable for choosing the next
gate, not for a paper's novelty claim.

## Primary sources used for the comparison

- [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060)
- [Do Reviews Matter: Evaluating the Role of Reviews in Recommender Systems](https://arxiv.org/abs/2512.12978)
- [Review-based Recommender Systems: A Survey of Approaches, Challenges and Future Perspectives](https://arxiv.org/abs/2405.05562)
- [A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data](https://arxiv.org/abs/2407.02112)
- [Benchmark Data Repositories for Better Benchmarking](https://arxiv.org/abs/2410.24100)
- [The Benchmark Leakage Trap: A Unified Contamination Framework for LLM-Based Recommendation](https://arxiv.org/abs/2602.13626)
- [On the Memorization Behavior of Large Language Models in Generative Recommendation](https://arxiv.org/abs/2606.17276)
- [Train Once, Answer All: Universal Pretraining Data Influence Estimation](https://arxiv.org/abs/2509.23383)
- [The Impact of Co-occurrence on Factual Knowledge of Large Language Models](https://arxiv.org/abs/2310.08256)
- [Automated Hypothesis Validation with Agentic Sequential Falsifications](https://arxiv.org/abs/2502.09858)
- [Investigating Data Contamination for Pre-training Language Models](https://arxiv.org/abs/2401.06059)
- [How Much Can We Forget about Data Contamination?](https://arxiv.org/abs/2410.03249)
- [Elephants Never Forget: Memorization and Learning of Tabular Data in Large Language Models](https://arxiv.org/abs/2404.06209)
- [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629)
- [The Generic Holdout](https://arxiv.org/abs/1809.05596)

## Preserved evidence

- The archived source run is under `idea-stage/alternative-run-20260910/`.
- Current search outputs are under `idea-stage/five-way-comparison-20260910/searches/`.
- No matching `replay_effect.py`, `rank_flip.py`, `check_n1.py`, or `check_n3.py` file was
  found in the project, temporary source directory, or broader user directory during recovery.
