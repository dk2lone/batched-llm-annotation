# Five-idea novelty recheck

Date: 2026-09-10
Scope: A1, A2, A3, I1, and A4
Sources: arXiv, OpenAlex, and Semantic Scholar

## Result

The source-coverage limitation in the earlier five-way comparison is resolved. Every idea now
has at least three successful concept searches covered by all three required sources. Some
Semantic Scholar responses are stored in paced retry files beside the matching arXiv and OpenAlex
results.
This makes the verdicts below coverage-complete under the project’s idea-generation protocol.
It does not prove that no unindexed or later paper exists.

| Rank | ID | Short title | Novelty | Contribution clarity | Venue fit | Novelty verdict |
|---:|---|---|---:|---:|---:|---|
| 1 | A2 | Content versus co-occurrence exposure | 7/10 | 8/10 | 9/10 | Novel experimental design located; pursue |
| 2 | A3 | Validation exposure and Type-I error | 7/10 | 7/10 | 7/10 | Novel endpoint and intervention located; repair design first |
| 3 | A1 | Repository provenance replay | 5/10 | 8/10 | 7/10 | Broad finding is not novel; narrowed attribution method may remain |
| 4 | I1 | Reviews from Tomorrow | 5/10 | 7/10 | 8/10 | Broad leakage claim is not novel; residual non-target future text remains incremental |
| 5 | A4 | Contamination-immune agentic tests | 3/10 | 4/10 | 5/10 | Not novel; reject |

The recommended research direction is A2. A3 has similar novelty, but its current one-model-per-
arm plan aliases the exposure effect with model-training variation. A2 has the cleaner identified
experiment and the strongest direct fit to IEEE BigData topic 5.

## Search completeness

The query definitions are in `QUERY_PLAN.tsv`; raw results are the JSONL files in this directory.
Keys were read from environment variables and were not written to the query plan, logs, or results.

| ID | Successful three-source rounds used | Unique titles after cross-round deduplication | Recovery notes |
|---|---|---:|---|
| A1 | R2, R3, replacement R4 | 98 | R2 Semantic Scholar was recovered; throttled R1 was replaced by a full three-source R4 |
| A2 | R1, replacement R4, replacement R5 | 129 | R1 Semantic Scholar was recovered; throttled or empty earlier rounds were replaced by full three-source R4 and R5 |
| A3 | R1, R2, R3 | 141 | No recovery needed |
| I1 | R1, R2, R3 | 145 | R1 Semantic Scholar was recovered with a source-only retry |
| A4 | R1, R2, R3 | 175 | Missing Semantic Scholar calls were recovered; R1 used a shorter equivalent query |

Several first-pass Semantic Scholar calls returned HTTP 429. Those failures were retained as
failures, and none was interpreted as zero results. Replacement queries were run when a failed or
empty round did not provide useful Semantic Scholar evidence. Exact-title supplement files were
also saved for each decisive closest paper.

## A2: content versus co-occurrence exposure

**Verdict: novel, 7/10.** The literature owns each neighboring component, but no retrieved paper
crosses semantic-content exposure and item-co-occurrence exposure as separately randomized
pretraining factors and estimates their effects on recommendation within one controlled catalogue.

Closest work:

- [Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?](https://arxiv.org/abs/2602.13626)
  simulates contaminated continued pretraining and measures recommendation inflation. It blends
  content and interaction evidence rather than randomizing them as crossed factors.
- [On the Memorization Behavior of LLMs in Generative Recommendation](https://arxiv.org/abs/2606.17276)
  shows that much of the LLM gain can come from memorized one-hop item transitions and separately
  discusses collaborative and semantic relations. It is observational with respect to exposure.
- *Content-Based Collaborative Generation for Recommender Systems* and related hybrid models ask
  how to combine content and collaborative signals architecturally. They do not identify causal
  effects of pretraining exposure.

Smallest defensible claim: a randomized factorial decomposition of recommendation gains caused by
pretraining exposure to item meaning versus item co-occurrence. Do not claim that semantic versus
collaborative recommendation is itself a new question.

## A3: validation exposure and agentic Type-I error

**Verdict: novel, 7/10, after a design repair.** The retrieved adaptive-data-analysis literature
shows that data-dependent hypothesis selection can invalidate inference, and POPPER supplies the
agentic testing framework. No retrieved paper randomizes a model’s training exposure to hidden
validation outcomes and measures the resulting realized Type-I error of an agent-selected test.

Closest work:

- [Automated Hypothesis Validation with Agentic Sequential Falsifications](https://arxiv.org/abs/2502.09858)
  introduces POPPER and claims strict Type-I error control for its sequential framework. A3 tests
  the consequence of violating its training-data independence condition.
- [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629)
  and [The Generic Holdout](https://arxiv.org/abs/1809.05596) establish the broader danger of
  adaptively reusing validation data. They do not study weight-level exposure in an LLM agent.

Required repair: randomize instance-level exposure within a shared model, or train multiple model
seeds per exposure arm and treat the model as the unit of analysis. The original one-model-per-arm
design cannot identify exposure separately from training-run variation.

Smallest defensible claim: a controlled measurement of how pretrained access to validation
outcomes changes the realized size of an agentic falsification test. Do not claim that adaptive
holdout reuse or selection-induced false positives are new.

## A1: repository provenance replay

**Verdict: not novel as framed, 5/10.** The broad headline that repository variants and
preprocessing choices can change performance and reverse the winning model already exists.

Closest work:

- [Unreflected Use of Tabular Data Repositories Can Undermine Research Quality](https://arxiv.org/abs/2503.09159)
  documents problematic OpenML dataset use, including inappropriate preprocessing. The paper also
  contains model-comparison reversals discussed in the earlier independent audit.
- [A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data](https://arxiv.org/abs/2407.02112)
  studies dataset-specific preprocessing effects on tabular-model comparisons.
- [Benchmark Data Repositories for Better Benchmarking](https://arxiv.org/abs/2410.24100)
  covers repository design and reproducibility.

Possible remaining contribution: checksum-anchored reconstruction of repository entries from one
canonical source, followed by a per-transformation causal attribution of the disagreement. That is
a method extension, not the discovery that repository variants change model rankings. It also
still lacks reproducible scripts or artifacts in this workspace.

## I1: Reviews from Tomorrow

**Verdict: not novel as broadly framed, 5/10.** Target-review leakage is a direct collision, and it
is likely the largest component of the proposed full-versus-past-only contrast.

Closest work:

- [TransNets: Learning to Transform for Recommendation](https://arxiv.org/abs/1704.02298) shows
  that much of review text’s predictive value comes from the target user’s review of the target
  item and designs a model for the case where that review is unavailable.
- [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060)
  studies temporal leakage in offline recommendation and its effect on model comparisons.
- [Do Reviews Matter for Recommendations in the Era of Large Language Models?](https://arxiv.org/abs/2512.12978)
  removes or changes review evidence, but does not isolate a historical text cutoff.

Possible remaining contribution: after deleting every held-out target review, isolate later
reviews by the same user and later reviews about the same item while holding the recommendation
task fixed. This is a clean residual audit, but it is incremental unless the residual effect is
large and a credible published protocol actually uses post-cutoff text.

## A4: contamination-immune agentic tests

**Verdict: not novel, 3/10.** The proposed guarantee is established post-selection and
multiple-testing theory applied to an LLM selector.

Closest work:

- [Valid Post-Selection Inference](https://arxiv.org/abs/1306.1059) reduces arbitrary
  model-selection inference to simultaneous inference and provides validity under all selection
  procedures.
- [Only Closed Testing Procedures are Admissible for Controlling False Discovery Proportions](https://arxiv.org/abs/1901.04885)
  gives a stronger admissibility result for a broad class that includes family-wise error control.
- [Max-Information, Differential Privacy, and Post-Selection Hypothesis Testing](https://arxiv.org/abs/1604.03924)
  quantifies valid correction after adaptive selection.

The constant-threshold necessity result follows by allowing the selector to choose the maximum
statistic. Max-statistic calibration is sufficient but not uniquely optimal because step-down and
closed-testing methods also exist. Rebranding the selector as an LLM does not create a new theorem.

## Boundary of the conclusion

These are literature judgments through 2026-09-10, not experimental results. “Novel” here means
that the completed searches located no direct collision with the specific contribution, while
the closest work leaves a concrete difference. A2 and A3 still need their feasibility gates before
either can support a submission claim.
