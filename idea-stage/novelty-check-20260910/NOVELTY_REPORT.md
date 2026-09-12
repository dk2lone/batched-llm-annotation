# Novelty-first audit of seven IEEE BigData candidates

Date: 2026-09-10  
Scope: K, A, T, A2, A3, A1, and I1  
Priority: novelty and contribution first; feasibility and venue fit were not used to determine the ranking

## Result

K has the strongest literature position. The source paper establishes the dimensionality failure
and explicitly leaves its mechanism open; no retrieved paper tests whether representation-space
distance concentration causally explains that failure in generic LLM tabular prediction.

A and A2 remain defensible, but for narrow reasons. A's potentially novel result is batching-induced
dependence across items, not cluster-robust inference itself. A2's potentially novel result is the
randomized causal decomposition, not the distinction between semantic and collaborative signals.

| Rank | ID | Exact contribution being scored | Novelty | Contribution strength | Recommendation |
|---:|---|---|---:|---:|---|
| 1 | K | Show that representation-space distance concentration, rather than dimension alone, causes generic LLM tabular prediction to collapse | **7.5/10** | **8.5/10** | **PROCEED** |
| 2 | A | Measure batching-induced across-item error correlation and its effect on downstream coverage | **7.0/10** | **8.5/10** | **PROCEED WITH CAUTION** |
| 3 | A2 | Randomly decompose pretraining gains into content exposure, co-occurrence exposure, and their interaction | **7.0/10** | **8.0/10** | **PROCEED** |
| 4 | A3 | Randomize exposure to hidden validation outcomes and measure realized Type-I error of an LLM-selected test | **6.5/10** | **7.5/10** | **REPAIR, THEN PROCEED WITH CAUTION** |
| 5 | A1 | Reconstruct repository lineages exactly and attribute a model-rank reversal to individual transformations | **5.0/10** | **6.0/10** | **NARROWED BACKUP** |
| 6 | I1 | Isolate leakage from post-cutoff non-target reviews after removing every target review | **5.0/10** | **6.0/10** | **NARROWED BACKUP** |
| 7 | T | Explain the tabular likelihood pathology using within-row redundancy distinct from feature correlation | **4.0/10** | **4.5/10** | **ABANDON CURRENT FORM** |

The novelty score estimates how much of the exact claim remains unclaimed. The contribution score
estimates how much the result would change what researchers believe or do if the experiment supports
it. Neither score estimates acceptance probability.

## Search coverage

The structured pass used 21 claim groups: three claims per idea and three query formulations per
claim. That produced 63 searches against arXiv, OpenAlex, and Semantic Scholar, restricted to 2024
through 2026 for the new pass. A separate seven-query sweep targeted ICLR, ICML, and NeurIPS work
from 2025 and 2026. Older foundational papers identified in the completed project audits were also
checked where they directly constrain a claim.

- 53 of the 63 structured searches returned all three sources on the first attempt.
- Ten first-pass searches were missing Semantic Scholar because of HTTP 429 responses.
- Eight were recovered with paced source-only retries.
- Two exact formulations remained unavailable after two retries. Both affected claims still have
  successful Semantic Scholar results for their other formulations, plus complete arXiv and
  OpenAlex results for all three formulations.
- All seven targeted venue sweeps returned successfully.

The query plans, raw JSONL results, failure logs, and recovery status are stored beside this report.
No failed request was interpreted as an empty result.

The novelty-check skill normally requests a fresh secondary Codex reviewer. Daniel explicitly asked
for this audit to be completed by the primary Codex session, so that phase was intentionally skipped.
No independence claim is made for the judgment below.

## K: distance concentration and generic LLM tabular failure

### Proposed method

Repeat the published dimensionality sweep with open-weight generic LLMs, measure neighborhood
contrast in the model's row representations, and intervene by removing nearest rows and by creating
high-dimensional datasets with deliberately low distance concentration.

### Core claims

1. Generic LLM tabular prediction behaves like local retrieval, and this behavior degrades with
   dimension. **Claim novelty: MEDIUM.** *Why Large Language Models Fail at Tabular Prediction*
   already reports strong behavioral agreement with distance-based methods, while recent tabular
   foundation-model work explicitly proposes a retrieval view.
2. Representation-space distance concentration predicts the accuracy collapse better than dimension
   itself. **Claim novelty: HIGH.** The concentration mechanism is established for vector retrieval,
   but no retrieved work connects it to the published generic-LLM tabular failure.
3. Low-concentration high-dimensional controls and nearest-row ablations causally distinguish the
   mechanism from a generic dimensionality effect. **Claim novelty: HIGH.** Retrieved tabular
   retrieval systems select useful examples but do not run this mechanism-separating intervention.

### Closest prior work

| Paper | Year | Overlap | Smallest defensible difference |
|---|---:|---|---|
| [Why Large Language Models Fail at Tabular Prediction](https://arxiv.org/abs/2608.02412) | 2026 | Establishes the dimensionality failure and distance-based behavioral agreement | Explicitly leaves the internal mechanism open |
| [Understanding the Surprising Generalization Properties of Tabular Foundation Models](https://arxiv.org/abs/2608.17957) | 2026 | Proposes that tabular in-context learning is retrieval-based | Studies specialized TFMs and does not connect concentration to generic-LLM failure |
| [High-Dimensional Concentration and Retrieval Instability in Embedding Spaces](https://arxiv.org/abs/2606.28330) | 2026 | Establishes concentration, contrast collapse, and unstable retrieval | Applies the mechanism to vector retrieval and RAG, not tabular prediction |
| [Scalable In-Context Learning on Tabular Data via Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2502.03147) | 2025 | Uses learned retrieval to select tabular demonstrations | Builds a retrieval-assisted model rather than explaining why an unmodified generic LLM fails |
| [ARASH: Adaptive Retrieval And Shot Selection for Tabular Prediction](https://arxiv.org/abs/2608.17856) | 2026 | Uses local neighborhoods to select tabular shots | Optimizes TabPFN efficiency; it does not test the proposed concentration mechanism |

### Assessment

- **Score:** 7.5/10
- **Recommendation:** PROCEED
- **Key differentiator:** the low-concentration/high-dimension intervention turns an open mechanism
  question into a falsifiable causal contrast.
- **Risk:** retrieval and concentration are both established lenses. A paper based only on attention
  correlation would look like a combination of known ideas. Nearest-row ablation and the decisive
  synthetic control must be primary evidence.
- **Suggested positioning:** "Distance concentration, not dimension alone, explains the observed
  collapse of generic LLM tabular prediction." Do not claim that retrieval-based tabular learning or
  high-dimensional concentration is new.

## A: batched LLM annotation and invalid independence

### Proposed method

Randomly assign the same documents to shared-context batches of different sizes, estimate
within-batch error correlation, separate co-batching from context length and position, and measure
the coverage of downstream estimators with and without batch-clustered uncertainty.

### Core claims

1. Shared-context annotation induces correlated errors across otherwise separate items.
   **Claim novelty: HIGH.** Batch-prompting work measures mean accuracy, utility, attacks, or refusal
   behavior; no retrieved paper estimates benign across-item annotation-error covariance.
2. This elective dependence causes i.i.d. confidence intervals for analyses using LLM labels to
   under-cover. **Claim novelty: MEDIUM.** The mathematical consequence of clustered errors is
   standard, and adjacent work already corrects inference with predicted labels under other forms of
   dependence.
3. Random batch assignment plus a length-matched filler condition identifies co-item influence
   separately from long-context degradation. **Claim novelty: HIGH.** Existing batch-interference
   studies do not use this design to estimate an annotation design effect.

### Closest prior work

| Paper | Year | Overlap | Smallest defensible difference |
|---|---:|---|---|
| [Batch Prompting: Efficient Inference with Large Language Model APIs](https://arxiv.org/abs/2301.08721) | 2023 | Same shared-prompt intervention | Measures efficiency and mean task performance, not across-item error covariance |
| [Efficient but Vulnerable: Benchmarking and Defending LLM Batch Prompting Attack](https://arxiv.org/abs/2503.15551) | 2025 | Demonstrates cross-query interference | Uses adversarial instructions and does not study statistical inference |
| [Safety in Batches?](https://arxiv.org/abs/2608.02681) | 2026 | Shows a batch effect not reducible to ordinary long-context effects | Studies harmful-answer refusal, not label covariance or interval coverage |
| [Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking](https://openalex.org/W7154539897) | 2026 | Shows omitted LLM-pipeline variance causes under-coverage | Varies model, temperature, and prompt between runs; it does not model items sharing one annotation call |
| [Spatially Robust Inference with Predicted and Missing at Random Labels](https://www.semanticscholar.org/paper/5a9495212beecd2c361c34f65133e8c6c5440a10) | 2026 | Handles dependence in inference with predicted labels | Dependence comes from spatial sampling, not an elective batching decision |
| [Nine Judges, Two Effective Votes](https://arxiv.org/abs/2605.29800) | 2026 | Uses correlated errors and effective sample size for LLM evaluation | Correlation is across models for one item, not across items in one prompt |

### Assessment

- **Score:** 7.0/10
- **Recommendation:** PROCEED WITH CAUTION
- **Key differentiator:** the newly measured object is within-run, across-item covariance created by
  the annotation call itself.
- **Risk:** the inferential correction is standard and under-coverage from LLM pipeline uncertainty
  already exists. If the filler control explains the effect, the result collapses to known long-context
  degradation.
- **Suggested positioning:** present batching-induced dependence as the discovery and the design
  effect as its practical consequence. Do not present cluster-robust standard errors as a new method.

## A2: factorial pretraining exposure for recommendation

### Proposed method

Use a synthetic catalogue and a randomized 2 x 2 continued-pretraining design that independently
exposes a model to item content and item co-occurrence documents, then estimate both main effects and
their interaction on held-out recommendation.

### Core claims

1. Content exposure and co-occurrence exposure can be independently randomized within one catalogue.
   **Claim novelty: HIGH.** No retrieved recommender study uses this crossed exposure design.
2. The design identifies separate causal contributions and an interaction, rather than observing a
   correlation between memorization and performance. **Claim novelty: HIGH.** Existing contamination
   studies blend exposure sources or probe fixed pretrained models.
3. Closed-book content and co-occurrence probes can connect acquired knowledge to recommendation
   gains. **Claim novelty: MEDIUM.** Content/collaborative decomposition and memorization probes are
   established separately.

### Closest prior work

| Paper | Year | Overlap | Smallest defensible difference |
|---|---:|---|---|
| [Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?](https://arxiv.org/abs/2602.13626) | 2026 | Continues pretraining on leaked recommendation data and measures inflation | Blends content and interaction evidence instead of crossing them as randomized factors |
| [Do LLMs Memorize Recommendation Datasets?](https://arxiv.org/abs/2505.10212) | 2025 | Probes item attributes, profiles, interactions, and their relation to recommendation | Observes existing memorization in fixed models; it does not manipulate exposure |
| [On the Memorization Behavior of LLMs in Generative Recommendation](https://arxiv.org/abs/2606.17276) | 2026 | Separates one-hop memorization from richer collaborative and semantic relations | Studies learned behavior and proposes training strategies, not a causal exposure decomposition |
| [Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals](https://arxiv.org/abs/2509.10468) | 2025 | Decomposes pretrained semantic and collaborative representations | Architectural decomposition rather than randomized pretraining exposure |

### Assessment

- **Score:** 7.0/10
- **Recommendation:** PROCEED
- **Key differentiator:** a randomized factorial estimate of where a pretraining benefit comes from.
- **Risk:** reviewers can reduce the story to the already-known semantic-versus-collaborative split
  unless the exposure intervention, interaction estimate, and held-out transitions are central.
- **Suggested positioning:** "A causal decomposition of recommendation gains from pretraining
  exposure." Do not claim that LLM recommenders memorize data or use semantic and collaborative
  signals as new findings.

## A3: validation exposure and agentic Type-I error

### Proposed method

Under many generated global-null datasets, randomize whether an LLM has training-time access to the
hidden validation outcomes, let it select tests from a fixed reachable family, and measure the
realized rejection rate.

### Core claims

1. Weight-level exposure to the exact validation outcomes increases realized Type-I error for an
   LLM-selected test. **Claim novelty: MEDIUM-HIGH.** No retrieved paper runs that exact intervention,
   but the direction follows established adaptive-holdout theory.
2. A global-null generator and fixed reachable test family distinguish false discovery from valid
   power. **Claim novelty: MEDIUM.** These controls are necessary applications of existing testing
   principles, not new statistical machinery.
3. Instance-level randomized exposure within one model, or multiple independently trained models per
   arm, identifies exposure separately from training-run variation. **Claim novelty: HIGH for this
   setting.** The current design has not yet selected and demonstrated a valid implementation.

### Closest prior work

| Paper | Year | Overlap | Smallest defensible difference |
|---|---:|---|---|
| [Automated Hypothesis Validation with Agentic Sequential Falsifications](https://arxiv.org/abs/2502.09858) | 2025 | Uses LLM agents and claims strict sequential Type-I control | Does not randomize prior model exposure to the hidden outcomes |
| [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629) | 2015 | Establishes that adaptive access can overfit a holdout | Studies interactive access, not information memorized in model weights |
| [The Generic Holdout](https://arxiv.org/abs/1809.05596) | 2018 | Prevents adaptive false discoveries by limiting holdout exposure | Provides the general theory but does not test LLM agents or weight-level exposure |
| [A Taxonomy for Data Contamination in Large Language Models](https://arxiv.org/abs/2407.08716) | 2024 | Organizes training-data contamination risks | Does not measure statistical test size under a controlled global null |

### Assessment

- **Score:** 6.5/10
- **Recommendation:** REPAIR, THEN PROCEED WITH CAUTION
- **Key differentiator:** training exposure is moved into the model before the agentic analysis and is
  manipulated as an experimental treatment.
- **Risk:** a reviewer can call the result a direct instance of adaptive holdout reuse. More seriously,
  one model per exposure arm confounds exposure with training variation and cannot support the causal
  claim.
- **Suggested positioning:** a controlled empirical stress test of an independence assumption in
  agentic hypothesis validation. Do not claim that adaptive test selection causes false positives as
  a new fact.

## A1: repository provenance replay

### Proposed method

Use checksums and a transformation graph to reproduce multiple repository versions of the same
canonical dataset, then attribute changes in a paired model comparison to individual preprocessing
transformations.

### Core claims

1. Repository versions and preprocessing choices can reverse which tabular model wins.
   **Claim novelty: LOW.** Recent work directly documents inappropriate repository preprocessing and
   preprocessing-sensitive model comparisons.
2. Checksum-anchored reconstruction can prove that two repository entries descend from one canonical
   source through specific transformations. **Claim novelty: MEDIUM.** This is a sharper provenance
   procedure than the retrieved empirical audits.
3. Transformation-by-transformation replay can causally attribute a rank reversal.
   **Claim novelty: MEDIUM.** The exact attribution may be new, but it narrows an established finding
   rather than opening a new research question.

### Closest prior work

| Paper | Year | Overlap | Smallest defensible difference |
|---|---:|---|---|
| [Unreflected Use of Tabular Data Repositories Can Undermine Research Quality](https://arxiv.org/abs/2503.09159) | 2025 | Directly studies OpenML misuse, preprocessing, and model-comparison problems | Does not provide the proposed checksum reconstruction and per-transformation causal replay |
| [A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data](https://arxiv.org/abs/2407.02112) | 2024 | Shows dataset-specific preprocessing affects tabular model evaluation | Compares pipelines rather than lineage-linked variants of one canonical source |
| [Benchmark Data Repositories for Better Benchmarking](https://arxiv.org/abs/2410.24100) | 2024 | Covers repository design and reproducibility | Does not attribute a concrete ranking change to a reconstructed transformation chain |

### Assessment

- **Score:** 5.0/10
- **Recommendation:** NARROWED BACKUP
- **Key differentiator:** exact lineage proof and per-transformation attribution.
- **Risk:** the headline result is already published. A single Adult/OpenML case may read as a
  replication note even if the reconstruction is meticulous.
- **Suggested positioning:** a provenance-forensics case study and attribution method. Never claim the
  discovery that preprocessing changes model rankings.

## I1: future non-target review text

### Proposed method

Freeze the historical recommendation task, remove every held-out target review, and independently
add post-cutoff reviews by the same user and about the same item to measure residual temporal leakage.

### Core claims

1. Review-aware evaluation can exploit text unavailable at the historical prediction time.
   **Claim novelty: LOW-MEDIUM.** Target-review leakage and temporal recommendation leakage are both
   established.
2. After every target review is removed, post-cutoff non-target reviews from the user and item create
   a separate measurable bias. **Claim novelty: MEDIUM.** No retrieved paper cleanly reports this
   residual decomposition, but it is an incremental extension of known leakage problems.
3. Freezing interactions, candidates, and model settings while changing only text availability
   yields a timestamp-safe audit protocol. **Claim novelty: MEDIUM.** The control is clean, but recent
   historical-review models already treat target-time availability as a design constraint.

### Closest prior work

| Paper | Year | Overlap | Smallest defensible difference |
|---|---:|---|---|
| [TransNets: Learning to Transform for Recommendation](https://arxiv.org/abs/1704.02298) | 2017 | Shows the unavailable target review supplies much of review text's predictive value | Does not isolate later non-target reviews after removing all target reviews |
| [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060) | 2023 | Shows future data can alter historical evaluation and model order | Adds later interactions rather than changing only non-target text availability |
| [Do Reviews Matter for Recommendations in the Era of Large Language Models?](https://arxiv.org/abs/2512.12978) | 2025 | Removes and distorts review evidence in recommendation benchmarks | Does not impose a historical cutoff or decompose later-user and later-item reviews |
| [HSP-GCN: Historical Semantic Proxy Graph Convolutional Networks for Review-Aware Recommendation](https://www.semanticscholar.org/paper/2942db029de28b456538c19083b0043e6a3140a7) | 2026 | Explicitly uses non-target historical reviews because target reviews are unavailable | Proposes a historical model rather than auditing accuracy inflation from post-cutoff text |

### Assessment

- **Score:** 5.0/10
- **Recommendation:** NARROWED BACKUP
- **Key differentiator:** the residual non-target future-text decomposition after target-review
  removal.
- **Risk:** the largest effect may be the already-known target-review leak. If no credible published
  protocol aggregates post-cutoff non-target text, the remaining experiment audits a hypothetical
  mistake rather than practice.
- **Suggested positioning:** an implementation-backed residual leakage audit. Do not use the broad
  title "reviews from tomorrow" without stating that target-review leakage is prior work.

## T: within-row redundancy and tabular likelihood pathology

### Proposed method

Compare native tabular datasets with high-dimensional CV/NLP embeddings and test whether a valid
within-row redundancy statistic explains the likelihood pathology beyond dimensionality and feature
correlation.

### Core claims

1. Within-row redundancy is the missing reason the likelihood pathology is rare in tabular data.
   **Claim novelty: LOW-MEDIUM.** The target paper already studies feature-correlation differences,
   and recent OOD work gives geometric and intrinsic-dimension explanations.
2. Native tables versus embedding tables provide a useful redundancy gradient while other causes can
   be controlled. **Claim novelty: MEDIUM.** The 2026 target paper already evaluates both groups, so
   the novelty is only the proposed reanalysis.
3. The proposed complexity measurement separates redundancy from feature correlation.
   **Claim novelty: LOW.** The original byte-compression measure has insufficient resolution, while a
   marginal-model codelength becomes the established likelihood-ratio correction.

### Closest prior work

| Paper | Year | Overlap | Smallest defensible difference |
|---|---:|---|---|
| [Why the Counterintuitive Phenomenon of Likelihood Rarely Appears in Tabular Anomaly Detection](https://arxiv.org/abs/2602.09593) | 2026 | Same phenomenon, datasets, model family, and explanations involving dimension and feature correlation | Does not name within-row redundancy, but the proposed variable may not be empirically distinct |
| [Input complexity and out-of-distribution detection with likelihood-based generative models](https://openalex.org/W2996156127) | 2019 | Establishes the complexity explanation and correction | Applies it outside ordinary tabular data |
| [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845) | 2019 | Uses a background model to correct likelihood | Directly collides with the proposed marginal-codelength score |
| [A Geometric Explanation of the Likelihood OOD Detection Paradox](https://arxiv.org/abs/2403.18910) | 2024 | Explains high density with low probability mass using local intrinsic dimension | Supplies an adjacent geometric explanation that T would need to beat |

### Assessment

- **Score:** 4.0/10
- **Recommendation:** ABANDON CURRENT FORM
- **Key differentiator:** none is presently validated. "Within-row redundancy" has not been separated
  from the feature-correlation variable already tested by the target paper.
- **Risk:** the usable estimator collides with prior work, the original estimator fails, and the
  remaining reanalysis uses dataset groups already present in the target paper.
- **Suggested positioning:** none yet. T needs a new measurable construct and a controlled example
  where redundancy changes while dimensionality and feature correlation do not.

## Final decision

K is the only idea whose main scientific question is both explicitly open in the source literature
and supported by a clean mechanism-separating experiment. A and A2 are credible second choices: A
offers the larger practical consequence if batching creates material dependence, while A2 offers the
cleaner causal design. A3 is interesting but one step more derivative and requires an identification
repair. A1 and I1 should remain backups only. T should not be built in its current form.

This is a literature judgment through 2026-09-10, not proof that no unindexed paper exists and not a
claim that any idea has a high acceptance probability.
