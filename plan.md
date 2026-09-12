# Research plan: dependence and valid inference in batched LLM annotation

Updated: 2026-09-11  
Status: Reviewer-revised PhD-depth project blueprint based on the completed Gate A pilot  
Target near-term venue: IEEE BigData 2026 High School Symposium  
Project state: Gate A passed one model-dataset cell; every broader claim remains untested

## 1. Executive decision

The project will study a statistical assumption hidden inside batched LLM annotation. When several
independently meaningful documents are placed in one prompt and the model returns all labels in one
response, the resulting errors may no longer be independent. Positive within-prompt dependence can
leave mean accuracy unchanged or even improve it while making ordinary standard errors, confidence
intervals, significance tests, and effective sample-size calculations too optimistic.

The full project has four connected goals:

1. Establish where batching-induced dependence occurs and how large it is.
2. Identify the mechanisms that create it.
3. Measure the damage to downstream statistical inference.
4. Evaluate batching and inference procedures that preserve efficiency without false certainty.

The intended contribution is not a new cluster-robust standard error. Clustered inference is
established. The potentially novel contribution is showing that the annotation call itself creates
a design-induced dependence structure, identifying its causes, quantifying its
estimand-specific consequences across a prespecified panel of models and tasks, and producing
evidence-based design rules for LLM annotation pipelines.

The program and the first conference paper are separate products. The program may pursue all four
goals. The IEEE BigData High School Symposium paper will make one primary phenomenon claim and one
directly linked inference claim only if both pass their prespecified evidence gates.

## 2. Working title and thesis

### Working title

**When the Batch Becomes the Cluster: Dependence, Mechanisms, and Valid Inference in LLM
Annotation**

The narrower symposium paper will use the provisional title **When the Batch Becomes the Cluster:
Measuring Error Dependence in Batched LLM Annotation**.

### Central research question

When an LLM labels multiple items in one prompt, when and why do the resulting labels become
statistically dependent, and what annotation and inference procedures remain valid under that
dependence?

### Intended thesis

Batched LLM annotation is a statistical intervention, not merely a cheaper implementation of
independent annotation. Joint prediction may create prompt-level error dependence through shared
context, batch composition, output order, and autoregressive label history. Whether that dependence
changes uncertainty depends on the downstream estimand and its probability space. Randomized
re-batching can identify the phenomenon; estimand-specific inference and direct channel
interventions can measure its consequences and causes.

This thesis is provisional. It becomes defensible only if the confirmatory studies below support
all three links in the chain:

```text
joint batch construction
        -> within-call dependence
        -> inferential miscalibration
        -> correction or mitigation restores calibration
```

## 3. Existing evidence and its limits

Gate A is discovery evidence from one fixed cell:

| Component | Frozen pilot choice |
|---|---|
| Model | Qwen3-8B, exact revision pinned |
| Dataset | Civil Comments validation split, exact revision pinned |
| Task | Binary toxicity classification |
| Decoding | Greedy, non-thinking, BF16 |
| Target documents | 2,000 naturally sampled records |
| Batch size | 16 |
| Randomized assignments | 3 |
| Conditions | solo, joint labeling, and matched-length filler |

The joint condition produced aggregate error ICC 0.0754 with a conservative 95% decision interval
of [0.0182, 0.1314]. All three permutation-placebo batch tests gave p = 0.0010. The matched-length
filler placebo ICC was 0.0048. Under the exchangeable-ICC approximation, batch size 16 gives a
design effect of 2.131 and an effective sample size of about 938 rather than 2,000. These are
heuristic summaries, not estimand-general variance results. The pilot findings are recorded in the
[Gate A report](gate-experiments/a/artifacts/full/analysis/report.md).

The pilot's paired error contrasts show a narrower accuracy story: joint labeling had 0.0542 lower
error than solo labeling, but 0.0282 higher error than the length-matched filler control. Parse
failures differed across conditions (solo 0.0090, filler 0.0012, joint 0.0003), so the confirmatory
accuracy analysis must include the frozen missing-output bounds in Section 10.6. The pilot did not
measure confidence-interval coverage. Applying a normal, equal-size exchangeable-cluster formula to
its ICC would predict about 82% coverage for a nominal 95% i.i.d. mean-error interval; that is a
planning approximation, not an observed result or a prediction for other estimands.

The pilot does **not** establish any of the following:

- that the effect generalizes beyond Qwen3-8B, Civil Comments, binary labels, or batch size 16;
- that the ANOVA ICC is the best estimator for binary errors;
- that ordinary confidence intervals actually under-cover in an end-to-end analysis;
- which part of joint batching causes the dependence;
- that every form of batching is unsafe;
- that cluster-robust inference is a new statistical method;
- that the effect is large enough to change published conclusions.

The pilot cell will remain labeled as exploratory discovery evidence. It will not be pooled into a
confirmatory result without an explicit sensitivity analysis showing how that choice changes the
conclusion.

## 4. Contribution boundaries

### Contributions the project may support

**C1. Phenomenon:** Measurement of benign, within-call, across-item output and error covariance
caused by joint LLM annotation, estimands not found in the completed literature audit.

**C2. Identification:** A randomized re-batching design with exact-context, position, composition,
and independent-call controls that separates co-batching from item difficulty and prompt length.

**C3. Mechanism:** Causal evidence about which channels create dependence, including batch
composition and autoregressive output history.

**C4. Consequence:** Direct, estimand-specific measurement of confidence-interval coverage and
Type-I error when batched labels are analyzed under an incorrect independence model.

**C5. Practice:** Tested annotation and analysis procedures that trade off accuracy, dependence,
throughput, and cost.

### Claims the project must not make

- “LLM labels are generally dependent” without model, task, prompt, and decoding boundaries.
- “Batching reduces accuracy.” The pilot's joint accuracy exceeded solo accuracy, even though joint
  errors were correlated; the joint condition was less accurate than matched-length filler.
- “Cluster-robust inference is new.”
- “Attention proves the mechanism.” Attention may be descriptive evidence but cannot carry a causal
  mechanism claim.
- “A zero ICC proves independence.” A negative study must report detectable-effect bounds.
- “The pilot proves a general problem.”
- “A positive error ICC invalidates every downstream analysis.” The variance consequence depends on
  the estimator's influence contributions and the study's sources of randomness.
- “Disagreement with one reference label is always an annotation error.” Subjective tasks require a
  separate analysis of output dependence and reference-relative error.

The completed [novelty audit](idea-stage/novelty-check-20260910/NOVELTY_REPORT.md) rates the idea
7/10. Its narrow novelty boundary is the within-run, across-item covariance produced by the
annotation call and the design used to identify it.

## 5. Formal framework

Let an experimental cell \(c\) fix the model checkpoint, task and dataset, prompt template,
decoding rule, response grammar, batch size, and serving engine. Within a cell:

- \(i\) indexes items from a declared finite population or sampling frame \(\mathcal{I}_c\);
- \(r\) indexes randomized assignments;
- \(\Pi_r\) is the complete partition of sampled items into annotation calls;
- \(O_r\) is the randomized document and output order;
- \(b\in\Pi_r\) identifies one globally unique annotation call;
- \(\omega\) denotes serving or numerical realization, analyzed separately from decoding draws;
- \(G_i\) is the reference label or reference-label distribution; and
- \(Y_i(\Pi_r,O_r,\omega;c)\) is the potential model output for item \(i\) under the complete
  assignment.

For tasks with one defensible discrete reference, define
\(E_i(\Pi_r,O_r,\omega;c)=1[Y_i(\Pi_r,O_r,\omega;c)\neq G_i]\). For subjective tasks, output
dependence and reference-relative disagreement are distinct outcomes. Neither is automatically
called error.

For condition (z), define the pair-weighted within-group covariance

\[
C_{c,z}=\frac{\sum_b\sum_{i\neq j\in b}(E_i-\bar E_z)(E_j-\bar E_z)}
{\sum_b |b|(|b|-1)}.
\]

Groups in the joint condition are real annotation calls. Groups in a control are the matched target
sets defined by the same frozen assignment matrix. The cell-level contrasts are

\[
\Delta^{exact}_{cov,c}=C_{c,joint}-C_{c,exact},\qquad
\Delta^{perm}_{cov,c}=C_{c,joint}-C_{c,perm}.
\]

The exact-context contrast is primary. The permutation-placebo contrast is a prespecified
falsification endpoint constructed from independently produced labels. The sign convention is
frozen: positive values mean more same-call co-error than the matched control. Pair weighting,
centering, incomplete-call handling, and stratification are fixed before confirmation. The
corresponding output-label covariance is co-primary for subjective tasks. When a statement applies
to either control, \(\Delta_{cov,c}\) denotes the named contrast rather than a third pooled
estimand.

For equal batch size \(k\), an exchangeable approximation gives the familiar design effect

\[
DEFF = 1 + (k-1)\rho,
\]

where \(\rho\) is the within-batch error ICC. The corresponding effective sample size is

\[
n_{eff}=n/DEFF.
\]

These equations are descriptive for a mean of exchangeable errors. They do not determine the
variance of group contrasts, regressions, prevalence estimates, or rankings. For those targets, the
analysis estimates covariance among the estimator's item-level influence contributions.
Throughout this plan, a variance ratio uses the same target, item frame, assignment policy, and
number of labels in its clustered and i.i.d. denominators. A ratio from one estimand cannot be
reused for another.

### Probability spaces

The project separates four sources of variation:

1. **Item sampling:** repeated samples from a declared finite population or sampling frame.
2. **Assignment:** repeated random partitions, batch compositions, and positions under a named
   batching policy.
3. **Decoding:** repeated draws under nonzero-temperature sampling.
4. **Serving:** disagreement under nominally identical greedy requests caused by numerical or
   provider variation.

The primary greedy-decoding phenomenon estimand averages over item sampling and randomized
assignment while holding the checkpoint, engine, prompt template, and decoding rule fixed. Decoding
and serving variation are reported as separate variance components. A claim about one probability
space cannot be transferred to another without a stated argument.

The identifying interference assumption is partial interference at the annotation-call level:
items may affect other items in the same call, but calls do not affect one another after conditioning
on a randomized execution block. This assumption is probed by interleaving conditions across fresh
sessions and repeating matched blocks.

### Primary estimands

**E1. Primary dependence:** Pair-weighted
\(\Delta^{exact}_{cov,c}\) for discrete errors or task-specific losses. The separately reported
\(\Delta^{perm}_{cov,c}\) is its falsification endpoint.

**E2. Output dependence:** Within-call covariance of predicted labels or task-specific score vectors,
reported separately from disagreement with a reference label.

**E3. Conditional dependence:** Residual covariance and call-level variance after cross-fitted item
difficulty adjustment and a fully specified crossed hierarchical model.

**E4. Accuracy effect:** Paired change in error probability between joint and control conditions.
This is separate from E1-E3.

**E5. Inferential damage:** Estimand-specific difference between nominal and empirical interval
coverage, plus excess Type-I error under an explicitly generated null, when call membership is
ignored.

**E6. Peer-composition effect:** Change in a target item's error probability or label log-odds when
the experimentally assigned batch composition changes.

**E7. Output-history effect:** Change in a later item's normalized class probability under
distribution-supported changes to earlier emitted labels, plus the reduction in \(\Delta_{cov,c}\)
when the history channel is disabled.

### Statistical perspectives

The same phenomenon will be analyzed through three compatible perspectives:

1. **Cluster sampling:** each annotation call can contribute covariance to an aggregate estimator;
   the effect is derived for that estimator rather than inferred from ICC alone.
2. **Crossed random effects:** repeated items and prompt batches contribute separate variance
   components.
3. **Partial interference:** an item's potential prediction may depend on its within-call peers,
   composition exposure, and output history, while separate randomized calls are assumed not to
   interfere within an execution block.

The randomization-based estimate of \(\Delta_{cov,c}\) is primary. Observed ICC, design effect,
overdispersion, residual covariance, and latent variance are prespecified secondary descriptions.
Disagreement among scales is reported and interpreted rather than converted into a requirement that
non-equivalent estimators have the same sign or significance.

## 6. Research questions and hypotheses

### Phenomenon and fixed-panel recurrence

**RQ1.** How often and how strongly does joint batching induce within-call output and
reference-relative error dependence across the prespecified model-task panel?

**H1.** Joint-label calls will show positive excess covariance relative to permutation-placebo groups and
independent calls with the same documents and comparable context length.

**RQ2.** Can mean accuracy and error dependence move independently?

**H2.** Some cells will improve or preserve mean accuracy while increasing the design effect. This
would show why accuracy-only batch evaluations miss the problem.

### Dose response and boundary conditions

**RQ3.** How does dependence change with batch size, prompt length, task difficulty, label
cardinality, semantic similarity, class composition, and output position?

The batch-size relationship will be modeled flexibly rather than assumed linear. A monotone trend is
a hypothesis to test, not a fact to build into the estimator.

### Mechanisms

**RQ4.** How much dependence comes from shared input context, simultaneous task framing,
autoregressive output history, batch composition, and serialization format?

**H3.** Exact-context target-only calls will have lower excess dependence than joint-label calls,
isolating a channel beyond prompt length and neighboring text.

**H4.** In prespecified open-model cells, a distribution-supported intervention on earlier label
tokens will change later normalized class probabilities, and disabling shared output history will
reduce excess covariance by the frozen mechanism margin.

**H5.** Experimentally increasing the proportion of one class or one difficulty level among batch
mates will shift target predictions in susceptible cells.

### Statistical consequences

**RQ5.** When does ignoring batch membership produce confidence-interval undercoverage, inflated
Type-I error, or incorrect model rankings?

**H6.** The size of inferential failure will track the call-level covariance of the corresponding
estimator's influence contributions, while methods that resample or model whole calls will approach
nominal coverage.

### Mitigation

**RQ6.** Which changes to batching, decoding, repeated assignment, or downstream analysis provide
the best accuracy-dependence-efficiency tradeoff?

**H7.** Recording call identifiers and using batch-aware inference will repair uncertainty without
changing predictions when enough independent calls support calibration; this is tested rather than
assumed. Independent decoding or repeated randomized re-batching may reduce dependence at additional
computational cost.

## 7. Confirmatory design principles

### 7.1 Discovery-confirmation separation

The project will use two evidence tiers.

- **Discovery tier:** Qwen3-8B/Civil Comments pilot, analysis development, mechanism exploration,
  and debugging.
- **Confirmation tier:** model families, datasets, prompt templates, and seeds that remain unopened
  until the design, exclusion rules, endpoints, and analysis code are frozen.

Any change made after seeing confirmatory outputs creates a new exploratory analysis. It cannot be
silently relabeled confirmatory.

### 7.2 Unit of randomization and independence

- Batch membership and item order are randomized independently within each assignment.
- The annotation call, not the individual label, is the primary cluster.
- Repeated occurrences of an item are modeled as repeated measures.
- Conditions are interleaved in randomized run blocks across time and fresh serving sessions.
- Model checkpoints, datasets, and prompt templates are fixed panel factors, not thousands of
  independent model-level observations.
- The primary claim is recurrence across the prespecified panel. Hierarchical partial pooling
  estimates heterogeneity but does not turn purposively selected models into a population sample.
- Prompt batching (several items in one request) is distinct from serving-engine execution batching
  (several requests processed together). Randomize and record both when the engine exposes its
  scheduler; otherwise report execution-batch membership as unobserved.

### 7.3 Operational and balanced re-batching policies

The project estimates two different objects and will not pool them:

1. **Operational-policy estimand:** items are partitioned by simple random assignment, subject only
   to context limits and prespecified length strata. This estimates dependence under an ordinary
   batching policy.
2. **Balanced experimental estimand:** assignments deliberately balance position, composition,
   difficulty, and co-occurrence. This increases precision for mechanism contrasts but describes
   that engineered policy only.

Within the balanced regime, each item will appear in multiple independently randomized batches. The
assignment generator should approximately balance:

- how often each item appears;
- each item's position in the output;
- pairwise co-occurrence frequencies;
- batch class composition;
- batch difficulty distribution; and
- total prompt length.

A balanced incomplete-block or near-balanced randomized design will reduce accidental confounding
between item difficulty and batch membership. Each regime has its own assignment generator,
potential-outcome target, randomization distribution, and estimate. Exact assignment matrices,
seeds, and checksums must be frozen before inference.

### 7.4 Capability and error floors

A model-task cell enters the confirmatory matrix based only on an independent, blinded capability
screen and fixed technical rules. The screen checks whether:

- parsing succeeds at a preregistered minimum rate;
- solo performance is above a capability floor;
- no condition is dominated by truncation or context-window violations.

Realized joint-condition errors are never used to exclude a confirmatory cell or continue sampling.
Near-perfect cells remain in the fixed-panel record with uncertainty bounds and an “insufficient
events for a precise covariance estimate” label. Fixed-panel summaries report both the full panel and
the subset that meets the prespecified information threshold. The primary fixed-panel target is the
equal-cell-weight mean excess covariance among cells admitted by the blinded capability screen,
using prespecified losses scaled to [0,1]. The screen's pass/fail record and all later uninformative
cells are published. Information-threshold subsets are labeled secondary and cannot replace the
primary screened-panel estimate. Subjective output-covariance outcomes are analyzed in a separate
family rather than pooled with reference-relative loss covariance.

### 7.5 Power analysis

Sample sizes will be determined by simulation before confirmatory runs using the exact assignment
generator, the signed excess-covariance statistic, and the locked randomization test. The design
target is at least 90% power to detect the preregistered minimum covariance effect after Holm family-
wise error correction, with adequate precision for the coverage and Type-I error targets in Study
4. Simulations must vary:

- marginal error rates from 0.02 to 0.30;
- ICC from 0 to 0.15;
- batch sizes from 2 to 32;
- unequal and incomplete batches;
- repeated items across assignments;
- parse failures that are independent, item-dependent, or batch-dependent; and
- heterogeneous effects across models and tasks.

The final preregistration will state a fixed number of items, calls, assignments, execution blocks,
and Monte Carlo randomizations per cell. Any blinded sample-size re-estimation may use only pooled
nuisance rates, must have a fixed maximum, and cannot inspect condition contrasts. Realized error
count is reported as an information diagnostic, not a stopping rule.

## 8. Experimental scope

### 8.1 Model axis

The confirmatory matrix is a fixed diversity panel, not a probability sample of all LLMs. It should
contain:

- at least four open-weight instruction-tuned model families;
- approximately three size levels per family where context limits permit;
- at least one base-versus-instruction-tuned comparison within a family;
- two independently operated commercial APIs as an external-validity layer, if reproducible access
  and model-version pinning are possible; and
- the original Qwen3-8B checkpoint as a replication anchor, kept outside the primary confirmatory
  meta-estimate.

Candidate open families include Qwen, Llama, Mistral, and Gemma. Exact checkpoints must be selected
through a blinded capability screen and frozen by immutable revision or weight checksum. Selection
rules, exclusions, and family weights are fixed before joint-condition outputs are opened. The study
must not substitute a newer checkpoint midstream. Conclusions are stated for this panel;
future-model transport is a hypothesis, not an identified population claim.

### 8.2 Task axis

The core task set should span different sources of ambiguity and output structure:

| Task family | Purpose |
|---|---|
| Binary toxicity or offensiveness | Replicate the pilot and study subjective boundary cases |
| Binary sentiment | Test a second binary text-labeling domain |
| Multiclass topic classification | Test more than two labels with relatively objective classes |
| Natural-language inference | Test relational reasoning with three-way labels |
| Multi-label emotion or content coding | Test non-exclusive labels and structured output |
| Controlled synthetic text task | Provide exact ground truth and manipulable difficulty |
| Structured information extraction | Test whether dependence extends beyond classification |

Each central task family used to support recurrence must contain at least two independently
constructed datasets so task type is not perfectly confounded with source domain, annotation
process, and label quality. Controlled synthetic tasks and structured extraction may be analyzed as
separate boundary families. Every confirmatory dataset must have a license permitting release of the
necessary identifiers, prompts, and derived outputs. Sensitive-text datasets require exposure
controls for researchers.

Task-specific primary losses are frozen before screening:

| Output type | Primary item-level outcome | Secondary outcomes |
|---|---|---|
| Binary or multiclass | Exact-match 0/1 loss | Per-class loss and normalized-score loss |
| Multi-label | Examplewise Hamming loss | Exact-set loss and label-level outcomes |
| Structured extraction | Prespecified field-level loss averaged within item | Exact-record match and field-specific loss |
| Subjective labels | Output-label or score-vector outcome | Reference-relative disagreement under each declared label aggregation |

Cross-task summaries operate on signed covariance of the prespecified item-level loss. They do not
silently equate exact-match, Hamming, and extraction errors.

### 8.3 Batch and decoding axis

The primary batch sizes are \(k\in\{1,2,4,8,16,32\}\), subject only to exact model context limits.
The main decoding condition is greedy label-only output with no chain-of-thought request. Secondary
conditions include:

- nonzero-temperature sampling with repeated seeds;
- constrained structured decoding versus ordinary text generation;
- labels-only versus short rationales followed by labels;
- forward, reverse, and randomized output order; and
- independent continuations after an identical shared prefix where the serving system supports it.

Temperature-zero prompts will still be repeated to measure numerical or serving nondeterminism.
Class-probability analyses use canonical single-token symbols where the tokenizer permits. Otherwise
they use normalized sequence probabilities under the same constrained grammar and prefix. Verbalized
confidence is never substituted for token probability. Cells without comparable probability access
remain eligible for label-level analyses but cannot support logit-level mechanism claims.

### 8.4 Prompt axis

Use at least three semantically equivalent prompt templates developed before confirmation. Template
wording is a fixed robustness factor, not a random sample from all possible prompts. Templates must
preserve the task definition and output schema while varying surface form. The main analysis reports
whether dependence is stable across the fixed templates rather than averaging away prompt
sensitivity.

## 9. Study program

### Study 0: locked replication of Gate A

#### Purpose

Verify that the original positive result can be regenerated from the frozen artifacts and repeated
on new randomized assignments without changing the model, dataset, labels, or analysis.

#### Design

1. Reproduce the saved analysis byte-for-byte from the existing raw outputs.
2. Repeat the original cell with new batch-assignment seeds.
3. Rotate every target through output positions using a balanced schedule.
4. Add an exact-neighbor target-only control: each joint batch's same 16 documents are shown, but
   each call requests only one target label.
5. Repeat a subset of identical prompts to quantify serving nondeterminism.
6. Run a prespecified serial-execution subset with one request in flight at a time, and compare its
   signed covariance contrast with the ordinary serving schedule.

#### Required result

The original result is considered replicated if the new assignments show positive signed excess
covariance, the Holm-adjusted randomization test rejects the sharp no-batch-effect null at the
preregistered level, and the exact-neighbor target-only control explains less than half of the joint
excess covariance. Compatibility with the pilot is assessed using a frozen prediction interval and
an absolute covariance margin selected by simulation before the new outputs are opened.
For the “less than half” rule, define attributable fraction
\(f=\max(0,C_{exact}-C_{perm})/(C_{joint}-C_{perm})\) on matched assignments, only when the
denominator is positive and exceeds the frozen absolute margin. The one-sided 95% upper confidence
bound for \(f\), obtained by resampling whole calls and repeated items under the frozen design, must
be below 0.5. Otherwise the exact-context result is inconclusive, not a passed control.

### Study 1: cross-model and cross-task recurrence

#### Purpose

Determine whether batch dependence is a recurring property of joint annotation or a narrow
Qwen-toxicity interaction.

#### Design

For every selected model-task-template cell:

1. Draw a natural-prevalence item sample and retain a separate, balanced diagnostic sample.
2. Run solo labels once per item.
3. Run multiple randomized joint assignments at each batch size.
4. Run exact-neighbor target-only calls using the same items, positions, and context.
5. Run matched-length unrelated-filler calls.
6. Construct permutation-placebo groups after inference while preserving batch sizes and design
   strata.
7. Record token counts, latency, GPU time, output length, parsing, and all call identifiers.
8. Record serving-engine execution-batch identifiers or concurrency windows where observable;
   repeat a serial-execution subset so shared server scheduling cannot silently explain the
   independent-call controls.

#### Primary analysis

- the signed joint-minus-exact-context and joint-minus-permutation excess covariance
  \(\Delta_{cov,c}\), using the frozen pair-weighting rule;
- an exact or Monte Carlo randomization test generated by the cell's actual assignment mechanism;
- paired accuracy contrasts, analyzed separately from dependence;
- observed ICC, design effect, residual covariance, and conditional call variance as secondary
  descriptions; and
- fixed-panel hierarchical partial pooling across the selected families and tasks, accompanied by
  leave-one-family-out and leave-one-task-out sensitivity analyses.

The pooled descriptive target is the finite, screened-panel equal-cell-weight mean of the frozen
[0,1]-loss covariance contrasts, not the effect in a hypothetical population of models. A crossed
family/task hierarchical fit estimates shrinkage and heterogeneity for this finite panel; its
weighting and priors are frozen at WP0. The global confirmatory test uses the equal-cell-weight mean
of cell statistics standardized by their own assignment-null standard deviations, with one
jointly replayed assignment draw per cell at each Monte Carlo step. All leave-one-family-out and
leave-one-task-out versions use the same rule. “Survives every leave-one-out” means each reduced
panel has a positive signed estimate and a one-sided p-value below 0.05 after Holm correction across
these prespecified sensitivity tests; failure limits the recurrence claim. A cell with zero
assignment-null standard deviation contributes zero to the global statistic and remains in the
equal-cell denominator; it is reported as uninformative, not silently removed.

#### Interpretation

- Recurrence across the fixed panel supports a bounded statement about the tested systems and tasks,
  not a population claim about all LLMs or annotation problems.
- Strong, predictably structured heterogeneity supports a boundary-condition result.
- Effects confined to one checkpoint and one task leave the work as a case study.

### Study 2: batch-size dose response and boundary map

#### Purpose

Map how dependence changes as the annotation workload and batch structure change.

#### Manipulations

- batch size;
- total prompt tokens, manipulated separately from item count;
- mean and variance of item difficulty;
- natural, balanced, and deliberately skewed label composition;
- semantic similarity among batch mates;
- homogeneous versus heterogeneous source domains;
- target output position;
- number of classes;
- item-length dispersion; and
- response-format complexity.

#### Design requirement

Manipulate item count and token length independently. Otherwise a batch-size effect cannot be
separated from a long-context effect. Rotate target items across conditions and positions so every
contrast is within-item wherever possible. Run operational-policy and balanced experimental
assignments as separate experiments with separate estimands; do not pool them into one dose curve.

#### Analysis

Fit flexible dose-response curves with uncertainty at the batch and model-task levels. Report
nonmonotonic relationships if present. Do not reduce the result to a single pooled ICC when cells
have materially different marginal error rates.

### Study 3: causal mechanism experiments

#### M1. Shared context versus joint output

Compare four conditions using identical documents and ordering:

1. solo target;
2. all batch documents visible, one target requested per independent call;
3. all batch documents visible, all labels requested in one response; and
4. all documents encoded together but labels produced through independent continuations, where
   technically supported.

These four conditions identify bundles of channels rather than a complete causal decomposition. A
matched factorial extension must independently vary document sharing, joint instruction, shared
output history, and requested label count wherever the serving interface permits. Unsupported
factor combinations are reported as unidentified rather than inferred from the nearest contrast.

#### M2. Autoregressive label-history intervention

For open models, teacher-force controlled prefixes of earlier labels and measure the normalized
probabilities of later labels. The confirmatory intervention samples truthful and shuffled prefixes
from the model's naturally observed support while holding documents fixed. Flipped and neutralized
prefixes are stress tests because they may create off-distribution histories. Use gold-known or
synthetic tasks for the confirmatory version and compare against an independent-continuation
condition that removes shared label history.

Evidence for this mechanism requires both a randomized prefix effect on later outputs and a
reduction in \(\Delta_{cov,c}\) when shared output history is disabled. Attention patterns and
forced-prefix effects alone are insufficient.

#### M3. Composition interference

Repeat fixed target items inside batches whose class prevalence, difficulty, and semantic similarity
are experimentally controlled. Estimate how each exposure changes target errors and prediction
probabilities. Use factorial randomization so class prevalence is not confused with semantic topic.
Before assignment, define each target's exposure as the other \(k-1\) items' reference-class
proportion, cross-fitted solo difficulty mean, and prespecified embedding-similarity mean, excluding
the target itself. Bin or contrast these continuous exposures at frozen cutpoints, and report the
assignment probability for each target-exposure level. Potential outcomes are indexed by target,
position, and this exposure vector under the partial-interference assumption; unmeasured peer
features and interactions remain a limitation of that mapping.

#### M4. Position and serialization

Use Latin-square position rotation and compare:

- target labels emitted first versus last;
- document order matched versus output order reversed;
- flat JSON versus one-record-per-line output;
- explicit per-item separators; and
- response segmentation or reset tokens where supported.

#### M5. Context-induced decision-boundary shift

Use carefully constructed borderline items and unambiguous anchors. Test whether early examples
shift the decision boundary applied to later examples. Estimate label-threshold movement from model
scores when available and from repeated binary outcomes otherwise.

#### Mechanism decision rule

A mechanism is supported only when a direct intervention changes the prespecified target outcome
and reduces \(\Delta_{cov,c}\) by both the frozen absolute and relative margins in held-out
model-task cells after multiplicity correction. Descriptive mediation, attention correlation, or a
local forced-prefix effect without covariance reduction is exploratory.

### Study 4: downstream inference and scientific conclusions

#### Purpose

Show whether batch dependence changes the answers researchers report, not only an ICC diagnostic.

#### Downstream estimands

Every retained estimand must name its target, source of truth, sampling frame, assignment policy,
and null generator before calls are made.

| Estimand | Target and source of truth | Confirmatory null generator |
|---|---|---|
| Reference-relative model accuracy | Finite-population mean accuracy on a frozen item frame with exact synthetic labels or an independently adjudicated reference bank | Repeated item samples and partitions under a fixed annotation policy |
| Emitted-label prevalence | Mean LLM-emitted label under the stated batching policy; this is not gold prevalence | Repeated item samples, partitions, and serving realizations under that policy |
| Gold prevalence or group difference | Population quantity identified through a probability sample with trusted labels or a frozen validation design and explicit misclassification correction | Stratified resampling or randomization from the declared population frame |
| Difference between annotation models | Difference in a common item-level loss on the same reference bank | Under the sharp null, exchange model assignments or compare prespecified exchangeable duplicate arms using independent batch partitions |
| Regression coefficient | Coefficient when the LLM label is the outcome in a prespecified population; the confirmatory null uses a randomized covariate hidden from the annotator prompt | Re-randomize the covariate using its recorded assignment mechanism; LLM labels used as covariates remain exploratory without a separate known-truth generator |
| Benchmark ranking | Rank fixed candidate classifiers by loss against batched LLM labels on common sampled items; the true reference ranking comes from exact synthetic or independently adjudicated labels | Use candidates with identical reference loss for the true-null experiment and freeze candidate order, tie region, scoring rule, and item-batching policy |

Gold-population targets and LLM-output targets are never treated as interchangeable. An estimand is
dropped from confirmation if its truth source or null generator cannot be implemented without using
the observed joint-condition outcomes.

Before any calls, freeze a separate variance prediction for each retained estimand. For mean
reference-relative accuracy and emitted-label prevalence, positive within-call covariance of the
corresponding item contributions predicts inflated variance under the operational batching policy.
For a regression with a covariate randomized independently at item level and hidden from the model,
the covariate's within-call correlation is approximately zero, so batch error ICC alone predicts
little slope-variance inflation; this is a prespecified negative-control prediction, not a guarantee.
Add a second regression experiment that randomizes the covariate by call or assigns a batch-correlated
covariate before annotation. Its contribution covariance may be larger, and the true coefficient
and null are generated independently of observed joint errors. For a paired model difference or
benchmark ranking, the relevant quantity is covariance of paired loss differences, not either
model's marginal error ICC. Gold prevalence additionally needs a separate label-bias analysis.

#### Repeated-sampling experiment

Select six cells before opening joint-condition outputs, spanning at least two model families and
two task families. Selection uses only the independent capability screen and design factors, never
realized dependence. For each selected cell:

1. Repeatedly sample items from a defined population.
2. Randomly form annotation batches using the frozen operational-policy generator.
3. Obtain fresh LLM labels for 5,000 independent repetitions per estimand, or the larger count fixed
   by Monte Carlo precision analysis before outputs are opened.
4. Construct nominal 90%, 95%, and 99% intervals using each candidate method.
5. Record whether each interval covers its corresponding target.
6. Use the estimand-specific null generator above to measure Type-I error and use frozen planted
   alternatives to measure power.

The item-sampling, batching-assignment, decoding, and serving probability spaces are stored
separately for every repetition. Actual repeated-call experiments are primary evidence. A larger
calibrated simulation study will map performance over error rates, call counts, covariance,
unequal batch sizes, repeated items, and effect heterogeneity using generators validated against
held-out observed cells.

The ranking experiment never asks the LLM to rank candidate systems in a shared response. It
creates candidate predictions on a common item frame, obtains batched LLM item labels, and then
computes each candidate's frozen loss against those labels. This keeps the tested dependence source
as item batching and defines ranking reversal against an independent reference ranking.

#### Methods compared

- i.i.d. analytic intervals;
- item-level bootstrap;
- batch-clustered sandwich variance;
- whole-batch bootstrap;
- small-cluster corrections and wild cluster bootstrap;
- hierarchical beta-binomial or logistic mixed models;
- crossed item-by-batch methods for repeated assignments; and
- randomization-based intervals when batch assignment is controlled.

For each estimator, compute the call-level covariance of its influence contributions. This tests
H6 on the scale of the downstream estimand instead of assuming that binary error ICC alone predicts
every inferential failure.

For mean accuracy, also compare joint and solo annotation at the same target estimand, item-frame
size, and label budget. Estimate their total repeated-sampling variance and bias separately. Report
the joint-to-solo mean-squared-error ratio alongside each arm's clustered-to-i.i.d. variance ratio;
a design effect above one within joint calls does not establish that joint annotation has worse
total precision than solo annotation.

#### Key distinction

Variance correction does not repair systematic labeling bias. Analyses estimating a gold-label
population quantity must separately model or validate label bias. The project will report
calibration of uncertainty and correction of bias as different problems.

#### Success criterion

For the primary 95% intervals, inferential damage is established only if the naive method has
coverage at or below 90% or Type-I error at or above 7.5% in at least two prespecified real cells,
after Holm correction across estimands and cells. Analysis-only correction meets the separate L5a
criterion only if a prespecified batch-aware method reaches 93% to 97% coverage and 4% to 6% Type-I
error in those cells, with median interval width no more than 1.5 times the oracle interval. The 90%
and 99% intervals, power, ranking
reversals, and alternative thresholds are secondary sensitivity analyses. These numerical gates
may be tightened by WP0 simulation before any confirmatory output is opened, but cannot be relaxed
afterward.

### Study 5: mitigation and systems tradeoffs

#### Mitigation families

**P1. Analysis-only correction:** retain efficient joint calls, record batch IDs, and use
batch-aware uncertainty. This changes interval construction, not the labels, covariance, or call
throughput. Its calibration must be measured at the actual call count and clustering structure;
asymptotic validity does not guarantee finite-call coverage.

**P2. Smaller batches:** reduce \(k\) until the dependence-throughput tradeoff is acceptable.

**P3. Independent decoding:** share input processing where possible but prevent one item's emitted
label from entering another item's output history.

**P4. Repeated randomized re-batching:** label each item in several independent batch contexts and
aggregate predictions using a prespecified rule.

**P5. Composition control:** construct batches to reduce harmful class, topic, or difficulty
imbalance.

**P6. Serialization control:** change delimiters, label order, or response segmentation when a
mechanism study supports that intervention.

#### Evaluation criteria

Every mitigation is scored on:

- mean accuracy and class-specific error;
- ICC, covariance, and design effect;
- confidence-interval coverage;
- token consumption;
- wall-clock latency and throughput;
- GPU time or API cost;
- parsing reliability; and
- reproducibility across model families.

The output is a Pareto frontier, not a claim that one configuration is universally optimal.
Analysis-only correction is eligible only when every label can be traced to a unique call and
assignment. If call lineage is missing, the plan reports the estimand as unrecoverable rather than
reconstructing clusters from row order or timestamps.
Report analysis-only calibration and operational mitigation as separate successes. A procedure
that fixes interval coverage while leaving label dependence and throughput unchanged meets L5a,
not L5b. Benchmark L5b against serial calls under the same model, hardware, serving policy, target
items, and accuracy metric; freeze the throughput and covariance margins before opening outcomes.

### Study 6: human and multilingual boundary studies

#### Human comparison

With ethics approval and informed consent, trained human annotators can label the same randomized
batches and solo items. This estimates whether dependence also occurs in that human protocol; it
cannot establish a universal human-versus-LLM distinction. Human and LLM effects must not be
compared as though their observations have identical sources of variation.

#### Multilingual comparison

Repeat selected tasks in multiple languages using native or professionally verified labels and
instructions. Separate language proficiency from batching by requiring a solo capability floor in
every language-model cell.

These are external-validity studies. They are not required for the first paper but strengthen a
doctoral-scale research program.

## 10. Statistical analysis plan

### 10.1 Descriptive reporting

Every cell reports:

- item and batch counts;
- marginal class and error rates;
- parse-failure rates and reasons;
- error counts per batch;
- position-specific performance;
- prompt and output token distributions;
- latency and compute;
- covariance, ICC, design effect, and effective sample size; and
- variation across randomized assignments.

### 10.2 Primary confirmatory model

Use a crossed logistic mixed model or an equivalent Bayesian hierarchical model as a secondary
conditional description:

\[
logit\{P(E_{icar}=1)\} = \alpha_c + u_{ic} + v_{car} + \gamma_a + \delta_{block}
 + \beta^T X_{icar},
\]

where \(c\) identifies the fixed model-task-template cell, \(a\) the randomized assignment,
\(r\) the globally unique annotation call, and `block` the randomized execution block. The item
effect \(u_{ic}\) captures repeated item difficulty, \(v_{car}\) captures call variation,
\(\gamma_a\) and \(\delta_{block}\) represent assignment and execution-block effects, and \(X\)
contains the frozen position, batch-size, condition, and composition terms. Model family, task
family, and template are modeled only at levels justified by the fixed panel.

Report the observed-scale ICC alongside the latent variance component. Reviewers should not need to
accept one modeling scale to accept the phenomenon. The preregistration fixes optimizer, priors or
penalties, convergence diagnostics, singular-fit handling, and the fallback descriptive model.
Failure to fit cannot trigger a switch of primary estimand, which remains randomization-based
\(\Delta_{cov,c}\).

### 10.3 Randomization inference

The primary cell statistic is the signed \(T_c=C_{c,joint}-C_{c,exact}\) from the matched
item sets and assignment matrix; the global statistic is fixed in Study 1. Test the sharp null that,
within each arm and execution block, every item's output would be unchanged by any allowable
reassignment of calls, peers, or positions. This null permits different fixed outputs, including a
marginal accuracy shift, between joint and exact-context arms. Conditional on the observed
item-arm-block output records under that null, replay the *paired* frozen assignment generator in
both arms, including length strata, position constraints, balance restrictions, and batch sizes.
Recompute the identical centered, pair-weighted \(T_c\) for each draw and use its upper tail for a
one-sided test. Keep repeated-item and execution-block restrictions in the replay; use the exact
distribution when enumerable or a preregistered Monte Carlo count otherwise.

This is a test of a strong no-assignment-effect null, not a test of the weaker statement
\(\Delta^{exact}_{cov,c}=0\). Rejection alone cannot identify the covariance mechanism; the
positive effect-size contrast, exact-context control, and missing-output bounds must also pass.
Freeze a design-respecting effect-size interval and simulation-check its coverage under nulls with
arm-specific mean shifts, heterogeneous item difficulty, and serving variation. Finite-sample bias
of the covariance statistic matters for effect-size interpretation, but is not by itself a reason an
otherwise correctly replayed sharp-null randomization p-value would be invalid. For composition
experiments, re-randomize only the factor assigned by the factorial design. Unrestricted label
shuffling is not a valid substitute.

### 10.4 Binary-outcome sensitivity

Because an ICC for binary errors depends on the marginal error rate, also report:

- raw within-batch pairwise covariance;
- standardized residual covariance after cross-fitted item-difficulty prediction;
- batch-level overdispersion;
- latent-scale variance from the hierarchical model; and
- results stratified by error-rate bands.

### 10.5 Repeated items and multiple clustering

Repeated re-batching creates dependence by item and by call. Use crossed random effects or multiway
cluster methods. Never pool repeated labels and treat them as new independent documents. Aggregate
assignment-level estimates with a model that recognizes that the same items recur.

### 10.6 Missing and malformed outputs

Every requested label follows a frozen state machine: `valid_first_response`, `malformed`,
`truncated`, `refused`, `missing`, or `retry_valid`. The original response is never replaced. A
retry is a new call with a new call identifier and a pointer to its parent, and confirmatory
endpoints use the prespecified first-response or retry policy without mixing them. Mandatory
sensitivity analyses treat unresolved labels as:

- errors;
- correct labels;
- a separate failure outcome; and
- batch-level failures when one malformed response affects several requested labels.

Analyze whether parse failures cluster by call. Dropping clustered failures can bias both accuracy
and dependence downward.

### 10.7 Multiplicity

Use a hierarchical testing order with confirmatory families frozen before outputs are opened:

1. test the global sharp no-assignment-effect null with the signed excess-covariance statistic at
   \(\alpha=0.05\), then test cell sharp nulls using Holm FWER at \(\alpha=0.05\); attach an
   excess-covariance claim only when the corresponding effect-size and control margins pass;
2. test fixed-panel recurrence across model and task families using Holm FWER at \(\alpha=0.05\);
3. test each frozen mechanism family using Holm FWER at \(\alpha=0.05\);
4. test downstream coverage and Type-I-error failures using Holm FWER at \(\alpha=0.05\); and
5. test the frozen mitigation contrasts using simultaneous 95% intervals.

The exact membership of each family and gatekeeping order appears in the preregistration. Benjamini-
Hochberg FDR is reserved for explicitly exploratory interaction maps. Exploratory interactions
remain exploratory regardless of p-value.

### 10.8 Practical significance

Statistical significance is insufficient. The default minimum dependence effect is an absolute
increase of 0.002 in within-call pairwise covariance relative to the exact-context control; WP0
simulation may raise but not lower it before confirmation. A percentage relative to control
covariance is not used because that control can be zero or change sign. A separate practical
mean-accuracy threshold is a clustered-to-i.i.d. variance ratio of at least 1.25, equivalent to at
least 20% effective-sample-size loss under the *same* mean estimand and sampling policy. Meeting the
covariance threshold does not imply meeting this variance threshold or Study 4's coverage gate.
Report consequences in units
researchers can use:

- percentage loss in effective sample size;
- change in interval width;
- empirical coverage deficit;
- excess false-positive probability;
- model-ranking reversal frequency; and
- added cost required to repair the problem.

## 11. Robustness and falsification tests

**F1. Permutation-placebo groups:** regroup independently produced labels using the exact frozen
assignment generator. This should not recreate the joint effect.

**F2. Exact-context target-only control:** hold documents, position, and length fixed while removing
joint label generation.

**F3. Re-randomization:** the effect must survive new batch assignments over the same items.

**F4. Position balance:** rotate items through positions so position cannot masquerade as a batch
effect.

**F5. Difficulty adjustment:** condition on cross-fitted solo error propensity and known item
features.

**F6. Label-preserving composition:** change semantic similarity while holding class counts fixed,
then change class counts while holding semantic similarity as stable as possible.

**F7. Parse-failure bounds:** recompute results under extreme assignments of missing labels.

**F8. Determinism audit:** repeat identical prompts and measure output disagreement.

**F9. Prompt-template replication:** repeat the primary contrast with independently written
templates.

**F10. Held-out confirmation:** run frozen analysis on unopened model-task cells.

**F11. Serving-batch audit:** compare ordinary concurrent serving with a serial-execution subset;
record observable engine batch composition and bound any residual covariance among independent calls.

## 12. Decision rules and claim ladder

### L0: no interpretable result

Capability, parsing, context, or error floors fail. Report the cell as uninformative.

### L1: replicated case study

The Qwen3-8B/Civil Comments cell meets Study 0's signed excess-covariance, Holm-adjusted
randomization, exact-context, prediction-interval, and absolute-margin rules. This supports a narrow
model-dataset claim.

### L2: fixed-panel recurrence

The global fixed-panel randomization test rejects at \(\alpha=0.05\), Holm-adjusted cell contrasts
show positive excess covariance meeting the practical margin in at least three model families and
three task families, and the result survives every leave-one-family-out and leave-one-task-out
analysis. This supports a claim about the frozen panel only.

### L3: explained phenomenon

At least one direct randomized intervention changes the target outcome and reduces excess
covariance by at least 0.002 and 25% of the *pre-intervention joint-minus-exact excess* in two
held-out model-task cells after Holm correction. A relative reduction is evaluated only when that
pre-intervention excess is positive and exceeds the frozen absolute margin; otherwise report an
absolute contrast without a percentage claim.

### L4: demonstrated scientific consequence

Naive 95% inference has coverage at or below 90% or Type-I error at or above 7.5% in at least two
prespecified actual repeated-call cells after Holm correction, and the failure tracks the relevant
influence-contribution covariance.

### L5a: calibrated analysis-only correction

At least one frozen batch-aware analysis procedure reaches 93% to 97% coverage and 4% to 6% Type-I
error in the affected held-out cells, with median interval width no more than 1.5 times the oracle.
This claim is conditional on the tested number and structure of calls; it does not imply changed
labels, reduced dependence, or faster annotation.

### L5b: dependence-reducing, efficient annotation

At least one frozen annotation procedure meets the L5a calibration bounds, reduces excess
covariance by at least 0.002 and 25% of the unmitigated joint excess, loses no more than one
percentage point of accuracy (or the stricter frozen task margin), and gains at least 25% throughput
or reduces cost per valid label by at least 25% against serial annotation on the same platform.
These margins are tested on held-out cells and cannot be relaxed after outcomes are opened.

A PhD-level flagship paper should reach L4. Reaching L5a strengthens its statistical guidance;
L5b would add a validated operational solution. The IEEE BigData High School Symposium paper is
defensible at L1 plus its prespecified
boundary-cell result; it does not require or imply L2.

## 13. Failure outcomes that remain scientifically useful

### Broad replication fails

If only the pilot cell remains positive, the general claim is rejected. The project becomes a
bounded case study or stops. It must not search post hoc for a model subset and call that the
original hypothesis.

### Dependence is real but small

Report upper bounds and identify conditions under which the effective-sample-size loss is
negligible. This can still produce a practical batching safety boundary.

### Dependence is heterogeneous

Develop a preregistered prediction model for susceptibility using batch size, task ambiguity,
output format, and model family. Validate it on held-out cells. Predictable heterogeneity is a
stronger result than unexplained averaging.

### No individual mechanism dominates

Report the failed interventions and retain the phenomenon as a systems-level effect. Do not claim a
mechanism from correlational probes.

### Corrections fail with few clusters

Map the failure region and recommend minimum cluster counts or independent calls. A correction is
not accepted merely because it is theoretically conventional.

## 14. Main threats to validity

**T1. Item difficulty confounding:** difficult items may accidentally cluster. Address with
randomized re-batching, repeated items, permutation-placebo groups, and crossed models.

**T2. Binary ICC dependence on prevalence:** report covariance, overdispersion, latent variance,
and stratified analyses.

**T3. Gold-label noise:** use multiple task types, controlled synthetic labels, and adjudicated or
high-agreement subsets.

**T4. Prompt-length confounding:** independently manipulate item count and token count; include
exact-context controls.

**T5. Position effects:** use balanced rotations and model position directly.

**T6. Autoregressive serialization:** treat output order as a mechanism, not nuisance noise.

**T7. Parse failures and truncation:** freeze token preflight rules, preserve raw outputs, and bound
missing outcomes.

**T8. Model or API drift:** pin exact revisions, record response metadata, and never combine silent
model updates.

**T9. GPU nondeterminism:** repeat identical greedy prompts and quantify run disagreement.

**T10. Selective model and task reporting:** preregister the full matrix and publish failed cells.

**T11. Pseudoreplication:** treat batches, items, model families, and tasks at their correct levels.

**T12. Statistical correction mistaken for bias correction:** evaluate variance and label bias
separately.

**T13. Researcher exposure to toxic text:** minimize raw-text inspection and document handling
procedures.

**T14. Training-data contamination:** contamination may affect mean accuracy, but random batching
still identifies prompt-level dependence. Use synthetic held-out tasks as a separate check.

**T15. Probability-space mismatch:** an interval can be calibrated over item sampling but fail over
serving realizations, or vice versa. Declare and simulate each source of randomness separately.

**T16. Policy mismatch:** balanced experimental batches do not identify performance under an
operational batching policy. Keep their assignment generators, estimands, and claims separate.

**T17. Serving-batch interference:** an engine may co-schedule independent prompts and create
numerical correlation in a control. Record serving composition where possible and validate a
serial-execution subset before attributing a control contrast solely to prompt batching.

## 15. Reproducibility and artifact standard

Every run must preserve:

- dataset identifier, revision, license, split, and row checksums;
- model identifier, exact revision or weight checksum, tokenizer revision, and serving engine;
- full environment lock, CUDA and driver versions, hardware type, and precision;
- system and user prompts with hashes;
- assignment matrix, batch membership, position, seeds, and plan checksum;
- decoding parameters and structured-output constraints;
- raw unedited responses, parser version, parse status, and latency;
- globally unique call ID, parent call ID, retry index, first-response status, assignment ID,
  execution block, serving session, observable execution-batch or concurrency-window ID, and
  response provenance for every requested label;
- analysis configuration, preregistration version, and code commit;
- costs, token counts, GPU time, and failure logs; and
- a clear distinction between exploratory and confirmatory outputs.

Existing artifact directories are immutable historical evidence. New studies must use new output
directories. No failed run is overwritten. Credentials must never appear in configs, commands,
logs, prompts, or reports.

Recommended future structure:

```text
batch-dependence/
  preregistrations/
  configs/
  manifests/
  data-provenance/
  raw/
  derived/
  analyses/
  figures/
  tests/
  run-cards/
  paper/
```

## 16. Work packages and evidence gates

The order below is the execution order. A later package starts only after the preceding package's
exit evidence is recorded, except that WP6 may run alongside WP5 after its own protocol is frozen.
The symposium is split into preparation before replication and a paper gate immediately after
Study 0; it does not wait for the full fixed-panel Study 1.

### WP0: literature and protocol freeze

- Update the novelty search immediately before preregistration.
- Build a claim-to-prior-work matrix.
- Freeze definitions, primary cells, exclusion rules, estimands, randomization statistics and nulls,
  practical margins, global panel weights, leave-one-out rules, and multiplicity handling.
- Simulate the exact assignment designs to fix sample sizes, the absolute covariance floor,
  test size, effect-size interval coverage, and the Study 0 control-fraction rule.
- Register the discovery-confirmation boundary.

**Exit evidence:** timestamped protocol, complete search log, simulation-based power report, and
immutable confirmatory manifest.

### WP-S0: symposium preparation

- Verify the live symposium rules, deadline, authorship eligibility, and page limit.
- Freeze the separate two-cell protocol from Section 17, including its boundary-cell selection rule,
  95% mean-accuracy coverage endpoint, and decision to omit that endpoint if L1 fails.
- Complete the data card, license audit, first-author contribution record, and analysis manifest.

**Exit evidence:** frozen symposium protocol and submission-eligibility record before new Study 0
outputs are opened.

### WP1: infrastructure generalization

- Refactor the Gate A package into model-, dataset-, task-, and batch-size-independent components.
- Add exact-context controls, balanced assignment generation, repeated-prompt audits, and crossed
  clustering.
- Add unit tests and simulation truth tests for every estimator.

**Exit evidence:** deterministic smoke suite, estimator recovery on known synthetic data, and exact
reproduction of the original Gate A analysis.

### WP2a: locked Gate A replication

- Complete Study 0.
- Publish the serial-serving validation and all parse-failure sensitivity results.

**Exit evidence:** L1 decision under the claim ladder, including a documented negative or
inconclusive result when the gate fails.

### WP-S1: symposium paper gate

- If L1 passes, run the frozen boundary cell and the bounded 95% mean-accuracy coverage experiment;
  otherwise record a no-submit decision under the current claim.
- Write the five-page paper using only supported, prespecified claims.

**Exit evidence:** symposium paper package or documented no-submit decision. The symposium schedule
cannot change the doctoral program's confirmatory definitions.

### WP2b: fixed-panel recurrence

- Run the frozen Study 1 matrix.
- Publish all interpretable and uninformative cells.

**Exit evidence:** L2 decision under the claim ladder.

### WP3: dose response and mechanisms

- Complete Studies 2 and 3 using discovery cells.
- Freeze the successful mechanism contrasts.
- Replicate them on held-out confirmation cells.

**Exit evidence:** L3 decision or an explicit mechanism-null result.

### WP4: inferential consequences

- Complete actual repeated-sampling experiments.
- Run calibrated simulations.
- Compare naive and batch-aware procedures.

**Exit evidence:** L4 decision with coverage and Type-I error results.

### WP5: mitigations

- Evaluate analysis-only, batching, decoding, and repeated-assignment procedures.
- Construct accuracy-dependence-efficiency frontiers.
- Validate recommendations on held-out cells.

**Exit evidence:** separate L5a and L5b decisions and bounded statistical or operational
recommendations as supported.

### WP6: external validity

- Complete selected human and multilingual comparisons.
- Separate confirmatory extensions from exploratory findings.

**Exit evidence:** documented boundary conditions and generalization limits.

### WP7: synthesis and release

- Freeze the final analysis.
- Generate tables and figures from machine-readable artifacts.
- Run citation, statistical, reproducibility, and claim-support audits.
- Release code, manifests, safe derived outputs, and a data-access guide.

**Exit evidence:** paper package in which every major claim maps to a preregistered result or is
explicitly labeled exploratory.

## 17. Paper architecture

### Full flagship paper

1. **Introduction:** batching is treated as an efficiency choice, but it changes the joint
   distribution of annotations.
2. **Related work:** batch prompting, cross-query interference, LLM annotation uncertainty,
   predicted-label inference, and clustered data.
3. **Formal problem:** annotation calls as clusters and batching as partial interference.
4. **Experimental design:** re-batching, exact-context controls, model-task matrix, and frozen
   analysis.
5. **Fixed-panel recurrence:** where dependence appears and its dose response.
6. **Mechanism results:** output history, composition, position, and context.
7. **Inferential consequences:** coverage, Type-I error, effective sample size, and ranking changes.
8. **Mitigation results:** correction and systems tradeoffs.
9. **Limitations and ethics:** scope, gold labels, model drift, sensitive content, and human study.
10. **Conclusion:** batch membership is part of the data-generating process and must be preserved in
    analysis.

### IEEE BigData 2026 High School Symposium slice

The symposium allows five pages including references, so it cannot contain the entire doctoral-scale
program. It is a separate protocol with one question:

> In the frozen Qwen3-8B/Civil Comments replication and one boundary cell chosen before joint
> outputs are opened, does joint labeling create signed excess error covariance relative to exact-
> context and permutation-placebo controls, and does ignoring call IDs miscalibrate a 95% interval
> for mean reference-relative accuracy?

The primary endpoint is Study 0's signed excess covariance under its locked randomization test. The
coverage experiment is a secondary endpoint and is reported as inconclusive if L1 fails. The one
boundary cell is selected using only the independent capability screen to differ on exactly one
scientifically useful axis, model family or task family. The symposium paper makes no claim about a
model population and excludes the doctoral program's mechanism decomposition, mitigation frontier,
human comparison, multilingual comparison, and benchmark-ranking study.

The five-page narrative contains only:

1. the estimand and nearest-work boundary;
2. the locked pilot replication and two controls;
3. the prespecified boundary cell;
4. one direct 95% coverage comparison if its claim gate is reached; and
5. the operational requirement to preserve call identifiers.

It uses two main visuals: one design diagram and one combined excess-covariance/coverage result
figure. Other analyses move to an appendix or artifact and cannot create claims absent from the
five-page paper. A data card, license audit, first-author contribution record, and frozen analysis
manifest are submission gates.

The work fits the symposium's topics on statistical methods for big data, mining text and
heterogeneous data, and data mining with large language models. Current rules and dates must be
rechecked at submission on the [official symposium page](https://bigdataieee.org/BigData2026/high-school-symposium/).

### Figure plan

**Figure 1:** Experimental design showing solo, exact-context target-only, joint, and permutation-
placebo conditions.

**Figure 2:** Dependence dose-response across batch sizes, with model-task heterogeneity.

**Figure 3:** Mechanism decomposition from direct interventions.

**Figure 4:** Nominal versus empirical coverage for naive and batch-aware methods.

**Figure 5:** Accuracy, dependence, and efficiency Pareto frontier for mitigations.

### Table plan

**Table 1:** Frozen model, dataset, task, prompt, and decoding matrix.

**Table 2:** Cell-level accuracy, ICC, design effect, and control contrasts.

**Table 3:** Coverage, Type-I error, interval width, and power by inference method.

**Table 4:** Mitigation costs and benefits.

## 18. Argument and evidence map

| Paper claim | Required evidence | Main falsifier |
|---|---|---|
| Joint batching creates excess dependence in the tested cells | Study 0 replication and Study 1 signed control contrasts | Exact-context or permutation-placebo groups reproduce the effect |
| The effect recurs in the fixed panel | Global randomization test, practical margins, and leave-one-family/task-out analyses | Result fails a leave-one-out analysis or is driven by one cell |
| A specific mechanism contributes | Target-outcome change plus covariance reduction from a randomized intervention | Intervention changes a local prediction but not excess covariance |
| Ignoring call IDs damages a declared estimand | Estimand-specific actual coverage, Type-I error, and influence covariance | Naive intervals remain within the frozen calibration region |
| Batch-aware analysis repairs uncertainty | Held-out coverage, Type-I error, and width gates under L5a | Corrected methods miss a frozen calibration or width gate |
| Efficient annotation mitigation is possible | Held-out L5b covariance, calibration, accuracy, and throughput or cost contrasts | No procedure meets all frozen L5b margins |

The paper should make only the claims whose required evidence survives its falsifier.

## 19. Literature plan

The latest audit searched arXiv, OpenAlex, and Semantic Scholar through 2026-09-10 using nine
formulations for Gate A. Before preregistration and again before submission:

1. rerun searches for “batch prompting,” “cross-query interference,” “within-prompt covariance,”
   “correlated LLM annotation errors,” and “clustered predicted labels”;
2. search venue proceedings for ICLR, ICML, NeurIPS, ACL, EMNLP, KDD, ICDM, and IEEE BigData;
3. trace citations forward and backward from the six closest papers in the existing audit;
4. record exact queries, dates, source failures, screening decisions, and nearest overlaps; and
5. narrow the contribution if another paper measures the same estimand.

The claim-to-prior-work matrix starts with these nearest audited papers and must be updated before
preregistration:

| Prior direction | What it establishes | Remaining distinction to test |
|---|---|---|
| [*Batch Prompting: Efficient Inference with Large Language Model APIs*](https://arxiv.org/abs/2301.08721) (2023) | Efficiency and mean task quality under batching | Same-call output and error covariance, controls, and inference |
| [*Efficient but Vulnerable: Benchmarking and Defending LLM Batch Prompting Attack*](https://arxiv.org/abs/2503.15551) (2025) | Adversarial cross-query influence in batched inference | Benign annotation dependence and downstream statistical calibration |
| [*Safety in Batches?*](https://arxiv.org/abs/2608.02681) (2026) | Safety and refusal behavior under batched or long contexts | Reference-relative label covariance under randomized re-batching |
| [*Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking*](https://openalex.org/W7154539897) (2026) | Dependence across prompts, models, temperatures, or repeated measurements | Across-item dependence induced within one annotation call |
| [*Spatially Robust Inference with Predicted and Missing at Random Labels*](https://www.semanticscholar.org/paper/5a9495212beecd2c361c34f65133e8c6c5440a10) (2026) | Inference under spatial dependence in generated outcomes | Batching as the randomized source of clusters and partial interference |
| [*Nine Judges, Two Effective Votes*](https://arxiv.org/abs/2605.29800) (2026) | Dependence among multiple judges evaluating the same item | Dependence among different items labeled in one call |

The related-work synthesis must keep five literatures distinct:

- efficient batch prompting;
- adversarial or safety-related cross-query interference;
- context and order effects in LLM inference;
- measurement error from LLM-generated labels; and
- clustered-data inference and partial interference.

## 20. Ethics, authorship, and venue compliance

- Review dataset licenses and terms before redistribution.
- Minimize direct researcher exposure to toxic or abusive text.
- Do not send private or identifying data to external APIs.
- Obtain ethics approval before recruiting human annotators.
- Report all excluded model-task cells and failed runs.
- Maintain a contemporaneous contribution log using CRediT roles.
- For the IEEE BigData High School Symposium, the high-school first author must remain the primary
  contributor and should lead the scientific decisions, analysis interpretation, and manuscript.
- If no eligible high-school first author is the actual primary contributor, do not submit this work
  to the High School Symposium; select a different venue rather than changing authorship order to
  satisfy eligibility.
- Freeze the submission dataset list, license findings, release permissions, and sensitive-data
  handling in a signed data card before confirmatory calls.
- Include Data Availability, Ethics, Author Contributions, Conflict of Interest, Funding, and AI-use
  statements in the final manuscript as required by the venue and research context.

## 21. Final standard for success

The project succeeds at PhD-paper depth if it produces a defensible answer to all four questions:

1. **Existence:** Does joint batching create dependence beyond context-length and item-difficulty
   controls?
2. **Scope:** Across which tested models, tasks, prompts, and batch sizes does it occur?
3. **Consequence:** When does it change uncertainty or scientific conclusions?
4. **Action:** Which design or analysis procedure fixes the problem, and at what cost?

The strongest possible result is not “the pilot replicated.” It is a validated account of batching
as part of the annotation data-generating process, supported by randomized evidence, direct
mechanism interventions, calibrated inference, and held-out validation.
