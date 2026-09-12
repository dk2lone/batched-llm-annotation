# Research ideas — IEEE BigData 2026

Target: IEEE BigData 2026 High School Symposium. Deadline 2026-09-20. 5 pages including
figures, tables and references. Public data, CPU only.

Status as of 2026-09-09. Five ideas formulated, novelty-checked across arXiv, OpenAlex,
Semantic Scholar, DBLP and Google Scholar, then audited by `gpt-5.6-sol` via `codex2`.
Four died. One survives and has verified results.

Design rule applied throughout: an idea must estimate a quantity with a defined floor or
ceiling, and must contribute a procedure with a validated property (coverage, bias, or a
proven bound). Not a measurement. Not a rank-order comparison.

---

## Verdict table

| Code | Idea | Novelty | Feasible | Status |
|------|------|---------|----------|--------|
| G2 | Repository self-disagreement via provenance replay | 6 → 7 | 7 | **Selected. Evidence in hand** |
| G1 | Calibrated equivalence test for benchmark headroom | 3 | 2 | Dead. Concurrent work |
| N1 | Feature-measurement-error ceiling via SIMEX | 3 | 3 | Dead. Target is not a ceiling |
| N2 | Effective test-set size, cluster-adjusted intervals | 2 | 5 | Dead. Direct prior work |
| N3 | Partial-identification bounds under missing features | 2 | 2 | Dead. Core claim false |

Novelty and feasibility are out of 10, assigned by the audit.

---

## G2 — Provenance replay  ✅ SELECTED

**Question.** Do public repositories publish entries for the same source dataset that differ
by undocumented preprocessing, and does that difference change which model wins?

**Estimand.** The causal effect of one named preprocessing transformation on measured model
performance, under a fixed evaluation population, fixed folds and a fixed seed.

**Method.** Reconstruct each published repository entry from one checksummed source file plus
a named sequence of transformations. Require the reconstruction to reproduce the entry
exactly. Then apply each transformation as a controlled intervention in a factorial design.

The replay is the contribution. Comparing entries as they exist is observational, because
entries differ on many dimensions at once and no single effect is attributable. Reproduction
converts a survey of differences into a measurement of causes.

**Verified results (2026-09-09).**

- Canonical source: UCI `adult.data` (sha256 `5b00264637dbfec3`) + `adult.test`
  (sha256 `a2a9044bc167a35b`), 48,842 rows. The untouched `fnlwgt` column agrees with
  OpenML `did=179` on 32,561 of 32,561 rows.
- OpenML `did=179` bins `age` and `hours-per-week` into 5 levels. The rule is neither equal
  width (31.07% match) nor equal frequency. Recovered cut points:
  - `age`: 17–25 / 26–35 / 36–45 / 46–55 / 56–90
  - `hours-per-week`: 1–20 / 21–35 / 36–45 / 46–60 / 61–99
  - Reproduces both columns at **100.0000%** over 48,842 rows.
- Effect is model-dependent. 5 identical folds, accuracy:

  | model | continuous | binned | paired diff | sd | folds agree |
  |---|---|---|---|---|---|
  | logistic regression | 0.8525 | 0.8531 | −0.0005 | 0.0010 | no |
  | decision tree, depth 8 | 0.8559 | 0.8558 | +0.0001 | 0.0005 | no |
  | random forest | 0.8567 | 0.8465 | **+0.0102** | 0.0025 | yes, 5/5 |
  | hist gradient boosting | 0.8735 | 0.8733 | +0.0002 | 0.0018 | no |

  Gradient boosting bins features internally, so it absorbs the change.

- **The transformation flips the winner.** Random forest vs depth-8 decision tree, 10 paired
  folds:
  - continuous lineage (`did=1590`): forest 0.8583, tree 0.8570, diff +0.0013,
    p = 0.475, 7/10 folds favour the forest. No decision.
  - binned lineage (`did=179`): forest 0.8463, tree 0.8571, diff −0.0108,
    p = 4.84e−06, 0/10 folds favour the forest. Tree wins.

  Both are active OpenML entries named `adult`.

**Variant inventory.** Nine active entries downloaded and compared:

| did | name | ver | rows | rows with ? | difference |
|---|---|---|---|---|---|
| 179 | adult | 1 | 48,842 | 3,620 | age and hours binned to 5 levels |
| 1590 | adult | 2 | 48,842 | 3,620 | continuous, test block ordered first |
| 43898 | adult | 3 | 48,790 | 3,615 | 52 rows absent, `?` kept as a category |
| 45068 | adult | 4 | 48,842 | 0 | missing values written as `nan` |
| 31 | credit-g | 1 | 1,000 | 0 | quoted levels, leading space in target |
| 44096 | credit-g | 2 | 1,000 | 0 | amounts cast to float |
| 45058 | credit-g | 3 | 1,000 | 0 | columns reordered, numerics first |
| 46378 | credit-g | 4 | 1,000 | 0 | target moved to column 1 |
| 46918 | credit-g | 5 | 1,000 | 0 | different level vocabulary and target name |

`46378` deserves care. Its declared target is `class`, and OpenML clients read that
correctly. But `class` is the first column and the last column is `foreign_worker`. Any
loader assuming the target is the final column trains a model to predict nationality instead
of credit risk. State the scope precisely: the failure hits last-column loaders, not the
OpenML API.

**Closest prior work.** No exact match in five sources.

- Ge, Nguyen, Arnaout, *lucie: An Improved Python Package for Loading Datasets from the UCI
  ML Repository* (2024). Documents broken imports and cleaned mirrors. Does not quantify
  performance effects.
- Tschalzev et al., *A Data-Centric Perspective on Evaluating Machine Learning Models for
  Tabular Data*, arXiv 2407.02112 (2024). Shows preprocessing changes rankings. Does not
  compare lineage-linked variants of one source.
- Longjohn et al., *Benchmark Data Repositories for Better Benchmarking*, arXiv 2410.24100
  (2024). Repository design and reproducibility. No row-linked performance discrepancies.
- Kohli et al., *Towards Quantifying the Effect of Datasets for Benchmarking* (2024).
  Dataset selection, not versions of one dataset.
- Olson et al., *PMLB* (2017). Standardizes datasets because repositories require
  inconsistent preprocessing.

**Known limit.** Data management venues are indexed poorly by arXiv, OpenAlex and Semantic
Scholar. DBLP and Google Scholar were searched with no exact match. State this in the paper
rather than claiming certain novelty.

**Plan and full evidence:** https://claude.ai/code/artifact/f6df9a5d-2291-474f-944f-a0e02084756a

---

## G1 — Calibrated equivalence test for benchmark headroom  ❌

**Question.** How much headroom remains between the best reported error on a tabular
benchmark and that benchmark's Bayes error?

**Intended method.** Bootstrap both the model error and the Bayes-error estimate on the same
resamples, run a two-one-sided-test for equivalence at a stated margin, validate coverage on
Gaussian mixtures with known Bayes error.

**Why it died.**

- Chowdhury et al., *The Ceiling Is in the Channel: Auditing Learner Gaps and Measurement
  Frontiers in Clinical Prediction*, arXiv 2609.01909 (September 2026). Separates the learner
  gap from the measurement-channel ceiling, gives a cross-fitted ceiling estimator with a
  partial-identification result, validates on three real cohorts. Occupies the territory.
- Ushio, Ishida, Sugiyama, arXiv 2505.20761 (2025). My brief claimed it gives only a point
  estimate. The audit reports it uses 1,000 bootstrap resamples with 95% intervals. Its
  abstract also proves that perfectly calibrated soft labels can still give a substantially
  inaccurate estimate, which breaks the plan to manufacture soft labels on CC18.
- Hawes and Priebe (2012) already bootstrap kNN error curves for a Bayes-error interval.
- Theisen et al., NeurIPS 2021, evaluates SOTA headroom against exact Bayes optimality.
- Paraschakis et al., AutoML 2024, shows selecting the best pipeline creates winner's-curse
  bias. "Best reported model" is a selected minimum, so resampling a fixed winner is
  optimistic.

**Route to 7 if revisited.** A selection-aware upper confidence bound for
`min_j R_j − R_Bayes` with finite-sample coverage over a prespecified model set. TOST alone
cannot get there.

---

## N1 — Feature-measurement-error ceiling via SIMEX  ❌

**Question.** For a dataset whose features carry published measurement uncertainty, how much
of the remaining error is sensor noise rather than model deficiency?

**Intended method.** Add extra noise at multipliers λ, fit at each λ, extrapolate the
performance curve to λ = −1 to estimate noise-free performance.

**Own experiment (2026-09-09).** On synthetic data with a known clean-feature ceiling:

| estimate | R² | error |
|---|---|---|
| true clean-data performance | 0.9804 | — |
| naive score on noisy features | 0.6014 | −0.3790 |
| SIMEX, quadratic extrapolant | 0.8389 | −0.1415 |
| SIMEX, rational extrapolant | 0.9329 | −0.0475 |

The method recovers 87% of the gap. But the number it recovers is the clean-data performance
of **one pipeline**, not a model-independent ceiling. My script literally computed the
pipeline's own clean-feature score.

**Why it died.**

- Same paper as G1: arXiv 2609.01909 covers the measurement-channel ceiling.
- Gell, Eickhoff, Omidvarnia et al., *How Measurement Noise Limits the Accuracy of
  Brain–Behaviour Predictions*, Nature Communications (2024). Measurement reliability as a
  limit on achievable accuracy, across datasets and algorithms.
- Luijken et al. (2019). Predictor measurement error and its effect on discrimination,
  calibration, Brier score.
- Kanu et al. (2023), SIMEX review. Documents sensitivity to the extrapolation function and
  degraded coverage.
- Extrapolation to λ = −1 is not identified. Many curves agree on [0, 2] and diverge at −1.
  Noise can also act as regularization, so performance need not be monotone.
- Parameterization trap: standard SIMEX adds variance `λσ²`. If λ scales the standard
  deviation instead, total variance is `(1+λ²)σ²` and λ = −1 removes nothing.

**Route to 7 if revisited.** Shape-constrained partial-identification bounds for zero-noise
risk under an explicit measurement-error class, with a proof that the interval covers the
clean-pipeline risk without assuming a curve shape.

---

## N2 — Effective test-set size  ❌

**Question.** Benchmark test items are not independent. How much does that inflate reported
confidence?

**Intended method.** Detect dependence groups, estimate the intraclass correlation, report
cluster-robust intervals, validate coverage by simulation.

**Why it died.**

- Hong, Lim, Bae, *Beyond Point Estimates: Reliable Evaluation of Prediction Performance
  Metrics under Clustered Data*, arXiv 2606.03656 (2026). Verified against the arXiv API.
  Cluster-robust sandwich intervals, hypothesis tests, paired model comparisons, power and
  sample size, simulation coverage, real-data application. The whole idea.
- Chandrahas, *evalci*, arXiv 2607.04429 (2026). Already ships `cluster_ci`, which resamples
  whole clusters.
- Rakotomalala et al. (2006). Accuracy estimation with clustered data.
- `deff = 1 + (m−1)ρ` assumes equal cluster sizes. With unequal sizes the correct form is
  `1 + ρ((Σm²/Σm) − 1)`.
- Near-duplicate detection establishes feature similarity, not shared sampling noise.
  Clustering deterministic model errors by input similarity mechanically creates within-group
  agreement.

**Route to 7 if revisited.** A dependence-graph sensitivity procedure giving bounds valid
uniformly over every cluster partition consistent with verified source links.

---

## N3 — Partial-identification bounds under missing features  ❌

**Question.** When a test set has missing feature values, how wide is the interval of test
accuracies the data admits?

**Intended method.** Propagate intervals through a tree ensemble, following both branches at
a split on a missing feature, to get exact min and max.

**Own experiment (2026-09-09).** The claim of exactness is false. Per-tree extrema do not
compose, because one latent value must satisfy every tree simultaneously. Measured on
synthetic data: the per-tree bound contains the exact set but is 1.1× too wide.

Counterexample: tree 1 outputs 1 when x ≤ 0 else 0; tree 2 outputs 0 when x ≤ 0 else 1. Each
has maximum 1, so summing gives 2. But the ensemble output is always 1.

**Why it died.**

- Chen et al., *Robustness Verification of Tree-Based Models*, arXiv 1906.03849, NeurIPS
  2019. Proves verification is linear-time for one tree and **NP-complete** for ensembles.
- Mišić, *Optimization of Tree Ensembles*, arXiv 1705.10883 (2020). MILP formulations for
  exact constrained ensemble maxima.
- Devos, Meert, Davis, *Versatile Verification of Tree Ensembles*, ICML 2021. The Veritas
  anytime solver converges to exact constrained optima.
- Liu, Zachariah, Stoica, *Robust Prediction When Features Are Missing*, arXiv 1912.07226
  (2019).
- Also: a Cartesian product of marginal feature ranges includes impossible combinations, so
  the bound is not sharp for the joint support. And comparing one model's interval width
  against a published A-vs-B gap is invalid; the right object is a joint bound on the
  difference using the same completion for both models.

**Route to 7 if revisited.** Restrict to bounded-treewidth feature-interaction graphs and
give a dynamic program with a proof of exact polynomial-time completion bounds. A genuine
algorithmic result, but ambitious for this deadline.

---

## Candidates killed before the audit

| Code | Idea | Killed by |
|---|---|---|
| K1 | Label-reliability ceiling on benchmark scores | *Has Automated Essay Scoring Reached Sufficient Accuracy? Deriving Achievable QWK Ceilings from Classical Test Theory* (2026); *Bias-Corrected Ceilings of Emotion Predictability from Human Label Variation Based on Instance-Level Fano Bounds* (2026) |
| K2 | Entropy ceiling for sequential recommendation | *On the Accuracy Limits of Sequential Recommender Systems: An Entropy-Based Approach* (2026) |
| K3 | Drift detector false alarm floor | *When Drift Detectors cry Wolf: False Alarm Rates in continuous ML Monitoring* (2026) |
| K4 | Fairness disparity detection floor | *Bias or Insufficient Sample Size? Improving Reliable Estimation of Algorithmic Bias for Minority Groups* (2025); *Size-adaptive Hypothesis Testing for Fairness* (2025) |
| K5 | Propensity-weighted imputation evaluation | Not killed by search. Killed on judgment: it ports inverse-propensity evaluation from recommender systems (Schnabel et al. 2016) |
| K6 | Dataset age as a predictor of which model wins | Age cannot be separated from domain, size and curation practice in an observational comparison |
| K7 | Temporal graph predictability ceiling | A Fano bound limits top-1 probability; TGB reports MRR, which depends on its negative-sampling protocol. Dividing one by the other is meaningless |
| K8 | Calibration error bias correction | Roelofs et al. (2022) — flagged from memory, not search-verified |

---

## Scripts

All in `scratchpad/provenance/` and `scratchpad/`:

- `replay_effect.py` — replay fidelity check plus the four-model paired effect table
- `rank_flip.py` — the ten-fold paired test that produces the rank flip
- `check_n1.py` — SIMEX recovery on synthetic data with a known ceiling
- `check_n3.py` — proof that per-tree bounds do not compose
- `ideas_v3.md` — the brief sent to the audit
- `codex_out.txt` — the full 300-line audit response

---

## Additional directions — 2026-09-10

Three additional topic 2/5/7 directions survived a separate OpenAlex, arXiv, Semantic
Scholar, and Claude Opus 5 audit. Full estimands, experiments, closest-work comparisons,
risks, search status, and killed candidates are in [`idea-generation-report.md`](idea-generation-report.md).

| Code | Direction | Novelty | Depth | Status |
|---|---|---:|---:|---|
| I1 | Factorial pretraining exposure: item content versus co-occurrence | 7 | 7 | Pass |
| I2 | Controlled validation-data exposure and agentic Type-I error | 7 | 7 | Pass |
| I3 | Contamination-immune agentic tests and the price of open-endedness | 6 | 7 | Pass, highest collision risk |

I1 is the recommended recommendation/LLM project. I2 is the strongest statistical critique.
I3 is cheapest to execute, but I2 and I3 should not be combined into one five-page paper.
