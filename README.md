# When LLM Labels Fail Together

**Dependence and uncertainty in batched LLM text annotation.**

HanRui Zhang · Marc Garneau Collegiate Institute

Accepted to the IEEE BigData 2026 High School Symposium.

[Paper](manuscript/main.pdf) · [Results](#results) · [Reproduction](#getting-started) · [Release artifacts](https://github.com/dk2lone/batched-llm-annotation/releases)

Batch prompting lets an LLM label several texts in one response. We ask whether those labels also share errors, and whether treating every labeled row as independent understates uncertainty.

Across four model families and four classification tasks at batch size 16, errors co-occurred more often in joint responses than in separate calls with the same visible context. Mean excess error covariance was **0.00961**, and 13 of 16 model–task combinations exceeded the prespecified practical threshold. A separate recovered repeated-sampling analysis found item-level 95% interval coverage as low as **77.8%**. That coverage result is secondary sensitivity evidence because it required a changed output contract after execution.

The practical implication is to retain a call identifier with each label and account for call membership when estimating uncertainty.

## Motivation

Annotation accuracy describes how often labels disagree with a reference. It does not tell us whether errors occur together. If several errors share one response, the number of labeled rows can overstate how much independent information the dataset contains.

Comparing batched annotation with one-text prompts alone cannot isolate this effect: the batched model also sees more text. Our main comparison therefore holds the neighboring texts and their order fixed while changing whether the requested labels share a generated response.

## Experimental Design

The primary study crosses four checkpoints with four English classification tasks. Every model sees the same 8,192 confirmation items for a task, under three randomized, length-balanced assignments. No model–task combination is removed based on accuracy or parse rate.

| Model | Toxicity | Natural-language inference | Sentiment | Topic |
|---|---|---|---|---|
| Qwen3-8B | Civil Comments | MultiNLI | IMDb | AG News |
| Granite-3.3-8B-Instruct | Civil Comments | MultiNLI | IMDb | AG News |
| Llama-3.1-8B-Instruct | Civil Comments | MultiNLI | IMDb | AG News |
| Mistral-7B-Instruct-v0.3 | Civil Comments | MultiNLI | IMDb | AG News |

The matched conditions are:

| Condition | Visible input | Requested output |
|---|---|---|
| Joint | A group of 16 texts | All 16 labels in one response |
| Exact-context target-only | The identical texts in the identical order | One target label per call, repeated for all 16 targets |
| Solo | One text | One label |
| Fake batches | Labels already generated in separate solo calls | Regrouped on CPU as a negative control |

An error means disagreement with the dataset reference label. The primary endpoint is **joint error covariance minus exact-context error covariance**, averaged equally across the 16 model–task combinations. Positive values indicate excess co-occurrence of errors when labels share a response. The analysis uses 4,999 paired randomization replays, Holm adjustment across the 16 tests, and a separately specified practical threshold of 0.002.

The matched comparison estimates the combined effect of requesting and generating multiple labels together. It does not isolate a particular decoding or reasoning mechanism.

## Results

### Dependence across models and tasks

All 16 excess covariance estimates were positive and rejected the sharp regrouping null after Holm adjustment. Thirteen also met the 0.002 practical threshold, spanning all four model families and all four tasks.

| Primary endpoint | Result |
|---|---:|
| Mean joint-minus-exact error covariance | 0.009610 |
| 95% randomization-based shift interval | [0.009360, 0.009857] |
| One-sided replay p-value | 0.0002 |
| Point estimates meeting the separate 0.002 threshold | 13 / 16 |

The interval applies to the fixed outputs and replay design. The p-value is at the 4,999-replay resolution floor. Neither quantity estimates variation across all possible LLMs or tasks.

All leave-one-model-out and leave-one-task-out means remained positive. Four prespecified malformed-output treatments gave combined effects from 0.00752 to 0.01550. A separate sensitivity substituting 3,002 linked recovered calls reduced the estimate to 0.00790 while preserving the recurrence conclusion.

See the [primary report](study-120h/extension-b3/analysis/report.md), [machine-readable results](study-120h/extension-b3/analysis/analysis.json), and [recovered sensitivity](study-120h/extension-b3/repaired-analysis-20260918T041906Z/REPORT.md).

### Batch size and information loss

The batch-size experiment repeats the matched comparison at sizes 2, 4, 8, 16, and 32 in four prespecified combinations: Qwen/Civil Comments, Granite/IMDb, Llama/Civil Comments, and Mistral/MultiNLI.

Average standardized excess dependence stayed positive at every tested size, from 0.07948 at size 2 to 0.04455 at size 32. Its slope was negative in three combinations and positive in Granite/IMDb. Pairwise dependence weakened on average after size 4, but the larger number of labels sharing each call can still increase the total variance penalty.

The total joint-call design effect, the joint-minus-control dependence contrast, and the variance ratio measured across fresh annotation runs are separate quantities. See the [analysis definitions](analysis/README.md#results-and-interpretation) for how each is calculated.

These batch-size summaries are descriptive. The saved report also contains block-bootstrap intervals, but resampling assignments separately breaks recurring item identity, so those intervals do not support the paper's main uncertainty claims.

See the [batch-size report](study-120h/extension-b5/run-01/analysis-final-a/REPORT.md), [result arrays and summaries](study-120h/extension-b5/run-01/analysis-final-a/ANALYSIS.json), and [manuscript discussion](manuscript/main.tex).

### Recovered sensitivity: confidence-interval coverage

Each selected combination has 250 reference replicates and 400 evaluation replicates. Every replicate uses 512 item appearances and 32 fresh batch-16 calls. The table shows how often nominal 95% intervals contained the estimated target disagreement rate.

| Model / task | Item-level Wald | Call-aware t | Whole-call bootstrap |
|---|---:|---:|---:|
| Granite / IMDb | 77.75% | 93.00% | 91.75% |
| Qwen / Civil Comments | 88.00% | 96.00% | 95.25% |
| Mistral / MultiNLI | 90.75% | 95.25% | 94.75% |
| Llama / Civil Comments | 96.75% | 96.50% | 94.00% |

The original analysis could not establish coverage because outputs were malformed or truncated. A post-execution amendment recovered 1,095 of 83,200 calls using structured decoding and revised output limits. The original result remains `coverage_unavailable_original`; the table is the separately reported recovered sensitivity. It does not establish that call-aware intervals are calibrated in every setting.

See the [revised coverage report](study-120h/extension-b4/revised-analysis-20260918T034636Z/REPORT.md), [machine-readable results](study-120h/extension-b4/revised-analysis-20260918T034636Z/REVISED_ANALYSIS.json), and [recovery amendment](study-120h/amendments/20260918T013521Z-structured-output-recovery.md).

## What's in This Repository

| Component | Location | Contents |
|---|---|---|
| Camera-ready paper | [`manuscript/`](manuscript/) | Five-page PDF, LaTeX source, references, figures, generated tables, and source archive |
| CPU reproduction package | [`analysis/`](analysis/) | Pinned dependencies, analysis commands, prompts, parsing, and numerical tests |
| Analysis-ready evidence | [`analysis/evidence/`](analysis/evidence/) | Error arrays, randomization replays, bootstrap seeds, dataset indices, and manifests |
| Protocol and revisions | [`analysis/protocol.json`](analysis/protocol.json) | Exact prompts, model and dataset revisions, decoding settings, and recovery constraints |
| Original implementation | [`study-120h/`](study-120h/) | Execution code, sealed protocols, schedules, compact results, and recovery records |
| Frozen designs | [`EXTENSION_PROTOCOL.md`](study-120h/EXTENSION_PROTOCOL.md), [`B5_PROTOCOL.md`](study-120h/B5_PROTOCOL.md) | Historical protocols preserved with their seals and amendments |
| Discovery pilot | [`gate-experiments/a/`](gate-experiments/a/) | Earlier Qwen/Civil Comments feasibility evidence, separate from confirmation |
| Research history | [`plan-120h-a100-rebased.md`](plan-120h-a100-rebased.md), [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md) | Planning history; older status passages describe their own dates |

Older sealed reports preserve their original wording. Use the camera-ready paper and [analysis instructions](analysis/README.md) for the current interpretation, including the distinction between tests against zero and the separate 0.002 reporting threshold.

## Getting Started

Reproducing the saved analyses requires **Python 3.11–3.13** and [uv](https://docs.astral.sh/uv/). The package includes the analysis-ready error arrays and needs no GPU, model weights, API key, dataset download, or new model calls.

1. Clone the repository and enter the analysis directory:

   ```sh
   git clone https://github.com/dk2lone/batched-llm-annotation.git
   cd batched-llm-annotation/analysis
   ```

   This gives you the code, stored evidence, protocols, and reports needed for CPU reproduction.

2. Install the pinned dependencies and reproduce the analyses:

   ```sh
   uv sync --locked --extra dev
   uv run reproduce all
   ```

   This writes covariance, batch-size, repeated-sampling, reference-label sensitivity, and figure outputs under ignored `analysis/outputs/`. Success produces the saved-result summaries and figures. Missing or changed evidence raises an error.

3. Run the numerical and code checks:

   ```sh
   uv run pytest
   uv run ruff check .
   ```

   Success is a passing test suite and no Ruff errors. These checks verify analysis and protocol consistency conditional on the stored evidence.

### Individual analyses and evidence limits

The [analysis README](analysis/README.md) documents individual commands, optional dataset checks, and interpretation limits. For example, from `analysis/`:

```sh
uv run reproduce covariance-grid
uv run reproduce batch-size
uv run reproduce repeated-sampling
```

These commands write their respective result summaries to `outputs/` without running models. The stored arrays reproduce the statistical analyses but do not independently validate the original model execution or parsing. Dataset text, raw model responses, and private cloud configuration are excluded from the CPU package.

Original execution code and compact provenance remain in `study-120h/`. Large earlier run archives are available through the [full-grid execution release](https://github.com/dk2lone/batched-llm-annotation/releases/tag/v0.4.0-extension-b3), [analysis release](https://github.com/dk2lone/batched-llm-annotation/releases/tag/v0.5.0-extension-a9), and [batch-size release](https://github.com/dk2lone/batched-llm-annotation/releases/tag/v0.6.0-b5). Rerunning the models is a separate operation requiring the pinned checkpoints, dataset access, serving environment, and compute.

## Build the Paper

From the repository root:

```sh
cd manuscript
python3 build_tables.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Table generation formats the included saved summaries and makes no model calls. A successful build produces the five-page `main.pdf`, including references. The [source archive](manuscript/source.zip) is also available.

## Scope and Limitations

- **Model and task scope.** The main grid covers four open-weight models near the 7–8B scale and four English classification tasks. It does not establish behavior for proprietary models, larger models, other languages, or generative annotation.
- **Reference labels.** Reported errors are disagreements with dataset labels. Those labels are not assumed to be perfect semantic ground truth.
- **Mechanism.** Exact-context controls separate joint annotation from context alone, but do not isolate which part of the joint-generation policy creates dependence.
- **Batch-size inference.** Four selected combinations and recurring items across assignments do not establish a universal scaling law.
- **Coverage recovery.** The repeated-sampling results mix retained original outputs with linked recoveries under a revised output contract. They remain secondary sensitivity evidence.

## Citation

```bibtex
@misc{zhang2026llmlabels,
  title  = {When LLM Labels Fail Together},
  author = {Zhang, HanRui},
  year   = {2026},
  url    = {https://github.com/dk2lone/batched-llm-annotation}
}
```

## License

The CPU analysis code is covered by [analysis/LICENSE](analysis/LICENSE). Dataset text and model weights must be obtained from their original providers under their respective licenses.
