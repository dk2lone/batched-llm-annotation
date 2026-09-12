# Small-gate plan for ideas K and A

Updated: 2026-09-11

Both gates have run on real open-weight models. Gate K failed its mechanism test, while Gate A
returned `PROCEED`. The checked-in smoke outputs remain pipeline checks rather than experimental
evidence.

## Execution order

Gate K is complete enough for a stop decision. Gate A completed its primary and triggered filler
control. Develop A as the surviving direction; do not rerun either completed gate.

| Gate | First GPU checkpoint | Continue when | Full prepared workload |
|---|---:|---|---:|
| K | 48 sparse 2D prompts | Failed mechanism gate after passing anchor; stop | 1,266/1,296 jobs preserved |
| A | Passed: 2,375 solo and joint prompts | Triggered and completed filler control | 8,375/8,375 prompts |

## K: representation mechanism

**Question:** Does generic LLM tabular failure follow distance concentration in the model's
internal representation, rather than raw column count alone?

Each latent 2D classification problem is rendered in three matched forms. They have the same
underlying rows, labels, order, feature names, and numeric width:

1. `sparse_zero_pad`: the two informative coordinates followed by zeros.
2. `dense_redundant_projection`: dense linear projections of those same two coordinates.
3. `independent_nuisance`: the two coordinates followed by label-irrelevant noise.

The dimensions are 2, 8, 32, and 64, with 60 demonstrations, 16 queries, and three random seeds.
The primary evidence combines accuracy, hidden-state distance contrast at layer 24, and removal
of the three latent-nearest demonstrations against matched random removal. Nearest-first ordering
is a behavioral control. Attention is excluded from the first gate.

The CPU construction gate has passed for all 36 arm/dimension/seed datasets. The frozen plan has
576 base jobs, 288 nearest-removal jobs, 288 random-removal jobs, and 144 ordering-control jobs.
The design follows the 60-row setup and unscaled random projections in the source study,
[Why In-Context Learning Fails for Tabular Data](https://arxiv.org/html/2608.02412), while adding
the matched causal arms needed to test the proposed mechanism.

Stop K when the 2D model-capability checkpoint fails. After a passing checkpoint, expand only if
the high-dimensional representation contrast, accuracy contrasts, nuisance comparison, and
nearest-row ablation agree under the thresholds frozen in `k/configs/full.toml`.

### K result

Mistral Small 3.1 24B Base passed the anchor at 89.6% accuracy with no parse failures, so the
mechanism result is interpretable. The frozen high-dimensional gate then returned
`mechanism_unsupported`: sparse-over-dense representation contrast was 0.305 against a 1.2
threshold, sparse-over-nuisance contrast was 0.578 against 1.2, and sparse-minus-dense accuracy
was -0.125 against a required +0.05. Modal stopped with 1,266 of 1,296 jobs saved, but all
rejecting conditions are complete and the missing ancillary jobs cannot change the verdict.
See `k/artifacts/mistral-24b-base/analysis/COMPLETION_AUDIT.md`.

## A: batch dependence

**Question:** When an LLM labels documents in batches, are errors within the same prompt
correlated strongly enough to invalidate the usual independent-item uncertainty estimates?

The gate freezes 2,000 naturally sampled Civil Comments validation records and 4,000 disjoint
filler records. Each target receives three conditions:

1. `solo`: the target appears alone and is labeled alone.
2. `joint`: 16 targets appear together and all 16 are labeled in one response.
3. `filler`: the target appears with 15 length-matched filler documents but only the target is
   labeled.

Three independent batch assignments separate a real shared-prompt effect from accidental item
grouping. The primary statistic is the mean within-batch error ICC across assignments, with
whole-batch bootstrap intervals and fake-batch permutation tests. Solo is the no-shared-context
placebo. Filler tests whether prompt length or neighboring text, rather than joint prediction,
explains the result. Position, accuracy change, parse failures, and a failures-as-errors
sensitivity analysis are also recorded.

The primary result is uninterpretable unless every joint assignment contains at least 20 errors
and the joint parse-failure rate is at most 0.02. A point ICC of at least 0.02 triggers the filler
control. A claim-ready `PROCEED` verdict additionally requires the conservative interval to
exclude zero and every assignment's fake-batch permutation p-value to be below 0.05.

Civil Comments is used from its 97,320-row validation split rather than its 1.8-million-row
training split; these sizes come from the
[official dataset card](https://huggingface.co/datasets/google/civil_comments/blob/main/README.md).

### A result

Qwen3-8B returned aggregate joint-error ICC 0.0754 with conservative 95% interval
[0.0182, 0.1314]. All three fake-batch permutation tests gave `p = 0.0010`; the matched-length
filler placebo ICC was 0.0048. The frozen verdict is `PROCEED`. The result is limited to one
model-dataset cell. See `a/artifacts/full/analysis/report.md`.

## Compute boundary

Gate A uses `Qwen/Qwen3-8B` in non-thinking, greedy mode. It uses vLLM and has a much shorter
8,192-token cap; 24 GB may work, while 48 GB gives useful batching headroom.

Gate K could not use Qwen3-8B because 576 of its frozen prompts exceeded the model's 32,768-token
context. Its final run used `mistralai/Mistral-Small-3.1-24B-Base-2503` on H100 GPUs. The Base
checkpoint was required because the Instruct checkpoint produced off-label first tokens despite
high restricted-choice accuracy.

Before loading K's model, run the tokenizer-only preflight. If any prompt exceeds 32,768 tokens,
reduce the design and rebuild the frozen plan. Do not enable YaRN for this gate because that would
change the model condition.

## Prepared files

- `a/README.md`: exact A setup, staged GPU commands, outputs, and stopping rules.
- `k/README.md`: exact K setup, tokenizer preflight, staged GPU commands, and outputs.
- `k/artifacts/full/`: original Qwen anchor evidence and frozen CPU-generated K plan.
- `k/artifacts/mistral-24b/`: failed-format Mistral Instruct anchor evidence.
- `k/artifacts/mistral-24b-base/`: real-model K results, analysis, and completion audit.
- `a/artifacts/full/`: real-model A data, plan, raw predictions, analysis, and failed-attempt archive.
- `a/artifacts/smoke/` and `k/artifacts/smoke/`: mock-only pipeline checks.

Gate A pins Civil Comments revision `f2970eb3a55777454c94069077cc8d9b5866312d` and Qwen3-8B
revision `b968826d9c46dd6066d109eabc6255188de91218`. Do not compare results across runs with
different resolved revisions.
