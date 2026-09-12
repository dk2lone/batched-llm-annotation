# Task: implement the small feasibility gate for idea A

You are the implementation owner for one isolated module in Daniel's IEEE BigData 2026 planning workspace. Read `/Users/dz/bigdata/AGENTS.md` first, then read the Candidate A sections in:

- `/Users/dz/tmp.bigdata-blind-idea-20260910/idea.md`
- `/Users/dz/tmp.bigdata-blind-idea-20260910/BLIND_IDEA_REPORT_20260910.md`
- `/Users/dz/bigdata/idea-stage/claude-novelty-20260911/CLAUDE_INDEPENDENT_NOVELTY_AUDIT.md`

Edit only `/Users/dz/bigdata/gate-experiments/a/**`. Do not edit `AGENTS.md`, `ideas.md`, `MANIFEST.md`, the shared prompt files, or the K module. Do not run an LLM, download model weights, download Civil Comments, provision a GPU, or make any paid API call. You may install small CPU-only development dependencies in an isolated `uv` environment and run local unit tests and mock-data smoke tests.

## Scientific target

Test whether asking one model to label 16 documents in one prompt induces positive within-batch covariance in binary annotation errors, beyond an equally long prompt that asks for only one target label.

The fixed first gate is:

- 2,000 Civil Comments documents from `google/civil_comments`
- binary toxicity gold label at `toxicity >= 0.5`
- configurable default model `Qwen/Qwen3-8B`
- greedy decoding
- solo `k=1`, joint batch `k=16`, and a length-matched filler condition
- three independent randomized assignments for the joint and filler conditions
- identical target documents in every condition
- primary output: within-real-batch ICC of the binary error indicator at `k=16`
- controls: fake-batch permutations, output position, parse-failure rate, and length-matched filler
- original decision thresholds: proceed at `rho >= 0.02`; bounded-negative-result candidate when the CI lies inside `[0, 0.01]` and filler shows no material accuracy gap; do not automatically claim batching when filler explains the observed accuracy loss

Preserve these thresholds, but distinguish a point-estimate threshold from evidence that the confidence interval excludes zero. Do not silently turn an ambiguous result into a pass or failure.

## Required implementation

Create a self-contained, typed Python 3.11 project in `gate-experiments/a` with a concise README, lockable dependency metadata, configs for a tiny mock smoke test and the full gate, and tests. Prefer a small package plus a single CLI entry point over notebooks.

The CLI must support these separate, resumable stages:

1. `prepare-data`: download only when the user explicitly runs it; deterministically select and freeze 2,000 target records plus a disjoint filler pool; write JSONL and provenance with dataset revision when available, row IDs, SHA-256 hashes, sampling seed, label prevalence, and selection rules. Make label-balanced versus natural-prevalence sampling configurable and explain why the full-gate default was chosen.
2. `plan`: make an immutable job manifest without loading a model. It must assign the same targets to three independently randomized joint batches, preserve slots, and produce filler prompts whose 15 unrelated context documents are drawn from the disjoint pool and approximately length-match the corresponding joint prompt. Prevent duplicates inside a prompt. Solo need not be repeated if its deterministic prompt is identical.
3. `run`: execute only the planned jobs and append raw results safely so interruption can resume without recomputing completed job IDs. Use a lazy-import vLLM backend suitable for a Linux CUDA host and a deterministic mock backend for CPU verification. Keep model ID, tensor parallelism, dtype, maximum context length, seed, batch scheduler settings, and output path configurable. Never import vLLM merely to run analysis or tests.
4. `analyze`: consume raw prediction JSONL without loading a model. Produce machine-readable JSON and a short Markdown report. Include accuracy by condition and assignment, parse-failure rates, paired accuracy deltas, per-assignment ICC, an explicitly documented aggregate ICC, whole-batch bootstrap intervals, design effects and effective sample sizes, fake-batch permutation nulls and p-values, and position effects. Avoid treating repeated appearances of the same document across assignments as independent without discussion or an appropriate aggregation strategy.
5. `smoke`: create synthetic local records, plan every condition, run the mock backend, analyze, and exit nonzero if schema/invariants fail. It must not access the network.

Each raw output row should contain enough information to audit the result: run and job IDs, condition, assignment, real batch ID, target document ID, slot, gold and predicted labels, parse status, error indicator, prompt/output hashes, context length in characters and tokens when available, model/decoding configuration, and timing. Retain raw model text in a separate field.

The joint prompt must request one label per stable document ID in strict JSON. The filler prompt must present the target plus 15 unrelated documents but request exactly one target label. Disable Qwen thinking through the chat template where supported. Parsing must be strict, report missing/duplicate/unknown IDs, and count malformed output as a parse failure rather than silently repairing it. Ensure prompt construction and parser behavior are unit-tested.

ICC code must handle binary outcomes, negative estimates, and equal or near-equal batch sizes without crashing. Resample whole batches for the cluster bootstrap. Fake-batch inference must permute batch membership within assignment while preserving batch sizes. Include a simulation test that plants shared batch effects and a null test; do not claim the estimator is unbiased merely because a single simulation passes.

## Documentation and handoff

Document exact CPU commands that demonstrate preparation without downloads and exact future GPU commands, expected files, restart behavior, and success/failure signals. State clearly that no pilot has run. Give a realistic token-count preflight and refuse before inference if a prompt exceeds the configured context budget. Do not add full-project experiments such as DSL/PPI coverage simulations; this module is only the small gate.

Before finishing, run formatting/static checks if configured, unit tests, and the network-free smoke command. Report what you created, commands run, and any unresolved scientific or engineering caveats in your final response.
