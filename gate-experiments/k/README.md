# Gate K: representation concentration and tabular ICL failure

This package tests whether a generic language model loses tabular in-context classification
ability when the same 2D signal is rendered in representations with different distance
geometry. It records both behavior and hidden-state distances.

The real-model gate has run and the mechanism is unsupported for Mistral Small 3.1 24B Base.
The model passed the 2D capability anchor at 89.6% accuracy with no parse failures, but failed
both frozen representation-contrast conditions and the sparse-minus-dense accuracy condition.
See `artifacts/mistral-24b-base/analysis/REPORT.md` and `COMPLETION_AUDIT.md`. Do not expand Gate K
to external datasets.

`artifacts/smoke/` still contains deterministic mock results that test only the pipeline.

## 1. CPU verification and frozen plan

From this directory:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy
uv run gatek prepare-synthetic --config configs/full.toml --out artifacts/full --strict
uv run gatek plan --config configs/full.toml --out artifacts/full
```

The first three checks validate the package. `prepare-synthetic` reconstructs the three matched
arms and checks their labels, latent rows, formatting, geometry, and signal-aware solvability.
`plan` freezes 1,296 paired jobs. Neither command uses the network or a model.

Expected plan counts are 576 base-order jobs, 288 nearest-removal jobs, 288 matched random-removal
jobs, and 144 nearest-first ordering controls.

## 2. GPU environment and exact context preflight

This section preserves the original Qwen execution procedure. Qwen was rejected after preflight
and anchor testing: 576/1,296 prompts exceeded its 32,768-token context, and 38/48 anchor outputs
had off-label greedy first tokens. The completed mechanism test instead used the pinned Mistral
Base configuration in `modal_gate.py` and `artifacts/mistral-24b-base/`.

On a Linux CUDA host:

```bash
uv sync --extra llm --extra dev
uv run gatek preflight-tokenizer --config configs/full.toml --out artifacts/full --yes-download
```

The preflight downloads tokenizer files only, renders every frozen prompt through Qwen's chat
template, records the resolved revision, and writes `artifacts/full/plan/tokenizer_preflight.json`.
It exits nonzero if any prompt exceeds the 32,768-token limit. Do not load the model until this
passes.

## 3. Model-capability anchor

```bash
uv run gatek run --config configs/full.toml --out artifacts/full --backend transformers --phase anchor
uv run gatek analyze --config configs/full.toml --out artifacts/full
```

The anchor executes 48 sparse 2D base-order jobs: 16 queries under each of three seeds. Stop if
accuracy is below 0.65 or parse failures exceed 0.02. A passing anchor produces
`insufficient_results`, which means the model is suitable and the high-dimensional jobs remain.

Results append to `artifacts/full/run/results.jsonl`; completed job IDs are skipped on restart.

## 4. Full mechanism gate

```bash
uv run gatek run --config configs/full.toml --out artifacts/full --backend transformers --phase all
uv run gatek analyze --config configs/full.toml --out artifacts/full
```

The run command skips the completed 48 anchors and executes the remaining 1,248 jobs. Analysis
writes `artifacts/full/analysis/analysis.json` and `REPORT.md`.

Expand the project only for `mechanism_supported_enough_to_expand`. This requires the frozen
high-dimensional representation-distance contrast, sparse-versus-dense accuracy gain, nuisance
direction, and nearest-removal selectivity to agree. `mechanism_unsupported` stops the idea;
`ambiguous` calls for diagnosing the failed condition before spending on external datasets.

Do not enable attention capture or OpenML in this first gate. They are optional follow-ups and
would add cost before the synthetic causal question is answered. Do not change thresholds after
looking at predictions; changed configs need a separate output directory.
