# Task: implement the small feasibility gate for idea K

You are the implementation owner for one isolated module in Daniel's IEEE BigData 2026 planning workspace. Read `/Users/dz/bigdata/AGENTS.md` first, then read the Candidate K sections in:

- `/Users/dz/tmp.bigdata-blind-idea-20260910/idea.md`
- `/Users/dz/tmp.bigdata-blind-idea-20260910/BLIND_IDEA_REPORT_20260910.md`
- `/Users/dz/bigdata/idea-stage/claude-novelty-20260911/CLAUDE_INDEPENDENT_NOVELTY_AUDIT.md`

Inspect the source paper `Why Large Language Models Fail at Tabular Prediction` (arXiv:2608.02412) and authoritative Transformers documentation as needed. Record the parts of its public protocol you reproduce and every divergence. Do not perform a new broad novelty review.

Edit only `/Users/dz/bigdata/gate-experiments/k/**`. Do not edit `AGENTS.md`, `ideas.md`, `MANIFEST.md`, the shared prompt files, or the A module. Do not run an LLM, download model weights, download OpenML data, provision a GPU, or make any paid API call. You may install small CPU-only development dependencies in an isolated `uv` environment and run local unit tests and synthetic/mock smoke tests.

## Scientific target

Test whether the dimensionality-related accuracy loss of a generic decoder-only LLM on in-context tabular prediction follows concentration of distances between query and demonstration-row representations, rather than ambient feature count alone.

The gate must contain two connected slices:

1. A source-paper-compatible random-projection sweep over five configurable OpenML datasets and dimensions from 2 through 64. This establishes that the implementation can reproduce the reported phenomenon later; do not execute it now.
2. A decisive matched synthetic intervention at fixed ambient dimension. The source paper already expands true 2D datasets into 4--64 dense random linear-combination columns and still observes LLM collapse, even though intrinsic dimension remains two. Therefore do not assume ambient dimension automatically concentrates raw distances. Starting from the exact same latent 2D points and labels, build three paired arms with identical ambient dimension, feature names/order, class balance, train/query rows, and fixed-width numeric formatting:
   - `sparse_zero_pad`: the two latent coordinates plus zero-valued padding columns. This is the proposed high-dimensional, low-complexity rescue condition.
   - `dense_redundant_projection`: a source-paper-style dense linear expansion of the same two coordinates. It adds no independent information and should preserve classical learnability, but may alter the LLM's row representations.
   - `independent_nuisance`: the same two informative coordinates plus independent label-irrelevant columns, scaled to create genuine raw-distance concentration.

Validate before GPU inference that the label rule and latent rows are identical across arms, token-length differences are recorded, signal-aware and flexible classical baselines remain strong, and the distance metrics have their expected directions. A missing raw-space contrast between sparse padding and dense redundant projection is not construction failure because both live on the same 2D manifold. Independent nuisance must produce a raw-space concentration contrast. If the arm invariants or the nuisance contrast fail, the future GPU run must stop.

Use the three arms to separate interpretations. A sparse-versus-dense-redundant difference tests serialization/integration and model-representation effects without adding independent noise. The independent-nuisance arm tests nuisance-induced raw-distance concentration, but any model effect there could also reflect failure to ignore nuisance. Representation-space measurements and causal row ablations are still required; do not overstate identification.

## Required implementation

Create a self-contained, typed Python 3.11 project in `gate-experiments/k` with a concise README, lockable dependency metadata, configs for a tiny network-free smoke test and the full gate, and tests. Prefer a small package plus a single CLI entry point over notebooks.

The CLI must support separate, resumable stages:

1. `prepare-synthetic`: generate all three matched arms deterministically, validate all construction invariants, compute concentration metrics in both full observed space and the shared latent 2D space, run classical baselines, and freeze JSONL/NPZ plus a provenance manifest. Full defaults should cover dimensions `2, 4, 8, 16, 32, 64`, at least five independent dataset seeds, and enough queries for paired inference. The smoke config must stay tiny.
2. `prepare-openml`: download only when explicitly invoked, apply the documented source-paper-compatible preprocessing/projection protocol, freeze source IDs/revisions/checksums and projection seeds, and never run during tests.
3. `plan`: build immutable inference jobs for random row order, nearest-first order, removal of the nearest informative-space row(s), and matched random-row removal. Use the same demonstrations and query across paired jobs. Add an optional attention-capture job but keep attention secondary.
4. `run`: load an open-weight causal model through a lazy-import Transformers backend, default configurable as `Qwen/Qwen3-8B`, plus a deterministic CPU mock backend. Run one prompt at a time if needed to capture hidden states. Disable Qwen thinking where supported. Record token spans for feature text only, pool a pre-specified hidden layer or set of layers into one representation per demonstration row and query row, and compute class prediction from a strict `A`/`B` response or well-documented label-token scores. Make dtype, device map, attention implementation, context budget, layer choice, and cache paths configurable. Refuse oversized prompts before inference.
5. `analyze`: produce JSON and a short Markdown report without loading a model. Join paired jobs and report accuracy, label probability or margin where available, raw-space and representation-space relative contrast, nearest/farthest ratio, coefficient of variation, Spearman rank agreement between informative-space and representation-space distances, paired concentrated-versus-deconcentrated effects at fixed dimension, and nearest-row-removal versus random-removal effects. If attention exists, report row-mass/rank correlations as descriptive only.
6. `smoke`: create tiny synthetic data, plan all intervention types, run a deterministic mock representation/prediction backend, analyze, verify invariants, and exit nonzero on failures. It must never access the network.

Use explicit terminology in code and reports: `distance_concentrated` means distances are more alike and therefore relative contrast is lower; `distance_deconcentrated` means relative contrast is higher. Unit-test this direction so labels cannot be accidentally reversed.

Row-span extraction must be auditable. Render a final prompt string, find unique row markers and feature-only character spans, tokenize with offset mappings when supported, and verify nonempty, nonoverlapping token spans. Demonstration representations must exclude gold label tokens. If a tokenizer cannot provide reliable offsets, fail with an actionable error instead of guessing. Store prompt hashes and span metadata, not full repeated prompts unless configured.

Treat attention as weak evidence. The causal check is removal of the nearest row(s), compared with paired random removals. Define nearest using informative-space distance, not the measured model representation, so the intervention is not selected by its own outcome. Include nearest-first ordering as a separate behavioral control, not proof of a metric mechanism.

Pre-register configurable gate summaries without forcing binary conclusions. At minimum distinguish:

- construction failure: arm invariants break, classical learnability unexpectedly collapses, or independent nuisance fails to produce the configured raw-distance contrast;
- mechanism unsupported: representation concentration differs but LLM accuracy and causal ablations do not follow it, or sparse zero padding fails as badly as dense redundant projection without a relevant representation contrast;
- mechanism supported enough to expand: stable representation-space contrast at fixed dimension, sparse padding improves accuracy over dense redundant projection in the predicted direction at high dimensions, the nuisance arm follows its concentration change, and nearest-row ablation is larger than matched random removal;
- ambiguous: any mixed pattern.

Do not use attention alone to pass the gate. Keep threshold values visible in config and report sensitivity near thresholds rather than selecting them after results.

## Documentation and handoff

Document exact CPU commands that validate construction without downloads, exact future GPU commands, expected artifacts, restart behavior, memory risks, and success/failure signals. State clearly that no pilot has run. Explain which measurements require hidden states and why vLLM is not the default backend for K. Do not implement the full 31-dataset paper grid; this module is only the small gate plus the five-dataset reproduction slice.

Before finishing, run formatting/static checks if configured, unit tests, and the network-free smoke command. Report what you created, commands run, and unresolved scientific or engineering caveats in your final response.
