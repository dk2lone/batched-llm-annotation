# IEEE BigData 2026 research-idea handoff

Updated: 2026-09-12.

## Current handoff: Step 1 complete; model and task claims narrowed

The current direction remains **A: dependence among errors when one LLM prompt labels multiple
text records**. Gate A passed as a one-cell feasibility pilot, Gate K is stopped, CPU-only Step 0
is sealed, and the paid solo-only Step 1 screen is **complete**. This folder has no Git repository,
branch, or PR. No joint, exact-context, filler, fake-batch, serial, covariance, or repeated-sampling
model output has been generated in the 120-hour study.

Read `AGENTS.md` first, then `study-120h/STEP1_STATUS.md`,
`study-120h/step1-out/screen/SCREEN.md`, and
`study-120h/step1-out/benchmark/THROUGHPUT.md`. The Step 1 exit gate passes 10 of 10 items.
The Step 0 freeze remains byte-identical: 70 files, manifest hash
`8c9ca2a6a18d3178b91245d3f44c2a2c3776724a6f70beabc5723553b6c38171`, and protocol
content hash `218c5bcaa30f4f93c8d516f85f29ad8b7fb3be90f40238c2f3faf837ce611e15`.

Step 1 produced 10,240 first-response solo calls across five executed model-task screens. All
eight exact-tokenizer preflight reports passed with zero prompts over the frozen context budget.
The frozen screen outcomes were:

| model | task | parsable | accuracy over all calls | outcome |
|---|---|---:|---:|---|
| Qwen3-8B | Civil Comments | 0.9897 | 0.8379 | PASS |
| Qwen3-8B | AG News | 0.9995 | 0.6270 | FAIL_CAPABILITY |
| Mistral-7B-Instruct-v0.3 | Civil Comments | 0.9951 | 0.6909 | FAIL_CAPABILITY |
| Mistral-7B-Instruct-v0.3 | AG News | 0.9946 | 0.6211 | FAIL_CAPABILITY |
| Qwen3-8B | MultiNLI fallback | 0.9990 | 0.7402 | PASS |

Mistral's failures activated the one permitted second-model replacement. The pinned OLMo-2
replacement generated zero outputs because vLLM refused the frozen `max_model_len=8192` against
the checkpoint's declared 4,096-token limit. The serving setting cannot be changed after model
outputs exist. The replacement limit is exhausted, so Phi-3.5 cannot be tried under this freeze.
The frozen fallback rule keeps AG News as the named second task but labels its failing cells
uninformative. Both the model and task claims are narrowed; do not write "across models and
tasks" from this design.

The conservative local meter is 0.190 A100-hours. Modal's billing report for 2026-09-12 records
$0.4519 of A100 cost and $0.6813 total Step 1 cost including CPU and memory. The corrected
throughput report reconstructs all timed chunks from the append-only ledgers and is safe to
regenerate without GPU work. An earlier tokenizer-report bug counted the fields of a
BatchEncoding instead of its token IDs; those invalid reports are preserved under
`study-120h/step1-out/preflight-invalid-20260912T204340Z-batchencoding-count/` and never enter the
final screen.

**Next action for a new session:** wait for Daniel's explicit authorization before Step 2. The
remaining viable frozen confirmation is the fresh Qwen3-8B/Civil Comments replication with
joint, exact-context, solo, and filler conditions. Do not rerun Step 1, try Phi-3.5, change the
8,192-token serving field, or claim a viable two-model/two-task panel. A broad panel would require
a new protocol freeze before any further joint output.

## Previous experiment handoff: Gate A passed, Gate K stopped

Work took place in `/Users/dz/bigdata`. There is no Git repository, branch, or PR. Daniel
authorized implementation and a paid Modal pilot for Gate K. No Claude CLI process was started
in this session.

The current project decision is **develop A: batched LLM annotation dependence**. Gate A returned
the pre-registered `PROCEED` verdict on Qwen3-8B and Civil Comments. Gate K remains novel as a
literature question, but its proposed representation-concentration mechanism failed the frozen
experiment on Mistral Small 3.1 24B Base. Do not spend more GPU money on Gate K.

### Gate A result

The completed 8,375-prompt run contains 14,000 target-level rows. Joint-prompt error ICC was
0.0754 with a conservative 95% decision interval of [0.0182, 0.1314]; every assignment's
fake-batch permutation test returned `p = 0.0010`. The matched-length separate-prompt filler
placebo ICC was 0.0048. The joint parse-failure rate was 0.0003, and each assignment had 208–237
errors, so every frozen interpretability condition passed. See
`gate-experiments/a/artifacts/full/analysis/report.md` and `analysis.json`.

The first joint attempt used a 256-token output cap and truncated every 16-label response. That
attempt is preserved under `artifacts/full/failed-attempt-20260911T235720Z/`. Exact pinned-tokenizer
counts showed complete responses need at most 310 tokens, so the cap was raised to 384 and only
the 375 joint prompts were rerun. Prompts, assignments, decoding, samples, and thresholds did not
change. All Gate A Modal apps are stopped with zero tasks.

### Gate K result

The tested question was whether generic LLM tabular failure follows distance concentration in
model representation space rather than ambient dimension alone.

- Model: `mistralai/Mistral-Small-3.1-24B-Base-2503`
- Revision: `ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc`
- Capability anchor: 43/48 correct, or 89.6%, with 0/48 parse failures
- Sparse-over-dense representation contrast: 0.305, below the required 1.200
- Sparse-over-nuisance representation contrast: 0.578, below the required 1.200
- Sparse-minus-dense high-dimensional accuracy: -0.125, below the required +0.050
- Verdict: `mechanism_unsupported`

Modal preserved 1,266 unique results from the frozen 1,296-job plan before reporting that the
workspace was disabled. The 30 missing jobs do not affect the stop decision. Twenty-eight are
8-dimensional checks outside the gate, one is a descriptive ordering control, and one belongs
to an ablation condition that passes even under its worst possible missing outcome. All three
conditions that reject K use complete 32- and 64-dimensional paired data.

Read these artifacts first:

| File | Purpose |
|---|---|
| `gate-experiments/k/artifacts/mistral-24b-base/analysis/COMPLETION_AUDIT.md` | Why the 30 missing jobs cannot change the decision |
| `gate-experiments/k/artifacts/mistral-24b-base/analysis/REPORT.md` | Full tables and frozen-condition verdict |
| `gate-experiments/k/artifacts/mistral-24b-base/analysis/analysis.json` | Machine-readable analysis |
| `gate-experiments/k/artifacts/mistral-24b-base/run/results.jsonl` | 1,266 raw result records |

### Model-selection history

`Qwen/Qwen3-8B` was unsuitable: 576/1,296 frozen prompts exceeded its 32,768-token context, and
38/48 greedy anchor outputs were off-label. Mistral 24B Instruct achieved 47/48 under restricted
A/B scoring but emitted an off-label greedy first token on 47/48 prompts, so it also failed the
unchanged format rule. Mistral 24B Base emitted literal A/B continuations and passed the anchor.

The Base run initially hit GPU-memory fragmentation. `backends.py` now captures only the input
to decoder layer 27 instead of retaining all hidden states, requests only final-token logits,
and `modal_gate.py` clears the CUDA cache after every job with expandable allocator segments.
The original 265 records used `hidden_states[27]`; the remaining 1,001 used the equivalent input
to decoder layer 27. The code passed 58 tests, Ruff, and mypy after this change.

### Runtime and security state

All `ieee-bigdata-gate-k` Modal apps are stopped with zero tasks. Two unrelated deployed apps,
`ncpe` and `knee-train`, each show zero tasks and were left untouched. Modal rejected the last
requests with `workspace ... is disabled`, likely because the workspace credit or spend limit
was reached. Do not retry paid work until Daniel confirms Modal is enabled and funded.

Daniel pasted a Modal token into chat. It was not used in a command and is not stored in project
files, but it should be revoked and regenerated because chat exposure is sufficient to treat it
as compromised. Never copy it into a handoff, command, prompt, or report.

### Earlier next-session instruction, superseded by the current handoff

At that point, the suggested next step was to turn Gate A's one-cell signal into a paper-scale
design. That design is now in `plan-120h-a100.md`; use the current handoff above instead.

The remainder of this file is historical C1-era context. Its old rankings and statements that no
experiment has run are superseded by `ideas.md` and the current handoff above.

## Location and current state

Shell working directory and research directory: `/Users/dz/bigdata`.

This is a research-planning folder with completed Gate K and Gate A experiments. No project Git branch or PR has been established in this workflow. The authenticated Semantic Scholar client was closed cleanly before handoff. No background work is continuing.

At the time of this handoff, the recommendation was C1, subject to a two-day pilot. The 2026-09-09 decision recorded in `ideas.md` supersedes that ranking. C1's contribution was refined through four internal rounds, independently reviewed with Claude Opus 5, and checked again after adding entrant provenance. Claude scored the broad method 4/10 and recommended `PROCEED WITH CAUTION`; the same-family provisional reviewer scored the provenance-resolved variant 6/10. These are bounded literature judgments, not experimental results or an exhaustive systematic review.

## User request and boundaries

Daniel requested the three strongest feasible research ideas for the [IEEE BigData 2026 High School Symposium](https://bigdataieee.org/BigData2026/high-school-symposium/), using the `idea-generation` skill exactly.

Priority areas are modeling, visualization, personalization, recommendation, and data mining with large language models. Stronger feasible directions elsewhere in the symposium scope are allowed.

Avoid human-subject approval, new data collection, extensive manual labeling, proprietary data, expensive training, and specialized hardware. Multiple coding agents may be available, but their availability does not solve novelty or evaluation problems. Optimize for the easiest defensible paper rather than the most elaborate system.

The user explicitly said not to write implementation code yet. A request to continue research or read this handoff does not itself remove that boundary. On 2026-09-08, the user explicitly requested a Claude CLI second opinion; that completed with canonical model `claude-opus-5`, and its review is saved under `refine-logs/`.

The requested pipeline is survey, propose, then verify. Survey OpenAlex, arXiv, and Semantic Scholar, cover 2023 onward, and separately emphasize 2025–2026. Generate at least five ideas across at least three directions, grounded in documented paper limitations. For every survivor, search direct phrasing, adjacent terminology, and the domain-free mechanism. Report rejected ideas and the prior papers that defeat them.

The final deliverable requires exactly three finalists with title, research question, hypothesis, documented gap, closest work and differentiation, linked public datasets and sizes, baselines, method, metrics, experiments and ablations, anticipated tables and figures, compute/API requirements, risks, a ten-day schedule, and 1–10 scores for novelty, feasibility, reviewer appeal, BigData fit, and completion. A detailed shortlist was delivered in the conversation. The files below preserve the evidence and designs, but are not a complete standalone reproduction of every requested final-output field.

Venue eligibility must be confirmed before submission. An earlier website check found a high-school first-author/primary-contributor requirement and a September 20 deadline. Recheck the live page for current requirements. Do not assume Daniel's eligibility from contextual descriptions.

## Read these files first

| File | Purpose |
|---|---|
| [novelty-check-report.md](novelty-check-report.md) | Latest claim-level novelty verdict for C1 and C2. Supersedes the earlier C2 novelty assessment after finding a March 2026 persistence study. |
| [verification-update.md](verification-update.md) | Latest evidence, API recovery, new competitors, and corrections to the secondary review. Overrides older coverage statements. |
| [c1-pilot-plan.md](c1-pilot-plan.md) | Concrete two-day pilot, fixed choices, attribution controls, measurements, and abandonment criteria. |
| [refine-logs/FINAL_PROPOSAL.md](refine-logs/FINAL_PROPOSAL.md) | Current refined C1 contribution, estimand, balance reference, and claim boundary. |
| [refine-logs/C1_NOVELTY_CHECK.md](refine-logs/C1_NOVELTY_CHECK.md) | Latest claim-level check of the nonincident-entrant mechanism extension. |
| [refine-logs/claude3-independent-review.md](refine-logs/claude3-independent-review.md) | Independent Claude Opus 5 novelty review and concrete reviewer attacks. |
| [finalists.json](finalists.json) | Three evaluated candidates, planning scores, designs, closest papers, and completed targeted-verification verdicts. |
| [research-audit.md](research-audit.md) | Survey themes, candidate ledger, rejected ideas, query families, related work, and verified data inventory. Read its final recovery update. |
| [s2-recovery-results.jsonl](s2-recovery-results.jsonl) | Fifteen captured success/failure responses from the recovery pass. Partial transcript, not an exhaustive API log. No credentials. |

Other `survey-*.jsonl`, `recent-*.jsonl`, `rec-*.jsonl`, `log-*.jsonl`, and `vis-*.jsonl` files preserve earlier searches. Reuse them. Do not restart the literature survey simply because this is a new session.

## Ranked candidates

### C1: The Future Chooses the Test Set

Question: Can post-test interactions change historical recommendation comparisons solely through the user/item eligibility filtering used to construct a benchmark?

The proposed contribution is a controlled future-suffix intervention with frozen historical targets and predictions. Future records may change only k-core eligibility. The pilot now distinguishes future-incident entrants from nonincident historical entrants admitted through the new k-core fixed point and decomposes their contributions to the pairwise model-gap shift. Dynamic k-core propagation and train-only filtering are already known and are not contributions.

Use raw Amazon 2014 Beauty ratings, approximately 2,023,070 records, followed by MovieLens 1M for replication. Do not accidentally start with the already filtered 198,502-record Amazon Beauty 5-core file. Start with popularity and item-item nearest neighbors on CPU. Add implicit ALS and BPR only after the pilot passes.

Important prior work: Zhao et al.'s filtering study, Time to Split, Don't Get Ahead of Yourself, A Critical Study on Data Leakage, SplitLight, RecNextEval, and the tgr train-window filtering implementation. Links and exact boundaries are in the evidence files.

Do not claim that filtering changes rankings for the first time. The older leakage study actually adds future records to training. Our narrower intervention must isolate future-dependent selection while keeping actual records historical. Freeze catalog handling and use a common-cohort analysis to avoid confusing cohort changes with per-user prediction changes.

Main risk: only the established facts that filtering changes dataset composition and k-core updates propagate are recovered, with no meaningful new evaluation consequence. The entrant-provenance analysis is a diagnostic extension unless nonincident entrants are common and materially affect replicated model comparisons. A ranking reversal is not required and is not sufficient on its own.

### C2: Silent Cache Hits

Question: Do incorrect accepted cache matches delay discovery of new log-event families, and how long do those errors persist?

Use existing labels in Loghub-2.0 Linux, Mac, and Hadoop, totaling approximately 304,228 log records. Separate naturally occurring errors from injected-cache stress tests. Instrument accepted-hit paths in existing parsers and compare budget-matched checks. Generic template verification, drift tests, and wrong-hit metrics are already published.

Mandatory competitors include EFParser, SCOPE, KELP, vCache, and GroundedCache. EFParser's accepted matches bypass the generation/correction path, but its reported over-general errors are relatively rare. SCOPE already performs matching-time semantic checks.

[VET-Parser](https://doi.org/10.1109/WCCIS70285.2026.11650850) was resolved through its Semantic Scholar paper ID. Its verified metadata and abstract describe variable-evidence risk detection, selective LLM arbitration, and verification before cache updates. This is medium overlap and makes VET-Parser a mandatory mitigation competitor, but the abstract does not report accepted-hit error persistence or event-family discovery delay. The full text remains closed-access.

Main risks: natural errors are too rare, strong existing checks eliminate the effect, discovery labels are ambiguous, or reproduction/API costs dominate. Complete Loghub archive download and checksums were not verified.

The stricter novelty check found a public March 2026 automated-research manuscript, *Persistent Demo-Pool Poisoning Attacks on Online LLM Log Parsers*. It already demonstrates long-lived degradation and silent trie absorption under adversarial injection. C2 must therefore be positioned only around naturally occurring accepted-hit failures and labeled event-family discovery delay, not persistence as a new mechanism.

### C3: Local Colors, Global Dependencies

Question: Can remote point movements change distortion colors on triangles whose geometry has not changed?

Audit the July 2026 Gap Index and its 2025 predecessor on Digits and a fixed Fashion-MNIST subset. Separate changed geometry, changed triangulation, and global normalization. Use rigid-motion and scale controls. Do not make human-perception claims without a human study.

Two secondary-review objections were wrong: the Gap Index paper does explicitly make the incident-triangle locality statement, and the predecessor does globally normalize triangle areas by the largest triangle. Both corrections were checked in the primary texts.

Reject this candidate in its current form. Shared-denominator dependence is established mathematics, and the Gap Index intentionally measures relative areas. A remote perturbation changing untouched triangles follows directly from the definitions, so the proposed audit lacks a sufficient contribution without an already-demonstrated material interpretation failure.

## Source coverage and recovery

OpenAlex and arXiv survey/recent searches and multiple three-angle novelty rounds were completed in the earlier work. Semantic Scholar was intermittent, including with the user-supplied key: HTTP 200, 429, and 500 responses occurred.

Minimal authenticated requests to `/graph/v1/paper/search` with `fields=title,year,url`, a small limit, and no year filter sometimes worked when richer requests failed. `/graph/v1/paper/search/bulk` also worked once, then returned rate limits. The cause was not isolated because parameters and service state both changed.

C1, C2, and C3 each received successful Semantic Scholar responses for direct, adjacent-vocabulary, and domain-free mechanism searches, complementing the completed OpenAlex and arXiv rounds. VET-Parser metadata and abstract were retrieved by paper ID. Some supplemental API calls still returned HTTP 429 or 500; failed calls were not treated as empty results.

The requested targeted three-source finalist verification is complete. Keep the scope precise: this was not a systematic review, and C1/C2 still need empirical pilots before either can support a paper claim.

The API key was used in process memory, not saved in these files or shell configuration. Do not copy the key from chat into handoff files, logs, code, or reviewer prompts. The prior interactive client's session is closed and cannot be resumed.

## Next actions

1. A1: Resume with `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/C1_NOVELTY_CHECK.md`, and `c1-pilot-plan.md`. Keep C1 as the preferred pilot candidate with a cautious, bounded novelty verdict.
2. A2: Treat C2 as a gated backup and include the Analemma persistence manuscript, VET-Parser, SCOPE, and EFParser as mandatory competitors. Stop early if natural wrong accepted hits and measurable family-discovery delays do not occur. Do not revive C3 without new evidence of a material interpretation failure.
3. A3: Once Daniel requests implementation, execute the two-day C1 pilot. Day 1 validates data, freezes windows, checks invariants, and measures entrant provenance. Day 2 runs two CPU baselines, computes the sole primary endpoint, and tests it against the historical-balance reference and power-based practical floor. Follow the plan's continue/stop criteria before expanding.

No user decision is needed to understand or preserve this handoff. Experiment implementation remains outside the currently authorized scope.
