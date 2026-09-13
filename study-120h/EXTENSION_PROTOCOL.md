# Cross-family extension (X1R4): CPU protocol before any model output

Status: CPU-only scientific protocol. The seal status and every hash live in the separate
`freeze/<stamp>-X1R4/` directory. This protocol draws no result from any model call.

The earlier `20260913T174936Z-X1` seal is preserved but superseded. It fixed only cell-level
execution order, recorded a power namespace that the simulator did not use, and encoded Llama
access as an immutable non-runnable policy. X1R1 corrects those defects before any extension model
output exists. The intermediate X1R1 seal is also preserved but superseded because its aggregate
B4 evaluation-prefix digests did not bind the sampled document order. X1R2 does.

X1R2 is preserved but superseded before any extension model output. Greedy decoding alone does
not guarantee identical GPU results when vLLM changes the active request batch. X1R3 inherits
every X1R2 frame, M1 call order, B3 plan, B4 schedule, statistical rule, and budget unchanged,
then adds the batch-invariant execution gate below.

X1R3 is preserved but superseded because its report validator did not compare the reported worker
revision, anchor ID, and request layout against the frozen plan. X1R4 adds those fail-closed checks
without changing any scientific schedule or analysis rule.

## Scope and separation

This extension is **not** part of the sealed Freeze B claim family. Freeze B admitted only Qwen
and Granite cells, and its admission lock records `recurrence_claim_not_supported`. The Qwen and
Granite joint results are already open. This extension is therefore disclosed as new work on new
untouched frames. It does not rewrite, extend, or reinterpret the Freeze B claim family.

Every Freeze B module default is unchanged. The extension adds `src/freezeb/extension.py` as a
wrapper and passes explicit parameters into the shared primitives. Every sealed Freeze B artifact
still reproduces from its own snapshot.

## Candidate panel

Five instruction checkpoints, one per family, at immutable pinned revisions.

| Order | Family | Repository | Revision | Access |
|---:|---|---|---|---|
| 1 | qwen | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | public |
| 2 | granite | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | public |
| 3 | llama | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | **manually gated** |
| 4 | mistral | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | public |
| 5 | phi | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | public |

All five revisions resolve on the Hub and match their pins. Llama file contents return HTTP 401
without an accepted licence and a token, so its configuration and chat template are currently
unverified. Access is a runtime observation rather than an immutable model-policy field. A signed
runtime/access lock must classify all five candidates as attempted or technically unavailable
before the screen; it may never silently omit Llama. A candidate may be replaced only while no
extension joint output exists. After the first joint output, a missing candidate narrows the panel
and never authorizes a substitute.

Phi-4-mini-instruct replaces Phi-3.5-mini-instruct, which failed every Freeze B solo screen. The
swap happened before any extension output existed and is recorded here for that reason.

## Tasks and frames

Four classification tasks at pinned dataset revisions: Civil Comments, MultiNLI, IMDb, AG News.
GoEmotions and MultiNERD are excluded from this extension.

Extension IMDb reads the **train** split. The test split holds 23,310 eligible rows and Freeze B
already used 14,336 of them, so only 8,974 remain against the 14,336 this extension needs. The
train split is sorted by label, so `scan_limit` must cover all 25,000 rows.

Each task supplies 2,048 screen rows, 8,192 confirmation rows, and 4,096 filler rows. Selection
deduplicates, removes every prior identifier, sorts by `SHA256(20260913:doc_id)`, then takes
contiguous screen, confirmation, and filler prefixes. The confirmation frame carries 16 equal
length strata.

Every extension row is disjoint from the Gate A pilot, the sealed Step 0 frames
(`20260912T041633Z`), and the Freeze B part-1 frames (`20260912T222618Z-B1`). The measured overlap
is zero for every task and every role. Only identifiers, hashes, source indices, and the
reconstruction rule may be published. References are dataset labels, not adjudicated human truth.

## Prompts, parsers, and serving

One prompt template, one response grammar, and one parser apply to every family. No model receives
a private prompt repair, a private response cap, or a private context allowance. All five
candidates are capped at a 32,768-token window. A revised grammar is allowed only if it is frozen
before the solo screen and applied to all five families, including the newly run Qwen and Granite
anchors. The Freeze B Qwen and Granite screens are **not** reused.

A first response is parsed once. The exact-context control shows identical documents in identical
order and requests one target label. Decoding is greedy at BF16 with `seed = 20260912`,
`max_num_seqs = 32`, and `tensor_parallel_size = 1`, verified per worker.

Every extension model worker must set `VLLM_BATCH_INVARIANT=1` before importing vLLM. The backend
fails closed if this setting is absent and records that invariant execution was active and
required. This controls numerical changes caused only by different active request batches, which
would otherwise be an alternative explanation for joint versus separate-call differences.

Before M1, each attempted checkpoint must pass the sealed synthetic matrix in
`BATCH_INVARIANCE_PLAN.json`: the same four task prompts are repeated at request batch sizes 1, 2,
8, 16, and 32, in forward and reverse order, across two fresh vLLM sessions. Anchor output bytes
must be identical across every layout and session. The gate also verifies the pinned checkpoint,
vLLM 0.28.0, BF16, one A100, tensor parallel size 1, `max_num_seqs = 32`, and seed 20260912. A
failure makes that checkpoint technically unavailable; it cannot enter M1 or receive a private
serving exception. The signed runtime/access lock must bind the passing report SHA256.

## Seeds and namespaces

| Purpose | Value |
|---|---|
| Frame selection seed | `20260913` |
| Assignment namespace | `bigdata2026-ext-v1` |
| Repeated-sampling namespace | `bigdata2026-ext-e5-v1` |
| Power namespace | `ext-power-v1` |

These namespaces do not collide with the Freeze B namespaces, so an extension anchor cell never
reuses a Freeze B assignment permutation.

## Budget

The extension holds a separate ceiling. It does not share or consume the Freeze B allowance, and
extension calls never count toward the original sealed queue.

| Setting | Value |
|---|---:|
| Meter origin | 0 metered A100-hours |
| Hard stop | 80 metered A100-hours |
| Protected reserve | 10 A100-hours |
| Start guard | 1.6x the base forecast |

A complete block starts only when `cumulative + 1.6 x base + 10 <= 80`. The meter counts all-worker
metered time, including model loads and failed attempts.

## Statistics

| Setting | Value |
|---|---|
| Primary contrast | family-equal `joint covariance - exact-context covariance` |
| Sensitivity contrast | cell-equal mean, reported beside the primary, never as the gate |
| Batch size | 16 |
| Assignments per cell | 3 |
| Replay draws | 4,999 |
| Alpha | 0.05 |
| Holm family | the sealed primary cells, 1 to 20 |
| Planning alpha | 0.0025, a Bonferroni proxy for a 20-cell family |
| Practical floor | 0.002 |

The family-equal weighting gives every admitted family one vote. A family with four admitted cells
cannot outweigh a family with one. The reported test is the design-respecting replay
randomization; the power table is a planning proxy only. The planning table pools three independent
2,000-simulation batches under the frozen `ext-power-v1`, `ext-power-v1-b`, and
`ext-power-v1-c` namespaces. It never enters the reported test.

### Item counts

The item count per cell is chosen from the frozen power table and the solo disagreement rate,
before any joint output. Under the 0.0025 planning alpha the table selects:

| Grid error rate | Items | Proxy power | Wilson lower bound |
|---:|---:|---:|---:|
| 0.05 | 2,048 | 0.9998 | 0.9991 |
| 0.10 | 2,048 | 0.9957 | 0.9937 |
| 0.20 | 4,096 | 0.9813 | 0.9776 |
| 0.35 | 4,096 | 0.8268 | 0.8171 |
| 0.50 | 8,192 | 0.9727 | 0.9682 |

The strict 20-cell planning alpha selects 8,192 items only at the 0.50 error grid. A selected
8,192-item cell has 34,304 calls rather than 17,152 at 4,096 items. Section "Budget feasibility"
states the consequence.

## Claim gates

**C1 cross-family recurrence.** All of: family-equal global contrast above zero; global replay
p at most 0.05; at least three families each holding a Holm-positive cell with effect at least
0.002; every leave-one-family-out family-equal mean above zero. A missing cell cannot pass.

**C2 practical uncertainty damage.** At least two prespecified E5 cells with at least 400
evaluation replicates, passing reference precision, item-level coverage at most 0.90, worst-case
upper coverage below 0.95 after reference uncertainty, and Holm-adjusted p at most 0.05.

The primary item-level Wald and 32-call t intervals use no finite-population correction. The
uncorrected factor is reported as a sensitivity only. The CPU analysis also freezes a 2,000-draw
whole-call bootstrap, Clopper-Pearson coverage intervals, target plus/minus 1.96 reference-SE
coverage variants, and the empirical-to-nominal i.i.d. variance ratio with a 1.25 practical floor.

**Anti-claim control.** Joint must exceed the exact-context control that shows identical neighbors
in identical order and requests one target.

## Run order

### M1: blinded solo capability screen

Order is model by model in the table order above, then task in this fixed order: Civil Comments,
MultiNLI, IMDb, AG News. 20 cells, 2,048 calls each, 40,960 calls total, solo only. Exact individual
call-ID and prompt-hash order digests are in `M1_CALL_ORDER.json`.

For every candidate cell and every allowed item count, `B3_PLANS.json` fixes the complete
joint/exact-context/solo interleaving and hashes each call's document order, targets, rendered
prompt, response cap, and execution position. `B4_SCHEDULES.json` fixes the sampled document order
for the 250-reference stream and the nested 400, 1,000, and 2,500 evaluation prefixes. These files
contain hashes and audit records, not dataset text.

Admission floors, identical for every family: parse rate at least 0.98; error rate between 0.05 and
0.80; accuracy at least 0.70 for Civil Comments, IMDb, and AG News; accuracy at least 0.55 for
MultiNLI; Civil Comments balanced accuracy at least 0.60 and toxic recall at least 0.25. Admission
uses solo capability and technical timing only. No joint result may enter the decision.

### Conditional B3 and B4 order

| Stage | Cells | Start condition |
|---|---|---|
| M2 B3 family core | first admitted cell of each family, in balanced order | every core cell completes before any C1 reading |
| M3 B3 remaining | remaining admitted cells in balanced order | the 1.6x guard plus the 10-hour reserve fits 80 |
| M4 B4 reference | one cell per admitted family | reference stream runs before any evaluation replicate |
| M4 B4 evaluation | the same cells, 400 replicates | reference precision passes before C2 is read |
| M5 B4 prefixes | the same cells, 1,000 then 2,500 | fixed round-robin technical expansion only |

The B4 cell for a family is its first admitted cell in the frozen task order. It is chosen from
solo evidence before any B3 output.

## Budget feasibility

Rates come from the completed Qwen/MultiNLI cell and the Freeze B screen. They are planning
figures, not quotes.

| Milestone | Work | Base hours | Guarded hours |
|---|---|---:|---:|
| M0 | synthetic runtime and batch-invariance lock, 5 models | 0.58 | 0.93 |
| M1 | solo screen, 40,960 calls | 0.93 | 1.48 |
| M2, M3 | 12 admitted cells, half at 8,192 items | 14.6 | 23.3 |
| M4 | 5 E5 cells, 250 reference plus 400 evaluation | 13.5 | 21.7 |
| | Subtotal | 29.6 | 47.4 |
| | Protected reserve | | 10.0 |
| | **Projected total** | | **57.4** |

The ceiling holds this shape with about 22 guarded hours of headroom. It does **not** hold all 20
cells at 8,192 items: that case projects about 86 guarded hours and exceeds 80. The block meter
stops admission before the ceiling is crossed, so the expected outcome is a narrowed E2 panel, not
an overspend. Family breadth precedes within-family depth. The 1,000 and 2,500 evaluation prefixes
are technical fills and are not in this total.

## Failure interpretation

A model that fails the screen is a reported screen failure, not a negative dependence result. A
technically unavailable checkpoint narrows the panel. A null or negative cell stays in the table.
Cross-family recurrence is unsupported unless the complete C1 gate passes. If reference uncertainty
is too large, report variance and width and make no measured-coverage claim.

## State at this seal

Zero scientific model outputs exist for this extension. No GPU work, no Modal app, no model load,
and no synthetic inference has run for it. The remaining work before paid science is, in order:
granted weight access or a terminal unavailability record for Llama, the sealed synthetic runtime
and batch-invariance checks for every attempted candidate, the signed runtime/access lock, the
blinded solo screen, and the X1R4 admission lock.
