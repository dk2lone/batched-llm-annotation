# Cross-family experiment extension: A0-A13 plan

**Problem:** Determine whether shared-call LLM text annotation creates excess across-item error
dependence and whether treating those labels as independent produces misleading uncertainty.

**Method thesis:** Compare joint size-16 labeling with an exact-context one-target-per-call control
across five model families, then measure confidence-interval behavior with independent repeated
calls in one prespecified cell per admitted family.

**Date:** 2026-09-14 UTC

**Status:** A0-A6 are complete. The active CPU protocol is sealed at
`study-120h/freeze/20260913T183353Z-X1R4/` with manifest SHA256
`3838c6e201388480eb08aa48e04a1ef3a5af6a86fac5d920e9837459382677d2`. It freezes
candidates, frames, prompts, seeds, budget, statistics, and exact M1/B3/B4 schedules. Earlier X1
seals are preserved but superseded. The synthetic runtime and batch-invariance preflight passed
all five models and used 1.498607 metered A100-hours. A6 completed all 40,960 sealed solo calls
in run `20260914T003735Z-5a8c63f2`; seven cells in four families passed the frozen admission
floors. See `study-120h/extension-m1/20260914T003735Z-5a8c63f2/REPORT.md` for every cell.
No extension joint output exists. A7, the admission lock, is sealed. A8 is next. This extension is separate
from Freeze B because Qwen and Granite joint results have already been opened. Nothing below
authorizes a further paid run.
See `study-120h/EXTENSION_PROTOCOL.md` for the full frozen protocol.

## Claim map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: cross-family recurrence | The current evidence covers only Qwen and Granite. | A positive **family-equal** joint-minus-exact covariance test; at least three families each have a Holm-positive prespecified cell with effect at least 0.002; every leave-one-family-out family-equal mean remains positive. The cell-equal mean is a sensitivity, never the gate. | B1-B3 |
| C2: practical uncertainty damage | Dependence alone does not show that ordinary confidence intervals fail. | At least two prespecified family cells have item-level coverage at most 0.90, worst-case upper coverage below 0.95 after reference uncertainty, and multiplicity-adjusted p at most 0.05. | B4 |
| Anti-claim: extra context explains the result | Joint and solo prompts differ in both context and requested outputs. | Joint exceeds an exact-context control that presents identical neighbors in identical order but requests one target. | B3 |

Claims apply only to the frozen checkpoints, tasks, prompts, and serving policy. This extension does
not identify a population-wide property of all LLMs.

## Candidate panel

One instruction checkpoint per family, pinned to an immutable revision and sealed in X1R4:

| Order | Family | Repository | Revision | Access |
|---:|---|---|---|---|
| 1 | qwen | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | public |
| 2 | granite | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | public |
| 3 | llama | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | verified |
| 4 | mistral | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | public |
| 5 | phi | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | public |

All five revisions resolve on the Hub, match their pins, and passed the signed runtime/access gate.
All five candidates are marked attempted in the frozen M1 schedule; none is technically
unavailable. Phi-4-mini-instruct replaces Phi-3.5-mini-instruct, which failed every Freeze B solo
screen. The swap happened before any extension output existed. After the first extension joint
output, a missing checkpoint narrows the panel and never authorizes a substitute.

Use four classification tasks: Civil Comments, MultiNLI, AG News, and IMDb. New screen,
confirmation, and filler rows are materialized and measured disjoint from the Gate A pilot, the
sealed Step 0 frames, the Freeze B part-1 frames, and one another. Extension IMDb reads the
**train** split: the test split holds 23,310 eligible rows and Freeze B used 14,336 of them, so the
8,974 that remain cannot supply the 14,336 this extension needs. Apply the same task prompt, label definitions, response grammar, parser, maximum
tokens, and reference-label rules to every family. Do not give Phi or another model a private prompt
repair. A revised grammar is allowed only if it is frozen before solo screening and applied to all
five families, including newly run Qwen and Granite anchors.

## Experimental blocks

### B1: Protocol, data, and runtime freeze

- **Claim tested:** The later comparisons differ by model family rather than hidden serving or data
  changes.
- **Runs:** CPU frame construction and assignment tests; synthetic prompts for every checkpoint at
  solo, size 16, exact-context, and size 32; cross-layout and cross-session deterministic checks.
- **Recorded controls:** immutable checkpoint and tokenizer hashes, vLLM/container revision, BF16,
  greedy decoding, prompt hashes, frame hashes, call IDs, assignment matrices, token limits, and
  all-worker A100 metering.
- **Success criterion:** Every primary model loads, fits the largest prompt, produces deterministic
  outputs across the frozen request layouts and sessions, and has a measured guarded cost.
- **Failure interpretation:** A technical failure narrows the extension before scientific output;
  it is not a negative dependence result.
- **Priority:** MUST-RUN.

### B2: Blinded solo capability screen

- **Claim tested:** Each model can perform each task well enough for disagreement covariance to be
  interpretable.
- **Runs:** 2,048 fresh solo calls for every model-task pair, 40,960 calls for five models and four
  tasks.
- **Admission rules:** Preserve the Freeze B floors: parse rate at least 0.98; disagreement between
  0.05 and 0.80; accuracy at least 0.70 for Civil Comments, AG News, and IMDb; accuracy at least
  0.55 for MultiNLI; plus Civil Comments balanced accuracy at least 0.60 and toxic recall at least
  0.25.
- **Selection:** Order tasks as Civil Comments, MultiNLI, IMDb, then AG News. Freeze all admitted
  cells and one repeated-sampling cell per family using the first admitted task in this order.
  Selection may use solo capability and technical timing only, never a joint-minus-exact result.
- **Success criterion:** At least three families, including at least one of Llama or Mistral, have
  an admitted cell. A five-family claim requires all five to admit at least one cell.
- **Failure interpretation:** A family with no admitted cells remains a reported screen failure.
  It does not proceed to E2 and is not replaced after joint outputs.
- **Priority:** MUST-RUN.

### B3: Full batched-versus-individual experiment

- **Claim tested:** Shared generation creates excess error covariance beyond context alone.
- **Runs per admitted cell:** Three independent length-balanced assignments; size-16 joint calls;
  exact-context one-target calls using identical documents and order; solo calls; fake batches
  built from independent-call labels.
- **Sample size:** Choose 2,048, 4,096, or 8,192 items from the sealed X1R4 power table and the solo
  disagreement rate, before joint outputs. At 4,096 items one cell holds 17,152 calls; at 8,192 it
  holds 34,304. Under the pooled 0.0025 planning-alpha table, the 0.50 error grid selects 8,192.
- **Primary metric:** Equal-assignment, pair-weighted
  `joint covariance - exact-context covariance`, weighted **family-equal**, with a 0.002 practical
  floor. The cell-equal mean is reported beside it as a sensitivity.
- **Inference:** 4,999 design-respecting replay draws, Holm correction over the full frozen cell
  family of 1 to 20 cells, family-equal global contrast, leave-one-family-out and
  leave-one-task-out sensitivity, and four malformed-output bounds. Preserve first responses and link rather than overwrite retries.
- **Run order:** First run one admitted cell per family in a balanced family order. Analyze the
  locked family core. Only then run remaining admitted cells in the frozen order.
- **Failure interpretation:** Null or negative families remain in the table. Cross-family recurrence
  is unsupported unless C1's complete gate passes.
- **Priority:** MUST-RUN.

### B4: Independent repeated-sampling experiment

- **Claim tested:** Ignoring call membership makes uncertainty estimates too narrow in practice.
- **Cells:** Exactly one cell per admitted family, chosen from B2 solo evidence before B3 outputs.
- **Initial runs per cell:** 250 reference replicates and 400 evaluation replicates. Each replicate
  samples 512 item appearances and makes 32 fresh size-16 joint calls. No E2 output is reused.
- **Intervals:** Item-level 95% Wald interval and call-aware 95% t interval over 32 call means;
  whole-call bootstrap and finite-population correction as sensitivities.
- **Metrics:** Empirical coverage with Clopper-Pearson interval, mean width, empirical variance to
  mean nominal iid variance ratio, and reference-target Monte Carlo uncertainty.
- **Expansion:** Extend evaluation prefixes to 1,000, then 2,500 only using the frozen round-robin
  family order and technical affordability. Do not expand a cell because its interim coverage is
  favorable.
- **Failure interpretation:** If reference uncertainty is too large, report variance and width but
  make no measured-coverage claim. If dependence recurs without undercoverage, claim dependence
  without demonstrated inferential damage.
- **Priority:** MUST-RUN through 400 evaluations; larger prefixes are NICE-TO-HAVE.

### B5: Batch-size, prompt, and mechanism boundaries

- **Claim tested:** The effect changes with batch size or composition rather than being a single
  size-16 artifact.
- **Cells:** One B2-selected representative cell per admitted family.
- **Runs:** Batch sizes 2, 4, 8, 16, and 32 with exact-context controls and three assignments; two
  uniformly frozen prompt variants at size 16; fixed-composition position rotation and
  homogeneous-versus-mixed batches when task labels support the intervention.
- **Metrics:** Covariance slope against `log2(batch size)`, prompt-minus-baseline contrasts,
  position/composition contrasts, accuracy, parse rate, throughput, token use, and metered cost.
- **Failure interpretation:** These are bounded mechanism and robustness results. They cannot rescue
  a failed C1 or C2 gate.
- **Priority:** NICE-TO-HAVE after B4 reaches 400 evaluations.

## A0-A13 execution order

The A labels are the human-facing plan. Existing sealed files retain their M1, B3, and B4 names so
their hashes and audit trail remain valid.

### A0 — Preserve existing evidence — complete

Keep the historical Gate A pilot, bridge, and Freeze B results unchanged and separate from this
extension's primary tests. The extension uses fresh frames, call IDs, and randomization streams.

### A1 — Lock five checkpoints — complete

Qwen3-8B, Granite-3.3-8B-Instruct, Llama-3.1-8B-Instruct,
Mistral-7B-Instruct-v0.3, and Phi-4-mini-instruct are pinned at the revisions above. The signed
X1R4 runtime/access lock records all five as attempted and none as technically unavailable.

### A2 — Build fresh data frames — complete

Civil Comments, MultiNLI, IMDb, and AG News each have 2,048 screen, 8,192 confirmation, and 4,096
filler rows. Their identifiers are disjoint from prior studies and from one another. Private text
remains outside the seal; identifiers, hashes, labels, exclusions, and reconstruction rules are
recorded.

### A3 — Freeze prompts, calls, and statistics — complete

X1R4 freezes the common prompt and parser policy, greedy decoding, response caps, three
length-balanced assignments, call IDs and order, malformed-response rules, family-equal covariance
analysis, 4,999 replay draws, Holm correction, admission criteria, item-count rules, and the rule
that selects one repeated-sampling cell per admitted family from solo evidence.

### A4 — Implement and CPU-test the unified runner — complete

The runner supports all five checkpoints and four tasks. It preserves exact order and immutable
first responses, links retries, resumes without duplicate calls or metering, quarantines invalid
payloads, verifies checkpoints, stops at the global first-1,000-attempt reforecast, sums time across
workers, and produces the frozen analysis inputs. The focused A4 suite has 78 passing tests; all
417 project CPU tests, Ruff, strict mypy, and ledger checks pass. The real plan-only entry point
verified 20 cells and 40,960 calls without paid execution.

### A5 — Run the synthetic runtime benchmark — complete

All five models passed checkpoint loading, 32,768-token context fit, vLLM 0.28.0 serving checks,
and deterministic batch-invariance comparisons across request sizes 1, 2, 8, 16, and 32 in both
orders and two fresh sessions. The two technical passes used 1.498607 metered A100-hours. This
evidence contains no scientific labels.

### A6 — Run the solo capability screen — complete (sealed M1 schedule)

Ran 2,048 individual calls in each of 20 model-task cells, 40,960 calls total. The global
first-1,000-attempt gate released the remaining work after zero failures and a 2.128643-hour
guarded full-screen reforecast. All calls returned `ok`; 1.383838 A100-hours were metered across
five workers. Seven cells in four families passed the frozen parse, disagreement, accuracy,
balanced-accuracy, and recall gates. Phi admitted none. The complete cell table is in the A6 report.

### A7 — Seal the admitted panel — complete

Report every pass and failure, select 2,048, 4,096, or 8,192 items with the frozen power rule,
select the first admitted task per family for repeated sampling, and seal cell order, worker
ownership, calls, and budget in a new timestamped admission lock. No extension joint output may be
generated before this lock exists.

The CPU-only X1R4 A7 lock was sealed at
`study-120h/admission-lock/20260914T012453Z-X1R4-A7/`. Its report lists all
20 pass/fail decisions. Seven cells in four families use the frozen 2,048/4,096
item power choices. The lock binds the family-core order, repeated-sampling
cells, selected B3/B4 schedule hashes and call counts, four logical family-owned
A100-40GB slots, and the 80-hour/10-hour-reserve/1.6x budget rule. No GPU or
extension joint output was generated.

### A8 — Run one main cell per admitted family — next paid stage (sealed B3 order)

Run one admitted model-task cell from each family before opening the family-core result. Each cell
uses three joint assignments, exact-context target-only calls with identical neighbors and order,
solo calls, and fake-batch controls. Reforecast each model after its first 1,000 attempts.

### A9 — Analyze the family core — pending A8

Calculate joint-minus-exact covariance, run 4,999 design-respecting draws, apply Holm correction,
calculate the family-equal global result, check leave-one-family-out results, and apply every frozen
malformed-output treatment. C1 requires qualifying results in at least three families.

### A10 — Complete remaining main cells — conditional (sealed B3 order)

Run the remaining admitted cells in frozen order while each complete block fits the 1.6x start
guard and 10-hour reserve under the 80-hour stop. Failed and null cells remain reported.

### A11 — Run the practical uncertainty experiment — conditional (sealed B4 order)

Use the first admitted task per family selected at A7. Run 250 independent reference replicates
and 400 independent evaluation replicates per selected cell, with 512 sampled items and 32 fresh
joint calls per replicate. Compare item-level Wald intervals, 32-call t intervals, and whole-call
bootstrap sensitivity. Report coverage, width, variance inflation, and reference uncertainty.

### A12 — Extend all four families equally to 1,000 evaluations — budget-conditional priority

After every selected family reaches the frozen 400-evaluation checkpoint, extend all four equally
to 1,000 using the already sealed nested B4 prefix and fixed round-robin order. This adds 600
replicates and 19,200 fresh joint calls per family, 76,800 calls overall. The incremental estimate
is 10–15 base or 16–24 guarded A100-hours; the provisional A8–A12 total through this checkpoint is
40–61 guarded hours, before prior use and the protected reserve. Reforecast from actual B4 runtime
and start only complete blocks that satisfy the unchanged 80-hour stop, 10-hour reserve, and 1.6x
guard. The 400-replicate C2 gate stays primary. If the equal 1,000 prefix cannot complete, report
the shortfall and do not claim four-family 1,000-replicate evidence. The 2,500 prefix and B5
sensitivities remain later conditional work. See the pre-output priority amendment at
`study-120h/amendments/20260914T013606Z-x1r4-e5-1000-priority.md`.

### A13 — Final analysis and paper — pending completed experiments

Freeze completed ledgers, rerun multiplicity-adjusted analyses, audit raw-output provenance and
claim scope, produce the family-by-task table and uncertainty figures, report failed screens and
incomplete cells, and keep historical, extension-confirmatory, and optional evidence separate.

The 4.824 cumulative A100-hours and the 150-hour stop belong to Freeze B. This extension holds a
**separate** ceiling, sealed in X1R4: meter origin 0, hard stop 80 metered A100-hours, 10 protected
reserve hours, and a 1.6x start guard. It does not share or consume the Freeze B allowance, and
extension calls never count toward the original sealed queue.

The ceiling supports the family core plus one 400-replicate E5 cell per family with roughly 22
guarded hours of headroom. It does **not** support all 20 admitted cells at 8,192 items, which
projects about 86 guarded hours. The block meter stops admission first, so the expected outcome is
a narrowed E2 panel, not an overspend. Family breadth precedes within-family depth.

## Paper storyline

- **Main paper:** common-policy solo screen, family-balanced E2 result, and 400-replicate E5 result.
- **Appendix:** every screen failure, every cell estimate, malformed-output bounds, larger E5
  prefixes, batch-size curves, prompt variants, position/composition tests, and cost ledgers.
- **Cut:** model-specific prompt tuning, result-dependent checkpoint replacement, commercial APIs,
  GoEmotions/MultiNERD, and additional parameter sizes until the five-family core is complete.

## Risks and controls

1. **Opened Qwen evidence:** Label this as a new extension and use new untouched frames. Do not
   rewrite the Freeze B claim family.
2. **Phi policy failure:** Apply one grammar to every family. If Phi fails again, report the failure
   and stop its scientific cells.
3. **Gated Llama weights:** Verify account access before sealing. Technical unavailability narrows
   the panel; post-output substitution is forbidden.
4. **Family-task confounding:** Screen all four tasks and require the family-weighted analysis;
   report task-specific effects rather than pooling records.
5. **Cheap E2 but expensive E5:** Prioritize one 400-evaluation cell per family before deeper
   prefixes or mechanism work.

## Final checklist

- [x] Extension is explicitly separate from Freeze B.
- [x] All candidate revisions, prompts, frames, call schedules, tests, and claims are sealed.
- [x] Every family receives identical preflight and solo admission rules.
- [ ] Every admitted cell receives joint, exact-context, solo, and fake-batch evaluation.
- [ ] One repeated-sampling cell is selected per family before batching outcomes.
- [ ] All first responses, failures, retries, worker time, and spend are retained.
- [x] All five checkpoints passed the signed runtime/access gate.
- [x] The unified A4 runner passes focused and repository-wide CPU validation.
- [ ] Family breadth is completed before optional depth.
- [ ] Claims are limited to the checkpoints and tasks actually completed.
