# Cross-family experiment extension: A0-A13 plan

**Problem:** Determine whether shared-call LLM text annotation creates excess across-item error
dependence and whether treating those labels as independent produces misleading uncertainty.

**Method thesis:** Compare joint size-16 labeling with exact-context one-target-per-call and solo
controls on the same texts across the complete four-family × four-existing-dataset grid, with no
accuracy admission gate. Use all 8,192 confirmation items from each dataset in every model cell
and preserve identical item IDs across models and conditions. Preserve the earlier seven-cell
X1R4 seal as audit history, not as a second active analysis. Measure confidence-interval behavior
with independent repeated calls in one prespecified cell per family. A later independent-dataset
panel tests replication.

**Date:** 2026-09-14 UTC

**Status:** A0-A8 execution are complete. The active CPU protocol is sealed at
`study-120h/freeze/20260913T183353Z-X1R4/` with manifest SHA256
`3838c6e201388480eb08aa48e04a1ef3a5af6a86fac5d920e9837459382677d2`. It freezes
candidates, frames, prompts, seeds, budget, statistics, and exact M1/B3/B4 schedules. Earlier X1
seals are preserved but superseded. The synthetic runtime and batch-invariance preflight passed
all five models and used 1.498607 metered A100-hours. A6 completed all 40,960 sealed solo calls
in run `20260914T003735Z-5a8c63f2`; seven cells in four families passed the frozen admission
floors. See `study-120h/extension-m1/20260914T003735Z-5a8c63f2/REPORT.md` for every cell.
Extension B3 completed all sixteen model-task cells: 548,864 scheduled first attempts and 127
required retries have terminal records, with no pending calls or irrecoverable technical failures.
The 127 outputs that remained truncated after their single retry are retained under the sealed
malformed-output treatments. No B3 scientific-result analysis has been performed. A7, the
historical admission lock, remains sealed. The active
sixteen-cell schedule and analysis are sealed at
`study-120h/full-grid-lock/20260914T032937Z-X1R4-A8/`. The active execution release at
`study-120h/execution-release/20260914T035244Z-X1R4-A8-model-major/` removes the compute-cost gate
and freezes model-major execution: Qwen, Granite, Llama, then Mistral, with Civil Comments,
MultiNLI, IMDb, then AG News inside each model. All 548,864 calls are released, with cost recorded
only for reporting. No A8 task is currently running. A9 analysis is the next stage; it requires no
new model calls. A12 orders
equal-family 1,000-replicate evaluation,
batch-size tests, and an optional independent-dataset 4 × 4. This extension is separate
from Freeze B because Qwen and Granite joint results have already been opened. Nothing below
authorizes a further paid run.
See `study-120h/EXTENSION_PROTOCOL.md` for the full frozen protocol.

## Claim map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: cross-family recurrence | The current evidence covers only Qwen and Granite. | A prospectively sealed joint-minus-exact covariance test over all sixteen cells with equal model and task weight, plus model margins, task margins, cell effects, and leave-one-model/task-out sensitivity. | B1-B3 |
| C2: practical uncertainty damage | Dependence alone does not show that ordinary confidence intervals fail. | At least two prespecified family cells have item-level coverage at most 0.90, worst-case upper coverage below 0.95 after reference uncertainty, and multiplicity-adjusted p at most 0.05. | B4 |
| Accuracy moderation | Solo accuracy is a property of the annotator, not an admission condition for comparing its call formats. | Report accuracy, parse rate, low-accuracy strata, and malformed-output bounds for every cell without dropping cells or defining the primary grid from those outcomes. | B2-B3 |
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
- **Selection:** Order tasks as Civil Comments, MultiNLI, IMDb, then AG News. Use the first cell
  passing the historical floors to select one repeated-sampling cell per family. Selection may use
  solo capability and technical timing only, never a joint-minus-exact result.
- **Interpretation:** Preserve the seven historical pass decisions and thirteen failures. These
  decisions select B4 cells and describe annotator capability; they do not select B3 grid cells.
- **Scope:** All sixteen combinations of Qwen, Granite, Llama, and Mistral with the four datasets
  proceed to the B3 grid regardless of solo accuracy or parse rate. Report every solo result.
- **Priority:** MUST-RUN.

### B3: Full batched-versus-individual experiment

- **Claim tested:** Shared generation creates excess error covariance beyond context alone.
- **Runs per grid cell:** Three independent length-balanced assignments; size-16 joint calls;
  exact-context one-target calls using identical documents and order; solo calls; fake batches
  built from independent-call labels.
- **Sample size:** Use all 8,192 confirmation items in every one of the sixteen model-task cells.
  The same 8,192 item IDs for a dataset are used across all four models and all three conditions.
  This fixed balanced choice is based on the prespecified hardest-case power simulation: at a 0.50
  error rate, 4,096 items provided 0.7322 proxy power, while 8,192 provided 0.9727 power with a
  0.9682 Wilson lower bound at planning alpha 0.0025 for an excess covariance of 0.004. Accuracy
  and parse rate cannot change inclusion or sample size.
- **Primary metric:** Equal-assignment, pair-weighted
  `joint covariance - exact-context covariance`, with equal model and task weight and a 0.002
  practical floor.
- **Inference:** Freeze the full sixteen-cell test, multiplicity adjustment, model and task margins,
  interaction, leave-one-model/task-out sensitivity, and malformed-output bounds before joint
  output. Preserve first responses and link rather than overwrite retries.
- **Run order:** Run Qwen, Granite, Llama, then Mistral. Within each model, complete Civil Comments,
  MultiNLI, IMDb, then AG News. Each model run contains 137,216 calls. Do not open results to
  change what continues.
- **Failure interpretation:** Null, negative, inaccurate, or poorly parsed cells remain in the
  table. Cross-family recurrence is unsupported unless the complete full-grid gate passes.
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
- **Expansion:** Extend evaluation prefixes to 1,000 at A12 step 1, then consider 2,500 only at
  A12 step 6, using the frozen round-robin family order and technical affordability. Do not expand
  a cell because its interim coverage is favorable.
- **Failure interpretation:** If reference uncertainty is too large, report variance and width but
  make no measured-coverage claim. If dependence recurs without undercoverage, claim dependence
  without demonstrated inferential damage.
- **Priority:** MUST-RUN through 400 evaluations; the equal-family 1,000 prefix is the next
  budget-conditional target under the 2026-09-14 pre-output amendment; 2,500 remains optional.

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
- **Priority:** Batch-size tests are A12 step 2; other B5 mechanisms have no priority over the
  ordered A12 work.

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

### A8 — Run the complete existing-dataset 4 × 4 — complete

The CPU-only full-grid lock was sealed at
`study-120h/full-grid-lock/20260914T032937Z-X1R4-A8/`. It binds the sixteen-cell schedule,
outcome rules, analysis, item IDs, selected-plan hashes, and budget-release rule. It launched no
GPU work. The active execution release is sealed at
`study-120h/execution-release/20260914T035244Z-X1R4-A8-model-major/`. It releases all sixteen cells,
supersedes the earlier alternating-model order and compute-budget policy, and fixes four sequential
model runs. There is no first-1,000-attempt pause, runtime reforecast requirement, A100-hour stop,
reserve, or cost-based block admission. Cost metering is descriptive only.

Run Qwen, Granite, Llama, then Mistral. Within each model, run Civil Comments, MultiNLI, IMDb,
then AG News. Solo accuracy and parse
rate are reported moderators, never eligibility gates. Use three joint assignments, exact-context
target-only calls with identical neighbors and order, solo calls, and fake-batch controls in every
cell. Every cell uses the same 8,192 confirmation items for its dataset. One cell requires 34,304
model calls: 1,536 joint calls across three assignments, 24,576 exact-context target-only calls,
and 8,192 solo calls. Each model run requires 137,216 calls and the full grid requires 548,864.
Only an irrecoverable technical failure may leave a cell incomplete;
scientific outcomes may not change execution order or reduce a cell below 8,192 items.

The terminal-truncation handling amendment at
`study-120h/amendments/20260914T173317Z-x1r4-b3-terminal-truncations.md` was recorded after
technical inspection of the completed Qwen cells and the Granite Civil Comments and MultiNLI
cells, but before B3 scientific-result analysis. A call whose single retry also reaches the frozen
output limit is retained as a terminal unresolved malformed output. It receives no third attempt
or changed setting and is included in all four sealed malformed-output treatments. A cell is
execution-complete when every scheduled first attempt and required retry has a terminal record;
parsed completeness and terminal malformed counts are reported separately. This uniform rule
applies to all completed and future B3 cells.

All sixteen cells are execution-complete. The four model runs contain 548,864 scheduled first
attempts and 127 required retries. Every attempt has a terminal record, no call is pending, and no
irrecoverable technical failure left a cell incomplete. The compact completion report is at
`study-120h/extension-b3/REPORT.md`; the full event ledgers are preserved separately from Git
history as a checksummed private release asset. These are execution facts, not B3 findings.

### A9 — Analyze the full 16-cell grid — next stage

Analyze all sixteen cells with equal model and task weight, cell effects, model and task margins,
interaction, leave-one-model/task-out checks, multiplicity adjustment, and every prospectively
sealed malformed-output treatment. Accuracy and parse rate are moderators only. The earlier
seven-cell X1R4 seal remains unchanged as audit history and is not a second active analysis or
claim gate. Report any technically or budget-incomplete grid without redefining its target.

### A10 — Merged into A8 — no separate stage

The former conditional full-grid stage is part of A8. This label remains only to preserve the
A0–A13 audit trail and references in earlier records.

### A11 — Run the practical uncertainty experiment — conditional (sealed B4 order)

Use the first admitted task per family selected at A7. Run 250 independent reference replicates
and 400 independent evaluation replicates per selected cell. Each replicate independently samples
512 item appearances and creates 32 fresh size-16 joint calls. Do not reuse B3 main-experiment
outputs. Compare ordinary item-level 95% Wald intervals, call-aware 95% t intervals based on the
32 call means, and whole-call bootstrap sensitivity. Report empirical coverage, interval width,
variance inflation, and reference uncertainty. Four families require 83,200 fresh joint calls;
the provisional cost is 10–15 metered or 16–24 guarded A100-hours, subject to measured B4 speed.

### A12 — Spend remaining compute in fixed order — budget-conditional

Only after every admitted family reaches 400 evaluation replicates, spend affordable compute in
this order. Do not select extra runs because an interim result looks favorable.

1. Extend all four families equally to 1,000 evaluation replicates using the sealed nested B4
   prefix and round-robin order. This adds 600 replicates and 19,200 fresh calls per family,
   76,800 calls overall. The incremental estimate is 10–15 metered or 16–24 guarded A100-hours.
   The 400-replicate C2 gate stays primary. Report any incomplete prefix as such. See
   `study-120h/amendments/20260914T013606Z-x1r4-e5-1000-priority.md`.
2. Test batch sizes 2, 4, 8, 16, and 32 in one prespecified cell per family, with exact-context
   controls and three assignments. Hold each family's item frame and assignment policy fixed
   across sizes; only the grouping and resulting call count change. Freeze the comparison and
   forecast before paid calls. The sealed size-16 condition may be analyzed once completed, but
   fresh calls are required for the other sizes.
3. Add one independent dataset for each of toxicity, sentiment, natural-language inference, and
   topic classification across all four families. Choose and freeze the four dataset sources for
   label quality, access, and task fit before model results, then run all sixteen combinations
   regardless of solo accuracy. Freeze fresh disjoint frames, prompts, response rules, schedules,
   statistics, and a separate GPU budget before joint output. Use batch-16 joint, exact-context,
   solo, and fake-batch conditions. At 4,096 items per cell, sixteen cells require 274,432 calls;
   a different item count requires a pre-output power and budget decision. Analyze this new-data
   grid separately as replication of the existing-dataset 4 × 4. Report accuracy and malformed
   responses for every cell. The earlier all-four-pass crossed-panel amendment is superseded as
   an active eligibility rule and retained as planning history:
   `study-120h/amendments/20260914T014726Z-crossed-4x4-plan.md`.
4. Add an exact-ground-truth synthetic task with a frozen generator, reference labels, and
   technical/output-format screen before scientific output. Seal its cell sizes and analysis
   separately; report accuracy without using it as a gate.
5. Add another checkpoint size per family only after fixing the checkpoints, common tasks,
   output rules, and budget before their output is seen. Report solo accuracy as a moderator.
6. Consider extending every admitted family's evaluation prefix to 2,500 last, in equal
   round-robin order. This adds 1,500 replicates and 48,000 fresh calls per family beyond 1,000.

At each step, reforecast from measured all-worker A100 time and start only complete blocks that
fit the applicable stop, reserve, and 1.6x guard. The current X1R4 ceiling covers only its sealed
work; new dataset, synthetic, and checkpoint experiments need separate pre-output budget seals.
The 512-item count remains fixed for every B4 evaluation replicate, regardless of B3 cell size.

### A13 — Final analysis and paper — pending completed experiments

Freeze all completed ledgers, rerun multiplicity-adjusted analyses, audit every reported number
against raw outputs, audit experiment integrity and claim scope, generate the family-by-task table
and covariance and confidence-interval figures, report failed screens and incomplete cells, and
write the paper. Keep historical, X1R4-confirmatory, and optional expansion evidence distinct.

The 4.824 cumulative A100-hours and the 150-hour stop belong to Freeze B. This extension holds a
**separate** ceiling, sealed in X1R4: meter origin 0, hard stop 80 metered A100-hours, 10 protected
reserve hours, and a 1.6x start guard. It does not share or consume the Freeze B allowance, and
extension calls never count toward the original sealed queue.

The A8 execution release removes compute-cost stopping from the full existing-dataset grid. Cost
is still metered and reported, but it cannot pause, reorder, resize, or cancel an A8 cell. Failed
or unrun cells remain reported. A12 remains a separate optional expansion ladder.

## Paper storyline

- **Main paper:** the sixteen-cell crossed comparison on the four existing datasets, with solo
  accuracy and parse rate shown for every cell;
  equal-family 400/1,000-replicate E5 coverage where completed. An independent-dataset 4 × 4 and
  batch-size results appear when their fixed budget-conditional steps finish.
- **Appendix:** every screen failure, every cell estimate, malformed-output bounds, larger E5
  prefixes, batch-size curves, prompt variants, position/composition tests, and cost ledgers.
- **Cut:** model-specific prompt tuning, result-dependent checkpoint replacement, commercial APIs,
  additional parameter sizes until the earlier A12 steps are resolved.

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
- [ ] Every one of the sixteen existing-dataset cells evaluates the same 8,192 items per dataset under joint, exact-context, solo, and fake-batch analysis, subject only to technical and budget stops.
- [x] One repeated-sampling cell is selected per family before batching outcomes.
- [x] All first responses, failures, and retries are retained; worker-time and spend reporting remains pending.
- [x] All five checkpoints passed the signed runtime/access gate.
- [x] The unified A4 runner passes focused and repository-wide CPU validation.
- [x] Family breadth is completed before optional depth.
- [ ] Claims are limited to the checkpoints and tasks actually completed.
