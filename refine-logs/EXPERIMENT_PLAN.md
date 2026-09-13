# Cross-family experiment extension

**Problem:** Determine whether shared-call LLM text annotation creates excess across-item error
dependence and whether treating those labels as independent produces misleading uncertainty.

**Method thesis:** Compare joint size-16 labeling with an exact-context one-target-per-call control
across five model families, then measure confidence-interval behavior with independent repeated
calls in one prespecified cell per admitted family.

**Date:** 2026-09-13

**Status:** The active CPU protocol is sealed at `study-120h/freeze/20260913T181826Z-X1R2/` with
manifest SHA256 `0d1e70e26d1a54c98ff708c82452678a262c7586aa62ef3fa3c25fe190daa9cc`.
It freezes candidates, frames, prompts, seeds, budget, statistics, and exact M1/B3/B4 schedules.
X1 and X1R1 are preserved but superseded. Zero model outputs exist for
this extension. No GPU, Modal, model load, or synthetic inference has run for it. This extension is
separate from the sealed Freeze B confirmation because Qwen and Granite results have already been
opened. Nothing below authorizes a paid run. See `study-120h/EXTENSION_PROTOCOL.md` for the full
frozen protocol.

## Claim map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: cross-family recurrence | The current evidence covers only Qwen and Granite. | A positive **family-equal** joint-minus-exact covariance test; at least three families each have a Holm-positive prespecified cell with effect at least 0.002; every leave-one-family-out family-equal mean remains positive. The cell-equal mean is a sensitivity, never the gate. | B1-B3 |
| C2: practical uncertainty damage | Dependence alone does not show that ordinary confidence intervals fail. | At least two prespecified family cells have item-level coverage at most 0.90, worst-case upper coverage below 0.95 after reference uncertainty, and multiplicity-adjusted p at most 0.05. | B4 |
| Anti-claim: extra context explains the result | Joint and solo prompts differ in both context and requested outputs. | Joint exceeds an exact-context control that presents identical neighbors in identical order but requests one target. | B3 |

Claims apply only to the frozen checkpoints, tasks, prompts, and serving policy. This extension does
not identify a population-wide property of all LLMs.

## Candidate panel

One instruction checkpoint per family, pinned to an immutable revision and sealed in X1R2:

| Order | Family | Repository | Revision | Access |
|---:|---|---|---|---|
| 1 | qwen | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | public |
| 2 | granite | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | public |
| 3 | llama | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | **manually gated** |
| 4 | mistral | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | public |
| 5 | phi | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | public |

All five revisions resolve on the Hub and match their pins. Llama file contents return HTTP 401
without an accepted licence and a token, so its configuration, chat template, and context limit are
unverified. All five candidates remain in the frozen M1 schedule. A signed runtime/access lock must
classify each as attempted or technically unavailable before the screen. Phi-4-mini-instruct
replaces Phi-3.5-mini-instruct, which failed every Freeze B solo screen;
the swap happened before any extension output existed. After the first extension joint output, a
missing checkpoint narrows the panel and never authorizes a substitute.

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
  solo, size 16, exact-context, and size 32; four-worker equivalence smoke.
- **Recorded controls:** immutable checkpoint and tokenizer hashes, vLLM/container revision, BF16,
  greedy decoding, prompt hashes, frame hashes, call IDs, assignment matrices, token limits, and
  all-worker A100 metering.
- **Success criterion:** Every primary model loads, fits the largest prompt, produces deterministic
  smoke outputs across workers, and has a measured guarded cost.
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
  exact-context one-target calls using identical documents and order; one solo call per selected
  item; fake batches built from independent exact-control labels.
- **Sample size:** Choose 2,048, 4,096, or 8,192 items from the sealed X1R2 power table and the solo
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

## Run order and decision gates

| Milestone | Work | Decision gate | Provisional guarded cost | Main risk |
|---|---|---|---:|---|
| M0 | Freeze extension, untouched frames, uniform prompts, revisions, tests, and synthetic benchmark. | No scientific output until every hash and runtime check passes. | Set by benchmark; target <=1 A100-hour. | Llama access or model-specific chat behavior. |
| M1 | Run all 40,960 solo screens. | Continue only with prespecified admitted cells; require at least three families for a cross-family study. | About 1.5 A100-hours from prior screen throughput, replaced by M0 forecast. | Phi repeats its formatting failure. |
| M2 | Run one complete E2 cell per admitted family. | Evaluate C1 only after every family-core cell is complete. | Roughly 1.3 A100-hours per 4,096-item cell, replaced model by model after 1,000 calls. | Cross-model timing transfer is inaccurate. |
| M3 | Complete remaining admitted E2 cells. | Start each cell only if its 1.6x forecast plus a 10-hour reserve fits the study stop. | Up to roughly 26 A100-hours for 20 cells at the current guard. | Too many low-value within-family cells. |
| M4 | Run 250 reference plus 400 evaluation replicates in one cell per family. | Assess C2 only after all prespecified family cells finish and reference precision passes. | Approximately 4-6 guarded A100-hours per family at current joint-call timing; benchmark again first. | Repeated sampling dominates cost. |
| M5 | Add larger E5 prefixes, then B5 sensitivity work. | Fixed round-robin technical expansion only; never choose from effect estimates. | Spend only within a separately approved extension ceiling. | Depth crowds out family breadth. |

The 4.824 cumulative A100-hours and the 150-hour stop belong to Freeze B. This extension holds a
**separate** ceiling, sealed in X1R2: meter origin 0, hard stop 80 metered A100-hours, 10 protected
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
- [ ] Llama weight access is granted, or Llama is reported technically unavailable.
- [ ] Family breadth is completed before optional depth.
- [ ] Claims are limited to the checkpoints and tasks actually completed.
