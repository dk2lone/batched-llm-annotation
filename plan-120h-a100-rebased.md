# Rebased 120–150 A100-hour study

Updated: 2026-09-12
Status: planning rebase; no Freeze B joint-condition output has been generated
Compute rule: **120 metered A100-hours is the target, 150 is the cumulative stop**
Historical protocol: [plan-120h-a100.md](plan-120h-a100.md) retains its substantive design; the sealed Step 0 freeze remains unchanged

## Decision

The earlier plan treated 120 hours as a hard ceiling and reduced the study to a 2 × 2 panel and two small repeated-sampling cells. Step 1 showed that its short-prompt ceilings were much more conservative than observed runtime: 10,240 solo calls used 0.190 metered A100-hours. This does not establish throughput for long prompts, different checkpoints, or million-call repeated sampling. It justifies preparing a broader, ordered study before measuring those workloads.

The active design now uses a call-count-first execution ladder. Scientific units, cell order, estimands, and stopping rules are frozen before new joint outputs. Metered runtime determines how far the predeclared ladder runs, never which scientific outcomes are retained. Complete useful, prespecified blocks until cumulative use reaches at least 120 hours; finish the current complete block if it fits below 150. Stop sooner only if a technical or scientific gate makes further calls uninformative, or no complete block can fit under 150. An early finish below 120 because the initial list ran quickly is not the intended use of this budget.

The original Gate A pilot remains discovery evidence. The existing Step 0 and Step 1 records remain an auditable first protocol attempt. Freeze B is a new protocol, not a silent amendment to their decisions.

## Paper question and claim ladder

The study asks whether placing multiple records in one LLM annotation call creates excess across-item error covariance beyond item difficulty and prompt length, whether that effect recurs across model and task families, how it changes with batch size, what part of the shared-call policy causes it, and whether ignoring call membership produces false precision for mean reference-relative accuracy.

The claims are nested:

1. **L1, replication:** fresh Qwen3-8B/Civil Comments records reproduce excess covariance relative to exact-context target-only calls.
2. **L2, recurrence:** the signed joint-minus-exact covariance recurs in a fixed, balanced panel spanning at least four open model families and four task families.
3. **L3, dose and channel:** dependence changes with batch size, and shared-context versus shared-output-history interventions distinguish competing explanations.
4. **L4, inferential consequence:** actual independent annotation runs measure variance and interval coverage for mean reference-relative accuracy in frozen cells. Six cells with 5,000 evaluation replicates each are the first full-precision milestone; held-out cells and further balanced precision rounds follow if capacity remains. Type-I error, power, prevalence, model differences, regression, and ranking require separately valid targets or generators and are conditional extensions, not implied by the mean-accuracy stream.
5. **L5, mitigation:** preserving call IDs and, where useful, reducing or reshuffling batches improves uncertainty estimates or the accuracy-throughput tradeoff in held-out evaluations.

Failure at a higher level does not erase a valid lower-level result. A positive pilot alone cannot support L2–L5.

## Discovery and confirmation boundary

Before any new joint output, create and seal a timestamped protocol containing exact checkpoint revisions, dataset revisions and row IDs, prompts, response grammars, item-count rules, assignments, cell order, exclusion rules, estimands, tests, multiplicity families, simulations, code hashes, and the expansion ladder. First draft it on CPU; after a separately authorized synthetic long-context benchmark, seal its runtime ledger before scientific calls. The completed Step 0 seal cannot be retroactively changed. Run its fresh Qwen/Civil replication exactly as frozen and report it separately from Freeze B; no Freeze B decision may use that replication result.

Use three evidence strata:

- **Historical discovery:** the completed Gate A pilot and the existing Step 1 solo screens. They may shape Freeze B and must be disclosed.
- **Freeze B screening:** solo-only capability and nuisance estimates. These may determine admission and prespecified item counts but cannot reveal a batching effect.
- **Freeze B confirmation:** all joint, exact-context, dose, mechanism, repeated-sampling, and mitigation outputs. No cell or analysis is promoted because its observed effect is favorable.

Freeze the complete order of eligible model-task cells before opening confirmation outputs. If compute stops the ladder, the executed prefix remains interpretable.

## Model and task scope

### Model policies

Screen an ordered candidate ladder that covers at least four open-weight families, roughly three parameter scales that fit one A100, and one matched base/instruction comparison. The ordered candidate registry is in [study-120h/FREEZE_B_PREP.md](study-120h/FREEZE_B_PREP.md): Qwen3, OLMo 2, Granite 3.3, and Phi 3.5 are public-access families; Gemma 3 is an access-contingent eighth policy. Freeze B must verify the pinned revisions, chat templates, licenses, and serving compatibility before screening. No model enters because a joint result is favorable.

The OLMo Step 1 failure was a serving-configuration failure, not a capability result: the frozen runner requested an 8,192-token model length from a checkpoint declaring 4,096. Freeze B may retest OLMo with a per-checkpoint limit no larger than the declared context window. The historical failure remains unchanged and reported.

Use BF16 where the checkpoint and hardware support it. Do not introduce quantization for only one family inside the primary comparison. If a checkpoint does not fit the selected A100 class under the frozen common policy, record a technical exclusion and advance to its named fallback.

### Task policies

Screen six public text-annotation families in a fixed order:

1. toxicity: Civil Comments;
2. topic: AG News;
3. natural-language inference: MultiNLI;
4. sentiment: a pinned public binary or ordinal sentiment frame;
5. multilabel classification: a pinned public emotion or intent frame;
6. structured extraction: a pinned public entity or schema-extraction frame.

Freeze B must resolve the last three exact datasets, label mappings, licenses, and redistribution rules before calls begin. For every task, report disagreement with the dataset reference label rather than claiming human-label error.

### Panel structure

Screen up to eight model policies across all six task policies using 2,048 solo calls per cell, for at most 98,304 screening calls. Admit cells using only frozen context-fit, parseability, capability, and error-information rules. CPU simulations set the error-information bands before screening.

The primary confirmatory panel is the first 16 admitted cells in a balanced order that covers at least four model families and four task families. The next 12 admitted cells form the held-out expansion panel. Freeze the balanced selection algorithm and candidate order before screening. If fewer than 16 cells qualify, report the achieved panel and narrow L2 before joint calls rather than replacing cells after seeing batching results. If a screened family or task cannot support the proposed coverage, do not claim it from a conveniently selected subset.

For each admitted cell, a blinded item-count rule chooses 2,048, 4,096, or 8,192 distinct records using solo-only nuisance estimates and a frozen power simulation. The same records are used across conditions within a cell. Three independent, position- and length-balanced batch assignments are required.

## Experiments

### E1. Fresh replication anchor

Run the already prepared Qwen3-8B/Civil Comments Step 2 conditions on fresh records: joint batch-16, exact-neighbor target-only, solo, matched-length filler, fake batches, and the frozen serving controls. Analyze it under the sealed Step 0 rules. This closes the original protocol and provides a bridge to Freeze B; it is not pooled as untouched evidence in the new panel. If Freeze B also admits Qwen/Civil, its panel records and calls must be disjoint from this bridge. The Step 1 solo screen has already consumed 2,048 of this cell's 14,720 planned calls, leaving 12,672 fresh Step 2 calls.

### E2. Fixed panel recurrence

For every primary panel cell, run at batch size 16:

- joint annotation;
- exact-neighbor target-only annotation with identical neighboring records and order;
- solo annotation;
- fake batches from independent-call labels.

Retain first responses and all failures. The primary endpoint is signed, pair-weighted excess error covariance, joint minus exact. Marginal accuracy change is separate. The confirmatory recurrence contrast gives cells equal weight and requires prespecified family/task coverage, not a majority chosen after inspection.

### E3. Batch-size dose response and prompt robustness

Use six fixed cells spanning model and task families. Run batch sizes 2, 4, 8, 16, and 32 with three independent assignments per size. Each size receives its own exact-context target-only control because prompt context changes with batch size. Batch size 1 is the solo baseline.

Run two additional prompt templates in the same six cells at batch size 16. Prompt variants test robustness of the phenomenon, not prompt selection. Their order, wording, and multiplicity family are frozen before outputs.

### E4. Mechanism interventions

Use four discovery cells and four held-out confirmation cells fixed before E2 output. The minimum mechanism set is:

- **shared context without shared output history:** exact-neighbor target-only calls;
- **shared output history intervention:** vary whether earlier item decisions are visible while holding item set and token budget as closely as possible;
- **composition intervention:** compare homogeneous versus deliberately mixed batches under fixed item marginals;
- **position intervention:** rotate target positions within the same batch composition.

Only the held-out four-cell results support a mechanism claim. If interventions do not isolate a channel, report that boundary and keep L1–L2 separate.

### E5. Repeated-sampling uncertainty experiment

Freeze an ordered six-cell primary list before any E2 result. The first two cells are the minimum direct-inference block; the remaining four and their order are fixed before joint results. For each completed cell, estimate uncertainty for mean reference-relative accuracy under the actual batch-16 policy:

- 250 independent reference replicates;
- 5,000 independent evaluation replicates;
- 512 sampled item appearances and 32 fresh joint calls per replicate;
- disjoint random streams for reference and evaluation;
- no reuse of an LLM output across replicates.

This first full-precision milestone requires 6 × (250 + 5,000) × 32 = **1,008,000 fresh joint calls**. It is not an assured 40-hour stage. Freeze checkpointable evaluation prefixes such as 400, 1,000, 2,500, and 5,000 replicates per cell. The first two cells must reach at least 400 evaluation replicates each before a measured-coverage claim is considered; each later complete cell is reported at its achieved frozen prefix. Use timing, never observed effects, to decide whether the next complete prefix fits. If the full block cannot fit, report the achieved precision and leave the unrun portion explicitly untested.

The target averages over independent uniform 512-item samples without replacement *within* each replicate, random size-16 partitions, and the frozen serving policy. Items may recur across replicates but an LLM output may not. Freeze the finite-population correction choice for both interval methods before outputs. The primary comparison is the ordinary item-level 95% interval versus a call-aware interval computed from the 32 call means. Report empirical coverage, interval width, empirical-to-i.i.d. variance ratio, Monte Carlo uncertainty, and sensitivity to reference-target uncertainty. Do not substitute an ICC-derived design effect for measured coverage. Do not call the call-aware method calibrated until its own measured coverage meets a frozen gate.

The same fresh outputs may support additional *descriptive* estimands only when their targets are frozen before calls and do not reuse observed outcomes to define truth:

- emitted-label prevalence under the stated annotation policy;
- paired differences between annotation models on synchronized item samples;
- regression coefficients for an item-randomized hidden covariate and a separately generated call-correlated covariate;
- rankings of fixed candidate classifiers against the batched labels, with an independent reference ranking.

For a **confirmatory** secondary estimand, Freeze B must also specify an independent truth source, valid null generator, planted alternative if power is claimed, required calls, and its place in the compute ladder. Paired model differences need synchronized fresh outputs from both models; a mean-accuracy stream from one model is insufficient. Type-I error and power are evaluated only for estimands with those generators. Mean accuracy remains the primary paper endpoint. Gold prevalence and group differences are excluded unless Freeze B identifies a trusted-label probability frame and an independently valid bias correction. Positive error ICC alone is not evidence that every secondary estimand is damaged.

The next six frozen cells repeat the same design only through the predeclared expansion ladder and a fresh cost projection. They extend external validity; they are not selected using E2 covariance.

### E6. Mitigation and operational tradeoff

Call-aware inference is evaluated on E5 without new generation. In six fixed cells, compare the full batch-16 policy with smaller batches and deterministic reshuffling already generated in E3. The frozen mitigation ladder also includes independent decoding, composition control, and serialization changes when E4 supplies a defensible intervention. Analysis-only correction is evaluated separately from changes to annotation policy.

Report accuracy, covariance, measured interval coverage, throughput, tokens, and metered cost together. Report Type-I error only for a secondary experiment with a valid frozen null generator. Do not claim a mitigation is better if it only reduces covariance by making accuracy or throughput materially worse. Report a Pareto frontier rather than naming a universal winner.

### E7. External-validity boundary

If the core and held-out ladders finish below the target, run a prespecified multilingual block on public datasets with native or professionally verified labels and a solo capability floor in every language-model cell. Language, model family, and task must be crossed enough to avoid treating one translated dataset as a language effect.

A human-annotation comparison remains part of the full research program but is outside this GPU rebase. It requires ethics review, informed consent, trained annotators, and new non-GPU resources. Available A100 time does not authorize or solve those requirements.

## Statistical commitments

Freeze the exact algorithms in executable tests before confirmation output. At minimum:

- preserve global call ID, item ID, batch, position, assignment, execution block, prompt/output tokens, latency, first response, parse state, and linked retry ID;
- use design-respecting randomization and bootstrap procedures that account for repeated items and call clusters;
- report signed covariance and intervals, not only ICC;
- keep accuracy, dependence, variance inflation, and coverage as distinct endpoints;
- use the 0.002 absolute excess-covariance floor and 1.25 variance-ratio floor unless Freeze B simulations justify a change before output;
- freeze the full-plan L1–L5 numerical claim gates, including the at-least-two-cell inferential-damage rule, or document any simulation-based change before joint outputs; distinguish an estimate of coverage from a passed claim gate;
- define one primary multiplicity family for L2 and separate secondary families for dose, mechanism, and mitigation;
- include malformed-output bounds that treat failures as correct, incorrect, and whole-call failures;
- publish every screened and admitted cell, including null, failed, and technically excluded cells.

## Execution ladder and compute ledger

The **target is at least 120 cumulative metered A100-hours**, including the existing 0.190 hours; 150 is the planning stop, including loading, failures, retries, and parallel workers summed together. These are design boundaries, not permission to start paid work. Additional calls must advance a frozen scientific endpoint, not merely burn time. Freeze B must store estimated hours for every complete block after a long-context benchmark; the following call counts are auditable before any duration is assigned.

| Block | Required calls or formula | Completion unit |
|---|---:|---|
| Freeze B | CPU planning and tests | sealed protocol |
| Long-context calibration and screen | up to 8 × 6 × 2,048 = 98,304 solo calls; synthetic calibration separate | complete screened cell |
| Historical Qwen/Civil anchor | 12,672 fresh Step 2 calls | all frozen Step 2 arms |
| Primary 16-cell panel | at 2,048 items/cell: 98,304 exact-context target calls, 6,144 joint calls, plus solo/control; scale linearly with the frozen item-count rule | complete model-task cell with all control arms |
| First two repeated-sampling cells | 2 × (250 + 400) × 32 = 41,600 fresh joint calls | complete 400-replicate prefix per cell |
| Six-cell first full-precision stream | 1,008,000 fresh joint calls total, including the first-two-cell calls | complete frozen replicate prefix per cell |
| Dose, mechanism, held-out panel and mitigation | formulas sealed in Freeze B after checkpoint and item-count selection | complete prespecified cell or intervention block |

At 0.15, 0.50, and 1.00 **metered A100-seconds per completed joint call**, the 1,008,000-call stream alone would take 42, 140, and 280 A100-hours. These are sensitivity examples, not measured joint throughput; the middle case already exceeds the total allowance once the panel and controls are included. The draft's former 40-hour allocation for this stream assumed about 0.143 seconds per call without evidence. The 150-hour ceiling therefore cannot promise the entire full-program ladder. A measured forecast decides how much of the predeclared order fits.

Step 1 recorded $0.6813 total billed cost for 0.190 metered A100-hours, including CPU and memory. Scaling that historical mixture gives about $431 at 120 hours or $538 at 150 hours, but future model loads, worker layout, and prices may differ. Recheck current Modal access, balance, and limits before any newly authorized paid run.

### Post-screen operational correction

The completed 73,728-call solo screen used 1.6643 incremental A100-hours; 2.7788 is the
cumulative study meter after including Step 1 and the synthetic benchmark. Its frozen 5.3189-hour
base forecast was 3.20× measured use. This corrects the screen accounting but does not change the
scientific protocol, the 120-hour target, the 150-hour stop, or forecasts for structurally
different joint work. The historical replication bridge remains the next minimum paid unit at
1.125 base, 1.800 guarded, and 3.375 incident A100-hours. Its measured joint and exact-context
timing must reforecast the ten-cell E2 panel and every later block. See
[study-120h/POST_SCREEN_COMPUTE_RECALIBRATION.md](study-120h/POST_SCREEN_COMPUTE_RECALIBRATION.md).

### Runtime checkpoints

1. Before any new scientific output, benchmark short and longest-context **synthetic** prompts for each checkpoint and condition, including exact-context and size-32 calls. No benchmark output enters scientific analyses. Freeze B sets a conservative per-call estimate and a 10-hour operating reserve from those observations. The reserve can fund protocol-allowed failures or later complete scientific blocks once core work is secure; it is not permanently stranded.
2. After the first 1,000 real calls for each model policy, update prefill, decode, loading, failure, and all-in metered projections from append-only logs. Timing may change only the next complete block, never its cell selection, outcomes, or analysis.
3. At every block boundary, project remaining blocks using actual prompt-length distributions and base, 1.6×, and 3× cost scenarios. Record predicted hours, uncertainty, cumulative use, and the decision in the ledger. Use the 1.6× case plus reserve for the start decision; treat 3× as an incident scenario that triggers a new measurement and forecast.
4. While cumulative use is below 120 hours, start the next complete frozen block or smaller prespecified prefix whenever its 1.6× projection plus the remaining operating reserve fits below 150. After 120, finish the current complete block, then stop. Do not exceed 150 to finish an in-flight block.
5. At the stop, preserve partial ledgers and analyze only complete prespecified units. Report incomplete attempts, unrun blocks, and any shortfall below 120 with its measured cause.

### Expansion order when execution is faster than forecast

Complete the following fixed order. Never use effect sizes to reorder it.

1. E1 anchor and all 16 primary panel cells, subject to the pre-joint capability gate.
2. First two frozen repeated-sampling cells through 400 evaluation replicates each, plus their 250-replicate reference streams.
3. Extend those two cells to 1,000, 2,500, then 5,000 evaluation replicates at frozen checkpoints.
4. Add the remaining four frozen repeated-sampling cells at 400, then 1,000, 2,500, and 5,000 replicates in a balanced round-robin order. This prevents precision from concentrating in whichever early cell happens to show a strong effect.
5. Six-cell batch-size dose response and prompt robustness, then four discovery and four held-out mechanism cells. If the long-context benchmark shows these blocks fit earlier without threatening steps 1–4, their execution may be interleaved in a schedule sealed before any joint output; reporting priority remains unchanged.
6. Twelve-cell held-out panel extension and prespecified model-size and base/instruction contrasts.
7. Secondary inferential experiments only where their independent truth/null/alternative generators and call budgets were frozen; then held-out mitigation and multilingual external validity.
8. Six held-out repeated-sampling cells, one complete checkpoint per cell in frozen balanced order.
9. If cumulative use is still below 120, run a deterministic capacity-fill cycle across the admitted cells in the frozen 12-cell repeated-sampling list: 1,000 fresh evaluation replicates per cell per cycle, with new partitions and calls, in the frozen balanced cell order. Freeze the seed stream, selection rule, cycle order, and analysis at 0B. Repeat complete cycles; near 120 or 150, use frozen 250-replicate-per-cell prefixes so the last complete unit fits. These rounds improve Monte Carlo precision and stress-test reproducibility across cells; no result can change their order. Do not call more precision a new model/task replication.

The final capacity-fill cycle makes the queue extend beyond any plausible fast-run forecast without inventing cells after results are seen. It makes 120–150 hours a **utilization target**, not a guaranteed runtime: capability failures, a technical stop, or a sudden throughput change can still prevent it. After any prefix, claim only the evidence actually completed; no outcome may change what runs next.

## Parallel A100 execution

Two to four independent A100 workers may be used after an equivalence smoke confirms the same container, engine, checkpoint hashes, tokenizer, deterministic settings, and output grammar. Parallelism reduces wall-clock time, not metered A100-hours.

Assign complete randomized execution blocks to workers. Interleave conditions within each block, record worker ID, GPU class and UUID, image digest, start/end timestamps, and charged duration, and include worker/block effects in sensitivity analyses. Do not split a single deterministic call across GPUs or silently mix A100 memory classes.

## Paper boundary

The six-page paper should center on the strongest *completed, gate-supported* chain among fresh replication, fixed-panel recurrence, and measured mean-accuracy uncertainty. A preliminary 400-replicate cell may estimate coverage but cannot inherit the full-plan L4 claim without meeting its multiplicity, precision, and at-least-two-cell rules. Include prespecified dose or mechanism results only as warranted by their own gates. Place full panel tables, prompt variants, intervention data, and failures in an archived supplement.

The target route is the [IEEE BigData 2026 Undergraduate and REU Consortium](https://bigdataieee.org/BigData2026/undergraduate-reu-consortium/). Its call, checked September 12, requires an undergraduate first author who is the primary contributor, allows six pages including references, and lists September 20, 2026 at 11:59 PM AoE. Verify Daniel's actual eligibility and recheck the live call before submission. Eight calendar days remain from this rebase; a 120–150 A100-hour program plus protocol, analysis, and writing may not fit before that deadline. The conference manuscript may therefore use only the completed prefix, with unsupported full-program claims deferred.

## Artifact and release gates

Create milestone releases rather than committing large immutable outputs:

1. **Freeze B prerelease:** protocol, hashes, sampled row identifiers, assignments, simulation reports, environment lock, and empty expected-output manifest.
2. **Screen release:** solo ledgers, context and capability decisions, technical exclusions, throughput model, billing snapshot, and the frozen admitted-cell order.
3. **Core-results release:** every completed anchor, panel, dose, mechanism, and uncertainty block with raw first responses and failure ledgers; identify unrun blocks explicitly.
4. **Final paper release:** analysis-ready tables, reconstruction code, figures, submitted manuscript version, machine-readable manifest, and SHA-256 checksums.

Do not publish credentials, model weights, local caches, or agent traces. If dataset text cannot be redistributed, publish row IDs, hashes, sampling metadata, and reconstruction code.

## Immediate next action

Do not launch Step 2 yet. Draft Freeze B on CPU, including the exact candidate ladder, three remaining dataset choices, power-based item-count rule, six primary repeated-sampling cells, intervention implementations, execution-block schedule, and 150-hour meter. When implementation and paid calibration are separately authorized, benchmark synthetic long contexts, seal the resulting cost projections and protocol, and only then make new scientific calls.
