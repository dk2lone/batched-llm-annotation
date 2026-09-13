# Freeze B protocol, part 1: scientific rules before model output

Status: CPU-only part-1 scientific protocol. Seal status and hashes are recorded in the separate
`freeze/` directory. The original `20260912T041633Z` Step 0 seal and its Step 1 output
remain historical records. This protocol draws no scientific result from a new model call.

## Scope and order

The Gate A pilot and Step 1 solo screen are disclosed discovery evidence. The original
Qwen3-8B/Civil Comments Step 2 is a separate frozen replication bridge, run on its original
rows and analyzed by its original code. Freeze B confirmation rows never enter that bridge.
Freeze B is a new panel selected from solo-only screens. The target is 120 cumulative metered
A100-hours, including historical Step 1 use; 150 is a stop, not a promised spend.

The six pinned task frames and eight ordered model policies are defined in
`src/freezeb/frames.py` and `src/freezeb/models.py`. All frame roles have been materialized
from pinned dataset revisions: 2,048 screen, 8,192 confirmation, and 4,096 filler records
per task, with disjoint IDs and historical exclusions. MultiNLI uses **training**, because
matched validation cannot support the independent planned frames. Only the IDs, hashes,
source indices, and reconstruction rules may be published without a separate rights check.
References are dataset labels, not adjudicated truth. For GoEmotions the endpoint is exact
label-set disagreement and label micro-F1 is a capability diagnostic. For MultiNERD it is
exact BIO-span-set disagreement and exact-span micro-F1 is a capability diagnostic. The
integer tag order and strict first-response grammars are in `src/freezeb/labels.py`.

Six candidates have a tokenizer chat template. The OLMo base checkpoint has none and is
excluded from this common chat-policy panel unless a separate, prespecified base-model
serving policy is sealed **before** a solo call. Gemma is gated and enters only after lawful
access and the same technical checks. Neither is a capability failure. A missing candidate
does not authorize a substitute selected after joint results. Model weights, tokenizer,
serving engine, precision, context limits, and access must be checked in the synthetic
benchmark/runtime lock. BF16 is primary. The OLMo Instruct limit is at most 4,096 tokens.

## Solo admission and panel lock

Every attempted model-task candidate gets the same 2,048-row screen frame, one label per
item in a target-only solo call. A response is parsed once, without repair. A technical
candidate must fit the largest applicable rendered prompt and response cap in its actual
tokenizer/context window and load in the pinned serving engine. Count load, context, and
generation failures separately. `src/freezeb/admission.py` then requires first-response
parse rate >=0.98 and parsed-row disagreement between 0.05 and 0.80. It requires overall
accuracy >=0.70 for Civil/AG/IMDb and >=0.55 for MultiNLI. Civil also needs balanced
accuracy >=0.60 and toxic-class recall >=0.25, because 93% of its new screen frame is
non-toxic. GoEmotions needs exact-set accuracy at least five percentage points above its
screen majority-label-set baseline and label micro-F1 >=0.35. MultiNERD needs exact-span
accuracy at least five points above its screen majority-span-set baseline and span
micro-F1 >=0.30. These are capability screens, not hypotheses about batch dependence.
Report all failures and thresholds with uncertainty; no joint output can override them.

`src/freezeb/selection.py` orders admitted cells by maximal family/task matching per round
with fixed model/task registry tie breaks. The first 16 are primary E2 cells, the next 12
held-out cells. E3 and primary E5 use the first six primary cells. E4 discovery uses the
first four primary cells; E4 held-out uses the first four held-out cells. Held-out E5 uses
the first six held-out cells. If a list is short, run only its available prefix and narrow
the claim; do not backfill based on E2 effects. Recurrence requires at least three model
families and three task families in the primary panel, including at least four of each
available before calls. Seal the exact admitted list and chosen item counts as part 2
after solo screens, before the replication bridge or any Freeze B joint output.

The item-count options are 2,048, 4,096, and 8,192 distinct confirmation rows. Apply
`src/freezeb/power.py::choose_n` to the solo disagreement rate and the sealed power table:
take the next higher error-rate grid row and the smallest size whose Wilson 95% lower
Monte Carlo power bound reaches 0.80 for a 0.004 covariance effect; otherwise use 8,192
and mark the target unmet. The table is a **planning proxy**, not a claim of calibrated
power for the final replay test. The test has repeated items and heterogeneous difficulty,
which this simple beta-binomial generator does not reproduce. If error is below 0.05, the
cell fails the information rule. Report actual precision and null behavior after data,
without silently changing sample size.

## Calls, comparisons, and failure handling

The same selected confirmation IDs occur in each condition. For batch sizes 2, 4, 8, 16,
and 32, `src/freezeb/assign.py` makes three length-balanced assignments. Each item occurs
once per assignment, in a distinct position where feasible. Assignment and call IDs,
execution block (call index mod 8), stratum, doc order, target ID, prompt variant, first
response, parse state, token counts, wall time, and all-worker A100-seconds belong in an
append-only call ledger. Interleave conditions within each execution block using the
frozen call ID order, then rotate condition order by block. Retries are linked new calls;
never overwrite first responses or quietly replace a failed result.

E2 runs batch-16 joint calls, a separate exact-neighbor target-only call for each item
appearance, and a solo call per distinct item. The exact control presents identical
neighbors and document order while requesting only the target's label. Fake batches regroup
independent exact-control labels using the frozen assignment generator. Pair-weighted
within-call covariance of binary reference disagreement is computed separately by arm;
the primary signed effect is joint minus exact, averaged equally across the three
assignments. Marginal accuracy, parse failure, and fake-batch contrast are separate.
`step0.stats` supplies the covariance, paired replay test, Holm adjustment, and unresolved
label bounds. Replay permutes within length stratum and execution block, preserving every
item's three appearances. Pass each call matrix transposed through the tested Freeze B
adapter. Use 4,999 fixed-seed draws per cell, one-sided for positive excess covariance,
and the replay-centred shift interval for descriptive uncertainty. Treat unresolved
labels as correct, as incorrect, dropped, and whole-call
discarded. The primary effect is reported with all four bounds and failure clustering.

The E2 primary multiplicity family contains all 16 preassigned primary cells, including
unrun/failed cells as no rejection. Apply Holm at 0.05 to completed cell tests. Also test
the equal-cell mean against independent, aligned replay draws over all completed primary
cells at 0.05, with unrun cells preventing the full-panel claim. L2 recurrence requires a
positive equal-cell mean with global p<=0.05, at least three Holm-positive families and
three Holm-positive tasks, every counted cell effect >=0.002, and a positive leave-one-
family-out and leave-one-task-out equal-cell mean. Report the entire signed panel. L1 is
limited to the separate historical bridge and its original sealed gate.

E3 uses the first six primary cells. It runs sizes 2, 4, 8, 16, and 32, each with its own
exact-context target-only control and three assignments. The E2 baseline size-16 calls
are reused only if prompts, assignments, and IDs match exactly; otherwise generate new
calls. Two frozen prompt variants at size 16 are `definition_last` and `brief_directive`
in `src/freezeb/prompts.py`. Baseline prompts remain fixed. Batch-size dose is a
prespecified descriptive slope of signed excess covariance against log2(size), with
cell-level slopes and equal-cell mean reported; Holm-adjust the six cell slopes as one
secondary family. Variant-versus-baseline differences form a separate 12-test Holm
family. A null or reversed pattern is reported as such.

E4 is limited to exact-context output targeting, fixed-composition position rotation,
and homogeneous-versus-mixed composition on the four discovery and four held-out cells.
`src/freezeb/mechanism.py` selects 256 confirmation items from each of two fixed
reference categories: toxic/non-toxic for Civil, positive/negative for IMDb, class 0/other
for AG and MultiNLI, neutral-only/other for GoEmotions, and person-entity-present/absent
for MultiNERD. In each of three assignments, 32 mixed calls contain eight items from
each category, while 32 homogeneous calls contain 16 from one category. Every arm uses
the same 512 items. Mixed call membership is held fixed while positions rotate by
0, 4, 8, and 12 slots. The same baseline prompt is used throughout. If 256 items in a
category are unavailable in a locked cell, that intervention is technically unrun,
not replaced by another category. A conventional decoder cannot hide
earlier generated decisions while still producing one shared autoregressive response;
therefore that proposed intervention is **not** an identified mechanism test in this
protocol. No output-history causal claim is available without a separate valid policy
sealed before joint calls. Held-out intervention contrasts form one Holm family per
intervention. Mechanism claims require a held-out adjusted p<=0.05 and a coherent effect
direction in at least three of four cells, with no model/task coverage overclaim.

E5 uses the first six primary cells, then six held-out cells in their frozen order.
`src/freezeb/repeated.py` selects 512 distinct item appearances per replicate, randomly
partitions them into 32 fresh size-16 calls, and uses separate seed streams for 250
reference replicates and 400/1,000/2,500/5,000 evaluation prefixes. A reference stream
estimates the mean reference-relative accuracy under the same item/partition/prompt/model
policy. No output is reused between replicates. The primary item-level 95% Wald interval
and call-aware 95% t interval over 32 call means use no finite-population correction;
finite-population correction and whole-call bootstrap are sensitivity checks. Report
empirical coverage and exact Clopper-Pearson interval, mean width, empirical variance
ratio to mean nominal iid variance, and reference-target Monte Carlo uncertainty using
`step0.stats`. A measured coverage claim needs at least 400 evaluation replicates and
reference SE <=25% of mean nominal half-width. Inferential damage L4 needs at least two
prespecified cells with item-level coverage <=0.90, reference-uncertainty worst-case
upper coverage bound <0.95, and Holm-adjusted one-sided binomial p<=0.05 across the six
primary cells. Call-aware calibration is a separate claim requiring its lower coverage
bound >=0.90 in at least those cells. Other outcomes remain descriptive unless their
truth/null generator and call schedule are separately frozen before any relevant output.

E6 uses E3/E4 generated policy arms and E5 interval methods. Report accuracy,
covariance, coverage where directly measured, throughput, token use, and metered cost.
A policy is not called a mitigation solely because covariance fell; compare the fixed
accuracy/throughput/cost frontier. E7 multilingual expansion is outside this part-1
confirmation lock until datasets, language-specific capability checks, and labels are
independently frozen. It cannot be selected from positive E2 results.

## Meter and runtime lock

Before scientific calls, a separately authorized synthetic benchmark must measure short,
size-16, and size-32 prompts and exact controls on each candidate in the pinned engine.
Its outputs never enter the study. Seal checkpoint access, tokenizer hashes, engine/image,
context fit, GPU memory, prompt/output tokens, wall time, A100-seconds including loads and
failures, and base/1.6x/3x projections for every complete block. The runtime lock may
exclude technical failures or shorten the fixed queue by affordability; it cannot rewrite
the scientific rules here.

The queue is: benchmark; solo screens; part-2 admission lock; original replication bridge;
E2 panel in balanced order; first two E5 cells through 400 evaluations; E3 six cells;
remaining E5 primary prefixes (1,000, 2,500, 5,000); E4 discovery/held-out; E2 held-out
and held-out E5 prefixes; E6 reused-output analysis; then fixed capacity-fill E5 cycles
of 1,000 fresh evaluations per cell with 250-replicate-per-cell final prefixes. Do not
start a complete unit if 1.6x forecast plus a 10-hour reserve would exceed 150 metered
A100-hours. Reforecast after each block using technical timing only. Check the meter after
every call and stop new calls at 150; preserve an incomplete unit but exclude it from its
confirmatory gate. At or above 120, stop after the current complete affordable unit.
If admissible candidates, scientific gates, or the 150-hour stop end the queue earlier,
report the actual shortfall rather than creating result-dependent calls.
