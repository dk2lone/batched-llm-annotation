# IEEE BigData 2026 research-idea handoff

Updated: 2026-09-14 UTC.

## Current handoff: A7 sealed; A8 family-core B3 work is next

The CPU-only X1R4 A7 admission lock is sealed at
`study-120h/admission-lock/20260914T012453Z-X1R4-A7/`. Its readable report lists
all 20 solo screen passes and failures. The seven admitted cells use frozen
power-rule sizes of 2,048 or 4,096 items. The lock binds the four-family core,
four repeated-sampling cells, selected B3/B4 call hashes, four logical
A100-40GB family-owned worker slots, and the separate 80-hour budget. It used
no GPU and produced no model output. A8 is the next paid stage and has not
started; its first four B3 cells are Qwen/MultiNLI, Granite/IMDb,
Llama/Civil Comments, and Mistral/MultiNLI in that sealed order. Each model
requires a first-1,000-attempt reforecast before its block continues.

## Previous handoff: A6 complete; A7 admission lock was next

The X1R4 A6 blinded solo screen completed on the `melissa21cn2016` Modal
workspace as run `20260914T003735Z-5a8c63f2`. Five A100-40GB workers returned
all 40,960 unique solo calls across 20 cells with `ok` status, no retries, and
no quarantined payloads. The first-1,000-attempt reforecast released the full
screen. The runner metered 1.383838 all-worker A100-hours; Modal showed $3.06
metered workspace cost after completion, offset by credits at that time. The
public result and per-cell failures are in
`study-120h/extension-m1/20260914T003735Z-5a8c63f2/REPORT.md`; private first
responses and ledgers remain under `study-120h/ext-work/extension-m1/`.
The code, report, runtime/access lock, and raw A6 payload are archived in the
private GitHub prerelease `v0.3.0-extension-m1` with verified SHA-256 asset digests.

Seven cells in four families passed the frozen solo gates: Qwen/Civil Comments,
IMDb, MultiNLI; Granite/IMDb; Llama/Civil Comments, IMDb; and Mistral/MultiNLI.
Phi admitted no cell because its output seldom matched the common grammar. A6
does not test joint-call dependence or interval coverage. No extension joint
output exists. A7 must seal the admitted panel and item counts before any B3
or B4 call. Do not rerun A6 or begin A7 merely because this handoff was opened.
The separate Freeze B Granite-8B/Civil Comments E2 cell is still unlaunched.

## Previous handoff: A0-A5 complete; A6 solo screen was next

The active human-facing execution plan now uses A0-A13 in
`refine-logs/EXPERIMENT_PLAN.md`. Existing sealed records keep their immutable M1, B3, and B4
identifiers. A0-A3 preserve prior evidence and freeze the five-checkpoint, four-task protocol. A4,
the unified runner, passes 78 focused tests, all 417 project CPU tests, Ruff, strict mypy, ledger
checks, and real plan-only validation. Its three implementation files remain untracked. A5 is the
completed synthetic preflight described below. A6 is the 40,960-call blinded solo screen; it has
not started and requires separate paid-work authorization.

The cross-family X1R4 synthetic batch-invariance preflight is complete. Run
`20260913T191010Z-4807e7d1` checked all five pinned model families across 360 events and 20
anchor comparisons in two fresh sessions per model. The report status is `passed`, its sealed plan
SHA256 is `167ce1fb6fd7e279559991b522df961529c87a8fdee29502ce20f9e7bd067188`, and it used
0.811174 metered A100-hours. This was synthetic validation only and produced no extension study
labels.

The separate synthetic runtime/context pass `20260913T200450Z-a1f3bd0f` also passed all five
candidates: 100 context-fit checks, 100 warmups, and 200 measured calls returned successfully.
The largest prompt plus response cap required 20,514 of the shared 32,768 tokens. Its report is
`study-120h/extension-runtime/20260913T200450Z-a1f3bd0f/report.json` (SHA256
`19302439372935b9735d0c88ed7f7056a4ee72a50959b81eb23f6c528074b1cc`); the batch
report is `study-120h/extension-batch-invariance/20260913T191010Z-4807e7d1/report.json`
(SHA256 `64d60961d9be472f094d082c00ab40b075e617c64401c235b4416f6ed1c8eec0`).
The runtime pass used 0.687433 metered-proxy A100-hours, bringing the two A5 technical passes to
1.498607 hours. The runtime report's `peak_gpu_memory_bytes = 0` is not a valid model-memory measurement:
vLLM allocates model memory in a child process. Do not use that field for a capacity claim.

The Llama, Mistral, and Phi checkpoints were staged at their pinned revisions without GPU use.
The staging record retains hashes and byte counts, not model weights or credentials. The technical
runtime amendment changes Modal's container scale-down window from one second to the supported
two-second minimum and does not change any scientific schedule.

The signed runtime/access lock is
`study-120h/runtime-lock/20260913T205646Z-X1R4/` (manifest file SHA256
`f96a96f6691c575f1004de91dde540e722d1dd42fb4db97d8a5f46886001066a`). It marks all
five candidates attempted and none technically unavailable. The signer field attributes this
operational attestation to Codex on Daniel's instruction; it is not a cryptographic or personal
signature. All 18 copied evidence and lock files hash-verify. Both paid preflight apps are
stopped, and all 55 X1R4 sealed files still match their hashes. The next scientific stage is A6,
the 40,960-call blinded solo screen recorded as M1 in the seal; it has not started. Freeze B
is unchanged: the Granite-8B/Civil Comments E2 cell remains implemented but unlaunched, and the
analysis code's hardcoded first-cell defect must be fixed before that second cell can be analyzed.

## Previous handoff: cross-family extension X1R4 CPU protocol sealed

Daniel authorized A1, the complete CPU freeze for the cross-family extension. That work is done.
The extension is a **new, separate** study from Freeze B. Freeze B admitted only Qwen and Granite
cells and its admission lock records `recurrence_claim_not_supported`. Qwen and Granite joint
results are already open, so the extension is disclosed as new work on new untouched frames. It
does not rewrite or extend the Freeze B claim family.

The active seal is [`study-120h/freeze/20260913T183353Z-X1R4/`](study-120h/freeze/20260913T183353Z-X1R4/),
with manifest SHA256 `3838c6e201388480eb08aa48e04a1ef3a5af6a86fac5d920e9837459382677d2` and
identity SHA256 `42d88c5212e46640caf49032264b420a71bdfc2064792eee6681e905c4201492`. It hashes 55
files and contains no dataset text, no model output, and no credential. The readable protocol is
[`study-120h/EXTENSION_PROTOCOL.md`](study-120h/EXTENSION_PROTOCOL.md).

Four earlier CPU seals remain preserved but are superseded. X1 fixed only cell-block order, used a
Freeze B power seed despite recording an extension namespace, and hardcoded Llama as non-runnable.
X1R1 fixed those defects but its aggregate B4 evaluation digests did not bind sampled document
order. X1R2 bound the exact M1 prompt order, every B3 prompt/composition/order, and every B4 sampled
document order. X1R3 added a vLLM batch-invariance gate but did not validate the reported worker
revision, anchor ID, or request layout against its frozen plan. X1R4 inherits every X1R2 scientific
schedule hash unchanged and adds the complete fail-closed batch-invariance gate before any
extension output exists.

X1R4 requires `VLLM_BATCH_INVARIANT=1` before vLLM imports and fails closed when it is absent. Its
sealed plan SHA256 is `167ce1fb6fd7e279559991b522df961529c87a8fdee29502ce20f9e7bd067188`.
For each attempted checkpoint, it compares identical synthetic prompts at request batch sizes 1,
2, 8, 16, and 32 in both orders across two fresh sessions. The signed runtime/access lock must
bind a passing report. Validation checks the checkpoint and tokenizer revisions, anchor identity,
request batch size and order, vLLM version, A100 type, and serving settings. This removes GPU
request-batch numerics as an uncontrolled difference between joint and separate-call conditions;
the hardware preflight itself has not run.

**Zero extension model outputs exist.** No GPU, Modal app, model load, or synthetic inference has
run for this extension. Its meter reads 0.000000 A100-hours.

The frozen candidate panel is five families at immutable revisions: `Qwen/Qwen3-8B`,
`ibm-granite/granite-3.3-8b-instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`mistralai/Mistral-7B-Instruct-v0.3`, and `microsoft/Phi-4-mini-instruct`. All five revisions
resolve on the Hub and match their pins. Phi-4-mini replaces Phi-3.5-mini, which failed every
Freeze B solo screen; the swap happened before any extension output existed.

The four frozen tasks are Civil Comments, MultiNLI, IMDb, and AG News. Each supplies 2,048 screen,
8,192 confirmation, and 4,096 filler rows at selection seed 20260913. Measured overlap with the
Gate A pilot, the sealed Step 0 frames, and the Freeze B part-1 frames is **zero** for every task
and every role, and the four tasks share no document ID. Extension IMDb reads the **train** split:
the test split holds 23,310 eligible rows and Freeze B used 14,336, so the remaining 8,974 cannot
supply the 14,336 needed. Private text lives in the gitignored
`study-120h/ext-work/20260913-cpu-draft/`; only identifiers and hashes are sealed.

The budget is separate and does not touch the Freeze B allowance: meter origin 0, hard stop 80
metered A100-hours, 10 protected reserve hours, and a 1.6x start guard. The statistics are frozen
at batch size 16, three assignments, 4,999 replay draws, alpha 0.05, a 0.002 practical floor, and a
**family-equal** primary contrast over a Holm family of 1 to 20 cells. The cell-equal mean is a
sensitivity, never the gate. The assignment namespace `bigdata2026-ext-v1` and the repeated-sampling
namespace `bigdata2026-ext-e5-v1` do not collide with Freeze B, so an anchor cell never reuses a
Freeze B permutation.

The planning and analysis code is [`study-120h/src/freezeb/extension.py`](study-120h/src/freezeb/extension.py),
built with [`study-120h/build_extension_frames.py`](study-120h/build_extension_frames.py) and sealed
by [`study-120h/seal_extension.py`](study-120h/seal_extension.py), with 55 focused tests in
[`study-120h/tests/test_extension.py`](study-120h/tests/test_extension.py). Freeze B modules gained
only optional parameters whose defaults reproduce every sealed value; a dedicated test asserts
that. X1R4 inherits 40,960 individual M1 calls, 60 possible B3 cell-size plans covering 1,200,640
calls, and 20 B4 schedules covering 1,760,000 calls. End-to-end CPU code composes the frozen Wald,
32-call t, whole-call bootstrap, Clopper-Pearson, reference-uncertainty, and variance-ratio analyses.
The final X1R4 integration passes Ruff, strict mypy, and all 64 focused extension and
batch-invariance CPU tests. All 55 sealed file hashes verify. No full-suite rerun followed the
final test-fixture-only correction; the preceding full run passed 338 tests.

**Three open blockers mean the extension cannot start scientific work yet:**

1. **Llama is manually gated.** Its config and chat template return HTTP 401 without an accepted
   licence and a token, so its context fit is unverified. All five candidates remain in the frozen
   M1 order; a later signed runtime/access lock must classify each one as attempted or technically
   unavailable. Without granted access the panel narrows to four families.
2. **No synthetic runtime or batch-invariance lock exists (X003b).** No candidate has a measured
   context fit, engine check, A100 timing, or passing cross-layout output check under this
   extension. The solo screen cannot be forecast or launched until those synthetic checks run.
   They need GPU authorization.
3. **The 80-hour ceiling cannot hold the widest panel.** The pooled CPU power table selects 8,192
   items at the 0.50 error grid. All 20 cells at 8,192 items project about 86 guarded hours. The
   block meter narrows the panel before the stop is crossed, so the expected outcome is a smaller
   E2 panel, not an overspend.

**Exact restart order:**

1. Read this section and [`study-120h/EXTENSION_PROTOCOL.md`](study-120h/EXTENSION_PROTOCOL.md).
2. Read `MANIFEST.json`, `BATCH_INVARIANCE_PLAN.json`, `CANDIDATES.json`, `M1_CALL_ORDER.json`,
   `B3_PLANS.json`, and `B4_SCHEDULES.json` in the X1R4 seal.
3. Resolve Llama access before the signed runtime/access lock; otherwise record it technically
   unavailable and narrow the panel to four candidate families.
4. Only then ask Daniel to authorize the combined synthetic runtime and batch-invariance lock. Do
   not launch any GPU before that.

Freeze B itself is unchanged. Its second sealed cell,
`ibm-granite/granite-3.3-8b-instruct::civil_comments`, is still implemented, reviewed, and
unlaunched at 4.824000 cumulative A100-hours against its own 150-hour stop. One Freeze B defect
remains open and is **out of A1 scope**: `study-120h/src/freezeb/e2_analysis.py` hardcodes
`e2.FIRST_CELL` for its assignment matrices and Holm family, so it can analyze only the
Qwen/MultiNLI cell and not the sealed Granite cell.

The workspace is `dk2lone/bigdata`, branch `main`, with no PR and a large uncommitted worktree.
Nothing from this session is committed. Preserve every timestamped or sealed artifact.

## Previous handoff: first fresh E2 cell analyzed; stop before another launch

The completed historical Qwen3-8B/Civil Comments bridge passes its original frozen Step 0
scientific gate. Mean joint covariance is `0.005031`, mean exact-context covariance is
`-0.000421`, and the errors-as-correct joint-minus-exact contrast is `0.005451` with a
design-respecting shift interval of `[0.004562, 0.006298]` and one-sided `p = 1/2001 = 0.000500`.
The errors-as-errors (`0.005442`), unresolved-dropped (`0.005451`), and whole-call-discarded
(`0.005455`) bounds all remain above the frozen `0.002` practical floor. Fake-batch covariance is
`-0.000421`, inside the frozen absolute `0.001` band, and filler covariance is `-0.000211`.

The analysis verified the sealed assignment matrices before calculation and used 2,000 frozen
replay draws and 2,000 fake-batch regroupings. See
[`study-120h/step2-out/bridge/analysis/report.md`](study-120h/step2-out/bridge/analysis/report.md)
and its machine-readable `analysis.json`. This is a bounded replication on one historical
model-task cell. It is not a fresh Freeze B panel result, recurrence evidence across models or
tasks, or a confidence-interval coverage result.

The bridge analysis CPU implementation is
[`study-120h/src/step2/analysis.py`](study-120h/src/step2/analysis.py), with the entry point
[`study-120h/analyze_step2_bridge.py`](study-120h/analyze_step2_bridge.py) and tests in
[`study-120h/tests/test_step2_analysis.py`](study-120h/tests/test_step2_analysis.py). Ruff passes,
the new files pass strict mypy, all 256 CPU tests pass, and `git diff --check` passes. A full-tree
mypy invocation still reports eight unrelated Freeze B errors in `src/freezeb/frames.py`,
`src/freezeb/power.py`, `src/freezeb/claims.py`, and `tests/test_freezeb.py`; do not describe the
entire current tree as type-clean.

The first fresh Freeze B cell, `Qwen/Qwen3-8B::multi_nli`, is complete on Modal profile
`daniel21cn2016`. Run `20260913T045515Z-7311206a` produced all 17,152 sealed first attempts: 768
joint, 12,288 exact-context, and 4,096 solo. All 17,152 generation statuses are `ok`; eight parse
states are malformed because the model altered target IDs, comprising three exact-context and five
solo calls. Call IDs and attempt IDs are unique; every one of the eight execution blocks contains
2,144 calls; and there were zero retries, pending calls, or execution failures. The exact pinned
checkpoint revision was verified against all 15 manifest hashes. The four-worker synthetic
equivalence smoke passed before scientific output.

The run used 0.811461 incremental A100-hours and ended at 4.824000 cumulative study A100-hours.
This was 37.9% below the 1.306764-hour base forecast and 61.2% below the 2.090822-hour guarded
forecast. The first-1,000-attempt reforecast passed before four-worker release. The final measured
per-attempt allocations were 0.468348 seconds joint, 0.104142 seconds exact-context, and 0.150530
seconds solo. These are mixed-chunk A100-time allocations, not standalone latency measurements.
The completed Modal app `ap-yGkfF4h11zjAyjelMCfGPy` stopped with zero tasks at 01:09:53 EDT.

The frozen 4,999-draw CPU analysis is complete. Mean joint covariance is `0.008068`, mean
exact-context covariance is `0.000108`, and their errors-as-correct contrast is `0.007959`, with a
design-respecting shift interval of `[0.006273, 0.009670]`, raw one-sided `p = 0.000200`, and
interim 16-slot Holm-adjusted `p = 0.003200`. The errors-as-errors (`0.007929`),
unresolved-dropped (`0.007948`), and whole-call-discarded (`0.007911`) sensitivities all exceed the
frozen `0.002` practical floor. Mean joint accuracy is `0.739421`, exact-context accuracy is
`0.638997`, and resolved solo accuracy is `0.742850`.

This is a positive and practically meaningful result for one fresh cell. It is not L1, which is
reserved for the historical replication bridge. It cannot support L2 recurrence because only one
cell is complete and the admission lock contains ten primary cells from two model families while
marking recurrence unsupported. It does not measure confidence-interval coverage; that requires
the independent E5 repeated-sampling stream.

The result ledgers and technical reports are under
[`study-120h/e2-out/qwen3-8b--multi_nli/`](study-120h/e2-out/qwen3-8b--multi_nli/). The summary is
`SUMMARY.json`, the 1,000-attempt decision is `reforecast-1000.json`, and the final timing record is
`reforecast-final.json`. The scientific outputs are `analysis/analysis.json` and
`analysis/report.md`.

The resumable first-cell runner is implemented in
[`study-120h/modal_freezeb_e2.py`](study-120h/modal_freezeb_e2.py), with its frozen planner and
ledger checks in [`study-120h/src/freezeb/e2.py`](study-120h/src/freezeb/e2.py). It preserves the
17,152 planned call IDs, assigns whole execution blocks to workers, retains immutable first
responses and linked retries, sums every worker's metered A100 time, and blocks parallel release
until a synthetic equivalence smoke and the first-1,000-attempt technical reforecast pass. The
pre-output concurrency record and its completed smoke record are under
[`study-120h/amendments/`](study-120h/amendments/).

The executed first-cell plan SHA256 is
`d642f6c3fd5deaaefce76aadcaeeef9d577a20df513c999d715fae8e21672409`, with 768 joint, 12,288
exact-context, and 4,096 solo calls across four fixed block owners. Ruff, strict mypy on the four
runner files, all 17 focused E2 tests, and all 275 project CPU tests pass. No paid work is running,
and no second E2 cell is authorized.

The second sealed cell, `ibm-granite/granite-3.3-8b-instruct::civil_comments`, now has a CPU-only
post-first-cell reforecast. Transferring the completed four-worker E2 layout with direct
Granite/Qwen Civil Comments runtime and loading ratios gives **0.808 base, 1.293 guarded, and 2.425
incident A100-hours**. The guarded start projection, including the current 4.824000-hour meter and
the required 10-hour reserve, is 16.117109 hours against the 150-hour stop. The derivation and its
cross-model limitation are in
[`study-120h/POST_E2_CELL1_REFORECAST.md`](study-120h/POST_E2_CELL1_REFORECAST.md). No paid run for
this cell has started, and no paid launch is authorized.

The shared E2 runner now implements this Granite/Civil Comments cell under the explicit selector
`--cell granite-civil`. Its deterministic plan contains the sealed 17,152 calls and has SHA256
`81159b7ea6ebb3fcf9ecd5e0a5a1ea26730519c8200cda8ce54f7926c7178f31`. The runner pins Granite
revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`, verifies all 15 checkpoint files, uses a distinct
result namespace and concurrency amendment, and retains the first-1,000-attempt release gate.
Saved release barriers must match the exact cell and plan and explicitly authorize continuation.
Recovered remote chunks retain original-session A100 metering. The Qwen/MultiNLI plan hash remains
unchanged. Returned payloads are metered before validation, while mismatches are quarantined and
excluded from scientific ledgers. Ruff, strict mypy, 17 focused E2 tests, and all 275 project CPU
tests pass. The independent blocking review in
[`refine-logs/EXPERIMENT_CODE_REVIEW.md`](refine-logs/EXPERIMENT_CODE_REVIEW.md) reports no
remaining launch blockers.

**Exact restart order:**

1. Read this section, the frozen queue in
   [`study-120h/freeze/20260913T022442Z-B2/ADMISSION_LOCK.json`](study-120h/freeze/20260913T022442Z-B2/ADMISSION_LOCK.json),
   and the first-cell `SUMMARY.json` and raw ledgers.
2. Read the first-cell `analysis/report.md`; preserve its single-cell scope and the distinction
   between covariance uncertainty and unmeasured interval coverage.
3. Before a paid launch, verify Modal profile `daniel21cn2016`, workspace availability, and current
   credit. Then stop for Daniel's explicit authorization of the guarded 1.293 A100-hour block.

The workspace is `dk2lone/bigdata`, branch `main`, with no PR and a large uncommitted worktree
containing the Freeze B implementation, sealed artifacts, bridge outputs, and this analysis.
Preserve every timestamped or sealed artifact and do not clean or reset unrelated changes.

## Previous handoff: historical Qwen/Civil replication bridge complete

Daniel authorized the historical Qwen3-8B/Civil Comments replication bridge and two concurrent
A100-40GB workers. Modal run `20260913T032819Z-f63f33a2` completed all 12,672 new first calls:
384 joint, 6,144 exact-context, and 6,144 filler. The existing 2,048 solo first responses were
reused. All 18,432 target rows are present. There were 17 malformed first responses, zero missing
or truncated calls, zero retries, and zero ledger conflicts. This is historical replication data,
not fresh Freeze B confirmation and not yet a dependence-analysis result.

The run took 41 minutes 45 seconds end to end and used 1.2338 incremental A100-hours. The study
meter is now 4.0125 cumulative A100-hours. The first 1,000-call reforecast was recorded before
the two-worker phase. The two-A100 deviation was written before model output. See
[`study-120h/step2-out/bridge/SPEED_REPORT.md`](study-120h/step2-out/bridge/SPEED_REPORT.md)
for condition counts, timing limits, the amendment, and raw evidence. The Modal app completed and
stopped normally. The frozen scientific analysis has not been run; do not cite a replication ICC
or confidence-interval coverage finding from these outputs yet.

**Next action:** analyze the completed historical bridge under its frozen statistics before
making a scientific claim, then use the real runtime when planning the first fresh Freeze B cell.

## Previous handoff: Freeze B solo admission complete and sealed

The IEEE BigData 2026 workspace is on branch `main` with no PR. Daniel authorized the full
36-cell Freeze B solo screen on the `daniel21cn2016` Modal profile. Run
`20260913-freezeb-screen` completed all 73,728 calls across six models and six tasks with six
parallel A100-40GB workers. Every event is terminal, final, `status: ok`, and solo-only; there
were zero retries and zero over-budget prompts. The app stopped normally.

The solo screen used 1.6643 incremental A100-hours. The 2.7788-hour local meter is cumulative:
it also includes 0.1900 Step 1 hours and the 0.9244-hour synthetic runtime benchmark. Daniel
reported an approximately $6 account charge for the screen; that is an observed all-in amount,
not a new invoice snapshot. The earlier roughly $12 screen forecast was 3.20× too high in base
A100-hours. Do not apply that solo correction to the structurally different joint prompts. The
auditable correction and next-stage budget are in
[`study-120h/POST_SCREEN_COMPUTE_RECALIBRATION.md`](study-120h/POST_SCREEN_COMPUTE_RECALIBRATION.md).
Raw ledgers, summaries, timing probes, and the first-1,000-call reforecasts are under
[`study-120h/freezeb-screen/20260913-freezeb-screen/`](study-120h/freezeb-screen/20260913-freezeb-screen/).

Part 2 is sealed at
[`study-120h/freeze/20260913T022442Z-B2/`](study-120h/freeze/20260913T022442Z-B2/).
The seal recomputed all 36 cells from the append-only ledgers and verified the complete call-ID
sets, saved summaries, plan hash, power-table hash, screen-file hashes, and source snapshot.
Claude Opus 5 independently reran the ledger, admission, panel, and hash checks in reused session
`003442dd-2fa0-4db7-bf00-5613c8a15d36` and returned PASS. Ruff and all 216 CPU tests pass.

Ten of 36 model-task cells passed the frozen admission rules. The ordered primary panel is:

1. `Qwen/Qwen3-8B::multi_nli`
2. `ibm-granite/granite-3.3-8b-instruct::civil_comments`
3. `Qwen/Qwen3-4B-Instruct-2507::ag_news`
4. `ibm-granite/granite-3.3-2b-instruct::imdb`
5. `Qwen/Qwen3-8B::civil_comments`
6. `ibm-granite/granite-3.3-8b-instruct::imdb`
7. `Qwen/Qwen3-8B::imdb`
8. `Qwen/Qwen3-4B-Instruct-2507::civil_comments`
9. `Qwen/Qwen3-4B-Instruct-2507::multi_nli`
10. `Qwen/Qwen3-4B-Instruct-2507::imdb`

The held-out panel is empty. Only Qwen and Granite appear in the primary panel; OLMo and Phi are
admitted nowhere. GoEmotions and MultiNERD fail for every model. The frozen lock therefore records
`recurrence_claim_not_supported`. The study may not claim recurrence across at least three model
families, a held-out mechanism result, or a full 16-cell E2 panel. Do not backfill, relax a frozen
threshold, or rerun a failed cell to recover those claims. Qwen3-8B on AG News missed the 0.70
accuracy floor narrowly at 0.69956; preserve that result and its interval rather than rounding it
into admission.

The Part 2 gate permits joint output to begin. **Next action:** wait for Daniel's separate explicit
paid-run authorization. The first frozen unit is the historical Qwen/Civil Comments replication
bridge. Its provisional budget is 1.125 base, 1.800 guarded, and 3.375 incident A100-hours, or
roughly $4.06, $6.49, and $12.17 using the observed screen ratio. The bridge's real joint timing
must reprice the later queue. After that, E2 runs one cell at a time in the sealed order, beginning
with `Qwen/Qwen3-8B::multi_nli`. Do not start any joint call merely because Part 2 passed.

## Previous handoff: Freeze B runtime benchmark and ledger complete

The authorized synthetic long-context benchmark is complete. Six concurrent A100 workers
covered all 180 planned model/case checks: 158 were feasible, 22 were context exclusions,
and all 316 measured requests in the feasible cases succeeded. Modal billed 0.924 A100-hours
and $2.010 total for benchmark app `ap-r3dVnR4pzw7gSUwEtclnKo`; the app is stopped.
Synthetic responses are runtime evidence only and never enter scientific analyses.

The runtime ledger is sealed at
[`study-120h/runtime-lock/20260913T004054Z-BRT1/`](study-120h/runtime-lock/20260913T004054Z-BRT1/).
Its manifest SHA256 is
`6aca413a59738e60305a755dfa43c07864a2d197ae6b283a3577296500106e0f`;
all 12 files verified. The original 51-file Freeze B part-1 seal also reverified with no
hash failures. Lint, strict typing, and all 208 CPU tests pass.

The full 36-cell solo screen forecasts to 5.32 base A100-hours, 8.51 hours under the frozen
1.6x start rule, and 15.96 hours under the 3x incident case. The historical Qwen3-8B/Civil
Comments replication bridge forecasts to 1.12, 1.80, and 3.37 hours respectively. Later
panel ranges remain conditional until solo admission fixes the selected cells and item counts.
The maximum repeated-sampling program does not fit in 150 hours; the frozen execution ladder
must stop at the last complete unit that fits.

Daniel clarified that aggregate capacity across multiple accounts is 120–150 actual A100-hours.
That covers the 120-hour study target and reaches the 150-hour hard stop. The earlier $14 figure
describes only the benchmark account and is not the total experiment budget. Parallel workers
reduce elapsed time but their A100-hours add together.

The ledger status is `sealed_with_declared_instrumentation_gap`: request wall time, prompt
tokens, output tokens, loading, and billed time were captured, but prefill and decode were not
timed separately. The mandatory first-1,000-real-calls checkpoint must add that split and
reforecast the next complete block. This gap does not invalidate the wall-time forecast.

**Next action:** run the solo-only 36-cell screen, checkpointing each completed cell and summing
usage across accounts. Do not start joint, exact-context, replication-bridge, mechanism, or
repeated-sampling calls before the solo admission results and Freeze B part-2 lock are sealed.
No scientific Freeze B output has run yet.

## Previous handoff: study scope rebased after Step 1

Freeze B scientific protocol part 1 is now sealed at
[`study-120h/freeze/20260912T222618Z-B1/`](study-120h/freeze/20260912T222618Z-B1/).
Its [`MANIFEST.json`](study-120h/freeze/20260912T222618Z-B1/MANIFEST.json) hashes the
protocol, executable rules, tests, six disjoint task-frame identifier lists, pinned metadata
audits, and CPU power planning table. The copied frame lists contain IDs and source indices,
not dataset text. The manifest SHA256 is
`94d33406dcede77eb5829c31975ea700cad7650faf162d9f5f1bf26b568f71f0`; all 51
snapshot file hashes verified. The original Step 0 seal remains untouched. The new scope uses strict
solo admission, balanced model-task selection, batch sizes 2/4/8/16/32, fixed mechanism
arms, repeated-sampling prefixes, and a 150-hour meter. The CPU suite passed in full,
including the tested adapter between Freeze B assignments and the original statistic.
No synthetic paid benchmark, new solo screen, or joint output has run under Freeze B.

**Superseded status:** the synthetic benchmark runner was implemented, tested, run, and sealed
as described in the current handoff above. The next experiment remains the solo-only screens
and part-2 admission lock. Only after that lock may the original Qwen/Civil Step 2 bridge and
Freeze B joint work begin.

The active study is now [plan-120h-a100-rebased.md](plan-120h-a100-rebased.md). It treats 120
metered A100-hours as a target and 150 cumulative hours as the stop. It prepares a 16-cell
multi-family, multi-task panel and an ordered expansion through repeated-sampling precision,
batch size, mechanisms, mitigation, and held-out cells. The full six-cell, 5,000-replicate
uncertainty stream requires 1,008,000 new joint calls; Step 1 measured only solo throughput, so
that stream is a conditional full-program target, not a promised 40-hour deliverable. Measured
long-context runtime determines how far the frozen ladder goes; observed effects never do. The
plan now has deterministic fresh-replicate cycles after its breadth and mechanism blocks, so a
faster-than-expected run continues through useful work until at least 120 hours when feasible.
The 150-hour planning stop and technical/capability gates still apply; exact utilization cannot
be guaranteed before long-context timing is measured.

The completed Step 0 freeze and Step 1 screen remain unchanged as audit history. Their scientific
outputs have not been reclassified. No Step 2 or Freeze B joint, exact-context, filler,
fake-batch, covariance, mechanism, mitigation, or repeated-sampling output exists.

This workspace is now the private Git repository `dk2lone/bigdata`, branch `main`, with no PR.
The current protocol and plan edits are uncommitted. Large immutable
protocol and result bundles follow [RESULTS_RELEASES.md](RESULTS_RELEASES.md).

**Current protocol:** [study-120h/FREEZE_B_PROTOCOL.md](study-120h/FREEZE_B_PROTOCOL.md)
contains the part-1 scientific rules. [study-120h/FREEZE_B_PREP.md](study-120h/FREEZE_B_PREP.md)
is superseded preparation history. The original Step 0 seal was reverified on September 12
(70 files, no mutations).

**Paid-work boundary:** Part 1 has fixed the scientific rules, but the runtime ledger and
solo-only admission lock are still required. The old frozen Qwen/Civil Step 2 remains a bridge
experiment and can run only after part 2 is sealed. Daniel asked to be consulted for Modal
access before Step 2; no new paid A100 execution has been authorized. A synthetic long-context
benchmark must fill in cost projections before the screens and part-2 lock.

## Previous handoff: Step 1 complete; model and task claims narrowed

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
