# Freeze B solo-screen runbook

Updated: 2026-09-13. Status: implementation complete and audited; paid execution not started.

This runner performs the frozen solo-only admission screen: six models by six tasks by 2,048
items, or 73,728 independent one-document calls. It cannot issue joint calls. The same run ID
and local output directory allow a stopped account to continue from completed terminal calls.

## 1. Inspect the plan without running a GPU

```bash
cd /Users/dz/bigdata/study-120h
uv run modal run modal_freezeb_screen.py
```

Success prints a 36-cell, 73,728-call plan with `"paid_execution": false` and exits. Failure
means a sealed frame, runtime-lock file, or local dependency is missing. This command does not
invoke a worker or download a checkpoint. Modal may register the app and build its container
images during this plan-only check; it does not request a CPU or GPU worker.

## 2. Start the screen only after paid execution is authorized

Use one stable run ID for every continuation. Replace `PROFILE_NAME` with a configured Modal
profile:

```bash
cd /Users/dz/bigdata/study-120h
MODAL_PROFILE=PROFILE_NAME uv run modal run modal_freezeb_screen.py \
  --execute \
  --run-id 20260913-freezeb-screen \
  --max-parallel-models 6
```

Success requires `MODAL_PROFILE`, acquires an exclusive local lock for the run directory, and
starts at most six A100-40GB workers, one per model. Each worker commits every chunk to the
account's result volume before the client records it locally. Failure leaves completed local and
remote chunks intact; rerunning the identical command resumes them after the first process exits.

The first successful call for each model runs alone as a timing probe. If vLLM does not expose
request timestamps, the runner makes one additional one-token technical call and records the
single-call wall-time difference. The runner stops if neither method yields positive timing; it
preserves the unusable probe under a unique filename, then retries the probe automatically on
resume. It does not report timing from the later 32-call batches as a prefill/decode decomposition.

The six workers reduce elapsed time, but their worker-seconds are added to the same A100-hour
meter. Before each complete cell, the runner applies the frozen 1.6x forecast, 10-hour reserve,
120-hour target, and 150-hour stop. It checks the hard stop after every returned chunk.

## 3. Continue on another account

Stop the first invocation before changing profiles. Then rerun the same command with a different
profile and the same run ID:

```bash
cd /Users/dz/bigdata/study-120h
MODAL_PROFILE=NEXT_PROFILE_NAME uv run modal run modal_freezeb_screen.py \
  --execute \
  --run-id 20260913-freezeb-screen \
  --max-parallel-models 6
```

The new account has a separate remote volume, while the shared local terminal-call ledger skips
completed calls. A second process against the same local run directory is rejected by the run
lock. Within one process, model workers already run concurrently.

To assign only particular models to an account, add a comma-separated `--models` value. The
saved plan remains the full frozen 36-cell plan:

```bash
MODAL_PROFILE=PROFILE_NAME uv run modal run modal_freezeb_screen.py \
  --execute \
  --run-id 20260913-freezeb-screen \
  --models Qwen/Qwen3-8B,ibm-granite/granite-3.3-8b-instruct
```

## 4. Seal admission lock part 2 after all cells finish

```bash
cd /Users/dz/bigdata/study-120h
uv run python seal_freezeb_part2.py freezeb-screen/20260913-freezeb-screen
```

Success creates a new immutable-style timestamped directory under `freeze/` containing the full
candidate outcomes, ordered primary and held-out panels, item counts, and complete block queue.
Failure names an incomplete, duplicate, unexpected, or non-solo call ledger. No joint experiment
may start until this command succeeds and the new seal is audited.

## Checkpoints and outputs

- `freezeb-screen/<run-id>/plan.json`: full frozen plan and content hash.
- `freezeb-screen/<run-id>/cells/`: append-only terminal events for all cells.
- `freezeb-screen/<run-id>/meter.jsonl`: all-in metered worker time by returned chunk.
- `freezeb-screen/<run-id>/meter-state.json`: cumulative A100-hours and worker sessions.
- `freezeb-screen/<run-id>/timing-probes/`: one usable component-timing record per model, plus
  uniquely named unusable attempts retained for audit.
- `freezeb-screen/<run-id>/reforecast-1000/`: per-model loading, single-probe timing, failure,
  and projected full-screen cost after its first 1,000 real calls.
- `freezeb-screen/<run-id>/summaries/`: solo admission metrics and uncertainty by cell.
