# B4 / A11 operations

B4 measures repeated-sampling uncertainty in the four cells selected by the A7 solo screen.
The initial prefix is 250 reference and 400 evaluation replicates per cell, with 32 fresh
size-16 calls per replicate: 83,200 first calls. Frozen protocols and their seals are unchanged.

## CPU preparation

From `/Users/dz/bigdata/study-120h`:

```sh
.venv/bin/python modal_extension_b4.py --output extension-b4/initial
```

This verifies all source seals, private frames, and full nested B4 schedules. It writes an
immutable `PLAN-400.json`. It does not import the Modal transport or contact a remote service.
Success prints `first_calls: 83200` and `gpu_work_launched: false`; changed/missing seals or
frames stop preparation. The private frame files are not bundled with this implementation.

## Execution boundary

Do not execute until GPU work is separately requested. Neither A8's execution release nor its
budget waiver authorizes B4. The B4 80-hour extension ceiling, 10-hour reserve, and 1.6 forecast
guard still apply. Prior extension use must be verified rather than assumed to be zero.

An operator budget input is a JSON object with these fields:

```json
{
  "verified_at_utc": "REPLACE_WITH_FRESH_TIMEZONE_AWARE_UTC_TIME",
  "workspace_enabled": true,
  "exclusive_gpu_work": true,
  "prior_extension_a100_hours": "REPLACE_WITH_METERED_PRIOR_EXTENSION_USE",
  "spend_limit_usd": "REPLACE_WITH_VERIFIED_LIMIT",
  "spend_used_usd": "REPLACE_WITH_CURRENT_SPEND",
  "a100_usd_per_hour": "REPLACE_WITH_VERIFIED_RATE",
  "base_forecast_hours": {
    "ibm-granite/granite-3.3-8b-instruct::imdb|reference": "REPLACE",
    "ibm-granite/granite-3.3-8b-instruct::imdb|evaluation": "REPLACE",
    "meta-llama/Llama-3.1-8B-Instruct::civil_comments|reference": "REPLACE",
    "meta-llama/Llama-3.1-8B-Instruct::civil_comments|evaluation": "REPLACE",
    "mistralai/Mistral-7B-Instruct-v0.3::multi_nli|reference": "REPLACE",
    "mistralai/Mistral-7B-Instruct-v0.3::multi_nli|evaluation": "REPLACE",
    "Qwen/Qwen3-8B::civil_comments|reference": "REPLACE",
    "Qwen/Qwen3-8B::civil_comments|evaluation": "REPLACE"
  }
}
```

Replace numeric placeholders with numbers. The prior-use baseline excludes this run directory,
whose charges are added from `meter.jsonl`. Spend used also excludes this run directory's charges.
Evaluation forecasts cover the requested full prefix. Evidence is operator-verified, not fetched
by preparation. Do not put account tokens or credentials in this file.
Forecasts include fresh model startup. Resumes retain the full forecast rather than prorating
startup costs by the number of missing calls.

```sh
.venv/bin/python prepare_extension_b4_budget.py --plan extension-b4/initial/PLAN-400.json --input budget-evidence.json --profile VERIFIED_PROFILE --output budget-record.json
```

This validates and hashes the local evidence, binds it to the exact plan/profile, and refuses
overwrites. Verification expires after one hour. It does not grant new execution authority.

Only after authorization, the execution command is:

```sh
MODAL_PROFILE=VERIFIED_PROFILE .venv/bin/python modal_extension_b4.py --output extension-b4/initial --execute --profile VERIFIED_PROFILE --budget-record budget-record.json
```

This is the only GPU-starting path. One GPU lane runs all references before evaluation, with
BF16, greedy decoding, vLLM 0.28.0 and required batch invariance. Each complete stream needs a
guarded forecast that fits the protected ceiling and verified spend capacity. Whole-phase wall
accounting includes model startup, generation, recovery and shutdown, plus 60 seconds of
settlement allowance. A timeout cancels the outstanding call and terminates its container.
An ambiguous failure leaves the meter unsettled and blocks further launches.

Reusing the command/output resumes missing attempts without reusing a response for a new
scientific call. Raw responses, serving receipts and linked technical retries are preserved.
There is one retry for generation failure or truncation, none for malformed JSON.

## Interrupted billing and analysis

```sh
.venv/bin/python reconcile_extension_b4_meter.py --run extension-b4/initial --phase-id UNSETTLED_PHASE_ID --billed-a100-hours VERIFIED_HOURS --billing-evidence billing-evidence.txt
```

This appends a settlement using verified billed time for one interrupted phase. It does not
overwrite old records or start workers. Verify that no remote call/container remains active
before reconciling. The evidence file is hashed, not copied into the ledger.

```sh
.venv/bin/python analyze_extension_b4.py --input extension-b4/initial --output extension-b4/analysis-400
```

This CPU-only command rebuilds the sealed plan, verifies response bytes/parses and matching
worker receipts, and runs the frozen independent-label, call-level t and whole-call-bootstrap
calculations. It writes immutable JSON, a report and a reproducibility manifest. Missing or
unresolved required labels make coverage unavailable, with no imputation or discarded replicates.
Reference precision and the four-cell Holm-corrected C2 gate govern any coverage claim.
An empty prepared run produces an unavailable-results report, never scientific findings.

## Nested prefixes

The runner accepts `--prefix 1000` or `--prefix 2500` in the same output directory and writes a
new immutable prefix plan. Preparation is CPU-only. Subsequent execution requires fresh budget
evidence bound to that new plan and separate authorization. The runner completes the 400 prefix
for every family before 1000, and every 1000 prefix before 2500. Existing nested-prefix responses
remain attached to the same call IDs. Analyze using the largest executed prefix and a new output
directory. Selecting expansions from favorable scientific results is not allowed.

## Local verification

```sh
.venv/bin/pytest tests/test_extension_b4.py tests/test_extension.py
```

These tests use synthetic labels and local files, including real schedule hashes when private
frames are available. They do not import the GPU transport or launch model work. Remote execution
still needs a separately authorized smoke run to verify the installed worker and infrastructure.
