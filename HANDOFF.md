# IEEE BigData 2026 handoff

Updated: 2026-09-18 03:02 UTC during the active B3/B4 output-recovery run.

## Current state

Project: dependence in batched LLM text annotation. Repository:
`dk2lone/batched-llm-annotation`, private, branch `main`.

The original four-cell B4 run is complete. It finished all 83,200 planned calls at
`2026-09-18T00:46:32Z`, deleted its final Runpod Pod, and left 1,095 terminal unresolved
calls: 1,091 malformed outputs and four exhausted truncations.

The completed 16-cell B3 grid has 3,002 terminal unresolved calls: 2,875 malformed outputs
and 127 exhausted truncations. The combined recovery inventory therefore contains exactly
4,097 calls. Successful original calls are not being rerun.

A structured-output recovery smoke passed for all four models and both observed request
shapes, one target and 16 targets. The full recovery is active now. At this handoff:

- 2,312 of 4,097 recovery events are durable locally;
- all 2,312 have `status=ok` and `parse_status=ok`;
- Qwen is complete at 211 of 211;
- Granite is complete at 2,097 of 2,097;
- Llama has only its two smoke calls recorded, with 172 left;
- Mistral has only its two smoke calls recorded, with 1,613 left;
- 1,785 calls remain overall.

These recovery outputs are a post-execution technical amendment. For B4, the original
result remains `coverage_unavailable_original`, and successful repairs feed a separate
revised coverage analysis. For B3, the original A9 malformed-output analysis remains
primary; repaired 16-cell results are secondary sensitivity evidence only.

## Active recovery run

Local runner PID: `83833`.

Command:

```bash
cd /Users/dz/bigdata/study-120h
PYTHONPATH=src .venv/bin/python run_output_recovery.py \
  --inventory output-recovery/20260918T013521Z-inventory/RECOVERY_INVENTORY.json \
  --output output-recovery/20260918T015605Z-smoke \
  --mode full
```

Do not start a second copy while PID `83833` is alive. If the process exits before
completion, first verify that no recovery Pod remains, then rerun the exact command above.
The runner uses the same recovery namespace, imports remote durable receipts, and skips every
locally recorded recovery ID. It never intentionally invokes a call twice.

Important paths:

- inventory: `study-120h/output-recovery/20260918T013521Z-inventory/RECOVERY_INVENTORY.json`
- inventory summary: `study-120h/output-recovery/20260918T013521Z-inventory/RECOVERY_SUMMARY.json`
- active output: `study-120h/output-recovery/20260918T015605Z-smoke/`
- local event ledger: `study-120h/output-recovery/20260918T015605Z-smoke/events.jsonl`
- receipts: `study-120h/output-recovery/20260918T015605Z-smoke/receipts/`
- meter: `study-120h/output-recovery/20260918T015605Z-smoke/meter.jsonl`
- lifecycle logs: `study-120h/output-recovery/20260918T015605Z-smoke/runpod-lifecycle/`
- watchdog audit: `study-120h/output-recovery/20260918T015605Z-smoke/runpod-watchdog.jsonl`
- final report: `study-120h/output-recovery/20260918T015605Z-smoke/FULL_REPORT.json`

Recovery namespace: `repair-05f46e56bd5c4d58ab398266`.

## Live Runpod state

Active model: `meta-llama/Llama-3.1-8B-Instruct`, currently booting.

- Pod: `9cfr5ssg5zt9aj`
- Pod name: `b4-4be8fcd1156e4105-33e2ed6ca285`
- Runtime status: `initializing`
- GPU: one Secure Cloud NVIDIA A100 PCIe 80GB in `CA-MTL-3`
- Price: `$1.59/hour`
- Watchdog workflow run: `35301662603`
- Watchdog lease: `4be8fcd1156e4105`
- Watchdog deletion target: `2026-09-18T04:32:12.753973Z`
- Runpod balance at handoff: `$9.6027180225`
- Current total spend: `$1.627/hour`, including `$0.031/hour` storage

The current rate and remaining work leave substantial balance margin. Qwen and Granite full
recovery are complete. Llama and Mistral run serially with separate Pods and separate
90-minute watchdogs.

## Recovery contract

The runner reconstructs each original prompt from the sealed frame and verifies its original
prompt hash, source event hash, source ledger hash, document order, target IDs, model revision,
and tokenizer revision. It changes only the output contract:

- vLLM structured decoding requires exactly `{"labels": {...}}`;
- every requested document ID is required and extra IDs are rejected;
- labels are restricted to the task's legal integer values;
- one-target calls receive 128 output tokens;
- 16-target calls receive 512 output tokens;
- each unresolved source call gets exactly one `repair-v1` attempt;
- an unsuccessful recovery remains unresolved and is not retried again.

The original B3 and B4 ledgers are immutable. Recovery events and receipts live only in the
separate output-recovery tree.

## Safety and monitoring

Every model runs on its existing offline cache volume. Pods run serially and are named with
the `b4-<watchdog lease>-` prefix so the existing GitHub watchdog can delete them remotely.
The local runner deletes and verifies the exact Pod ID before moving to the next model.

Use this read-only status check:

```bash
cd /Users/dz/bigdata/study-120h
ps aux | rg '[r]un_output_recovery.py'
wc -l output-recovery/20260918T015605Z-smoke/events.jsonl
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path('output-recovery/20260918T015605Z-smoke/events.jsonl')
rows = [json.loads(line) for line in path.read_text().splitlines()]
print('events', len(rows), '/ 4097')
print('models', Counter(row['model_id'] for row in rows))
print('outcomes', Counter((row['status'], row['parse_status']) for row in rows))
PY
```

Use the existing credential loader for live Runpod checks; do not print credentials:

```bash
cd /Users/dz/bigdata/study-120h
PYTHONPATH=src .venv/bin/python - <<'PY'
import json, subprocess
from extension_b4_runpod import credential_env

for args in (['runpodctl', 'pod', 'list', '--all'], ['runpodctl', 'user']):
    result = subprocess.run(args, env=credential_env(), capture_output=True,
                            text=True, check=True)
    value = json.loads(result.stdout)
    if args[-1] == 'user':
        value = {key: value.get(key) for key in
                 ('clientBalance', 'currentSpendPerHr')}
    print(json.dumps(value, indent=2))
PY
```

Do not kill a healthy runner or Pod while receipts advance. If the runner dies, its `finally`
block normally deletes the Pod and releases the watchdog. If a Pod remains after a crash, do
not launch another runner until the exact remaining Pod is reconciled or deleted. The active
watchdog is the independent deletion fallback.

## Code and validation

Latest local commits:

- `d6ee79d add guarded output recovery runner`
- `89fa1b7 plan structured output recovery`
- `8990c57 allow no-pod b4 continuations`

The recovery implementation passed 39 targeted tests, Ruff, strict mypy, and reconstruction
of all 4,097 unique recovery jobs. The four-model paid smoke recorded eight successful calls
and deleted every smoke Pod. Smoke evidence is committed in:

`study-120h/output-recovery/20260918T015605Z-smoke/SMOKE_REPORT.json`

The commits are local and have not been pushed. Many timestamped B4 run directories are
untracked user data. Do not delete, add wholesale, or overwrite them.

## Next action

Monitor the active recovery until `FULL_REPORT.json` exists and reports 4,097 completed
calls. Then verify zero Runpod Pods, all watchdog leases released, the final balance, exact
event uniqueness, source linkage, and counts by model/status. Commit only the compact full
report and the updated handoff, not raw events or receipts. Do not begin revised B4 coverage
analysis until those completion checks pass.
