# IEEE BigData 2026 handoff

Updated: 2026-09-18 after completed B3/B4 structured-output recovery.

## Current state

Project: dependence in batched LLM text annotation. Repository:
`dk2lone/batched-llm-annotation`, private, branch `main`.

The four-cell B4 execution and the post-execution B3/B4 output recovery are complete. No
runner or Runpod Pod remains. Runpod now bills only `$0.031/hour` for retained storage; the
verified balance is `$8.9472095511` against the unchanged `$80` spending limit.

The original B4 run finished all 83,200 planned calls at `2026-09-18T00:46:32Z`. Its final
records contained 1,095 unresolved outputs: 1,091 malformed and four exhausted truncations.
The completed 16-cell B3 grid contained 3,002 unresolved outputs: 2,875 malformed and 127
exhausted truncations.

The separate `repair-v1` recovery therefore covered exactly 4,097 calls:

- B3: 3,002
- B4: 1,095
- malformed sources: 3,966
- truncated sources: 131
- recovered successfully: 4,097
- still unresolved after recovery: 0

## Recovery artifacts

Inventory:

`study-120h/output-recovery/20260918T013521Z-inventory/RECOVERY_INVENTORY.json`

Inventory SHA-256:

`0d917c3a4674bf1b152aff9b5651884f01a2cfa5be1cb2468ca843b20fcbe27e`

Recovery output:

`study-120h/output-recovery/20260918T015605Z-smoke/`

Important files:

- `FULL_REPORT.json`: compact completed-run report
- `SMOKE_REPORT.json`: passing four-model smoke report
- `events.jsonl`: 4,097 recovered events, kept local and ignored by Git
- `receipts/`: 138 durable chunk receipts, kept local and ignored by Git
- `meter.jsonl`: nine matched start/settled records
- `runpod-lifecycle/`: nine matched create/ready/termination records
- `runpod-watchdog.jsonl`: nine matched arm/release records

Recovery namespace:

`repair-05f46e56bd5c4d58ab398266`

The final event ledger SHA-256 is:

`71969ef8d96bd586b52219029c00dae72470c4a372f5c1f94d599cf323c7d649`

## Final integrity audit

The final audit verified:

- 4,097 ledger rows and 4,097 unique recovery IDs;
- exact equality between the inventory and recovered ID sets;
- source-event, JSON-schema, and inventory-record hashes for every event;
- exact model, task, target-ID, and prediction-ID linkage for every event;
- all 4,097 events have `status=ok` and `parse_status=ok`;
- all raw output hashes reproduce;
- all 138 receipts reproduce the complete event ledger event by event;
- the model totals are Qwen 211, Granite 2,097, Llama 174, and Mistral 1,615;
- all nine created Pods have matching verified terminations;
- all nine watchdog leases were released after local deletion;
- all nine meter starts have matching settlements.

The conservative meter upper bound, including a one-minute allowance per Pod, is
1.4713153039 A100-hours. Live balance movement included retained storage and was consistent
with the bounded run. No paid GPU resource remains.

## Scientific interpretation

Recovery does not overwrite or retroactively change either original run.

For B4, preserve the original outcome as `coverage_unavailable_original`. A new revised
analysis may substitute only a successfully linked `repair-v1` event for its exact unresolved
source call, then run the original reference-target, interval, bootstrap-sensitivity, and
coverage calculations. Report the original unavailable result and revised repaired result
separately.

For the 16-cell B3 study, retain the original A9 malformed-output bounds as primary. Use the
repaired 16-cell results only as a secondary sensitivity analysis. Report whether the family
recurrence conclusions change after exact linked replacement; never overwrite the original
A9 artifacts.

## Recovery contract

Each repaired call used the original sealed prompt, document order, target IDs, checkpoint,
tokenizer revision, seed, zero temperature, and batch-invariant serving. The amendment changed
only the output contract: exact JSON-schema decoding and output caps of 128 tokens for
single-target calls and 512 for 16-target calls. Every unresolved call received one recovery
attempt. Successful original calls were never rerun.

The recovery design is documented in:

`study-120h/amendments/20260918T013521Z-structured-output-recovery.md`

## Code and commits

Relevant local commits:

- `d6ee79d add guarded output recovery runner`
- `89fa1b7 plan structured output recovery`
- `5092dfc refresh recovery handoff state`

The recovery implementation passed 39 targeted tests, Ruff, strict mypy, and reconstruction
of all 4,097 unique jobs. Commits remain local and have not been pushed. The branch is ahead
of `origin/main`. Many timestamped B4 execution directories are untracked user data; do not
delete, add wholesale, or overwrite them.

## Next action

Do not launch another GPU recovery. The next scientific task is to implement and run two
read-only derived analyses from the completed linked recovery ledger:

1. revised B4 coverage analysis, explicitly paired with `coverage_unavailable_original`;
2. repaired B3 16-cell sensitivity analysis, while keeping original A9 primary.

Before either analysis, verify the inventory and event-ledger hashes above. Preserve the
original ledgers, recovery events, receipts, seals, and locks. Do not expose credentials or
invoke Claude CLI unless Daniel explicitly requests it.
