# B3/B4 structured-output recovery plan

Recorded after B4 execution completed and before any repaired-output coverage analysis.
The original B3 and B4 ledgers remain immutable. This document defines a separate technical
recovery pipeline; it does not retroactively change the original execution protocols.

## Why recovery is needed

B4 completed 83,200 planned calls. Its final records contain 1,091 malformed responses and four
responses that remained truncated after the original retry. Under the original no-imputation rule,
all four B4 cells are coverage-unavailable.

The completed 16-cell B3 grid contains 2,875 malformed final responses and 127 responses that
remained truncated. B3's sealed malformed-output sensitivity analysis remains the primary B3
result. Repaired B3 results are secondary sensitivity evidence only.

The combined recovery inventory therefore contains exactly 4,097 calls: 1,095 from B4 and 3,002
from B3. Successful calls are never rerun.

## Recovery contract

1. Rebuild each exact original prompt from its call ID, frame, document order, target IDs, and
   prompt hash. Keep the pinned model revision, tokenizer, bfloat16 serving, deterministic seed,
   temperature, and batch-invariant runtime unchanged.
2. Change only the output contract. Each request receives a JSON schema whose `labels` object has
   exactly the requested IDs as required properties, rejects additional properties, and restricts
   values to the task's legal integer labels.
3. Use compact structured decoding. Allow 128 output tokens for one-target B3 calls and 512 for
   sixteen-target B3/B4 calls. These replace the original 84/384 caps for recovery only.
4. Make one `repair-v1` attempt per unresolved call. Store it outside the original event ledgers
   with the source attempt ID, source event hash, schema hash, prompt hash, runtime evidence, and
   raw repaired response. No second recovery attempt is allowed.
5. A repair succeeds only when generation ends normally and the existing parser returns
   `status=ok`, `parse_status=ok`, and exactly the requested IDs. Failed repairs remain unresolved.

The schema directly addresses the observed failures: 3,560 ID-set mismatches, 366 non-object
responses, 40 illegal labels, and 131 truncations across B3 and B4.

## Execution order and safety gates

Group B3 and B4 recovery calls by model and use one model-cache pod per family, avoiding duplicate
model boots. Run Qwen, Granite, Llama, and Mistral serially on one approved 80GB A100 lane.

Before paid recovery:

1. Generate and hash the complete 4,097-call inventory from the final local ledgers.
2. Pass CPU tests for exact schemas, source hashes, retry linkage, duplicate rejection, and merge
   behavior.
3. Run a bounded GPU smoke for each model covering one-target and sixteen-target schemas, legal
   label ranges, prompt reconstruction, structured-output batch invariance, and durable receipts.
4. Record a new budget and watchdog. Recovery spend may not consume or modify original B3/B4
   meters. Stop before the full run if any smoke output fails the unchanged parser.

## Analysis use

For B4, keep the original result as `coverage_unavailable_original`. A separate revised analysis
may replace an unresolved call only with its successful linked `repair-v1` output, then compute the
same reference targets, intervals, bootstrap sensitivity, and coverage gate. Report the recovery
rate and results with and without recovered calls. This is a post-execution technical amendment,
not the originally frozen no-imputation analysis.

For B3, retain the existing A9 analysis and malformed-output bounds as primary. Add a repaired
secondary analysis using the same successful-replacement rule across all 16 cells. Report whether
the family recurrence conclusions change; never overwrite the original A9 artifacts.
