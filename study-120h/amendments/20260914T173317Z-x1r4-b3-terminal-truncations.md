# Protocol amendment: B3 terminal truncation handling

- Recorded: `2026-09-14T17:33:17Z`
- Branch: `main`
- Scope: every X1R4 B3 model-dataset cell
- Analysis status when recorded: no B3 scientific-result analysis performed

## Reason

After all four Qwen cells and the first two Granite cells had run, the technical ledger showed
68 Granite exact-context calls whose first attempt and single permitted retry both reached the
frozen output-token limit. Civil Comments had 61 such calls and MultiNLI had 7. The raw responses
show that Granite began returning labels for documents other than the one requested. Qwen had no
terminal failures in its four cells.

This amendment records how those observed responses and any equivalent future responses are
handled. It does not replace or alter any recorded attempt.

## Handling rule

1. Preserve every first attempt and retry, including raw response bytes, status, prompt hash,
   call identity, and retry link.
2. A first attempt with a retryable technical status receives the one retry already allowed by
   the frozen policy.
3. If that retry also ends because the output-token limit was reached, record the call as a
   terminal unresolved malformed output. It is an executed call, not an absent invocation or a
   valid parsed prediction.
4. Do not issue a third attempt, change the prompt, increase the output-token limit, change
   decoding, or substitute a prediction.
5. Treat a cell as execution-complete once every scheduled first attempt and every required retry
   has a terminal record. Report parsed completeness and terminal malformed counts separately.
6. Analyze unresolved outputs under all four treatments fixed in the A8 full-grid lock:
   `unresolved_as_correct`, `unresolved_as_error`, `unresolved_dropped`, and
   `whole_call_discarded`.
7. Apply this rule uniformly to completed and future B3 cells for every model and dataset.

## What remains unchanged

The models, datasets, 8,192 item samples, three batch assignments, call IDs and order, prompts,
output-token limits, deterministic decoding, parser, one-retry maximum, and full-grid analysis
remain unchanged. Terminal malformed counts and rates must be reported by model, dataset, and
condition.
