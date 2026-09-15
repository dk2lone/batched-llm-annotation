# X1R4 B3 Llama interrupted-run exclusion

Recorded: 2026-09-14T23:01:50Z

## Scope

This amendment applies only to the Llama 3.1 8B B3 execution attempts in Modal apps
`ap-3NPckE7YfRFq9COEC7ERyS` and `ap-FENtkafpe1tx3Hwq15mWZT`, stored remotely under
namespace `752b2bc54b8bd7bb`.

## Cause

The first app was administratively stopped after its smoke-test containers overlapped
execution containers and exceeded the requested four-GPU ceiling. The local launcher then
converted queued calls that never reached a model into `worker_failed` attempt records. A
resume consequently assigned attempt-2 identifiers to calls whose first real model
generation occurred only on the resume. The resumed launcher also retained an entire owner
lane of returned payloads before appending them locally.

## Treatment

All local and remote outputs from these two apps are excluded from scientific analysis.
They must not be relabeled, merged into the clean run, or counted toward B3 completeness.
The local audit copy is preserved at
`extension-b3/interrupted/20260914T230150Z-meta-llama--Llama-3.1-8B-Instruct/`.

The clean rerun uses the unchanged sealed calls, prompts, model revision, decoding policy,
parser, and retry policy under remote namespace `752b2bc54b8bd7bb-clean1`. Runtime-only
safeguards cap the class at four containers, serialize chunks within each owner lane, append
each returned chunk immediately, and leave an uncommitted remote-call exception pending for
resume instead of creating a synthetic attempt.

## Clean rerun outcome

Modal app `ap-d2AWKii6rBD0ZwOJBBLaYw` completed the clean two-dataset scope under namespace
`752b2bc54b8bd7bb-clean1` at 2026-09-14T23:47:16Z. Civil Comments and MultiNLI each contain
34,304 first attempts. Civil Comments required no retries and has 12 unresolved parsed
outputs. MultiNLI required one retry, which remained a terminal truncation, and has 31
unresolved parsed outputs. Both execution ledgers are complete with zero pending attempts
and zero irrecoverable technical failures. Model IDs, task IDs, attempt IDs, retry linkage,
and prompt hashes passed the post-run audit.
