# B4 Runpod SSH stdin recovery

Recorded before any post-stall model output.

The one-lane Runpod B4 attempt in
`extension-b4/runpod-pcie-auto-20260917T030957Z/` stopped after 28 durable
receipts and 896 accepted first attempts. The local SSH subprocess retained its
stdin pipe after a timed `communicate()` poll. The remote request client waited
for EOF before opening its loopback HTTP connection, while the worker server
waited for that connection. The GPU remained idle with the model resident.

The exact stuck SSH child was terminated. The existing fail-closed cleanup
deleted Pod `u8due56secjnmr` and verified termination. The 28 local and remote
receipt counts matched, and all 896 recorded events had status `ok`.

The transport now supplies subprocess stdin from a finite temporary file. This
preserves bounded polling while giving SSH deterministic EOF. The remote worker,
model revisions, prompts, schedules, decoding, call identities, and analysis are
unchanged.

A continuation must use a newly generated plan that binds the changed transport
hash. Before launching a Pod, it may carry forward the 28 old receipts only after
validating every event against the unchanged scientific plan, confirming that
`extension_b4_runpod.py` and the resume entry point in `runpod_extension_b4.py`
are the only implementation-hash differences, confirming
all old Pods have verified termination, and confirming the interrupted meter was
reconciled. The continuation record must bind both plan hashes, the source meter,
and every imported receipt. Imported calls remain first attempts and must not be
generated again.

If a continuation fails before new output, a direct retry may use the same
implementation hashes. Its interrupted phase must first be conservatively
settled, and every created Pod must have a matching verified termination record.

The interrupted phase is charged conservatively from its recorded phase start
through launcher failure, plus the standard 60-second settlement allowance.
This amendment changes transport and recovery only. It does not change any
scientific condition or outcome rule.
