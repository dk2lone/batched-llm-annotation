# Batched LLM annotation dependence

Research workspace for an IEEE BigData 2026 study of whether annotating multiple texts in one LLM call creates dependence between item-level errors and changes uncertainty estimates.

The completed Gate A pilot, historical Qwen3-8B/Civil Comments bridge, and first fresh
Qwen3-8B/MultiNLI cell all support excess same-call error covariance. The 36-cell Freeze B solo
screen is complete, but only one of ten admitted fresh cells has been run, so recurrence and
confidence-interval coverage claims remain unavailable.

The separate five-family X1R4 extension has a sealed protocol and a passing synthetic
batch-invariance preflight. Its 40,960-call solo screen is complete: seven cells in four
families passed the frozen capability gates. The CPU-only A7 admission lock is
sealed in [`study-120h/admission-lock/20260914T012453Z-X1R4-A7/REPORT.md`](study-120h/admission-lock/20260914T012453Z-X1R4-A7/REPORT.md).
No extension joint results exist. Before A8 family-core B3 work, the planned
full four-model × four-existing-dataset comparison needs its separate
pre-output schedule, outcome, analysis, and budget seal.

Start with [`HANDOFF.md`](HANDOFF.md), then read the active
[`plan-120h-a100-rebased.md`](plan-120h-a100-rebased.md), the sealed part-1
[`study-120h/FREEZE_B_PROTOCOL.md`](study-120h/FREEZE_B_PROTOCOL.md), and the held execution
instructions in [`study-120h/FREEZEB_SCREEN_RUNBOOK.md`](study-120h/FREEZEB_SCREEN_RUNBOOK.md).
The original [`plan-120h-a100.md`](plan-120h-a100.md) remains historical audit context.

Large immutable protocols and raw outputs are stored as checksum-protected GitHub Release assets.
See [`RESULTS_RELEASES.md`](RESULTS_RELEASES.md) for the publication boundary and archive contents.
