# A12 expansion order correction

Decision recorded 2026-09-14 01:59:53 UTC, before X1R4 joint output.
This planning correction supersedes the priority and A13–A16 numbering in
`20260914T014726Z-crossed-4x4-plan.md`. That file remains unchanged as history.
The active human-facing execution order is A0–A13 in
`../../refine-logs/EXPERIMENT_PLAN.md`.

After all four admitted families reach 400 independent evaluation replicates,
the budget-conditional order is: extend all equally to 1,000; test batch sizes
2, 4, 8, 16, and 32; add independent datasets across all four families with a
four-shared-dataset × four-family target; add an exact-ground-truth synthetic
task; add another checkpoint size per family; consider 2,500 evaluations last.
No extension is selected based on favorable interim results. A13 is final
analysis and paper.

The existing X1R4 B4 schedule remains fixed at 512 item appearances and 32
fresh size-16 joint calls per replicate, with no B3 output reuse. The existing
B3 main-cell item counts remain the sealed 2,048/4,096 power-rule selections.
The new crossed panel uses the same B3 power-rule ladder, selected and sealed
from solo evidence before any new joint output. Four new shared datasets must
pass the uniform solo gates in each of Qwen, Granite, Llama, and Mistral to
support a complete 4 × 4 claim. The nine old-dataset A6 failures remain
failures.

This record changes no frozen protocol, call schedule, claim gate, or GPU
ceiling. The existing 80-hour X1R4 stop and 10-hour reserve remain in force;
any new scientific block needs a pre-output schedule and budget seal. No GPU
work is authorized by this planning record.
