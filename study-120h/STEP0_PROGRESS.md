# Step 0 progress, session of 2026-09-12

Final update: Step 0 is **COMPLETE** in `STEP0_STATUS.md`. The sealed freeze is
`freeze/20260912T041633Z`; all 12 exit-gate items pass. The checkpoint notes below are
preserved to show what remained before the final simulation and freeze run.

Checkpoint verdict: **INCOMPLETE**. The code was finished and checked. The run was not.

## Done in this session

- Audited the partial implementation against the Step 0 exit gate.
- Fixed the protocol defects: the serving engine is now the exact pin `vllm==0.28.0` with
  PyPI evidence, the gated-repo claim now points at stored evidence, the MultiNLI fallback
  frame is fully specified, and decisions D17, D18, and D19 record why.
- Fixed the code defects: the filler condition now builds the target plus 15 length-matched
  peers by the pilot's greedy nearest-fit rule; solo calls are spread over the 8 execution
  blocks; the randomization test now replays the frozen generator itself instead of a
  private shuffler, and refuses unresolved labels; the Gate A report regex no longer
  swallows the sentence stop; the venue extractor now reads a date placed on the next line;
  a gated or disabled dataset repo is now a hard blocker.
- Added tests for data freeze, simulation, reproduction, integrity, venue, and the CLI, and
  corrected two wrong assertions in the existing tests.
- Added the `audit-pins`, `checks`, and `status` stages, and moved `simulate` before the
  freeze so no field stays `SET_BY_SIMULATION`.
- Wrote README.md with the numbered commands and the success and failure signal of each.
- Verified out of band: the Gate A reproduction passes 24 of 24 key metrics with the
  historical artifact unchanged; the live venue pages parse with all four consistency
  checks true; all five pinned repositories return gated false with matching revisions;
  all three datasets stream from their pinned revisions.

## Checks

- pytest: 76 of 77 pass. The one failure is `test_no_design_field_is_left_for_a_later_stage`,
  which is correct: the simulation had not yet resolved the threshold.
- ruff: all checks passed.
- mypy: no issues in 21 source files.

## What remains

1. Wait for `step0 simulate --out simulations` to finish, then read section 4 of
   `simulations/REPORT.md`.
2. Write the resolved values into `src/step0/lock.py`, replacing the five
   `SET_BY_SIMULATION` markers.
3. Run steps 2 to 11 of README.md.

A probe at 40 trials per point already showed the shape of the power curve: detection power
at the 0.002 floor is near 1.0 for marginal error rates 0.02 to 0.08, about 0.72 at 0.113,
and about 0.50 at 0.20. The high-error-rate loss is a real limit of the fixed design and
must be reported, not designed away.

No new experimental model labels exist. No GPU, Modal, paid compute, or study-model
inference was used. No file outside study-120h was modified.
