# IEEE BigData 2026 research project

Updated: 2026-09-13

## Start here

This is a research workspace for an IEEE BigData 2026 paper on dependence in batched LLM text annotation. It is a Git repository on branch `main`, pushed to the private GitHub repo `dk2lone/batched-llm-annotation`. Two feasibility gates have run. The budgeted study is in execution: Step 1 screening, the historical Qwen3-8B/Civil Comments bridge, the first fresh Qwen3-8B/MultiNLI cell, and the 36-cell Freeze B solo screen are complete. Nine of the ten admitted fresh cells are still unrun. The separate five-family X1R4 extension has a sealed runtime/access lock and has produced no scientific model output.

Read in this order:

1. `AGENTS.md` for the current state and work boundaries.
2. `plan-120h-a100-rebased.md` for the active ordered, budgeted study, ceilings, and stopping rules.
3. `refine-logs/EXPERIMENT_PLAN.md` for the cross-family A0-A13 extension design.

The repository now carries experiment code, run data, and these plans. The prose record that used to live beside them, including the handoff, the manifest, the idea ledger, the novelty audits, the refinement rounds, and the per-stage status reports, was removed from the working tree on 2026-09-13. Every one of those files is still in Git history. Recover one with `git show 624c708:<path>`.

## Current decision and evidence

**Develop A: batched LLM annotation dependence.** The completed Gate A pilot used Qwen3-8B and Civil Comments. Across 8,375 prompts and 14,000 target-level rows, joint-prompt error ICC was 0.0754 with a conservative 95% decision interval of [0.0182, 0.1314]. Three fake-batch tests returned `p = 0.0010`; the matched-length separate-prompt filler placebo ICC was 0.0048. The frozen verdict was `PROCEED`. This is one model-task cell, not a paper-scale replication or a measured confidence-interval coverage result. Read `gate-experiments/a/artifacts/full/analysis/report.md` before citing the pilot.

**Stop K: distance concentration and LLM tabular failure.** The tested Mistral Small 3.1 24B Base model passed the capability anchor but failed the frozen representation-contrast and accuracy-mechanism conditions. The result is `mechanism_unsupported`, not proof that every related research direction is impossible. Do not spend more GPU time on K without new evidence or a changed hypothesis. Thirty jobs never completed, and none of them can reverse the stop decision. Read `gate-experiments/k/artifacts/mistral-24b-base/analysis/analysis.json` for the frozen numbers.

The 120-hour plan tests whether the shared annotation call creates excess across-item error covariance beyond an exact-context target-only control and whether that dependence changes uncertainty for mean reference-relative accuracy. It uses a fresh 2 × 2 model-task panel and two selected repeated-sampling cells. The A0-A13 extension in `refine-logs/EXPERIMENT_PLAN.md` widens that to a four-family by four-dataset grid with no accuracy admission gate.

## Work boundary

- Research planning and read-only checks are allowed. Do not write experiment code, launch a pilot, or start paid GPU work unless Daniel explicitly requests implementation.
- Do not present planned analyses as findings. Keep the old Gate A pilot separate from fresh confirmation data.
- Do not delete or overwrite timestamped run data, seals, or locks. The sealed protocols `study-120h/EXTENSION_PROTOCOL.md`, `study-120h/FREEZE_B_PROTOCOL.md`, and `study-120h/FREEZE_B_PREP.md` are hashed by the seal scripts. Editing one invalidates its seal.
- Do not copy credentials into files, commands, prompts, traces, or reports. A Modal token was exposed in chat and should be rotated. The Semantic Scholar key is available through `SEMANTIC_SCHOLAR_API_KEY` when needed.
- Semantic Scholar permits at most one cumulative request per second. Pace requests below that rate, and treat HTTP 429 or 500 responses as unavailable evidence rather than empty results.
- Invoke a Claude CLI only when Daniel explicitly asks. His default is Claude Opus 5; verify the available identifier before launching and do not silently substitute a model.
- The last recorded Modal workspace state was disabled. Verify availability and spend limits before any newly authorized paid run; do not assume the old state is current.

## Venue and authorship

The study is aimed at an IEEE BigData 2026 undergraduate-appropriate venue. Earlier checks found that the High School Symposium requires a high-school first author who is the primary contributor, while the Undergraduate and REU Consortium may be the suitable route for Daniel. Recheck the live call, deadlines, page limits, and actual author eligibility before preparing a submission. Never assign authorship merely to satisfy a track rule.

## File map

| File | Role |
|---|---|
| `plan-120h-a100-rebased.md` | Active 120-hour single-A100 execution plan, order, ceilings, claim gates |
| `refine-logs/EXPERIMENT_PLAN.md` | Cross-family A0-A13 extension plan |
| `study-120h/` | Execution tree: Modal drivers, `src/`, tests, seals, locks, run outputs |
| `study-120h/EXTENSION_PROTOCOL.md` | Sealed X1 extension protocol, hashed by `seal_extension.py` |
| `study-120h/FREEZE_B_PROTOCOL.md` | Sealed Freeze B protocol, hashed by `seal_freezeb.py` |
| `study-120h/FREEZE_B_PREP.md` | Sealed Freeze B preparation record, hashed by `seal_freezeb.py` |
| `gate-experiments/a/` | Gate A pilot code and artifacts |
| `gate-experiments/a/artifacts/full/analysis/report.md` | Completed Gate A pilot and frozen verdict, checked by `test_repro.py` |
| `gate-experiments/k/` | Gate K code and artifacts; the direction is stopped |
| `idea-stage/`, `refine-logs/` | Raw literature logs and refinement state as JSON and JSONL |

## Recommended next action

On a new session, read `plan-120h-a100-rebased.md` and `refine-logs/EXPERIMENT_PLAN.md`, then ask what Daniel wants done next. Do not rerun a completed stage and do not start a GPU merely because this directory was opened.
