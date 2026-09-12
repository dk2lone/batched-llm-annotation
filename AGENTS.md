# IEEE BigData 2026 research project

Updated: 2026-09-11

## Start here

This is a research workspace for an IEEE BigData 2026 paper on dependence in batched LLM text annotation. There is no Git repository, branch, or PR here. Two feasibility gates have run, but the 120-hour A100 study is a plan only: no experiment from that plan has started.

Read in this order:

1. `AGENTS.md` for the current state and work boundaries.
2. `HANDOFF.md` for the current session handoff, pilot evidence, and restart instructions. Its later C1-era sections are history.
3. `plan-120h-a100.md` for the ordered, budgeted paper study and stopping rules.
4. `ideas.md` for the current idea ranking and the status of Gates A and K.
5. `plan.md` only when expanding the budgeted study into the original, unconstrained program.

For the active study design, `plan-120h-a100.md` takes priority. For idea ranking, `ideas.md` takes priority. The old recommender comparison in `idea-stage/IDEA_REPORT.md` and the C1 material in the lower part of `HANDOFF.md` are superseded.

## Current decision and evidence

**Develop A: batched LLM annotation dependence.** The completed Gate A pilot used Qwen3-8B and Civil Comments. Across 8,375 prompts and 14,000 target-level rows, joint-prompt error ICC was 0.0754 with a conservative 95% decision interval of [0.0182, 0.1314]. Three fake-batch tests returned `p = 0.0010`; the matched-length separate-prompt filler placebo ICC was 0.0048. The frozen verdict was `PROCEED`. This is one model-task cell, not a paper-scale replication or a measured confidence-interval coverage result. Read `gate-experiments/a/artifacts/full/analysis/report.md` before citing the pilot.

**Stop K: distance concentration and LLM tabular failure.** The tested Mistral Small 3.1 24B Base model passed the capability anchor but failed the frozen representation-contrast and accuracy-mechanism conditions. The result is `mechanism_unsupported`, not proof that every related research direction is impossible. Do not spend more GPU time on K without new evidence or a changed hypothesis. Read `gate-experiments/k/artifacts/mistral-24b-base/analysis/COMPLETION_AUDIT.md` for the 30 missing jobs and why they cannot reverse the stop decision.

The 120-hour plan tests whether the shared annotation call creates excess across-item error covariance beyond an exact-context target-only control and whether that dependence changes uncertainty for mean reference-relative accuracy. It uses a fresh 2 × 2 model-task panel and two selected repeated-sampling cells. The original `plan.md` remains the continuation program, not the current execution checklist. Neither plan's new experiments have been run.

## Work boundary

- Research planning and read-only checks are allowed. Do not write experiment code, launch a pilot, or start paid GPU work unless Daniel explicitly requests implementation.
- Do not present planned analyses as findings. Keep the old Gate A pilot separate from fresh confirmation data.
- Preserve historical and timestamped files. Do not delete or overwrite them.
- Do not copy credentials into files, commands, prompts, traces, or reports. A Modal token was exposed in chat and should be rotated. The Semantic Scholar key is available through `SEMANTIC_SCHOLAR_API_KEY` when needed.
- Semantic Scholar permits at most one cumulative request per second. Pace requests below that rate, and treat HTTP 429 or 500 responses as unavailable evidence rather than empty results.
- Invoke a Claude CLI only when Daniel explicitly asks. His default is Claude Opus 5; verify the available identifier before launching and do not silently substitute a model.
- The last recorded Modal workspace state was disabled. Verify availability and spend limits before any newly authorized paid run; do not assume the old state is current.

## Venue and authorship

The study is aimed at an IEEE BigData 2026 undergraduate-appropriate venue. Earlier checks found that the High School Symposium requires a high-school first author who is the primary contributor, while the Undergraduate and REU Consortium may be the suitable route for Daniel. Recheck the live call, deadlines, page limits, and actual author eligibility before preparing a submission. Never assign authorship merely to satisfy a track rule.

## File map

| File | Role |
|---|---|
| `HANDOFF.md` | Current restart handoff at top; older C1 research history below |
| `plan-120h-a100.md` | Current 120-hour single-A100 execution plan, order, ceilings, claim gates |
| `plan.md` | Preserved full research program for later expansion |
| `ideas.md` | Current idea ledger; A recommended, K stopped |
| `gate-experiments/a/artifacts/full/analysis/report.md` | Completed Gate A pilot and frozen verdict |
| `gate-experiments/k/artifacts/mistral-24b-base/analysis/COMPLETION_AUDIT.md` | Completed Gate K stop audit |
| `idea-stage/novelty-check-20260910/NOVELTY_REPORT.md` | Claim-level literature novelty audit |
| `MANIFEST.md` | Append-only research-artifact index |

Older recommender files, raw literature logs, and `.aris/traces/` are audit history, not the current starting point. The traces are project-local and should not be committed.

## Recommended next action

On a new session, read `HANDOFF.md` and `plan-120h-a100.md`, then ask what Daniel wants done next. If he authorizes implementation, start with Step 0 of the budgeted plan: freeze protocol, sample frames, analysis tests, and the venue/authorship route before new joint outputs. Do not rerun Gate A or start a GPU merely because this directory was opened.
