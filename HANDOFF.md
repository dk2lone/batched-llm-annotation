# IEEE BigData 2026 handoff

Updated: 2026-09-18 after the CPU-only A1–A7 decision audit.

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

The revised B4 coverage analysis is also complete. It preserves the original result as
`coverage_unavailable_original`, substitutes only the 1,095 exact linked B4 repairs, and
passes the prespecified C2 gate with Granite/IMDb and Qwen/Civil Comments as the two supported
undercoverage cells.

The repaired B3 sensitivity analysis is complete. It leaves the sealed A9 result primary,
substitutes only the 3,002 exact linked B3 repairs, and reaches the same recurrence verdict:
the 16-cell recurrence gate passes.

The post-hoc A1–A7 decision audit is complete in
`study-120h/decision-audit/20260918T054237Z-a1-a7/`. Its recommendation is no additional
experiment for the current paper if repaired C2 is presented as sensitivity evidence. A
uniform-contract Granite/IMDb rerun is conditional on needing a clean C2 claim; it is not
authorized or running.

## Repaired B3 recurrence sensitivity

Artifacts:

`study-120h/extension-b3/repaired-analysis-20260918T041906Z/`

The sealed A9 analysis reproduced exactly before repair substitution. After all 3,002 exact
repairs were substituted, the equal-cell joint-minus-exact covariance effect was 0.0078988744
with one-sided replay `p = 0.0002` and a 95% shift interval of [0.0076483472, 0.0081461906].
The original primary estimate was 0.0096104285, so repair reduced the estimate by 0.0017115540,
or 17.81%, without changing the conclusion.

The repaired recurrence gate passes every criterion:

- 13 of 16 cells remain Holm-positive at the 0.002 practical floor;
- positive cells still cover all four model families and all four tasks;
- every leave-one-model-out and leave-one-task-out mean remains positive;
- every repaired sensitivity treatment has the same global effect, 0.0078988744, because no
  unresolved output remains.

Cell membership changed in one place while the total stayed 13: Granite/MultiNLI fell below
the 0.002 practical floor at 0.001632, and Granite/Civil Comments rose above it at 0.002739.
Llama/Civil Comments and Llama/IMDb are the other two repaired cells below the floor.

The repaired sensitivity SHA-256 is:

`0086f0a1adeccfae9ac66046c437749a018a31d9abf1b7179be3f92de3f5901f`

All three scientific artifacts reproduced byte-for-byte in a second independent CPU run. No
GPU work was launched.

## Revised B4 coverage result

Artifacts:

`study-120h/extension-b4/revised-analysis-20260918T034636Z/`

The revised IID Wald coverage estimates across 400 evaluation replicates are:

- Granite/IMDb: 0.7775; call-aware t 0.9300; whole-call bootstrap 0.9175
- Llama/Civil Comments: 0.9675; call-aware t 0.9650; bootstrap 0.9400
- Mistral/MultiNLI: 0.9075; call-aware t 0.9525; bootstrap 0.9475
- Qwen/Civil Comments: 0.8800; call-aware t 0.9600; bootstrap 0.9525

All four reference targets passed the frozen precision check. The revised C2 gate passes
because Granite and Qwen meet every gate condition, including IID coverage at or below 0.90,
least-favourable upper coverage below 0.95, and Holm-adjusted significance. Mistral's IID
coverage is significantly below 0.95 but is 0.9075, above the prespecified 0.90 practical
cutoff. Llama does not show undercoverage.

The revised analysis SHA-256 is:

`f7b5e9ac5bbeace107765a702fe2a0bdfcec34ee4c9ee51cb0c1913e22326443`

The complete analysis reproduced identically in a second independent CPU run. No GPU work
was launched.

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

For B4, preserve the original outcome as `coverage_unavailable_original`. Report the completed
revised repaired analysis separately as post-execution sensitivity evidence. Its C2 gate
passes, with strong IID undercoverage in Granite and Qwen and substantially improved coverage
from the call-aware intervals.

For the 16-cell B3 study, retain the original A9 malformed-output bounds as primary. Report
the completed repaired analysis as a secondary post-execution sensitivity. Exact linked
replacement reduces the global effect by 17.81% but does not change the family recurrence
conclusion. Never overwrite the original A9 artifacts.

## A1–A7 decision audit

Regenerate with
`cd study-120h && PYTHONPATH=src .venv/bin/python analyze_decision_audit.py --output <new-directory>`.
The output contains `DECISION.md`, `B3_AUDIT.json`, `B4_AUDIT.json`, `A3_TABLE.csv`,
`A4_TABLE.csv`, `A6_TABLE.csv`, and `MANIFEST.json`. It is CPU-only and post-hoc. The sealed
A9 result, original B4 unavailable status, and revised repaired C2 result remain separate.

Granite/IMDb has 133 repair-free evaluation replicates out of 400 and 82 complete original
reference replicates out of 250. Qwen/Civil Comments has 354 and 220. IID coverage on the
repair-free evaluation subsets is 0.7744 for Granite and 0.8842 for Qwen, close to the revised
all-replicate estimates of 0.7775 and 0.8800. Neither subset meets the frozen C2 minimum of
400 evaluation replicates. Granite's original-only reference target is 0.19293 versus 0.19816
after repair, a −0.00523 shift outside the latter's Monte Carlo uncertainty band; coverage
stays near 0.78–0.79 across the target/evaluation sensitivity combinations.

Granite repaired calls are not measurably more label-homogeneous than original calls in the
within-cell reference or evaluation comparisons. Standardized B3 joint-minus-exact dependence
is positive in all 16 cells and all four families. The three assignment-level equal-cell raw
excesses are 0.01013, 0.00899, and 0.00971. The assignments are seeded regroupings of the
same items, not a composition intervention, so A6 gives no evidence motivating a new
composition experiment. The old “exact-context fake covariance” column duplicates exact
covariance in all 16 cells; solo fake covariance is the separate control. A9 permutation
`p = 0.0002` is the 4,999-draw resolution floor, not an effect size.

The strongest remaining validity threat is the mixed original/recovery output contract in
Granite/IMDb. Current robustness checks preserve the substantive undercoverage pattern but
cannot turn the repaired C2 result into a clean confirmation. A single clean Granite cell
would address the largest repair objection but would not create a fully clean two-cell C2
confirmation. At least two uniformly generated cells are needed for that.

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
of all 4,097 unique jobs. The repaired B3 and B4 analyses add fail-closed exact-replacement
tests, independent parsing and hash verification, manifest verification, and deterministic
full-data reproduction. Commits remain local and have not been pushed. The branch is ahead of
`origin/main`. Many timestamped B4 execution directories are untracked user data; do not
delete, add wholesale, or overwrite them.

## Next action

Do not launch GPU work. Use the A1–A7 decision report to frame the paper's A9 primary result,
repaired B3 sensitivity, original B4 unavailable result, and revised B4 repaired coverage
claim. If a clean C2 claim becomes necessary, decide explicitly between one uniform-contract
Granite/IMDb rerun to address the largest objection and a two-cell Granite+Qwen rerun for fully
clean confirmation. No such paid run is authorized by this handoff.

Preserve the original ledgers, recovery events, receipts, seals, locks, and both original
analysis directories. Do not expose credentials or invoke Claude CLI unless Daniel explicitly
requests it.
