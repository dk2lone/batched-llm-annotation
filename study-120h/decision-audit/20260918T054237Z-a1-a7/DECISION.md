# B3/B4 post-hoc decision audit

The sealed A9 recurrence result remains primary. The original B4 result remains `coverage_unavailable_original`; the repaired C2 result is a separate post-execution sensitivity. A1–A6 below are new post-hoc checks and do not change either gate.

## Provenance and repair classification

The recovery inventory and linked receipts identify 4,097 repaired calls: 3,002 B3 and 1,095 B4. The B4 analysis uses the authenticated inventory mapping, then excludes every evaluation replicate containing at least one mapped repair. Successful original calls remain original. All 4,097 repairs used the later structured-output contract; this is a technical amendment, not uniformly generated confirmation data.

Existing-report reconciliation: sealed A9 has global raw excess 0.009610; the separately repaired B3 sensitivity has 0.007899, a 17.81% reduction caused by replacing 3,002 unresolved source calls. Both report 13/16 positive cells. Original B4 has 1,095 unresolved outputs and no coverage estimate; revised B4 uses exactly 1,095 linked repairs and reports 400 evaluation replicates per cell. This audit reproduces those revised IID and call-aware coverage counts exactly. Its smaller repair-free totals follow whole-replicate exclusion, not missing execution.

## A1 — Repair-free B4 coverage

| cell | clean evaluation / 400 | clean reference / 250 | IID coverage [exact 95%] | call-aware coverage | variance ratio [95%] | frozen C2 status |
|---|---:|---:|---:|---:|---:|---|
| Granite/IMDb | 133 | 82 | 0.774 [0.694–0.842] | 0.925 | 2.595 [2.037–3.187] | below frozen 400-replicate minimum |
| Llama/Civil Comments | 373 | 229 | 0.965 [0.941–0.981] | 0.962 | 0.916 [0.792–1.035] | below frozen 400-replicate minimum |
| Mistral/MultiNLI | 277 | 180 | 0.906 [0.865–0.938] | 0.953 | 1.295 [1.099–1.489] | below frozen 400-replicate minimum |
| Qwen/Civil Comments | 354 | 220 | 0.884 [0.846–0.916] | 0.963 | 1.668 [1.413–1.946] | below frozen 400-replicate minimum |

Granite/IMDb retains only 133 clean evaluation and 82 clean reference replicates; its original C2 support is **too imprecise under the frozen gate**, rather than disproved. Qwen/Civil Comments is also below the frozen sample minimum, though its repair-free IID coverage remains 0.884. Thus no repair-free subset can pass the unchanged C2 gate. Mean widths and exact intervals are in `B4_AUDIT.json`.

## A2 — Original-only reference targets

Each original-only target averages the means of complete, unrepaired reference replicates. This keeps the frozen equal-replicate weighting and 512 appearances per replicate; partly repaired reference replicates are excluded whole.

| cell | existing target | original-only target (SE) | shift | inside existing 95% MC band | IID coverage: existing/all, original/all, existing/clean, original/clean |
|---|---:|---:|---:|---|---|
| Granite/IMDb | 0.19816 | 0.19293 (0.00297) | -0.00523 | no | 0.777, 0.782, 0.774, 0.789 |
| Llama/Civil Comments | 0.29018 | 0.29019 (0.00125) | 0.00001 | yes | 0.968, 0.968, 0.965, 0.965 |
| Mistral/MultiNLI | 0.61383 | 0.61503 (0.00205) | 0.00120 | yes | 0.907, 0.907, 0.906, 0.906 |
| Qwen/Civil Comments | 0.12502 | 0.12490 (0.00148) | -0.00012 | yes | 0.880, 0.880, 0.884, 0.884 |

Granite's original-only target shifts by −0.00523, outside the existing target's Monte Carlo band. Its IID coverage stays about 0.78–0.79 across the four combinations; Qwen stays about 0.88. With all 400 evaluation replicates, changing the target does not change the revised supported-cell set. Clean subsets cannot satisfy the frozen sample minimum. The original-only reference precision check passes in all cells.

## A3 — Repair/schema homogeneity

Calls are the bootstrap units, stratified by cell and reference/evaluation stream. B4 has no assignment arm beyond its single joint condition. The full four-metric, eight-stratum results are in `A3_TABLE.csv`.

| cell | evaluation repaired/original calls | pairwise label agreement, repaired minus original [95%] |
|---|---:|---:|
| Granite/IMDb | 469/12331 | 0.003 [-0.015, 0.019] |
| Llama/Civil Comments | 29/12771 | 0.049 [0.006, 0.093] |
| Mistral/MultiNLI | 147/12653 | -0.068 [-0.103, -0.037] |
| Qwen/Civil Comments | 49/12751 | -0.049 [-0.103, 0.001] |

Granite repaired calls are not measurably more homogeneous: their reference and evaluation agreement intervals both include zero. These observational contrasts cannot identify the effect of the output schema. `B4_AUDIT.json` also records all-call, original-call, repaired-call, and clean-replicate error covariance, plus full versus clean variance ratios.

## A4 — Standardized B3 dependence

`A4_TABLE.csv` gives all 16 cells in one table: marginal error/accuracy, raw covariance, covariance divided by p(1−p), joint-minus-exact and joint-minus-solo contrasts, and conditional assignment-replay intervals. Original A9 unresolved outputs use its frozen unresolved-as-correct treatment. Marginal p is fixed under this replay, so it has a point estimate only. This is post-hoc, not a new gate.

The standardized joint-minus-exact contrast is positive in 16 of 16 cells and all 4 model families. The direction and cross-family recurrence therefore survive standardization; the A9 practical floor remains on the raw scale.

## A5 — B4 variance inflation

| cell | call-aware / IID variance | empirical / IID ratio [95%] | IID / call-aware coverage | IID / call-aware mean width | effective n (approx.) |
|---|---:|---:|---:|---:|---:|
| Granite/IMDb | 0.000908 / 0.000305 | 2.711 [2.328, 3.078] | 0.777 / 0.930 | 0.068 / 0.122 | 189 |
| Llama/Civil Comments | 0.000393 / 0.000401 | 0.899 [0.783, 1.009] | 0.968 / 0.965 | 0.078 / 0.080 | 569 |
| Mistral/MultiNLI | 0.000710 / 0.000462 | 1.346 [1.156, 1.553] | 0.907 / 0.953 | 0.084 / 0.108 | 381 |
| Qwen/Civil Comments | 0.000434 / 0.000215 | 1.702 [1.456, 1.949] | 0.880 / 0.960 | 0.057 / 0.084 | 301 |

The existing 1.25 ratio floor is exceeded by Granite, Qwen, and Mistral point estimates; Granite and Qwen have intervals wholly above it. Llama does not. For equal batch size 16 and common marginal p, a raw pair covariance of 0.002 implies approximate inflation 1 + 15×0.002/[p(1−p)]. The cell-specific values are in `B4_AUDIT.json`. This formula assumes exchangeable within-call covariance and no additional between-call dependence. Effective n = 512/observed ratio is only the corresponding design-effect interpretation.

## A6 — Three existing B3 assignments

The sealed protocol defines three seeded item regroupings, not composition arms. `A6_TABLE.csv` reports every cell and assignment: joint, exact fake, and solo fake covariance, marginal errors, standardized contrasts, joint-call error homogeneity, and assignment-replay intervals. The equal-cell raw excess by assignment is 0.01013, 0.00899, 0.00971. All assignments reuse the same items, so this between-assignment range is descriptive; no independent composition-arm test is available.

**Composition decision: no evidence motivating a new composition experiment.** Assignment variation alone does not identify composition effects.

## A7 — Report integrity

In both stored B3 analyses, the exact-context fake covariance duplicates the exact-context covariance in all 16 cells. It is the same regrouping, not a second control. Solo fake covariance is separate in all 16 cells and is labeled as such in the new tables. A9 replay p = 0.0002 is the 4,999-draw resolution floor, not an effect-size measure. No sealed or original artifact was edited.

## Conclusions across sensitivities

| analysis | B3 positive dependence across families | B4 IID undercoverage in Granite and Qwen | frozen C2 confirmation |
|---|---|---|---|
| sealed A9 / original B4 | yes | unavailable original | unavailable original |
| repaired B3 / revised B4 | yes, 13/16 positive | yes, 0.778 / 0.880 | passes as repaired sensitivity |
| repair-free B4 evaluations | unchanged B3 | descriptive signal persists | cannot pass 400-replicate minimum |
| original-only reference target | unchanged B3 | persists with all evaluations | revised supported set unchanged |
| schema homogeneity | unchanged B3 | no Granite homogeneity shift detected | repair contract remains a threat |
| standardized B3 dependence | positive in all families | unchanged B4 | no new gate |

## Decision and paid-work trigger

The strongest remaining threat is the mixed original/recovery output contract in Granite/IMDb: only 133 clean evaluation replicates remain, and its original-only reference target lies outside the revised target's Monte Carlo band. Its coverage estimate is stable, but that does not turn the repaired C2 result into a clean confirmation.

Ranked recommendation: (1) no additional experiment, reporting repaired C2 only as sensitivity evidence; (2) uniform-contract Granite/IMDb rerun if a clean C2 claim is required; (3) uniform-contract two-cell Granite+Qwen rerun for a fully clean C2 confirmation; (4) A11 extension to 1,000 replicates only if a new substantive precision need is identified; (5) batch sizes 2, 4, 8, 16, 32 only after the repair concern is resolved and batch-size dependence becomes the question; (6) batch-composition intervention, which A6 does not motivate.

No GPU work is triggered by the current robustness results. Paid GPU work would be triggered by a decision to claim clean C2 confirmation: the existing Granite cell has 133/400 repair-free evaluations and 82/250 repair-free references, so it cannot supply that claim. One clean Granite cell addresses the largest repair objection but does not create a fully clean two-cell C2 confirmation; that requires at least two uniformly generated cells. Do not extend A11 merely to narrow intervals. A batch-size experiment requires the repair concern to be resolved and a genuine question about dependence versus batch size.
