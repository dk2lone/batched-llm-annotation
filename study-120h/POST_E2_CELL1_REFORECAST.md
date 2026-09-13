# Reforecast after first fresh E2 cell

Date: 2026-09-13
Status: technical planning record; no second-cell output or paid execution authorized

## Decision

The second sealed primary cell is
`ibm-granite/granite-3.3-8b-instruct::civil_comments`. Under the same four-worker,
batch-16 execution layout used for the completed first cell, its forecast is **0.808 base,
1.293 guarded, and 2.425 incident A100-hours**.

The cell passes the study-wide start rule. The current cumulative meter is 4.824000 A100-hours,
so the guarded cell plus the required 10-hour reserve projects to 16.117109 hours, below the
150-hour stop. This record does not authorize implementation or launch.

## Raw timing inputs

| Input | Qwen3-8B | Granite 3.3 8B | Granite/Qwen ratio |
|---|---:|---:|---:|
| Civil Comments synthetic size-16 joint median | 2.247907 s | 2.424750 s | 1.078670 |
| Civil Comments synthetic size-16 exact-context median | 0.209500 s | 0.225469 s | 1.076225 |
| Civil Comments synthetic short median | 0.201331 s | 0.215862 s | 1.072170 |
| Solo-screen model loading | 184.082519 A100-s | 161.247154 A100-s | 0.875950 |

The completed Qwen3-8B/MultiNLI E2 cell used 2,921.261127 A100-seconds in total. Its four
equivalence-smoke workers used 665.302023 seconds, and its scientific chunks used 2,255.959105
seconds. Three scientific workers incurred a second cold start after the first-1,000-call gate;
their first chunks were the three large 166.975--176.096-second chunks.

## Calculation

The estimate preserves that observed execution layout. It treats the mean first-cell smoke charge,
`665.302023 / 4 = 166.325506` seconds, as one Qwen worker load. The seven expected worker loads
(four smoke workers and three post-gate reloads) are scaled by the direct Granite/Qwen loading ratio
of 0.875950. The remaining first-cell generation time is scaled by 1.075506, the first-cell
condition-time-weighted mean of the three Civil Comments synthetic speed ratios above.

| Component | Forecast A100-seconds |
|---|---:|
| Four initial Granite worker loads and smoke calls | 582.772 |
| Three post-gate Granite worker reloads | 437.079 |
| 17,152 scientific calls | 1,889.645 |
| Total | 2,909.495 |

| Case | A100-hours | Rough dollars at the existing $3.605/hour planning ratio |
|---|---:|---:|
| Base | 0.808 | $2.91 |
| Frozen 1.6x guard | 1.293 | $4.66 |
| 3x incident | 2.425 | $8.74 |

The dollar conversion preserves the project's conservative planning ratio. It is not a current
Modal quote or invoice.

## Findings and limits

1. **F1:** Granite's synthetic Civil Comments generation timings are 7.2%--7.9% slower than
   Qwen3-8B, but its measured model load is 12.4% faster. Those effects nearly cancel at whole-cell
   scale.
2. **F2:** The new 0.808-hour base estimate is 38.2% below the 1.307-hour estimate used before the
   first E2 cell. The reduction comes from replacing bridge-era timing with the completed production
   E2 layout.
3. **F3:** The condition timings are model-transfer estimates, not Granite E2 measurements. The
   runner must repeat the frozen first-1,000-attempt reforecast before releasing the remaining
   blocks.

## Next experiment

Implement and CPU-test a runner for only the second sealed cell. Before model output, record its
worker-ownership amendment and run the four-worker equivalence smoke. After 1,000 real attempts,
replace this forecast with Granite E2 timing and apply the existing start/stop rule before releasing
the rest of the cell.

## Evidence

- `e2-out/qwen3-8b--multi_nli/meter.jsonl`
- `e2-out/qwen3-8b--multi_nli/timing.jsonl`
- `e2-out/qwen3-8b--multi_nli/reforecast-final.json`
- `freezeb-screen/20260913-freezeb-screen/reforecast-1000/Qwen--Qwen3-8B.json`
- `freezeb-screen/20260913-freezeb-screen/reforecast-1000/ibm-granite--granite-3.3-8b-instruct.json`
- `runtime-lock/20260913T004054Z-BRT1/benchmark/report.json`
- `freeze/20260913T022442Z-B2/ADMISSION_LOCK.json`
