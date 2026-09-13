# Post-bridge compute reforecast

Date: 2026-09-13
Status: technical planning record; no fresh E2 output or paid execution authorized

## Decision

The historical Qwen3-8B/Civil Comments bridge is complete and its final measured timing can
replace the synthetic timing for the next Qwen3-8B E2 start decision. The next sealed cell is
`Qwen/Qwen3-8B::multi_nli`. Its provisional forecast is 1.307 base A100-hours, 2.091 hours under
the frozen 1.6x guard, and 3.920 hours under the 3x incident case.

The cell fits the study-wide start rule. The current cumulative meter is 4.0125 A100-hours, so
the guarded cell plus the required 10-hour reserve projects to 16.1034 hours, below the 150-hour
stop. Account-specific credit and GPU availability still have to be checked immediately before a
paid launch.

## Calculation

| Condition | Planned calls | Measured A100 seconds/call | Forecast A100 seconds |
|---|---:|---:|---:|
| Joint, batch 16 | 768 | 0.402788 | 309.341 |
| Exact context, batch 16 | 12,288 | 0.340605 | 4,185.352 |
| Solo | 4,096 | 0.051185 | 209.655 |
| Total | 17,152 |  | 4,704.349 |

The joint and exact-context rates are the final condition allocations in
`step2-out/bridge/SUMMARY.json`. The solo rate is the measured Qwen3-8B/MultiNLI screen time:
104.827607 A100 seconds for 2,048 calls. `4,704.349 / 3,600 = 1.306764` base A100-hours.

| Case | A100-hours | Rough dollars at $3.605/hour |
|---|---:|---:|
| Base | 1.307 | $4.71 |
| Frozen 1.6x guard | 2.091 | $7.54 |
| 3x incident | 3.920 | $14.13 |

The dollar conversion is a planning ratio inferred from Daniel's reported solo-screen charge,
not a current Modal quote or invoice.

## Timing limits

vLLM supplied no per-request component timing during the bridge. Joint and exact-context values
are allocations of mixed-chunk A100 time and include the bridge's serving-session overhead. They
are suitable for a guarded next-block forecast, not a standalone latency claim. MultiNLI solo
timing is directly measured, but transferring joint and exact timing from Civil Comments assumes
their comparable frozen batch-16 prompt sizes remain operationally representative.

Do not multiply every primary-panel forecast by the bridge's aggregate actual/base ratio. The ten
cells use different models and task lengths. Reforecast after the first 1,000 calls and after each
complete cell using that model's technical timing, as the frozen runtime protocol requires.

## Worker layout

One call still runs on one A100. Up to four independent workers may shorten wall-clock time while
leaving summed A100-hours unchanged, provided each frozen execution block has one owner and its
within-block order is preserved. Because the existing two-worker amendment covers only the
historical bridge, a four-worker E2 run requires a separate pre-output operational amendment and
an equivalence smoke check. This record does not authorize that launch.

## Evidence

- `step2-out/bridge/SUMMARY.json`
- `step2-out/bridge/SPEED_REPORT.md`
- `freezeb-screen/20260913-freezeb-screen/meter.jsonl`
- `runtime-lock/20260913T004054Z-BRT1/LEDGER.json`
- `freeze/20260913T022442Z-B2/ADMISSION_LOCK.json`
