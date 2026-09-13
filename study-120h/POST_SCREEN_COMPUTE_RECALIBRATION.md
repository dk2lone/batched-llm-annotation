# Post-screen compute recalibration

Date: 2026-09-13
Status: operational planning record; no paid execution authorized by this document

## Decision

The 120-hour study does **not** need a scientific or protocol rebase. Its compute accounting does
need this operational correction: the completed solo screen used 1.6643 incremental A100-hours,
not 2.7788 hours. The larger number is the cumulative study meter after adding the earlier
0.1900-hour Step 1 run and 0.9244-hour synthetic runtime benchmark.

Do not transfer the solo-screen error ratio to joint work. Joint, exact-context, and repeated-
sampling prompts have different lengths and serving behavior. Keep their sealed synthetic
forecasts until the historical replication bridge supplies real joint-work timing, then reforecast
every later complete block from that evidence as the runtime protocol requires.

## Corrected solo-screen accounting

| Quantity | A100-hours |
|---|---:|
| Cumulative meter before solo screen | 1.1144 |
| Solo-screen incremental use | 1.6643 |
| Cumulative meter after solo screen | 2.7788 |
| Remaining to the 120-hour target | 117.2212 |
| Remaining to the 150-hour stop | 147.2212 |

The incremental value is independently reproducible in two ways:

1. `2.7787800390 - 1.1144443667 = 1.6643356724` A100-hours.
2. Summing `metered_a100_seconds` across the screen meter gives
   `5,991.6084206 / 3,600 = 1.6643356724` A100-hours.

The frozen synthetic forecast was 5.3189 base hours, 8.5102 guarded hours, and 15.9567 hours
under the 3× incident case. Base was 3.196× the measured screen use. The mandatory first-1,000-
call reforecasts summed to 1.4363 base and 2.2981 guarded hours; final use was 15.9% above that
updated base and 27.6% below its guarded value. This supports block-level measurement and
reforecasting. It does not supply a correction factor for joint prompts.

Daniel reported that the account charge for the screen was about $6. Treat that as an observed
all-in account charge, not an invoice-derived GPU price. Dividing it by the incremental screen use
gives a rough planning ratio of $3.605 per metered A100-hour. The earlier roughly $12 screen
forecast was wrong because it used a conservative synthetic runtime and the lower benchmark-era
dollar ratio; the cumulative 2.7788-hour meter must not be presented as screen-only use.

## Minimum, desired, and maximum compute

These tiers describe scientific scope. Dollar conversions use the rough $3.605 ratio only; Modal
CPU, memory, model loading, worker layout, and current prices can change the actual bill.

| Tier | Scientific work | Provisional additional A100-hours | Rough additional dollars |
|---|---|---:|---:|
| Minimum now | Historical Qwen3-8B/Civil Comments replication bridge | 1.125 base; 1.800 guarded; 3.375 incident | $4.06; $6.49; $12.17 |
| Desired core | Bridge plus all ten sealed E2 primary cells | 12.929 base; 20.686 guarded; 38.786 incident | $46.61; $74.58; $139.82 |
| Maximum study | Continue through the fixed queue toward the target; stop no later than 150 cumulative hours | 117.221 remaining to target; 147.221 to stop | about $422.59; about $530.74 |

The desired-core figures combine the bridge with the old synthetic forecasts for the ten cells at
their sealed item counts. They are deliberately provisional. The bridge must replace the joint
and exact-context rates before any primary-panel funding decision; do not reserve or launch the
whole desired tier from this table.

The maximum is a compute envelope, not a promise that every planned experiment fits. The fixed
queue stops at the last complete unit allowed by the scientific gates, the 1.6× start rule, the
remaining operating reserve, and the 150-hour cumulative stop. At the observed screen ratio, a
full 120-hour cumulative program is about $433 and the 150-hour stop is about $541. This agrees
with the active plan's previous roughly $431/$538 whole-study estimate despite the bad solo-screen
forecast.

## Next start decision

The primary experiment remains the historical Qwen3-8B/Civil Comments replication bridge. It is
the minimum useful next paid unit and measures the joint/exact-context workload needed to price the
rest of the study. Use 1.800 A100-hours, roughly $6.49 at the observed ratio, as the planned amount.
Keep access to the 3.375-hour, roughly $12.17 incident amount if the account balance permits, but
stop and reforecast if timing approaches that case.

No paid run was started while producing this recalibration.

## Evidence boundary

Inputs are the immutable runtime ledger at
`study-120h/runtime-lock/20260913T004054Z-BRT1/LEDGER.json`, the append-only solo meter and
first-1,000-call reforecasts at `study-120h/freezeb-screen/20260913-freezeb-screen/`, and the
sealed admission and item-count lock at `study-120h/freeze/20260913T022442Z-B2/ADMISSION_LOCK.json`.
This record changes no scientific threshold, admission decision, execution order, or sealed file.
