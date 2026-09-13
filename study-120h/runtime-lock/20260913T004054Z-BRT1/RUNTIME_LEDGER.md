# Freeze B runtime and compute ledger

Benchmark run: `20260912T233301Z-5738ea84`. Status: **sealed_with_declared_instrumentation_gap**.
Synthetic outputs are runtime evidence only and never enter scientific analyses.

## Measured benchmark

| quantity | value |
|---|---:|
| model workers | 6 |
| planned model/case checks | 180 |
| successful measured requests | 316 |
| failed measured requests | 0 |
| context-excluded model/cases | 22 |
| billed A100 hours | 0.924 |
| total benchmark cost | $2.01 |

## Immediate queue forecast

| block | base h | 1.6x h | 3x h |
|---|---:|---:|---:|
| `all_36_solo_screen_cells` | 5.32 | 8.51 | 15.96 |
| `historical_replication_anchor` | 1.12 | 1.80 | 3.37 |

The screen estimate covers 36 runnable model-task candidates and one model load per six-task bundle.
The anchor estimate covers the 12,672 still-unrun Qwen3-8B/Civil Comments calls.

## Conditional panel ranges

Admission and solo-only item-count selection have not happened. These are low/high ranges over
currently forecastable candidate cells, not promises about the final selected panel.

| block | base h | 1.6x h | 3x h |
|---|---:|---:|---:|
| `e2_primary_16_n2048` | 9.43–103.28 | 15.09–165.25 | 28.29–309.84 |
| `e2_primary_16_n4096` | 18.05–205.71 | 28.88–329.14 | 54.15–617.13 |
| `e2_primary_16_n8192` | 35.29–410.57 | 56.46–656.92 | 105.87–1231.72 |
| `e5_first_two_through400` | 14.25–1722.80 | 22.80–2756.48 | 42.75–5168.40 |
| `e5_six_cells_through5000` | 366.17–22205.51 | 585.87–35528.82 | 1098.51–66616.54 |
| `e4_eight_cells` | 0.52–3.75 | 0.83–6.00 | 1.56–11.26 |

## Account decision

At this run's all-in rate, 120 A100-hours is about **$261** and 150 hours is about **$326**.
If the stated $14 was available before this benchmark, about **$11.99** remains, equivalent to roughly **5.51 A100-hours** at the observed all-in rate.
The required 10-hour operating reserve alone is about **$21.75** at that rate.
That single account cannot fund the study by itself. This is not the overall funding decision.
Across the user's accounts, declared capacity is **120–150 A100-hours**. That covers the 120-hour target and reaches the 150-hour hard stop.

## Limits and next action

The base rate uses the slower of two measured repetitions and reconciles request time to Modal's billed A100 time. The 1.6x case is the start-decision forecast; 3x is the incident case. Batch sizes 2, 4, and 8 are interpolated because only 1, 16, and 32 were measured.

The benchmark did not separately capture time-to-first-token/prefill and decode time. That gap is sealed here rather than reconstructed. Capture those components in the mandatory first-1,000-real-calls reforecast, and do not describe this ledger as a component-level timing measurement.

Next: run the solo-only screens within the declared aggregate capacity. The admission lock will replace conditional ranges with the exact selected cells and item counts before any joint scientific output.
