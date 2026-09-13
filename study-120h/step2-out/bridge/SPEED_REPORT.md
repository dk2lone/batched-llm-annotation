# Historical Qwen3-8B / Civil Comments replication bridge

Run `20260913T032819Z-f63f33a2` completed on 2026-09-13 at 04:10 UTC in Modal workspace `daniel21cn2016`. It executed the sealed Step 0 `qwen_civil` bridge with two concurrent A100-40GB block workers after a 1,000-call one-worker timing barrier. The operational deviation is recorded in `../../amendments/20260913T032743Z-step2-bridge-concurrency.md`. The 2,048 Step 1 solo first responses were reused and not rerun. This is historical replication output, separate from the fresh Freeze B confirmation panel.

| Condition | First-response calls | Target rows | Valid | Malformed | Allocated A100 seconds/call |
|---|---:|---:|---:|---:|---:|
| Joint | 384 | 6,144 | 384 | 0 | 0.403 |
| Exact context | 6,144 | 6,144 | 6,137 | 7 | 0.341 |
| Filler | 6,144 | 6,144 | 6,134 | 10 | 0.357 |
| Total | 12,672 | 18,432 | 12,655 | 17 | 0.351 |

Every planned first call and target row is present. No response was missing or truncated, no retry was needed, and no ledger conflict was recorded. The malformed first responses remain in the ledger under the frozen rule. Each of eight blocks used one fresh serving session; blocks 1 and 2, 3 and 4, and 5 and 6 had overlapping inference timestamps. All 15 checkpoint files matched the pinned revision and SHA256 manifest before generation.

The bridge used **1.2338 incremental A100-hours** across its two workers. The cumulative study meter moved from 2.7788 to **4.0125 A100-hours**. The run lasted **41 minutes 45 seconds** from local start to final summary, or **5.06 calls/second** end to end. This includes model loads and compilation. The sum of GPU time corresponds to **10,271 calls/A100-hour**. The first 1,000 calls used 0.1440 A100-hours, and the mandatory condition-specific reforecast was written before the two-worker phase.

The condition figures in the table are **allocations of measured chunk GPU time**, not measured standalone request latencies. vLLM returned no per-request queue, prefill, or decode timing for this run. The frozen conditions were interleaved within chunks; equal shares were assigned within each chunk. A final chunk-composition regression had a positive but implausible exact-context coefficient, so it should not be read as a separate condition benchmark. The whole-run wall time and summed A100-hours are directly measured operational speed. No scientific dependence statistic or confidence-interval coverage result is claimed here.

Evidence: `SUMMARY.json`, `calls.jsonl`, `timing.jsonl`, `meter.jsonl`, `meter-state.json`, `reforecast-1000.json`, `checkpoint-verification.json`, and the pre-output concurrency amendment. The Modal app completed normally: `ap-VD4cKZVRToqEG2SLZSftBw`.
