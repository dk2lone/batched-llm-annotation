# B5 batch-size dose-response: calibrated, full run not launched

The 1,080-call GPU calibration smoke completed on 2026-09-18. The 586,752-call experiment has
not launched. No Pod remains running.

## 1. What is ready

| Item | Path |
|---|---|
| Protocol | `study-120h/B5_PROTOCOL.md` |
| Seal | `study-120h/freeze/20260919T170405Z-B5/` |
| Preparation artifacts | `study-120h/extension-b5/preparation/20260918T185042Z/` |
| CPU test evidence | `study-120h/extension-b5/CPU_TESTS.json` |
| CPU dry-run artifacts | `study-120h/extension-b5/dry-run-20260918-calibration-digest/DRY_RUN-full.json`, `DRY_RUN-smoke.json` |
| Schedule, contract, forecast | `study-120h/src/freezeb/b5.py` |
| Estimands and readings | `study-120h/src/freezeb/b5_analysis.py` |
| Power and precision | `study-120h/src/freezeb/b5_power.py` |
| Preparation builder | `study-120h/prepare_b5.py` |
| Seal tool | `study-120h/seal_b5.py` |
| Launcher | `study-120h/run_b5.py` |
| Remote worker | `study-120h/b5_worker.py` |
| Analysis | `study-120h/analyze_b5.py` |
| Tests | `study-120h/tests/test_b5.py`, `test_b5_analysis.py`, `test_b5_runner.py`, `test_batch_invariance.py` |

Seal manifest SHA-256: `f239a63ac05b93e35e4c3b0d7edec7d08b3a2b77b2c239e227e1a720a0315171`.
Schedule SHA-256: `6145f5088a56fc869929257daa9b145fc1351ce34eb03cafa7101448bd547a25`.
Confirmation token: `f239a63ac05b`. The execution seal covers 24 files and fails if any changes.

## 2. Exact design and call count

Four cells, five batch sizes, three independently seeded assignments, the existing sealed
8,192-item confirmation frame per cell. Size 16 is generated fresh.

Groups are contiguous slices of one canonical permutation per assignment, so a size-`2k`
group is exactly the union of two adjacent size-`k` groups.

| | Per cell | All four cells |
|---|---:|---:|
| Joint calls | 23,808 | **95,232** |
| Exact-context calls | 122,880 | **491,520** |
| Total | 146,688 | **586,752** |

The launcher fails closed if the generated schedule differs from those three numbers. The
CPU dry run produced exactly 586,752 calls in 1,212 chunks.

Every scientific call uses structured JSON-schema decoding from the first attempt. Output
token limits are derived before generation from the serialized maximum valid response.
Size 32 fits the 32,768-token window in every cell: Qwen 8,736, Granite 17,671, Llama 9,146,
Mistral 7,665.

## 3. Runtime and cost

Fitted from the four completed B4 A100 PCIe 80GB lanes only. Three coefficients: seconds per
prompt token, seconds per generated token, seconds per dispatched chunk of 32 calls. The fit
reproduces all four measured lanes within 0.73 percent. No unexplained multiplier is used.

| Quantity | Value |
|---|---:|
| Planning point estimate, metered A100 hours | **18.09** |
| Upper bound, no prefix reuse at all | 70.21 |
| Ceiling, metered A100 hours | **35.00** |
| Wall time on one A100 PCIe 80GB | equal to metered hours |
| Smoke budget | 2.00 h / **$5.18 including reserve** |
| Calibrated full-run projection | **27.49 metered A100 hours** |
| Full-run required balance | **$45.71 including $2 reserve** |

Per cell, at the point estimate: Qwen 3.71 h, Granite 7.24 h, Llama 3.66 h, Mistral 3.48 h.
Each cell includes a 0.25 h boot allowance and a 2 percent retry allowance.

The completed smoke used 0.4194 metered A100 hours. All 20 blocks observed prefix reuse, the
overall cache-hit fraction was 83.41%, and measured generation was 1.5504 times the fitted
forecast. The live balance after teardown is `$7.9940765883`, so the funding gap is `$37.72`.
The phased launcher uses `extension-b5/smoke-01/SMOKE_CALIBRATION_EXECUTION_ONLY.json`. It points to
the original smoke record and transfers its projection because the schedule hash is unchanged.

The point estimate sits inside the 35-hour ceiling. The no-reuse upper bound does not, so the
run carries a live meter and a per-block projection gate. If measured throughput projects past
35 metered hours, the run stops with a complete ledger instead of overspending.

This differs from the earlier working figure of 29 to 30 metered A100-hours. The 18.09-hour
estimate is recalculated from the locked B4 runtime evidence and the generated B5 schedule.

## 4. Commands for later

Run every command from `/Users/dz/bigdata/study-120h`.

### Step 1 — complete. The calibration smoke passed.

### Step 2 — reload at least $37.72, then run read-only preflight.

```
.venv/bin/python run_b5.py --preflight --output extension-b5/run-01 \
  --calibration extension-b5/smoke-01/SMOKE_CALIBRATION_EXECUTION_ONLY.json \
  --confirm f239a63ac05b --usd-per-hour 0 --reserve-usd 0
```

Success: `"passed": true` and `"pod_created": false`. The printed
`required_balance_usd` is the funding target.

### Step 3 — launch phase 1: Qwen and Granite, all five batch sizes.

```
.venv/bin/python run_b5.py --launch --output extension-b5/run-01 \
  --model-phase 1 --calibration extension-b5/smoke-01/SMOKE_CALIBRATION_EXECUTION_ONLY.json \
  --confirm f239a63ac05b --usd-per-hour 0 --reserve-usd 0
```

Success: the command exits after both model cells, with every completed call durable.

### Step 4 — resume phase 2: Llama and Mistral, all five batch sizes.

```
.venv/bin/python run_b5.py --resume --output extension-b5/run-01 \
  --model-phase 2 --calibration extension-b5/smoke-01/SMOKE_CALIBRATION_EXECUTION_ONLY.json \
  --confirm f239a63ac05b --usd-per-hour 0 --reserve-usd 0
```

Success: the combined ledger reaches `"completed_calls": 586752`.

### Step 5 — progress, at any time, from another terminal. Creates no Pod.

```
.venv/bin/python run_b5.py --status --output extension-b5/run-01
```

### Step 6 — resume after any stop.

```
.venv/bin/python run_b5.py --resume --output extension-b5/run-01 \
  --model-phase <1-or-2> \
  --calibration extension-b5/smoke-01/SMOKE_CALIBRATION_EXECUTION_ONLY.json \
  --confirm f239a63ac05b --usd-per-hour 0 --reserve-usd 0
```

Resume reconciles the durable receipts first, then dispatches only the outstanding calls.

### Step 7 — analysis, CPU only.

```
.venv/bin/python analyze_b5.py \
  --events extension-b5/run-01/events.jsonl \
  --output extension-b5/run-01/analysis
```

The smoke selection read the same generator as the full schedule and completed 1,080 calls.
It differed only in which chunks it selected.

## 5. Success and failure indicators

**The preparation is healthy when:**

- `seal_b5.py --verify freeze/20260918T190501Z-B5` prints the manifest hash above;
- `run_b5.py --dry-run` prints 586,752 calls and `"gpu_work_launched": false`;
- the 126 targeted tests pass.

**The run is healthy when:**

- `--status` shows `percent_complete` rising and `open_lanes` at 0 between blocks;
- `projected_total_metered_hours` stays below 35;
- `terminal_unresolved_attempts` stays a small fraction of `stored_attempts`.

**Stop and investigate when:**

- preflight reports `"passed": false`;
- `projected_total_metered_hours` exceeds 35, which stops the run by design;
- `--status` shows an open metered lane while nothing is running, which means a Pod may
  still exist; check the Runpod console before relaunching;
- terminal failures cluster in one cell or one size, which would bias that condition.

**Result reading.** All six outcome readings are frozen in the protocol. Success is not
defined as a positive slope. A null result, a flat result, a non-monotonic result, and an
inconclusive result are all valid outcomes, and an inconclusive result is never reported as
equivalence.

## 6. What this experiment can and cannot resolve

From 200 simulated replicates per shape, using completed B3 rates as planning input only:

| Shape | True slope | Mean estimate | Slope SE | 95% half width | Rejection rate |
|---|---:|---:|---:|---:|---:|
| positive_monotonic | 0.00417 | 0.00412 | 0.00130 | 0.00254 | 0.89 |
| threshold | 0.00444 | 0.00442 | 0.00128 | 0.00251 | 0.93 |
| decreasing | −0.00417 | −0.00407 | 0.00134 | 0.00262 | 0.84 |
| flat_positive | 0.00000 | 0.00004 | 0.00136 | 0.00267 | 0.04 |
| non_monotonic | 0.00000 | 0.00002 | 0.00124 | 0.00244 | 0.05 |
| null | 0.00000 | 0.00003 | 0.00124 | 0.00243 | 0.06 |

The 95 percent half width is below the practical slope of 0.0041667 in every shape, so the
design resolves a practical-size dose response.

Per-size precision is uneven. The standard error of a per-size contrast is 0.0059 at size 2,
0.0034 at size 4, 0.0024 at size 8, 0.0016 at size 16, and 0.0013 at size 32. Size 2 is the
least precise because a size-2 call contributes only one pair. The design can resolve a
per-size contrast at or above the practical contrast of 0.0167 at every size, and cannot
resolve a per-size contrast smaller than its own standard error.

The design was not tuned toward the observed B3 direction. The simulated shapes include a
decreasing arm and a null arm, and the false-positive rate under the null, flat and
non-monotonic shapes stays between 0.04 and 0.06.

## 7. Verification already run

| Check | Result |
|---|---|
| Targeted B5 tests | 126 passed, 0 failed |
| Full repository suite | 659 passed, 0 failed |
| Ruff | 23 errors, all inside sealed source snapshots, 0 in live code |
| Strict mypy | 8 errors, all pre-existing, 0 in any B5 file |
| Seal verification | passes against live files and snapshot, 25 files |
| Independent analysis regeneration | two processes, byte-for-byte identical |
| Full CPU schedule build and dry run | 586,752 calls, 1,212 chunks, no Pod |
| Smoke dry run | 1,080 calls, 20 chunks, no Pod |

## 8. Statement about GPU work

No Pod was created. No model was called. No paid GPU benchmark ran. No Runpod write
operation was made. Every number in this document comes from CPU computation over sealed
B3 and B4 evidence.
