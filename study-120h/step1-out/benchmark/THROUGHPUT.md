# Step 1 throughput benchmark and forecast

Freeze: `20260912T041633Z`
Written: 2026-09-12T20:55:26.233000+00:00

Every number in this file is a performance fact measured on solo calls. None of it is
a scientific finding. No number here may enter a claim about error covariance,
variance, or interval coverage. Those endpoints belong to Steps 2 to 4.

## Measured solo throughput

| model | calls | generation s | calls/s generation | calls/s with overhead | output tokens/label |
|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-8B` | 6144 | 95.6 | 64.237 | 15.387 | 22.7 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 4096 | 68.5 | 59.815 | 20.858 | 22.6 |

## Fitted cost model

| model | s/prefill token | s/decode token | s/call fixed | mean abs error s | measured prompt tokens/call |
|---|---:|---:|---:|---:|---|
| `Qwen/Qwen3-8B` | 5.970e-05 | 7.725e-05 | 0.0009 | 0.008 | 201 to 248 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 3.072e-05 | 9.397e-05 | 0.0074 | 0.013 | 218 to 265 |

## Forecast for the remaining frozen calls

| cell | calls | measured-proportional h | measured-flat h | frozen-cap h |
|---|---:|---:|---:|---:|
| `qwen_civil` | 14720 | 0.36 | 0.36 | 0.38 |
| `qwen_secondtask` | 8576 | 0.16 | 0.16 | 0.17 |
| `mistral_civil` | 8576 | 0.13 | 0.13 | 0.14 |
| `mistral_secondtask` | 9632 | 0.13 | 0.13 | 0.15 |
| `qwen_secondtask` | 8576 | 0.15 | 0.14 | 0.16 |

The frozen-cap column is the hard upper bound: no call may generate past its cap.

## Extrapolation limit

Step 1 generates solo calls only, by protocol. A joint call carries about sixteen times
the prefill tokens of a solo call, so the fitted prefill coefficient is applied far
outside its measured range. Read every joint projection as an order of magnitude.

## Metered A100 hours so far

- runs recorded: 4
- total metered A100 hours: 0.190
- longest single worker: 0.111 h

metered A100 hours are summed over workers. Two concurrent single-A100 workers cost the same total hours as one worker running twice as long.

## Frozen call ledger that remains

- Steps 2 and 3 planned calls: 41504
- solo calls already produced in Step 1: 8192
- Steps 2 and 3 remaining: 33312
- Step 4 minimum across the two frozen cells: 41600
- Step 4 maximum across the two frozen cells: 80000

Trim order if the projection exceeds the budget:

1. delete step 5
2. reduce the evaluation-replicate maximum toward its frozen 400 minimum using timing data only
3. stop before confirmatory calls and publish a revised, newly frozen smaller design

Never trimmed:

- the exact-context control
- a primary cell, based on an observed effect
- the second model or task, after joint outputs are seen
- the reserve, before the two core result streams complete
