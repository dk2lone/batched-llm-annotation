# Protocol amendment: Step 1 execution budget and worker layout

- Amendment id: `step1-execution-budget`
- Stamp (UTC): `20260912T202451Z`
- Amends freeze: `20260912T041633Z`
- Protocol content hash: `218c5bcaa30f4f93c8d516f85f29ad8b7fb3be90f40238c2f3faf837ce611e15`
- Content hash: `1d81dce775639c91f62103305e2125cbeac41f8790108df8f3bd1672d856c89b`

## Status

This amendment records an authorisation the user gave before any model output existed.
It changes the execution budget and the worker layout only. The Step 0 freeze is
unchanged and stays sealed. No sample frame, prompt, threshold, estimand, test, or claim
gate is touched. A change to any of those would require a new freeze directory, never an
edit of the sealed one.

## Clauses

| id | topic | clause |
|---|---|---|
| A1 | planning target | 120 metered A100 hours is a planning target, not the hard limit of the study. |
| A2 | overall ceiling | 150 metered A100 hours is the overall ceiling. Work stops at that total. |
| A3 | Step 1 allocation | the 8-hour Step 1 allocation is a checkpoint, not a hard cap. Reaching it requires a report and a decision, not an automatic stop. |
| A4 | concurrent workers | Step 1 may run independent workers concurrently, one model on one A100 per worker. Tensor parallelism stays forbidden and tensor_parallel_size stays 1. |
| A5 | budget accounting | concurrency shortens the calendar and never the budget. Metered A100 hours are summed over workers, so two concurrent single-A100 workers cost the same total as one worker running twice as long. |

## What this amendment does not change

- every sample frame, item id, and dataset revision
- every prompt template, response grammar, and parser rule
- every model and tokenizer revision, and the vllm==0.28.0 engine pin
- every serving field: max_model_len 8192, bfloat16, tensor_parallel_size 1, gpu_memory_utilization 0.9, max_num_seqs 64, greedy decoding
- every screen criterion, threshold, floor, estimand, test, and claim gate
- the discovery and confirmation separation, and the replacement and fallback policies

## Frozen stage ceilings, for reference

| step | frozen ceiling in A100 hours |
|---|---:|
| step_0_freeze | 0 |
| step_1_screen_and_benchmark | 8 |
| step_2_anchor_replication | 14 |
| step_3_three_fresh_cells | 32 |
| step_4_repeated_sampling | 46 |
| step_5_optional_sensitivity | 5 |
| step_6_reserve | 15 |
| total | 120 |

The frozen table above is unchanged. Clauses A1 to A3 place it inside a 150-hour overall
ceiling and make the Step 1 row a reporting checkpoint. Clause A5 keeps the accounting
honest: two concurrent single-A100 workers cost two A100 hours per wall-clock hour.
