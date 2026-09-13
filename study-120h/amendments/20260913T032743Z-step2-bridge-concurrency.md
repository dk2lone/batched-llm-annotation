# Protocol amendment: two concurrent A100 workers for the replication bridge

- Amendment id: `step2-bridge-concurrency`
- Stamp (UTC): `20260913T032743Z`
- Amends freeze: `20260912T041633Z`
- Protocol content hash: `218c5bcaa30f4f93c8d516f85f29ad8b7fb3be90f40238c2f3faf837ce611e15`
- Content hash: `24be20aabdcf934d683f0398f3aa377895910ba5cc7f5001690b671303c5ef33`

## The deviation

- Sealed field: `panel.serving.hardware`
- Sealed text: "one NVIDIA A100, one at a time"
- Operational practice: at most 2 A100 workers run at the same time, one model process on one A100 each

This is a deviation from a recorded serving statement. It is written before any bridge
output exists. It changes no scientific field.

## Clauses

| id | topic | clause |
|---|---|---|
| B1 | recorded deviation | the sealed protocol records the serving hardware as 'one NVIDIA A100, one at a time'. This bridge runs at most two A100 workers at the same time. That is an operational deviation from that sentence and is recorded here before any bridge output exists. |
| B2 | one A100 per model process | each worker owns exactly one A100 and one model process. Tensor parallelism stays forbidden and tensor_parallel_size stays 1. No call is ever split across GPUs. |
| B3 | block ownership | a frozen execution block is never split across workers. One worker owns one block for the whole life of that block, so the fresh serving session per block and the ascending execution_order_key inside the block are both preserved. |
| B4 | budget accounting | concurrency shortens the calendar and never the budget. Metered A100 hours are summed over workers, so two concurrent single-A100 workers cost two A100 hours per wall-clock hour. The 150-hour stop is applied to that sum and is checked conservatively, counting the work both workers may still bill. |
| B5 | timing evidence | throughput measured under two concurrent workers is not the single-worker rate of the sealed forecast. Every reforecast written by this bridge records the worker count that produced it, and per-condition rates are reported as measured, never rescaled to a one-worker equivalent. |
| B6 | scientific separation | no scientific field changes. The worker count is not a treatment, is not recorded as a condition, and never enters an estimand, a test, or a claim gate. The frozen serial-serving control keeps its own definition and is not affected. |

## What this amendment does not change

- the sealed Step 0 freeze, which stays byte-identical and read-only
- every sample frame, item id, filler pool, and dataset revision
- every prompt template, response grammar, and parser rule
- every call id, assignment, execution block, and within-block execution order
- the model and tokenizer revision, and the vllm==0.28.0 engine pin
- every serving field: max_model_len 8192, bfloat16, tensor_parallel_size 1, gpu_memory_utilization 0.9, max_num_seqs 64, greedy decoding at seed 0
- the first-response policy, the one-retry rule, and the retry triggers
- every estimand, test, threshold, and claim gate

## Reading the timing evidence

Measured throughput under two workers is a two-worker rate. Summed A100 hours are the
budget quantity and are unaffected by the worker count. Elapsed wall-clock time is
roughly halved and is never reported as a cost saving.
