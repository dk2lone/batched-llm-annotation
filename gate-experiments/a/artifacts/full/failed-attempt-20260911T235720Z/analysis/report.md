# Gate A result: within-batch error correlation

- Config: `full` (fingerprint `0030582ab7b320b7`)
- Backend: vllm; model: Qwen/Qwen3-8B
- Rows: 8000 over 2375 jobs and 2000 documents

## Verdict: UNDETERMINED

- the aggregate ICC is undefined, usually because a condition has no variance

Aggregate ICC n/a, decision interval [n/a, n/a], largest fake-batch p n/a.

## Accuracy and parsing by condition

| condition | rows | docs | accuracy | error rate CI | parse failure rate |
|---|---|---|---|---|---|
| solo | 1982 | 1982 | 0.8325 | [0.1509, 0.1837] | 0.0090 |

## Paired error deltas, per document

| contrast | docs | mean error delta | CI | excludes zero |
|---|---|---|---|---|

## Caveats carried by this report

- Parse failures are dropped from the ICC. If they cluster by batch, that deletion is not ignorable. See `icc.sensitivity_failures_as_errors`.
- The aggregate interval treats the three assignments as independent. The decision interval widens it to the conservative envelope.
- A single gate on one model and one dataset bounds nothing beyond that cell.
