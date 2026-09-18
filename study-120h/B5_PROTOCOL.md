# B5 batch-size dose-response: CPU protocol before any model output

Status: CPU-only scientific protocol. No B5 model output exists. Every decision below is
frozen before generation. The seal status and every hash live in the separate
`freeze/<stamp>-B5/` directory.

This protocol is new work. It does not edit, extend, or reinterpret the sealed X1R4
extension protocol, the sealed Freeze B protocol, the sealed structured-output recovery
amendment, or any timestamped run directory. It reuses their frames, their pinned model
revisions, and their prompt and parser code without modification, and adds its own
schedule, output contract, estimands, thresholds, and budget.

## 1. Question

Does pairwise error dependence inside a batched annotation call change with batch size, or
is the size-16 effect measured in B3 peculiar to size 16?

The design measures the same estimand at five batch sizes in four model and task cells, and
reports how it moves with `log2(batch size)`.

## 2. Cells and item frames

Four cells, one per model family. Each cell uses the existing sealed 8,192-item
confirmation frame of its task.

| Cell | Model revision | Frame | Ordered doc ID SHA-256 | Document SHA-256 |
|---|---|---|---|---|
| `Qwen/Qwen3-8B::civil_comments` | `b968826d9c46dd6066d109eabc6255188de91218` | `civil_comments` confirm | `0384663a83ec28d3…` | `adfc6d1d19e5c8be…` |
| `ibm-granite/granite-3.3-8b-instruct::imdb` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | `imdb` confirm | `f932668ba6008d19…` | `f97558d2f65466cf…` |
| `meta-llama/Llama-3.1-8B-Instruct::civil_comments` | `0e9e39f249a16976918f6564b8830bc894c89659` | `civil_comments` confirm | `0384663a83ec28d3…` | `adfc6d1d19e5c8be…` |
| `mistralai/Mistral-7B-Instruct-v0.3::multi_nli` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | `multi_nli` confirm | `55650209e7b05f41…` | `74ae367e059b5454…` |

The full 64-character hashes, the frame file hash, and the sealed frame identifier hash of
every cell are recorded in `SCHEDULE_DIGEST.json` under `frames`. The frames are verified
against the sealed `freeze/20260913T183353Z-X1R4` identifiers on every load. The tokenizer
revision equals the model revision for every cell, because each pin names one immutable
repository commit that carries both.

The four-family scope is fixed. It is never reduced to save money.

## 3. Batch sizes, assignments, and the nesting rule

Sizes: 2, 4, 8, 16, 32. Assignments: 3, independently seeded.

Size 16 is generated fresh. No earlier size-16 output is reused as a primary observation.

Grouping rule, frozen:

1. Each cell and assignment has one canonical permutation of the 8,192 frame positions,
   drawn from `numpy.random.default_rng(derived_seed("bigdata2026-b5-v1", cell_key,
   "permutation", assignment))`.
2. A size-`k` group is a contiguous `k`-slice of that permutation.
3. Therefore each size-`2k` group is exactly the union of the two adjacent size-`k` groups,
   and every item appears exactly once at every size in every assignment.

The nesting is verified for every cell, every assignment, and every adjacent size pair
before the seal. `SCHEDULE_DIGEST.json` records the check under `nesting`.

## 4. What changes and what is held fixed

Only the grouping size changes, together with the prompt fields that the size necessarily
implies: the number of documents shown, the joint instruction count `{n}`, the requested
target list, the JSON schema, and the output token limit.

Held fixed across every size: the item frame, item marginals, prompt template and wording,
document preprocessing, model revision, tokenizer revision, temperature, seed policy,
serving configuration, parser, and analysis code.

## 5. Conditions

Two conditions per group, dispatched together:

- `joint`: one call that shows the `k` documents in group order and requests a label for
  every one of them.
- `exact_context`: `k` calls that show the same `k` documents in the same order and request
  a label for exactly one target.

No fresh solo arm is generated. The solo arm is not part of this estimand.

Per cell: 23,808 joint calls and 122,880 exact-context calls, 146,688 in total.

| Size | Joint per cell | Exact-context per cell |
|---:|---:|---:|
| 2 | 12,288 | 24,576 |
| 4 | 6,144 | 24,576 |
| 8 | 3,072 | 24,576 |
| 16 | 1,536 | 24,576 |
| 32 | 768 | 24,576 |

Totals across the four cells: **joint 95,232, exact-context 491,520, total 586,752**.
`b5.verify_counts` fails closed if the generated schedule differs from these three numbers.

## 6. Output contract

Contract version `b5-structured-output-v1`. It is a new contract with its own hashes. The
sealed `repair-v1` amendment is a reference only, and no B5 artifact is pooled with it.

Structured JSON-schema decoding applies to every scientific call from the first call. There
is no selective repair path and no second contract.

- One schema per call: an object with a single required property `labels`, whose properties
  are exactly the requested target IDs, each an integer restricted to that task's legal
  label values (`civil_comments` and `imdb`: 0, 1; `multi_nli`: 0, 1, 2).
  `additionalProperties` is `false` at both levels.
- Extra, missing, or duplicated IDs are impossible under the schema, and a response whose
  parsed key set differs from the requested set is never counted as resolved.
- Parsing and validation are deterministic and reuse the sealed `labels.py` parser.
- The output token limit is derived before any model output. It is the UTF-8 byte length of
  the compact serialized maximum-valid response, plus a fixed allowance of 16 tokens. One
  token never covers less than one UTF-8 byte, and structured decoding runs with
  `disable_any_whitespace`, so this bounds the longest legal response.
- Size-32 context fit is proved on CPU from the pinned-tokenizer prompt lengths measured in
  the sealed B3 run. `CONTEXT_AUDIT.json` records the worst case per cell and size. Every
  cell fits the 32,768-token window: Qwen 8,736, Granite 17,671, Llama 9,146, Mistral 7,665.

Every scientific event stores the raw output, the parsed output, prompt and output token
counts, the finish reason, the prompt SHA-256, the schema SHA-256, the assignment, the
document order hash, and the target ID hash.

### Serving configuration

vLLM 0.28.0, `VLLM_BATCH_INVARIANT=1`, `tensor_parallel_size=1`, `dtype=bfloat16`,
`max_model_len=32768`, `gpu_memory_utilization=0.90`, `max_num_seqs=32`, `temperature=0.0`,
`top_p=1.0`, `top_k=-1`, `seed=20260912`,
`StructuredOutputsParams(json=…, disable_any_whitespace=True,
disable_additional_properties=True)`. One A100 PCIe 80GB Pod runs at a time.

### Retry and failure policy

- A call is retried at most once, and only when its first attempt status is
  `generation_failed`, `truncated`, or `worker_failed`.
- A retry repeats the identical prompt, schema, token limit, sampling seed, and serving
  configuration. The attempt ID changes and records its parent attempt ID.
- The decoding contract is never changed after a malformed response is seen.
- A terminal failure stays in the scientific ledger as an unresolved row. It is never
  deleted and never silently excluded. Unresolved rows shrink the affected call in the
  pair-weighted estimator.

## 7. Estimands

For one cell, one size `k`, and one condition, the within-call error covariance is the
pair-weighted estimator
`C = sum_b sum_{i != j in b} (E_i - Ebar)(E_j - Ebar) / sum_b |b|(|b| - 1)`,
where `E` is the 0/1 disagreement against the reference label and `Ebar` is the mean over
the resolved rows of that condition.

Standardized dependence is `C / (p (1 - p))`, where `p` is the marginal error probability of
that condition. This is the within-call error correlation.

**Primary per-cell, per-size estimand:** standardized joint dependence minus standardized
exact-context dependence. The raw joint-minus-exact covariance is reported beside it.

**Primary cross-size statistic:** inside each cell, the ordinary least squares slope of the
standardized contrast on `log2(batch size)` over the five sizes; then the equal-weight mean
of the four cell slopes.

Per-size and per-cell values average the three assignments. Uncertainty preserves the
assignment structure, and target rows are never treated as independent.

## 8. Inference, frozen before any output

- **Cluster bootstrap**, 2,000 draws. The resampling unit is the size-32 super-group, drawn
  with replacement independently inside each cell and each assignment, 256 blocks per
  assignment. A super-group is the top of the nesting hierarchy, so resampling it carries
  its subgroups at every size together.
- **Intervals** are 2.5 and 97.5 percentiles of the bootstrap distribution.
- **Prespecified test against zero**: the primary cross-family slope rejects zero at
  `alpha = 0.05` when its 95 percent bootstrap interval excludes zero. A two-sided bootstrap
  p value is reported beside it.
- **Regrouping permutation test**, 999 draws, within cell and size. It re-groups the observed
  labels under a fresh permutation, which is a valid null when no within-call dependence
  exists. Its smallest reachable p value is 0.001.
- **Multiplicity**: the primary statistic is single. The per-size categorical reading applies
  Holm across the four cells within each size. Per-size contrasts and secondary analyses are
  reported with intervals and are not used to claim the primary result.

Seeds are derived from the namespace `b5-bootstrap-v1`, so every interval and every p value
regenerates byte for byte on CPU.

## 9. Practical thresholds

The sealed B3 floor of 0.002 is a raw covariance. It is not transferred to the standardized
scale.

The standardized contrast is the excess within-call error correlation. The approximate
variance inflation of a size-`k` call is `1 + (k - 1) * correlation`. The existing B4
practical inflation floor is 1.25 at the sealed batch size 16, which needs a correlation of
`0.25 / 15`.

- **Practical contrast** = `(1.25 - 1) / (16 - 1)` = **0.0166667**.
- **Practical slope** = practical contrast divided by the four doublings from size 2 to
  size 32 = **0.0041667** per doubling.

## 10. The six prespecified readings

All six are frozen. Success is not defined as a positive slope.

1. **positive_monotonic** — the cross-family slope interval lies wholly above zero, every
   cell slope is positive, and the per-size contrasts are ordered in size. Pairwise error
   dependence strengthens with batch size.
2. **flat_positive** — every per-size contrast is at or above the practical contrast, the
   slope interval contains zero, and its upper bound is below the practical slope. Batching
   introduces dependence, but dependence does not materially grow with size.
3. **threshold** — the two smallest sizes lie below the practical contrast and the largest
   size lies at or above it. Evidence for a practical batch-size boundary.
4. **non_monotonic** — the per-size contrasts are not ordered in size, and at least one
   per-size interval excludes zero. Dependence varies with size without a simple dose
   response.
5. **null** — every per-size interval contains zero and every per-size contrast is below the
   practical contrast. The earlier size-16 effect does not replicate under the uniform
   contract.
6. **inconclusive** — the cross-family slope interval is wider than twice the practical
   slope, or a cell slope interval is wider than four times it. The experiment is
   inconclusive. The absence of significance is not evidence of equivalence.

Readings are descriptive labels applied by `b5_analysis.classify`. More than one may apply,
and the report prints every match.

## 11. Secondary analyses, all prespecified

Raw covariance slope on `log2(size)`; per-size joint, exact-context, and excess covariance;
per-size standardized dependence; marginal accuracy and error rate per condition; parse
failure and terminal failure rates per size and condition; within-call label homogeneity;
approximate variance inflation `1 + (k - 1) * standardized covariance`; model-specific
slopes; assignment sensitivity through the per-assignment range; leave-one-family-out
slopes; the nonlinear categorical-size reading with Holm across cells; and a clearly
labelled descriptive comparison against the sealed A9 size-16 B3 result.

The historical B3 values are on the raw covariance scale, were produced under the earlier
output contract, and are never pooled into this experiment.

## 12. Completeness and stopping rules

- The block is complete when every one of the 586,752 scheduled calls has a stored terminal
  attempt.
- A call with no first attempt is scheduled. A call whose first attempt is retryable and has
  no second attempt is retried once. Nothing else is dispatched.
- The run stops without completing when the measured metered hours reach the ceiling, or
  when the measured throughput projects past the ceiling. It stops with a complete ledger of
  the work already done, never by discarding calls.
- A stopped run is resumable. Resume reconciles the durable local receipts first, then
  dispatches only what is still outstanding.
- Analysis runs only on a complete block. A partial block is reported as partial.

## 13. Reporting rules

The analysis writes `ANALYSIS.json`, `REPORT.md`, and `MANIFEST.json`. The report always
states the primary slope, its interval, its two-sided p value, the prespecified test
outcome, the per-size table, the cell table, every matched reading, both practical
thresholds, and the descriptive historical comparison. An inconclusive result is reported as
inconclusive and never as equivalence. Two independent CPU regenerations of the analysis must
produce the same `result_sha256`.

## 14. Runtime, budget, and the launch boundary

Runtime is fitted from the four completed B4 A100 PCIe 80GB lanes only, as three
coefficients: seconds per prompt token, seconds per generated token, and seconds per
dispatched chunk of 32 calls. The fit reproduces all four measured lanes within 0.73 percent.
No unexplained multiplier is used.

| Quantity | Value |
|---|---:|
| Point estimate, metered A100 hours | 18.09 |
| Upper bound with no prefix reuse | 70.21 |
| Wall time on one A100 PCIe 80GB | equal to metered hours |
| Ceiling, metered A100 hours | 35.00 |
| Required balance at $1.59 per hour | $30.76 |
| Explicit reserve inside that balance | $2.00 |

The point estimate assumes the engine reuses a group's document prefix across that group's
exact-context calls. The chunk order enforces that condition: a joint call and its `k`
exact-context calls always reach the engine inside one dispatched chunk. The upper bound
assumes no prefix reuse at all and exceeds the ceiling, so the run carries a live meter and a
per-block projection gate instead of a multiplier. The ceiling is specific to this
experiment and is kept separate from historical usage.

**Launch authorization boundary.** Only `--launch` and `--resume` may create a Pod.
`--dry-run`, `--preflight`, `--status`, and `--verify-only` never create one. A launch is
refused unless every gate passes:

1. the seal verifies against the live files and the snapshot;
2. the schedule totals, prompt hashes, and schema hashes rebuild and match the seal;
3. the sealed context audit fits every scheduled prompt;
4. the CPU test evidence artifact is present;
5. no metered lane of this run is still open;
6. no Pod is running on the account;
7. the live Runpod balance covers the remaining block plus the explicit reserve;
8. the ceiling covers the point estimate;
9. Daniel supplies the confirmation token printed by the seal.

No credential, balance, Pod identifier, or API key is hard-coded. The launcher reads the
account through the Runpod CLI at run time.

## 15. Preservation

Every sealed protocol, lock, original artifact, timestamped run directory, recovery
artifact, and untracked user file is preserved. This protocol edits none of them.
