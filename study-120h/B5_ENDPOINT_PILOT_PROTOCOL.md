# B5 endpoint pilot v1

Status: separate exploratory spending decision. This protocol and its seal precede all pilot model output. The existing five-size B5 protocol, seal, smoke, and future confirmatory output are separate and must never be pooled with this pilot.

## Design

Use the four cells and their complete 8,192-item confirmation frames listed in `B5_PROTOCOL.md`. One fresh canonical permutation per cell is derived from the seed namespace `b5-endpoint-pilot-v1`. Contiguous groups of 2 and 32 are formed from that permutation, so each size-32 super-group contains exactly 16 adjacent size-2 groups. Every item appears once at each endpoint in each condition. The two conditions are joint and exact-context, with the same group content and order.

The exact schedule per cell is 4,096 size-2 joint, 8,192 size-2 exact-context, 256 size-32 joint, and 8,192 size-32 exact-context calls. Across four cells this is 17,408 joint and 65,536 exact-context calls, exactly **82,944**. Count or coverage mismatch stops execution. Call IDs, assignment seeds, and output namespace are distinct from B5.

The pinned checkpoint and tokenizer revisions, prompt renderer, structured JSON schema, parser, zero-temperature vLLM 0.28.0 serving configuration, first-response preservation, and single identical retry for generation failures, truncation, or worker failure are the B5 policy. Every terminal failure remains in the ledger. The B5 smoke's successful size-2 and size-32 schema and context observations are supporting technical evidence; this pilot's schedule is checked separately before launch.

## Estimands and uncertainty

For each cell and size, compute B5's pair-weighted within-call error covariance against the dataset reference label for joint and exact-context. Divide each condition's covariance by its own marginal error probability times one minus that probability. The standardized contrast is standardized joint minus standardized exact-context dependence. Report raw covariance contrast, condition error and accuracy, approximate variance inflation, and technical failures beside it.

The primary pilot statistic is the equal-cell mean size-32 standardized contrast minus the equal-cell mean size-2 standardized contrast. Report per-cell differences and leave-one-family-out differences. Use 2,000 deterministic cluster-bootstrap draws. Resample 256 size-32 super-groups within each cell, carrying the 16 nested size-2 groups and both conditions together. Use percentile 95% intervals for each endpoint and their difference. Never resample individual documents. Derived seed namespace: `b5-endpoint-pilot-v1-bootstrap`. Two independent CPU regenerations must yield identical scientific result hashes.

## Exploratory spending decision

Set `tau = 0.0166667`, `c2` to the equal-cell size-2 contrast, `c32` to the equal-cell size-32 contrast, and `delta = c32 - c2`. Apply the following rule mechanically, in order:

1. `GO` if `abs(delta) >= tau` and at least three cell differences have delta's sign; or the 95% delta interval excludes zero; or both endpoint contrasts reach tau with at least three cells positive at both endpoints; or one endpoint reaches tau with its interval excluding zero while the other point estimate is below tau/2.
2. `NO_GO` only if both endpoint 95% intervals lie wholly inside `[-tau, tau]`.
3. Otherwise `INCONCLUSIVE`.

This classifies whether to consider spending on the full B5 experiment. It is not a confirmatory gate. `GO` does not prove dose response. `NO_GO` and `INCONCLUSIVE` do not prove equivalence. No outcome authorizes automatic full B5 launch.

## Execution and preservation

Run in `extension-b5-endpoint-pilot/` only. Use one A100 PCIe 80GB Pod at a time at no more than $1.59/hour. Expected pilot runtime is 4–5 metered A100 hours. The hard ceiling is 7.0 metered A100 hours, or $11.13 in GPU charges. Before any Pod creation, require live available balance of at least $13.00, verify no Pod is running, verify the seal, complete CPU tests and dry run, and install the watchdog before model setup. Store local chunk receipts and append-only events. Resume from receipts without replacing completed receipts. Delete a Pod on completion or unrecoverable failure and verify deletion through the Runpod API. Stop if measured time or its projection would exceed 7.0 hours.

Analyze only after all 82,944 scheduled calls have terminal records and zero are missing. Write `ANALYSIS.json`, `REPORT.md`, `MANIFEST.json`, and an execution and cost summary in the pilot run namespace. Record the full technical failure and retry counts. Commit the preparation and completed result locally; do not push.
