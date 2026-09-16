# IEEE BigData 2026 handoff

Updated: 2026-09-16 00:25 America/Toronto (2026-09-16 04:25 UTC).

## Current state

Project: dependence in batched LLM text annotation. Repository:
`dk2lone/batched-llm-annotation`, private, branch `main`. The latest execution
evidence commit before this handoff is `606b616` (`record runpod capacity retries`).

Daniel authorized the initial B4 / A11 scientific run: 250 reference and 400
evaluation replicates for four model-task cells, or 83,200 first calls. The run is
not active. Live verification at this handoff showed:

- no Runpod Pods;
- no detached `b4-a11` screen;
- no scientific model calls or results from the attempted launch;
- Runpod balance `$26.9081152783`;
- four retained 40GB model-cache volumes costing `$0.016/hour` total.

The only current launch blocker is provider capacity. `EUR-IS-1` repeatedly
allocated one Secure Cloud A100 SXM 80GB and rejected the next allocation. All
advertised A100 regions reported low or no stock. Do not describe this as an
experiment failure. The scientific experiment has not begun.

## Scientific execution lock

The initial checkpoint uses four separate single-GPU Secure Cloud A100 SXM 80GB
Pods, one per cell:

- `ibm-granite/granite-3.3-8b-instruct::imdb`
- `meta-llama/Llama-3.1-8B-Instruct::civil_comments`
- `mistralai/Mistral-7B-Instruct-v0.3::multi_nli`
- `Qwen/Qwen3-8B::civil_comments`

All four reference streams must finish before evaluation begins. Preserve the
sealed prompts, schedules, model revisions, one-GPU-per-lane design, 80-A100-hour
extension ceiling, 10-hour reserve, and 1.6 forecast guard. Do not substitute
Community Cloud, a 40GB GPU, a different model revision, a sequential scientific
runner, or a different accelerator without a new documented decision. Planned
comparisons are not findings.

## Completed preparation

The pinned private runtime image is:

`ghcr.io/dk2lone/b4-a100-runtime@sha256:1cf3be1dfb50dbb800a658b371e8ec58f1ea485b55bae5b92ee281af2feba898`

Runpod registry authentication is configured as `cmu3hayz6002fmkgiko27jzi0`.
Do not place registry or Runpod credentials in files, commands, logs, or chat.

Four exact offline model caches exist in `EUR-IS-1`:

| Cell | Volume ID | Size |
|---|---|---:|
| Granite / IMDb | `fnsdztl4tk` | 40GB |
| Llama / Civil Comments | `scqgo0j9sr` | 40GB |
| Mistral / MultiNLI | `l8doxgmrtn` | 40GB |
| Qwen / Civil Comments | `r4m3d3hmg2` | 40GB |

The exact Qwen, Granite, Llama, and Mistral revisions each passed a supervised
synthetic A100 smoke test using the offline worker. Receipt replay, durable
recovery, exact serving identity, image identity, A100 80GB hardware, and a seeded
CUDA witness passed. The GitHub Actions watchdog passed a live CPU deletion test.
A four-separate-A100 capacity probe also passed once before launch, showing that
the required allocation is possible but intermittent.

The launch-path change that serializes Pod creation passed Ruff, strict mypy, 65
focused tests, and the full 519-test suite. An independent CPU-only audit passed
all plan, source, configuration, preflight, budget, and evidence bindings. The
change and launch package were pushed in `99a8fb4`.

Auto-pay is disabled by Daniel's dashboard attestation. The GitHub watchdog is a
remote deletion fallback, not a provider-enforced spending cap. Normal local
cleanup and the prepaid balance remain independent controls.

## Active immutable records

The most recent clean launch package is:

`study-120h/extension-b4/runpod-20260916-serialized-launch-v2/`

Important identities:

- plan SHA-256: `cc83c70acfe9dffd94cb495812124860346b3ac5fe6181b71b727c3ae1cbb6d2`
- Runpod config canonical SHA-256:
  `48d51ecc6c50e3bfe275d035cb476f11c754533450c0282d5a3665267f683809`
- launch budget record SHA-256:
  `0a9f65d0007ce221ea8a2e470ad1f5e31ccf322f62930ca390e5eaad84e32a42`
- account ID: `user_3GCQlNiLDruYedpe87pTOrbYANt`
- verified A100 price: `$1.59/GPU-hour`

The old budget evidence is expired. Never edit or overwrite it. Prepare fresh
budget evidence and a new immutable launch directory before another paid attempt.
Use the same scientific plan, image, model revisions, cache volumes, and preflight
evidence unless a live check proves one changed.

`meter.jsonl` in the v2 directory is intentionally ignored by Git but remains in
the local workspace. It has all failed phases reconciled. The committed
reconciliation JSON files are the durable evidence if that ignored file is lost.

## Launch attempts and accounting

No attempted launch produced a scientific request.

1. The original runner issued four concurrent create calls. Three Pods were
   returned and the fourth failed with `graphql_error`. All three were deleted.
   Settled charge: `0.00978336830188609` A100-hours.
2. A retry returned no Pods and failed immediately. Settled charge: zero.
3. After serializing creation, Runpod returned one Pod and rejected the next.
   The returned Pod was deleted before inference. Settled charge: zero.
4. The final retry again returned one Pod and rejected the next. The returned Pod
   was deleted before inference. Settled charge:
   `0.003787499999998527` A100-hours, about `$0.00602`.

Total billed capacity-failure time is `0.013570868301884617` A100-hours. The B4
budget baseline used before these attempts was 0.5 A100-hours for preparation, so
fresh budget evidence should use at least `0.5135708683018846` prior extension
A100-hours. Refresh the balance and storage spend rather than copying the old
`spend_used_usd` value.

The final two watchdog runs were `35055189703` and `35055401994`. Both were armed
before Pod creation, then cancelled only after local deletion verified zero Pods.
Evidence is under the v2 launch directory and commit `606b616`.

## Recovery behavior

Once scientific calls begin, the worker writes accepted chunk receipts atomically
to the attached network volume before returning them. The local runner also
absorbs receipts into append-only event files. A disconnected session can resume
completed chunks without repeating accepted calls. A request that was in flight
at the exact interruption point may run again, but receipt identity and ingestion
deduplication prevent it from becoming two accepted scientific records.

The failed capacity attempts have no scientific progress to recover. Keep all
timestamped launch directories, reconciliation records, seals, and locks.

## Next action

1. Read `study-120h/extension-b4/runpod/OPERATIONS.md`, this handoff, and the v2
   reconciliation files.
2. Verify live state before writing anything: no Pods, four volumes, account ID,
   balance, `$0.016/hour` storage spend, A100 price, and per-datacenter stock.
3. Do not retry while Runpod can allocate only one A100. The current runner tears
   down the whole reference round if all four lanes cannot start.
4. When four A100s appear feasible in `EUR-IS-1`, create a new timestamped output
   directory, regenerate the deterministic plan, bind fresh preflight and budget
   evidence, run the CPU-only consistency audit, then resume the authorized
   initial A11 run. A fresh authorization is not needed for the unchanged initial
   run, but any protocol, accelerator, cloud type, or spending-scope change does
   need Daniel's decision.
5. Confirm the remote watchdog step is active before any Pod appears. Verify every
   returned Pod's image, one A100 SXM 80GB, `$1.59/hour` rate, data center, and
   attached cell-specific volume. Monitor until scientific receipts appear.

Moving caches is not currently recommended. Four replacement 40GB volumes would
add about `$0.016/hour`, or roughly `$11.20/month`, while retained, plus small CPU
staging costs. Other A100 regions also showed low or no stock, and network-volume
support must be verified before any move. Never delete the current volumes until
replacement caches pass exact-revision and offline smoke checks.

Do not rerun completed preparation, edit sealed protocols, overwrite timestamped
data, or present the Gate A pilot as fresh confirmation. Preserve the unrelated
untracked `.flash/` directory and the empty untracked Qwen
`local-watchdog.log`. The Runpod key previously pasted in chat should be rotated.
Do not invoke Claude CLI unless Daniel explicitly requests it.
