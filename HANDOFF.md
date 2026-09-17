# IEEE BigData 2026 handoff

Updated: 2026-09-17 03:25:54 UTC during the active one-A100 B4 run.

## Current state

Project: dependence in batched LLM text annotation. Repository:
`dk2lone/batched-llm-annotation`, private, branch `main`. The branch matches
`origin/main`; the latest commit is `8e2df6e` (`reject incompatible runpod drivers early`).

Daniel authorized the B4 / A11 scientific run: 250 reference and 400 evaluation
replicates for four model-task cells, or 83,200 first calls. The execution plan uses
one Secure Cloud NVIDIA A100 PCIe 80GB lane in `CA-MTL-3`. GPU concurrency changes
wall-clock time, not the frozen prompts, schedules, calls, or analysis.

The scientific run is active. Live verification at this handoff showed:

- Pod `u8due56secjnmr` is `RUNNING` in `CA-MTL-3`;
- image digest:
  `sha256:1cf3be1dfb50dbb800a658b371e8ec58f1ea485b55bae5b92ee281af2feba898`;
- one NVIDIA A100 PCIe 80GB at `$1.59/hour`;
- compatible driver `580.159.04`, PyTorch `2.13.0+cu130`;
- Granite / IMDb reference phase active;
- 14 durable receipts covering 448 events;
- all 448 recorded events have status `ok`;
- Runpod balance `$24.4061117831`;
- total current spend `$1.627/hour`, including `$0.031/hour` retained storage;
- spend limit `$80`.

These are execution-integrity facts, not scientific findings. Do not analyze or cite
partial outputs as results. The reference round must complete across all four cells
before evaluation begins.

## Active run

Output directory:

`study-120h/extension-b4/runpod-pcie-auto-20260917T030957Z/`

Important identities:

- plan SHA-256: `b8132b467997a6b8ec330a324af552dfa06708688735c7027a70089e8bdd591f`;
- budget record SHA-256: `4a87ed5d9c01a2ff347de1a14f6baef6ac839d73f2bb8507a8da8d692850010f`;
- current phase ID: `3df56fb9852b4e33a600fe28d2da90f4`;
- current remote namespace: `b4-18559bf79867ad13f4fca5ee`;
- current cell: `ibm-granite/granite-3.3-8b-instruct::imdb`;
- current stream: `reference`;
- watchdog run: `35177153106`;
- watchdog shutdown target: `2026-09-17T05:58:36.283701+00:00`.

The launcher state file is:

`/Users/dz/.local/state/runpod-pcie-b4-launch.json`

Its `attempt_status` is `running`. The top-level `status` string may still read
`waiting_for_verified_caches_capacity_and_balance`; that field is not cleared when a
live attempt starts. Use `attempt_status`, the live Pod list, lifecycle records, meter,
and receipts to determine the actual state.

Do not start a second launcher or execute `runpod_extension_b4.py --execute` manually.
Do not interrupt Pod `u8due56secjnmr` while receipts continue to advance.

## Durable progress and recovery

The remote worker persists each accepted chunk atomically before replying:

`/workspace/b4-results/b4-18559bf79867ad13f4fca5ee/imdb/`

At the handoff snapshot it contained 14 JSON receipts, 448 events, and no non-`ok`
statuses. The local runner ingests completed chunks into the timestamped output tree.
If the connection or Pod fails, a replacement worker can recover completed chunks from
the attached volume without accepting them twice. Preserve the network volume and every
timestamped run directory.

The Granite worker runs from the content-addressed runtime directory beginning with:

`/workspace/b4-runtime/d82aa08424c345ca3ebc84de7349636b96a4c309e01ef43350aa1ed94af4c21a/`

Old worker logs and smoke receipts remain on the volume under other content hashes and
namespaces. Do not count them as current progress. Derive the namespace from the active
plan, stream, and output directory before inspecting receipts.

## Fixes made before execution

Runpod first returned many allocation stubs with no machine. When real machines began
appearing, two separate startup defects were identified and fixed:

- `9a86689`: wait for delayed Runpod SSH readiness;
- `0f9cb25`: stop waiting immediately when a pre-SSH Pod exits or disappears;
- `508006b`: include the existing Runpod registry-auth resource when pulling the private
  GHCR runtime image;
- `8e2df6e`: initialize CUDA immediately after SSH and reject incompatible drivers before
  uploading code or loading a model.

The registry-auth omission caused real A100 Pods to remain at `awaiting_container`.
After that fix, two hosts exposed driver `570.172.08`, which cannot initialize the pinned
CUDA 13.0/PyTorch runtime. They were deleted before scientific calls. The successful
preparation hosts and the active host use driver `580.159.04`.

The driver gate and registry-auth change pass 71 focused Runpod/B4 tests. Strict mypy and
Ruff pass for the changed code. Failed Pods were deleted and their watchdogs released.
No failed attempt produced accepted scientific output.

## Automatic launcher and safety

LaunchAgent `com.dk2lone.runpod-pcie-b4-launch` runs:

`/usr/bin/caffeinate -i /Users/dz/.local/bin/launch-runpod-pcie-b4.py`

It checks every five minutes when idle. It requires verified caches, the expected account,
at least `$8` balance, no existing running Pod, acceptable stock and price, and all four
volumes. It creates fresh immutable plan, preflight, configuration, and budget records for
each attempt. A file lock prevents overlapping automatic checks.

The GitHub watchdog is a remote deletion fallback. The local runner remains responsible
for normal phase cleanup. If an incompatible host reaches SSH, the CUDA gate fails and the
runner deletes that exact Pod before retrying.

## Runtime and budget

The one-GPU plan forecasts exactly 15 A100-hours of scientific work. Expected wall time is
about 15 hours; the 1.6 guard permits 24 hours. At `$1.59/hour`, the base GPU forecast is
`$23.85`, excluding storage and failed startup attempts. The study-wide extension ceiling
remains 80 A100-hours with a 10-hour reserve.

The balance of `$24.4061117831` is too close to the base forecast to safely cover the run.
Top up to approximately `$40` as soon as possible. Auto-pay is disabled. Do not raise or
change the study's `$80` spending ceiling without Daniel's decision.

## Scientific execution lock

The four frozen cells are:

- `ibm-granite/granite-3.3-8b-instruct::imdb`
- `meta-llama/Llama-3.1-8B-Instruct::civil_comments`
- `mistralai/Mistral-7B-Instruct-v0.3::multi_nli`
- `Qwen/Qwen3-8B::civil_comments`

All four reference streams must finish before evaluation begins. Preserve the sealed
prompts, schedules, model revisions, exact decoding, reference/evaluation counts, round
barriers, 80-A100-hour ceiling, 10-hour reserve, and 1.6 forecast guard.

The PCIe/data-center change is documented in:

`study-120h/amendments/20260916T213900Z-b4-a100-pcie-fallback.md`

The current generated plan narrows actual execution to one lane, as Daniel authorized
after two-GPU capacity remained unavailable.

## Verified preparation

Runtime image:

`ghcr.io/dk2lone/b4-a100-runtime@sha256:1cf3be1dfb50dbb800a658b371e8ec58f1ea485b55bae5b92ee281af2feba898`

Exact offline caches:

| Cell | Volume ID | Model revision |
|---|---|---|
| Granite / IMDb | `nm92sgw9zy` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |
| Llama / Civil Comments | `871iih1km4` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Mistral / MultiNLI | `16nchs8xkt` | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| Qwen / Civil Comments | `pxk7r2wj35` | `b968826d9c46dd6066d109eabc6255188de91218` |

`/Users/dz/.local/state/runpod-pcie-b4.json` reports
`all_caches_and_smokes_ready`. Every exact revision passed the supervised synthetic A100
PCIe smoke using driver `580.159.04`. Do not rerun cache preparation unless live evidence
shows a cache or volume is invalid.

## Next action

1. Top up the Runpod balance to approximately `$40` without changing the `$80` spending
   ceiling.
2. Verify Pod `u8due56secjnmr` remains `RUNNING`, watchdog `35177153106` remains active,
   and current-namespace receipt counts continue to increase.
3. Monitor `meter.jsonl`, lifecycle records, local receipts, and remote durable receipts.
   Report stalls, non-`ok` events, worker exits, cleanup failures, or budget pressure.
4. Let the runner advance through all reference cells and then the equal-family evaluation
   prefixes. Do not manually launch another lane.
5. Only analyze results after the planned phase completes and the integrity checks pass.

Keep all timestamped run data, lifecycle logs, meter records, seals, locks, and caches.
Do not edit sealed protocols, overwrite run data, rerun completed preparation, or present
the Gate A pilot as fresh confirmation. Do not expose credentials. The previously exposed
Modal token and Runpod key should be rotated. Do not invoke Claude CLI unless Daniel
explicitly requests it.
