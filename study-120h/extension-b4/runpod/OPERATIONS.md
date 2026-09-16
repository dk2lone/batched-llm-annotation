# Runpod B4 / A11

The Runpod runner uses four separate Secure Cloud A100 SXM 80GB Pods, one per model.
Each round creates only the Pods it needs and deletes each Pod when its lane finishes.
References finish for all four families before evaluation starts. The initial study remains
250 reference and 400 evaluation replicates per cell, or 83,200 first calls.

The implementation is locally tested, not remotely validated. The pinned runtime image was
published successfully; no network volumes, model caches or GPU smoke tests have been provisioned.
Do not mark preflight checks passed without evidence.

Runpod CLI 2.14.0 removed `--terminate-after`, and Runpod's backend did not enforce the old field.
The runner now requires an independently active GitHub Actions watchdog before every Pod creation.
Five hours is its shutdown target, not a guaranteed lease. Normal runner deletion remains the
first cleanup path. Auto-pay must be disabled and prepaid credit must not exceed the authorized
remaining budget; that financial ceiling is the backstop if both deletion paths fail.

## Local preparation

From `/Users/dz/bigdata/study-120h`:

```sh
.venv/bin/python runpod_extension_b4.py --output extension-b4/runpod-initial
```

This verifies source seals, private frames and the frozen nested schedules, then writes an
immutable Runpod-specific `PLAN-400.json`. It does not contact Runpod or import GPU libraries.
Success prints `first_calls: 83200` and `gpu_work_launched: false`. Changed seals or an existing
plan with different code hashes stop preparation. Use a new directory after changing code.

## Before separately authorized paid testing

Use the official Runpod PyTorch base specified in `Dockerfile`, build for `linux/amd64`, and
publish a derived image containing the pinned Python 3.11/vLLM environment. Record its immutable
image digest. The runner uses `/opt/b4/bin/python`; the base image alone is not sufficient.
The hosted build is tracked in GitHub Actions as `build b4 runtime`. The template and its image
were checked live on 2026-09-15. A started build is not a verified published runtime.

Prepare four distinct network volumes in the same data center, one per selected cell. Cache
each exact sealed model/tokenizer revision under `/workspace/hf` using the HF hub cache layout.
The inference worker is offline and never downloads a model. Gated model access must already be
approved. Do not embed Runpod, Hugging Face or registry tokens in the image or JSON evidence.
Volumes survive Pod termination and keep billing until an operator removes them.

Verify CLI authentication, registered SSH keys, actual prepaid balance, current on-demand
price and capacity for four separate one-GPU Pods. A four-GPU Pod availability query alone
does not establish four separate Pod allocations. The installed CLI used for preparation was
runpodctl 2.11.0; recheck live CLI help and smoke-test response shapes before execution.

Run a separately authorized small GPU smoke test using synthetic text, never the selected
scientific schedules. Check each model's pinned backend, BF16, greedy decoding, batch invariance,
80GB GPU, raw response persistence, recovery and Pod termination. Charge that paid work to the
extension budget. Save content-addressed evidence tied to this implementation and image.

A secret-free configuration has exactly these fields:

```json
{
  "schema_version": "b4-runpod-config-v2",
  "account_id": "VERIFIED_RUNPOD_ACCOUNT_ID",
  "cloud_type": "SECURE",
  "gpu_type_id": "NVIDIA A100-SXM4-80GB",
  "image": "REGISTRY/IMAGE@sha256:VERIFIED_IMAGE_DIGEST",
  "data_center_id": "VERIFIED_DATA_CENTER",
  "network_volumes": {
    "ibm-granite/granite-3.3-8b-instruct::imdb": "VERIFIED_VOLUME_ID_1",
    "meta-llama/Llama-3.1-8B-Instruct::civil_comments": "VERIFIED_VOLUME_ID_2",
    "mistralai/Mistral-7B-Instruct-v0.3::multi_nli": "VERIFIED_VOLUME_ID_3",
    "Qwen/Qwen3-8B::civil_comments": "VERIFIED_VOLUME_ID_4"
  },
  "watchdog": {
    "repository": "dk2lone/batched-llm-annotation",
    "workflow": "b4-pod-watchdog.yml",
    "ref": "main",
    "max_target_seconds": 18000
  }
}
```

The preflight JSON must contain `runpod_config_sha256`, the plan's `implementation_sha256` and
`serving_policy`, and a `checks` object. Its required true checks are `runtime_smoke_passed`,
`four_single_gpu_capacity_verified`, `model_caches_verified`, `balance_verified`, `ssh_verified`,
`watchdog_cpu_test_passed`, and `auto_pay_disabled_verified`.
Include the underlying sanitized receipts/catalog/account evidence, not just unsupported flags.

The standard B4 budget evidence described in `../../EXTENSION_B4_OPERATIONS.md` still applies.
For Runpod, its `profile` is the verified account ID. Add `runpod_config_sha256`, a finite
nonnegative `runpod_storage_reserve_usd`, and `runpod_preflight` containing those seven true checks
plus `evidence_sha256`, the SHA-256 of the exact preflight JSON file. Storage reserve is withheld
from spend capacity. The GPU hourly rate must be a verified upper bound on the allocated rate.
Forecasts include provisioning, model loading, recovery, generation and resume startup costs.

Use `prepare_extension_b4_budget.py` with the Runpod plan and account ID to hash the local budget
evidence. It does not verify an account, perform a smoke test or authorize GPU work. Fresh budget
verification expires after one hour. No valid paid-execution configuration/evidence was prepared
by this local implementation step.

## Full execution, only after separate authorization

```sh
.venv/bin/python runpod_extension_b4.py --output extension-b4/runpod-initial --execute --profile VERIFIED_RUNPOD_ACCOUNT_ID --budget-record budget-record.json --runpod-config runpod-config.json --preflight-evidence preflight.json
```

This command can start paid Pods. All four guarded lane forecasts plus 60 seconds per lane must
fit both the remaining 80-A100-hour ceiling with its 10-hour reserve and verified spend capacity.
The 1.6 forecast multiplier is unchanged. No fallback to Community Cloud, 40GB or another model
is permitted. The returned Pod rate/configuration and actual serving backend are checked before
scientific generation. A changed or unverifiable returned price triggers teardown, not continuation.

The worker serves only on loopback, reached through validated SSH details. SSH is noninteractive
and retains accepted host keys. Each cell uses its own volume and durable atomic raw receipts;
recovery does not substitute one response for a different scientific call. Runtime code is
uploaded without private source frames or credentials. Only rendered selected prompts are sent.

Before a round, the runner dispatches `.github/workflows/b4-pod-watchdog.yml` and waits until its
deletion step is actively running. Pods use the watchdog's unique lease prefix. Normal/error
teardown deletes only the exact ID returned by that lane's create call and verifies `not_found`;
only then may the runner cancel the watchdog. Network volumes are never deleted automatically.
If the local process disappears, the independent workflow deletes matching Pods at the target.
The prepaid balance remains the financial ceiling because GitHub is not guaranteed to execute.

GPU-hour accounting sums each lane's wall time, including provisioning and verified termination,
plus the existing conservative 60-second settlement allowance. Uncertain creation, cancellation,
termination, or deadline overruns leave the whole round unsettled. Verify no Pods remain active
and reconcile every phase against actual billing before resuming. Failed rounds never launch
later evaluation prefixes. Persistent-volume storage must also be checked against its reserve.

Analysis and billing reconciliation use the existing CPU-only B4 commands. The analyzer rebuilds
the Runpod-specific plan and verifies 80GB serving evidence. Existing Modal preparations remain
preserved but cannot be used to execute the updated Runpod code.

## Local checks

```sh
.venv/bin/pytest tests/test_extension_b4.py tests/test_extension_b4_runpod.py
```

These checks use fake Pods and model responses. Success verifies local scheduling, evidence
gates, receipt handling and cleanup logic. It does not establish remote compatibility or results.
