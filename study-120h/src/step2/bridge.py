"""Plan, guard, resume, and time the historical Qwen3-8B/Civil Comments Step 2 bridge.

The sealed Step 0 freeze planned 14,720 calls for the `qwen_civil` cell. Step 1 already
produced its 2,048 solo first responses, so the bridge owns the remaining 12,672 joint,
exact-context, and filler calls. Every design field is read from the seal and none is
recomputed here: the prompt, the assignment, the call ids, the execution blocks, and the
within-block order all come from `freeze/20260912T041633Z`.

This module decides nothing scientific. It refuses to start on a changed seal, a wrong
model, or a wrong serving setting; it rebuilds the pending work from the append-only
ledger; and it records the real timing the rest of the study needs to reprice its queue.
The 150-hour meter is shared study-wide, so its rules come from `freezeb.meter`. Nothing
else is shared with Freeze B.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from freezeb.meter import FORECAST_MULTIPLIER, STOP_HOURS, may_start, must_stop
from step0.hashing import read_jsonl, sha256_file, sha256_json
from step1.frozen import Frozen
from step1.ledger import rows_by_call, validate
from step1.plan import job_rows, retry_call_id, retry_job, solo_jobs

SCHEMA_VERSION = "step2-bridge-plan-v1"

#: The one cell this bridge finishes, and the pins it must find in the seal.
CELL_ID = "qwen_civil"
FRAME_TASK_ID = "civil_comments"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
FREEZE_STAMP = "20260912T041633Z"
PROTOCOL_CONTENT_HASH = "218c5bcaa30f4f93c8d516f85f29ad8b7fb3be90f40238c2f3faf837ce611e15"
ENGINE_VERSION = "0.28.0"

#: Conditions the bridge executes. `solo` is reused, never rerun. `fake_batch` creates no call.
BRIDGE_CONDITIONS: tuple[str, ...] = ("joint", "exact_context", "filler")
SEALED_CALL_COUNTS: dict[str, int] = {
    "joint": 384,
    "exact_context": 6144,
    "filler": 6144,
    "solo": 2048,
}
FRESH_CALLS = 12672
SOLO_CALLS = 2048
N_EXECUTION_BLOCKS = 8

EXPECTED_SERVING: dict[str, Any] = {
    "dtype": "bfloat16",
    "engine": "vllm",
    "engine_version_pin": "vllm==0.28.0",
    "gpu_memory_utilization": 0.9,
    "max_model_len": 8192,
    "max_num_seqs": 64,
    "tensor_parallel_size": 1,
}
EXPECTED_DECODING: dict[str, Any] = {
    "max_tokens_base": 64,
    "max_tokens_per_label": 20,
    "rule": "greedy",
    "seed": 0,
    "temperature": 0.0,
    "top_k": -1,
    "top_p": 1.0,
}

#: The frozen retry policy: one retry, for an infrastructure error or a truncation only.
MAX_RETRIES = 1
RETRY_STATUSES = frozenset({"missing", "truncated"})

#: The mandatory technical checkpoint. It counts every real call, retries included.
FIRST_REFORECAST_CALLS = 1000

#: Daniel authorised two concurrent A100 workers. One worker owns one whole execution block.
MAX_CONCURRENT_WORKERS = 2

#: The Freeze B model volume already holds the pinned checkpoint in the selected workspace.
MODEL_VOLUME = "ieee-bigdata-freezeb-models"
CHECKPOINT_MANIFESTS = Path(
    "runtime-lock/20260913T004054Z-BRT1/benchmark/checkpoint-manifests.json"
)

#: Read-only inputs outside this package.
RUNTIME_LOCK = Path("runtime-lock/20260913T004054Z-BRT1/LEDGER.json")
RUNTIME_LOCK_BLOCK_ID = "e1::qwen3-8b::civil_comments"
PRIOR_METER_STATE = Path("freezeb-screen/20260913-freezeb-screen/meter-state.json")
SOLO_LEDGER = Path("step1-out/results/qwen_civil__civil_comments__Qwen_Qwen3-8B/solo.jsonl")
PREFLIGHT_REPORT = Path("step1-out/preflight/qwen_civil__civil_comments__Qwen_Qwen3-8B.json")


class BridgeGuardError(RuntimeError):
    """Raised when a guard refuses to let the bridge run."""


# ------------------------------------------------------------------------------------
# Guards
# ------------------------------------------------------------------------------------


def verify_freeze(frozen: Frozen) -> dict[str, Any]:
    """Refuse a changed seal, a changed model pin, or a changed serving setting."""
    problems: list[str] = []
    if frozen.stamp != FREEZE_STAMP:
        problems.append(f"freeze stamp is {frozen.stamp}, not the sealed {FREEZE_STAMP}")
    if frozen.content_hash != PROTOCOL_CONTENT_HASH:
        problems.append("protocol content hash differs from the sealed Step 0 protocol")
    cell = frozen.cell(CELL_ID)
    if str(cell["model_id"]) != MODEL_ID:
        problems.append(f"{CELL_ID} is pinned to {cell['model_id']}, not {MODEL_ID}")
    if str(cell["task_id"]) != FRAME_TASK_ID:
        problems.append(f"{CELL_ID} is pinned to task {cell['task_id']}, not {FRAME_TASK_ID}")
    if frozen.model_revision(MODEL_ID) != MODEL_REVISION:
        problems.append(f"{MODEL_ID} revision differs from the sealed pin")
    serving = frozen.serving
    for key, expected in EXPECTED_SERVING.items():
        if serving.get(key) != expected:
            problems.append(f"serving.{key} is {serving.get(key)!r}, not {expected!r}")
    decoding = frozen.decoding
    for key, expected in EXPECTED_DECODING.items():
        if decoding.get(key) != expected:
            problems.append(f"decoding.{key} is {decoding.get(key)!r}, not {expected!r}")
    if problems:
        raise BridgeGuardError("the bridge refuses this freeze: " + "; ".join(problems))
    return {
        "freeze_stamp": frozen.stamp,
        "protocol_content_hash": frozen.content_hash,
        "freeze_dir": str(frozen.freeze_dir),
        "seal_verified": True,
        "cell_id": CELL_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "task_id": FRAME_TASK_ID,
        "serving": serving,
        "decoding": decoding,
    }


def sealed_meta(frozen: Frozen) -> dict[str, Any]:
    """The sealed assignment metadata of this cell."""
    path = frozen.freeze_dir / "assignments" / CELL_ID / "meta.json"
    if not path.exists():
        raise BridgeGuardError(f"{path} is missing from the freeze")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def sealed_condition_counts(frozen: Frozen) -> dict[str, int]:
    """Count the sealed skeletons by condition and refuse an unexpected condition."""
    counts: dict[str, int] = {}
    for row in frozen.job_skeletons(CELL_ID):
        key = str(row["condition"])
        counts[key] = counts.get(key, 0) + 1
    if counts != SEALED_CALL_COUNTS:
        raise BridgeGuardError(
            f"sealed condition counts are {counts}, not the expected {SEALED_CALL_COUNTS}"
        )
    return counts


def serving_controls(frozen: Frozen) -> dict[str, Any]:
    """Report whether the frozen serial-serving control creates a call for this cell."""
    serial = dict(frozen.protocol["conditions"]["serial_serving"])
    cells = dict(serial.get("cells", {}))
    applies = CELL_ID in cells
    return {
        "control": "serial_serving",
        "applies_to_this_cell": applies,
        "cells_named_by_the_freeze": sorted(cells),
        "planned_calls_for_this_cell": dict(cells.get(CELL_ID, {})),
        "engine_setting": str(serial["engine_setting"]),
        "note": (
            "the freeze gives the serial-serving control to mistral_secondtask only, so it "
            "creates no call in this bridge and max_num_seqs stays at the frozen value"
        ),
    }


def verify_preflight(report: dict[str, Any]) -> dict[str, Any]:
    """Refuse a stored preflight that is not the exact-tokenizer pass for this cell."""
    problems: list[str] = []
    if not report.get("ok") or not report.get("exact_tokenizer"):
        problems.append("the stored preflight did not pass with an exact pinned tokenizer")
    if str(report.get("cell_id")) != CELL_ID:
        problems.append(f"preflight cell is {report.get('cell_id')}")
    if str(report.get("model_id")) != MODEL_ID:
        problems.append(f"preflight model is {report.get('model_id')}")
    if str(report.get("frame_task_id")) != FRAME_TASK_ID:
        problems.append(f"preflight frame is {report.get('frame_task_id')}")
    if int(report.get("n_over_budget", 1)) != 0:
        problems.append(f"{report.get('n_over_budget')} prompts are over the frozen budget")
    if int(report.get("n_calls_checked", 0)) != sum(SEALED_CALL_COUNTS.values()):
        problems.append(f"preflight checked {report.get('n_calls_checked')} prompts, not 14720")
    by_condition = dict(report.get("by_condition", {}))
    for condition in BRIDGE_CONDITIONS:
        counted = int(by_condition.get(condition, {}).get("n_calls", 0))
        if counted != SEALED_CALL_COUNTS[condition]:
            problems.append(f"preflight counted {counted} {condition} prompts")
    if problems:
        raise BridgeGuardError("the stored preflight is unusable: " + "; ".join(problems))
    return {
        "source": "step1_exact_tokenizer_preflight",
        "checked_at_utc": report.get("checked_at_utc"),
        "counter": report.get("counter"),
        "n_calls_checked": report.get("n_calls_checked"),
        "n_over_budget": report.get("n_over_budget"),
        "max_prompt_tokens_by_condition": {
            condition: by_condition[condition]["max"] for condition in BRIDGE_CONDITIONS
        },
    }


def verify_worker(host: dict[str, Any] | None, backend: dict[str, Any] | None) -> None:
    """Refuse output that a wrong model, engine, or GPU produced."""
    if not isinstance(host, dict) or not isinstance(backend, dict):
        raise BridgeGuardError("a successful chunk carries no worker provenance")
    expected = {
        "backend": "vllm",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": EXPECTED_SERVING["dtype"],
        "max_model_len": EXPECTED_SERVING["max_model_len"],
        "max_num_seqs": EXPECTED_SERVING["max_num_seqs"],
        "tensor_parallel_size": EXPECTED_SERVING["tensor_parallel_size"],
        "seed": EXPECTED_DECODING["seed"],
    }
    mismatched = [key for key, value in expected.items() if backend.get(key) != value]
    if mismatched:
        raise BridgeGuardError(f"worker backend differs from the freeze: {sorted(mismatched)}")
    if str(host.get("installed_engine_version_string")) != ENGINE_VERSION:
        raise BridgeGuardError(
            f"worker runs vLLM {host.get('installed_engine_version_string')}, not {ENGINE_VERSION}"
        )
    gpu_name = str(host.get("gpu_name") or "")
    if "A100" not in gpu_name.upper():
        raise BridgeGuardError(f"worker is not an A100: {gpu_name or 'missing GPU name'}")


def verify_profile(requested: str, environment: str | None) -> str:
    """Refuse a paid run unless the named Modal profile is the active one."""
    named = requested.strip()
    active = (environment or "").strip()
    if not named:
        raise BridgeGuardError("a paid bridge run requires an explicit --modal-profile")
    if not active:
        raise BridgeGuardError("a paid bridge run requires MODAL_PROFILE in the environment")
    if named != active:
        raise BridgeGuardError(f"MODAL_PROFILE is {active!r} but --modal-profile names {named!r}")
    return named


def verify_checkpoint(actual: dict[str, Any], manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the materialised checkpoint against the sealed runtime-lock manifest.

    The runtime lock recorded the revision, the file list, the byte size, and the SHA256 of
    every file of the pinned Qwen3-8B snapshot. That is the strongest identity check the
    existing evidence permits, and it is applied before any call.
    """
    expected = next(
        (row for row in manifests if str(row.get("model_id")) == MODEL_ID), None
    )
    if expected is None:
        raise BridgeGuardError(f"the runtime lock records no checkpoint manifest for {MODEL_ID}")
    if str(expected["revision"]) != MODEL_REVISION:
        raise BridgeGuardError("the runtime-lock manifest pins a different model revision")
    if str(actual.get("model_id")) != MODEL_ID or str(actual.get("revision")) != MODEL_REVISION:
        raise BridgeGuardError("the materialised checkpoint is not the pinned model revision")
    checked = list(actual.get("files", []))
    wanted = {str(row["path"]): row for row in expected["files"]}
    seen = {str(row["path"]): row for row in checked}
    missing = sorted(set(wanted) - set(seen))
    if missing:
        raise BridgeGuardError(f"the volume is missing {len(missing)} pinned files: {missing[:3]}")
    hashed = 0
    for path, row in sorted(wanted.items()):
        found = seen[path]
        if int(found["bytes"]) != int(row["bytes"]):
            raise BridgeGuardError(f"{path} differs in size from the runtime-lock manifest")
        if found.get("sha256") is None:
            continue
        if str(found["sha256"]) != str(row["sha256"]):
            raise BridgeGuardError(f"{path} differs in content from the runtime-lock manifest")
        hashed += 1
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "volume": MODEL_VOLUME,
        "n_files": len(wanted),
        "n_files_hash_verified": hashed,
        "n_files_size_verified_only": len(wanted) - hashed,
        "source": "runtime-lock/20260913T004054Z-BRT1 checkpoint manifest",
    }


def verify_chunk_payload(
    payload: dict[str, Any], jobs: Sequence[dict[str, Any]], chunk_id: str
) -> None:
    """Fail closed on a returned or recovered chunk that does not match what was requested."""
    if str(payload.get("chunk_id")) != chunk_id:
        raise BridgeGuardError(f"a chunk payload carries id {payload.get('chunk_id')!r}")
    wanted = {str(job["call_id"]): {str(t) for t in job["target_ids"]} for job in jobs}
    rows = list(payload.get("rows", []))
    seen: dict[str, set[str]] = {}
    for row in rows:
        seen.setdefault(str(row["call_id"]), set()).add(str(row["doc_id"]))
    if set(seen) != set(wanted):
        raise BridgeGuardError(f"chunk {chunk_id} returned call ids that were not requested")
    for call_id, targets in wanted.items():
        if seen[call_id] != targets:
            raise BridgeGuardError(f"chunk {chunk_id} returned wrong target rows for {call_id}")
    measurements = list(payload.get("measurements", []))
    if [str(row["call_id"]) for row in measurements] != [str(job["call_id"]) for job in jobs]:
        raise BridgeGuardError(f"chunk {chunk_id} returned timing for a different call order")


def conflicts_recorded(ledger: Path) -> int:
    """Rejected conflicting outputs beside this ledger. Any at all stops a paid run."""
    path = ledger.with_suffix(ledger.suffix + ".conflicts.jsonl")
    return sum(1 for _ in read_jsonl(path)) if path.exists() else 0


def stop_now(cumulative_hours: float, *, in_flight_hours: float = 0.0) -> bool:
    """The conservative cumulative stop. Count what running workers may still bill.

    With two concurrent workers the meter can advance twice per wall-clock second, so the
    check reserves the in-flight cost of every worker before it lets the next chunk start.
    """
    if cumulative_hours < 0 or in_flight_hours < 0:
        raise ValueError("hours cannot be negative")
    return cumulative_hours + in_flight_hours >= STOP_HOURS


# ------------------------------------------------------------------------------------
# The plan
# ------------------------------------------------------------------------------------


def planned_jobs(frozen: Frozen) -> list[dict[str, Any]]:
    """The 12,672 fresh calls of this cell, in frozen block and within-block order."""
    sealed_condition_counts(frozen)
    rows = job_rows(
        frozen, CELL_ID, FRAME_TASK_ID, model_id=MODEL_ID, conditions=BRIDGE_CONDITIONS
    )
    ordered = sorted(
        rows, key=lambda row: (int(row["execution_block"]), str(row["execution_order_key"]))
    )
    if len(ordered) != FRESH_CALLS:
        raise BridgeGuardError(f"the plan holds {len(ordered)} fresh calls, not {FRESH_CALLS}")
    call_ids = [str(row["call_id"]) for row in ordered]
    if len(set(call_ids)) != len(call_ids):
        raise BridgeGuardError("the frozen plan repeats a call id")
    for row in ordered:
        if str(row["model_id"]) != MODEL_ID:
            raise BridgeGuardError(f"a planned call names model {row['model_id']}")
        if int(row["retry_index"]) != 0 or row["parent_call_id"] is not None:
            raise BridgeGuardError("a planned call is not a first response")
    meta = sealed_meta(frozen)
    expected_blocks = meta["planned_calls_by_condition_and_execution_block"]
    observed: dict[str, dict[str, int]] = {}
    for row in ordered:
        per_condition = observed.setdefault(str(row["condition"]), {})
        block = str(int(row["execution_block"]))
        per_condition[block] = per_condition.get(block, 0) + 1
    for condition in BRIDGE_CONDITIONS:
        if observed.get(condition) != {
            key: int(value) for key, value in expected_blocks[condition].items()
        }:
            raise BridgeGuardError(f"{condition} block counts differ from the sealed metadata")
    return ordered


def blocks_of(jobs: Sequence[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    """Group calls into the frozen execution blocks, in block order."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault(int(job["execution_block"]), []).append(dict(job))
    return [(block, grouped[block]) for block in sorted(grouped)]


def verify_solo_reuse(frozen: Frozen, solo_ledger: Path) -> dict[str, Any]:
    """Prove the 2,048 Step 1 solo rows are complete, so the bridge never reruns them."""
    expected = solo_jobs(frozen, CELL_ID, FRAME_TASK_ID, model_id=MODEL_ID)
    if len(expected) != SOLO_CALLS:
        raise BridgeGuardError(f"the freeze plans {len(expected)} solo calls, not {SOLO_CALLS}")
    report = validate(solo_ledger, expected, rows_per_call=1)
    if not report["ok"]:
        raise BridgeGuardError(
            f"the Step 1 solo ledger cannot be reused: {report['problems']}"
        )
    rows = list(read_jsonl(solo_ledger))
    offenders = sorted(
        {
            f"{row.get('model_id')}/{row.get('condition')}/{row.get('cell_id')}"
            for row in rows
            if str(row.get("model_id")) != MODEL_ID
            or str(row.get("condition")) != "solo"
            or str(row.get("cell_id")) != CELL_ID
        }
    )
    if offenders:
        raise BridgeGuardError(f"the Step 1 solo ledger holds foreign rows: {offenders}")
    return {
        "reused": True,
        "rerun": False,
        "path": str(solo_ledger),
        "n_calls": int(report["n_calls"]),
        "n_rows": int(report["n_rows"]),
        "ledger_sha256": sha256_file(solo_ledger),
        "call_ids_sha256": sha256_json(sorted(str(job["call_id"]) for job in expected)),
        "note": "Step 1 produced these first responses; the bridge reads them and adds none",
    }


def runtime_forecast(lock: dict[str, Any], *, n_serving_sessions: int) -> dict[str, Any]:
    """The sealed synthetic forecast, raised by the extra model loads the freeze requires."""
    block = next(
        (row for row in lock["blocks"] if str(row.get("block_id")) == RUNTIME_LOCK_BLOCK_ID), None
    )
    if block is None or block.get("status") != "forecastable":
        raise BridgeGuardError(f"the runtime lock cannot forecast {RUNTIME_LOCK_BLOCK_ID}")
    base = float(block["base_a100_hours"])
    load = float(block["load_a100_hours"])
    extra_loads = max(0, int(n_serving_sessions) - 1)
    adjusted = base + load * extra_loads
    return {
        "block_id": RUNTIME_LOCK_BLOCK_ID,
        "sealed_base_a100_hours": base,
        "single_load_a100_hours": load,
        "extra_serving_session_loads": extra_loads,
        "adjusted_base_a100_hours": adjusted,
        "adjusted_guarded_a100_hours": adjusted * FORECAST_MULTIPLIER,
        "fresh_calls": int(block["fresh_calls"]),
        "synthetic_only": True,
        "note": (
            "the freeze requires a fresh serving session for every execution block, so the "
            "sealed one-load forecast is raised by one model load per extra block"
        ),
    }


def plan(
    frozen: Frozen,
    *,
    solo_ledger: Path,
    preflight_report: dict[str, Any],
    runtime_lock: dict[str, Any],
    chunk_size: int,
    max_concurrent_workers: int = MAX_CONCURRENT_WORKERS,
) -> dict[str, Any]:
    """The complete execution plan. It generates nothing.

    `plan_sha256` covers the frozen scientific content only. Worker count and chunk size are
    execution settings: they are reported and guarded separately, and they never change the
    plan identity, because they change no call, no prompt, and no order.
    """
    if not 1 <= chunk_size <= 64:
        raise BridgeGuardError("chunk size must be 1 through 64")
    if not 1 <= max_concurrent_workers <= MAX_CONCURRENT_WORKERS:
        raise BridgeGuardError(f"workers must be 1 through {MAX_CONCURRENT_WORKERS}")
    jobs = planned_jobs(frozen)
    by_condition: dict[str, int] = {}
    for job in jobs:
        by_condition[str(job["condition"])] = by_condition.get(str(job["condition"]), 0) + 1
    core = {
        "schema_version": SCHEMA_VERSION,
        "stage": "historical_replication_bridge",
        "separate_from_freeze_b": True,
        "freeze": verify_freeze(frozen),
        "sealed_condition_counts": sealed_condition_counts(frozen),
        "fresh_calls_total": len(jobs),
        "fresh_calls_by_condition": dict(sorted(by_condition.items())),
        "fresh_call_ids_sha256": sha256_json([str(job["call_id"]) for job in jobs]),
        "execution_blocks": [block for block, _ in blocks_of(jobs)],
        "fresh_serving_session_per_block": bool(
            frozen.protocol["conditions"]["interleaving"]["fresh_serving_session_per_block"]
        ),
        "within_block_order": "ascending execution_order_key",
        "solo_reuse": verify_solo_reuse(frozen, solo_ledger),
        "serving_controls": serving_controls(frozen),
        "fake_batch": {
            "creates_calls": False,
            "note": str(frozen.protocol["conditions"]["fake_batch"]["definition"]),
        },
        "preflight": verify_preflight(preflight_report),
        "retry_policy": {"max_retries": MAX_RETRIES, "triggers": sorted(RETRY_STATUSES)},
        "forecast": runtime_forecast(runtime_lock, n_serving_sessions=N_EXECUTION_BLOCKS),
        "stop_a100_hours": STOP_HOURS,
    }
    plan_sha256 = sha256_json(core)
    return {
        **core,
        "plan_sha256": plan_sha256,
        "execution": {
            "chunk_size": chunk_size,
            "max_concurrent_workers": max_concurrent_workers,
            "blocks_are_never_split_across_workers": True,
            "results_namespace": plan_sha256[:16],
            "first_reforecast_barrier_calls": FIRST_REFORECAST_CALLS,
            "deviation_from_sealed_serving_hardware": max_concurrent_workers > 1,
            "sealed_serving_hardware": str(frozen.serving["hardware"]),
            "note": (
                "execution settings are excluded from plan_sha256 because they change no call, "
                "no prompt, and no order. Concurrency above one worker is recorded in the "
                "step2-bridge-concurrency amendment before any output exists."
            ),
        },
    }


# ------------------------------------------------------------------------------------
# Resume
# ------------------------------------------------------------------------------------


def stored_rows(ledger: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(ledger)) if ledger.exists() else []


def resume_state(
    planned: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Describe what the append-only ledger already holds. It never edits a stored row."""
    planned_ids = {str(job["call_id"]) for job in planned}
    targets_by_call = {str(job["call_id"]): {str(t) for t in job["target_ids"]} for job in planned}
    first_by_call: dict[str, dict[str, Any]] = {}
    stored_targets: dict[str, set[str]] = {}
    retries_by_parent: dict[str, set[str]] = {}
    foreign: list[str] = []
    for row in rows:
        call_id = str(row["call_id"])
        if str(row.get("model_id")) != MODEL_ID or str(row.get("cell_id")) != CELL_ID:
            foreign.append(call_id)
            continue
        if str(row.get("condition")) not in BRIDGE_CONDITIONS:
            foreign.append(call_id)
            continue
        if int(row["retry_index"]) == 0:
            if call_id not in planned_ids:
                foreign.append(call_id)
                continue
            first_by_call[call_id] = dict(row)
            stored_targets.setdefault(call_id, set()).add(str(row["doc_id"]))
        else:
            parent = str(row["parent_call_id"])
            expected_retry = retry_call_id(parent, int(row["retry_index"]))
            if parent not in planned_ids or call_id != expected_retry:
                foreign.append(call_id)
                continue
            retries_by_parent.setdefault(parent, set()).add(call_id)
    if foreign:
        sample = sorted(set(foreign))[:5]
        raise BridgeGuardError(
            f"{len(foreign)} stored rows do not belong to this bridge plan: {sample}"
        )
    incomplete = sorted(
        call_id
        for call_id, seen in stored_targets.items()
        if seen != targets_by_call[call_id]
    )
    statuses: dict[str, int] = {}
    for call_id, row in first_by_call.items():
        if call_id in incomplete:
            continue
        key = str(row["first_response_status"])
        statuses[key] = statuses.get(key, 0) + 1
    complete = {call_id for call_id in first_by_call if call_id not in set(incomplete)}
    return {
        "n_planned": len(planned_ids),
        "n_first_responses_stored": len(complete),
        "n_incomplete_first_responses": len(incomplete),
        "incomplete_call_ids_sample": incomplete[:10],
        "n_missing_first_responses": len(planned_ids) - len(complete),
        "n_retries_stored": sum(len(value) for value in retries_by_parent.values()),
        "first_response_status_counts": dict(sorted(statuses.items())),
        "first_by_call": first_by_call,
        "complete_call_ids": complete,
        "retries_by_parent": {key: sorted(value) for key, value in retries_by_parent.items()},
    }


def pending_work(
    planned: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Split the remaining work into pending first responses and permitted retries.

    A complete stored first response is never repeated and never replaced. A call whose
    requested labels are not all stored is not a first response yet, so it is offered again:
    the append then completes it from the identical output or rejects a different one.

    A truncated or missing first response earns exactly one linked retry with a new call id.
    Retries are listed after the pending first responses, so a retry can never take the place
    of a planned call that has not run yet.
    """
    state = resume_state(planned, rows)
    first_by_call = state["first_by_call"]
    complete: set[str] = state["complete_call_ids"]
    retries_by_parent = state["retries_by_parent"]
    pending: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []
    for job in planned:
        call_id = str(job["call_id"])
        if call_id not in complete:
            pending.append(dict(job))
            continue
        if str(first_by_call[call_id]["first_response_status"]) not in RETRY_STATUSES:
            continue
        if len(retries_by_parent.get(call_id, [])) >= MAX_RETRIES:
            continue
        retries.append(retry_job(dict(job), 1))
    return {"first_responses": pending, "retries": retries}


def retry_jobs_for(
    jobs: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One linked retry for each first response of this chunk that the seal permits retrying.

    The seal permits a retry after an infrastructure error or a truncation only. A malformed
    response is a scientific result and is never retried. A retry is itself never retried,
    because the frozen policy allows one.
    """
    status_by_call = {
        str(row["call_id"]): str(row["first_response_status"])
        for row in rows
        if int(row["retry_index"]) == 0
    }
    scheduled: list[dict[str, Any]] = []
    for job in jobs:
        if int(job["retry_index"]) != 0:
            continue
        if status_by_call.get(str(job["call_id"])) in RETRY_STATUSES:
            scheduled.append(retry_job(dict(job), 1))
    return scheduled


def completeness(planned: Sequence[dict[str, Any]], ledger: Path) -> dict[str, Any]:
    """The final completeness report. It never reports a partial run as complete.

    A joint call is complete only when every one of its 16 requested labels holds a row, so
    a call that lost target rows is counted as missing, not as done.
    """
    report = validate(ledger, [dict(job) for job in planned], rows_per_call=None)
    stored = rows_by_call(ledger)
    expected_rows = sum(len(job["target_ids"]) for job in planned)
    incomplete = sorted(
        str(job["call_id"])
        for job in planned
        if stored.get(str(job["call_id"]))
        and set(stored[str(job["call_id"])]) != {str(t) for t in job["target_ids"]}
    )
    problems = list(report["problems"])
    if incomplete:
        problems.append(f"{len(incomplete)} stored calls are missing a requested label row")
    conflicts_path = ledger.with_suffix(ledger.suffix + ".conflicts.jsonl")
    return {
        "complete": (
            bool(report["ok"])
            and int(report["n_missing_calls"]) == 0
            and not incomplete
            and not conflicts_path.exists()
        ),
        "n_expected_calls": int(report["n_expected_calls"]),
        "n_expected_label_rows": expected_rows,
        "n_first_response_calls": int(report["n_first_response_calls"]),
        "n_missing_calls": int(report["n_missing_calls"]),
        "missing_call_ids_sample": list(report["missing_call_ids_sample"]),
        "n_incomplete_calls": len(incomplete),
        "incomplete_call_ids_sample": incomplete[:10],
        "n_retry_rows": int(report["n_retry_rows"]),
        "n_rejected_conflicts": int(report["n_rejected_conflicts"]),
        "problems": problems,
    }


# ------------------------------------------------------------------------------------
# Timing checkpoint and reforecast
# ------------------------------------------------------------------------------------


def attribute_seconds(
    total_seconds: float, weights: Sequence[float | None]
) -> tuple[list[float], str]:
    """Split one chunk's metered A100 seconds over the calls of that chunk.

    The engine serves a chunk as one mixed batch, so a per-call share is an attribution,
    not a measurement. Per-request engine timing is used when vLLM exposes it for every
    call of the chunk; otherwise the share is equal and the method says so.
    """
    if not weights:
        raise ValueError("a chunk holds at least one call")
    usable = [value for value in weights if value is not None and value > 0.0]
    if len(usable) == len(weights):
        total_weight = sum(usable)
        return (
            [total_seconds * float(value or 0.0) / total_weight for value in weights],
            "per_request_engine_timing",
        )
    share = total_seconds / len(weights)
    return [share] * len(weights), "equal_share_within_chunk"


def timing_records(
    jobs: Sequence[dict[str, Any]],
    measurements: Sequence[dict[str, Any]],
    *,
    run_id: str,
    chunk_id: str,
    serving_session: str,
    execution_block: int,
    chunk_wall_seconds: float,
    metered_a100_seconds: float,
    cumulative_a100_hours: float,
) -> list[dict[str, Any]]:
    """One operational timing row per call. These rows never enter the frozen ledger."""
    if len(jobs) != len(measurements):
        raise ValueError("every call of a chunk needs one measurement")
    weights: list[float | None] = []
    for measurement in measurements:
        prefill = measurement.get("prefill_seconds")
        decode = measurement.get("decode_seconds")
        weights.append(
            float(prefill) + float(decode) if prefill is not None and decode is not None else None
        )
    shares, method = attribute_seconds(metered_a100_seconds, weights)
    recorded_at = dt.datetime.now(dt.UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for job, measurement, weight, share in zip(jobs, measurements, weights, shares, strict=True):
        rows.append(
            {
                "call_id": str(job["call_id"]),
                "parent_call_id": job["parent_call_id"],
                "retry_index": int(job["retry_index"]),
                "condition": str(job["condition"]),
                "execution_block": execution_block,
                "run_id": run_id,
                "chunk_id": chunk_id,
                "chunk_size": len(jobs),
                "serving_session": serving_session,
                "status": str(measurement.get("status", "unknown")),
                "finish_reason": measurement.get("finish_reason"),
                "request_seconds": weight,
                "queue_seconds": measurement.get("queue_seconds"),
                "prefill_seconds": measurement.get("prefill_seconds"),
                "decode_seconds": measurement.get("decode_seconds"),
                "prompt_tokens": measurement.get("prompt_tokens"),
                "output_tokens": measurement.get("output_tokens"),
                "chunk_wall_seconds": chunk_wall_seconds,
                "chunk_metered_a100_seconds": metered_a100_seconds,
                "metered_a100_seconds": share,
                "attribution_method": method,
                "cumulative_a100_hours": cumulative_a100_hours,
                "recorded_at_utc": recorded_at,
            }
        )
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _column(group: Sequence[dict[str, Any]], name: str) -> list[float]:
    return [float(row[name]) for row in group if row.get(name) is not None]


def checkpoint_by_condition(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate real-call timing by condition: wall, queue, prefill, decode, tokens, hours."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    per_condition: dict[str, Any] = {}
    total = 0.0
    for condition, group in sorted(grouped.items()):
        metered = _column(group, "metered_a100_seconds")
        total += sum(metered)
        per_condition[condition] = {
            "n_calls": len(group),
            "n_ok": sum(row.get("status") == "ok" for row in group),
            "n_with_component_timing": len(_column(group, "prefill_seconds")),
            "request_seconds_mean": _mean(_column(group, "request_seconds")),
            "queue_seconds_mean": _mean(_column(group, "queue_seconds")),
            "prefill_seconds_mean": _mean(_column(group, "prefill_seconds")),
            "decode_seconds_mean": _mean(_column(group, "decode_seconds")),
            "prompt_tokens_mean": _mean(_column(group, "prompt_tokens")),
            "output_tokens_mean": _mean(_column(group, "output_tokens")),
            "prompt_tokens_sum": sum(_column(group, "prompt_tokens")),
            "output_tokens_sum": sum(_column(group, "output_tokens")),
            "metered_a100_seconds_sum": sum(metered),
            "metered_a100_seconds_per_call": _mean(metered),
            "attribution_methods": sorted({str(row["attribution_method"]) for row in group}),
        }
    return {
        "n_calls": sum(len(group) for group in grouped.values()),
        "metered_a100_seconds_total": total,
        "metered_a100_hours_total": total / 3600.0,
        "by_condition": per_condition,
        "written_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def condition_rates(
    rows: Sequence[dict[str, Any]], *, min_chunks: int = 8
) -> dict[str, Any] | None:
    """Estimate A100 seconds per call for each condition from whole-chunk measurements.

    A chunk mixes arms, because the frozen within-block order mixes them. The measured
    quantity is therefore the chunk total, and the per-condition rate is the least-squares
    solution of chunk seconds against chunk composition. It needs no assumed constant.
    """
    by_chunk: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk = by_chunk.setdefault(
            str(row["chunk_id"]),
            {"counts": {}, "seconds": float(row["chunk_metered_a100_seconds"])},
        )
        counts = chunk["counts"]
        condition = str(row["condition"])
        counts[condition] = counts.get(condition, 0) + 1
    if len(by_chunk) < min_chunks:
        return None
    conditions = sorted({key for chunk in by_chunk.values() for key in chunk["counts"]})
    matrix = np.array(
        [[float(chunk["counts"].get(c, 0)) for c in conditions] for chunk in by_chunk.values()]
    )
    observed = np.array([float(chunk["seconds"]) for chunk in by_chunk.values()])
    solution, _residuals, rank, _singular = np.linalg.lstsq(matrix, observed, rcond=None)
    if int(rank) != len(conditions):
        return None
    rates = {c: float(value) for c, value in zip(conditions, solution, strict=True)}
    return {
        "method": "chunk_least_squares",
        "n_chunks": len(by_chunk),
        "rank": int(rank),
        "seconds_per_call": rates,
        "all_positive": all(value > 0.0 for value in rates.values()),
    }


def missing_rates(
    rows: Sequence[dict[str, Any]], remaining_by_condition: dict[str, int]
) -> list[str]:
    """Conditions that still have work planned but no measured call to price it with."""
    measured = {str(row["condition"]) for row in rows}
    return sorted(
        condition
        for condition, count in remaining_by_condition.items()
        if count > 0 and condition not in measured
    )


def reforecast(
    rows: Sequence[dict[str, Any]],
    remaining_by_condition: dict[str, int],
    *,
    cumulative_a100_hours: float,
    remaining_load_seconds: float = 0.0,
    n_concurrent_workers: int = 1,
) -> dict[str, Any]:
    """Reprice the rest of the bridge from real calls, then apply the frozen start rule.

    Rates measured under concurrent workers are the rates of that layout. They are reported
    as measured and are never rescaled to a one-worker equivalent.
    """
    if not rows:
        raise ValueError("a reforecast needs at least one measured call")
    checkpoint = checkpoint_by_condition(rows)
    least_squares = condition_rates(rows)
    if least_squares is not None and least_squares["all_positive"]:
        rates = dict(least_squares["seconds_per_call"])
        method = str(least_squares["method"])
    else:
        rates = {
            condition: float(summary["metered_a100_seconds_per_call"] or 0.0)
            for condition, summary in checkpoint["by_condition"].items()
        }
        method = "per_condition_attributed_mean"
    missing = missing_rates([{"condition": key} for key in rates], remaining_by_condition)
    if missing:
        raise ValueError(f"no measured rate yet for {missing}")
    projected = sum(rates.get(c, 0.0) * count for c, count in remaining_by_condition.items())
    projected += remaining_load_seconds
    base_hours = projected / 3600.0
    decision = may_start(cumulative_a100_hours, base_hours)
    return {
        "real_calls_used": len(rows),
        "method": method,
        "n_concurrent_workers_when_measured": n_concurrent_workers,
        "rates_are_rescaled_to_one_worker": False,
        "seconds_per_call_by_condition": dict(sorted(rates.items())),
        "least_squares": least_squares,
        "remaining_calls_by_condition": dict(sorted(remaining_by_condition.items())),
        "remaining_load_a100_seconds": remaining_load_seconds,
        "projected_remaining_base_a100_hours": base_hours,
        "projected_remaining_guarded_a100_hours": base_hours * FORECAST_MULTIPLIER,
        "cumulative_a100_hours": cumulative_a100_hours,
        "may_continue": decision.start,
        "decision_reason": decision.reason,
        "projected_total_a100_hours": decision.projected_total_hours,
        "stop_a100_hours": STOP_HOURS,
        "must_stop_now": must_stop(cumulative_a100_hours),
        "checkpoint": checkpoint,
        "technical_only": True,
        "written_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def remaining_counts(
    planned: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> dict[str, int]:
    """Planned first responses that still have no stored row, counted by condition."""
    pending = pending_work(planned, rows)["first_responses"]
    counts: dict[str, int] = {condition: 0 for condition in BRIDGE_CONDITIONS}
    for job in pending:
        counts[str(job["condition"])] += 1
    return counts
