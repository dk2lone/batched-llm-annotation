"""X1R4 M1 solo-screen execution rules: seal binding, resume, quarantine, metering.

Every rule that decides what may run, what may be recorded, and what must stop
lives here on the CPU side. The Modal runner only moves data through it, so the
whole gate is testable without a GPU, a Modal account, or a model weight.

Nothing in this module loads a model, starts a container, or reads a credential.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from step0.hashing import canonical_json, read_jsonl, sha256_file, sha256_json

from . import extension as ext
from .admission import evaluate
from .frames import N_SCREEN
from .labels import normalize_label
from .prompts import build_messages
from .runtime_benchmark import RuntimePolicy
from .screen import MAX_TECHNICAL_ATTEMPTS, parse_event
from .screen import pending_calls as _screen_pending_calls

SCHEMA_VERSION = "ext-x1r4-m1-screen-v1"
RECEIPT_SCHEMA = "ext-x1r4-m1-meter-receipt-v1"
X1R4_SEAL = "study-120h/freeze/20260913T183353Z-X1R4"
X1R4_RUNTIME_LOCK = "study-120h/runtime-lock/20260913T205646Z-X1R4"
EXPECTED_VLLM_VERSION = "0.28.0"
EXPECTED_SEED = 20260912
EXPECTED_MAX_NUM_SEQS = 32
FIRST_REFORECAST_ATTEMPTS = 1000
GATE_NAME = "first_1000_attempt_global_reforecast"
M1_TOTAL_CALLS = len(ext.EXT_MODEL_IDS) * len(ext.EXT_TASK_KEYS) * N_SCREEN
TERMINAL_STATUSES = ("ok", "context_excluded", "generation_failed", "worker_failed")
RETRYABLE_STATUSES = ("generation_failed", "worker_failed")
IDENTITY_KEYS = (
    "call_id",
    "model_id",
    "task_id",
    "condition",
    "doc_id",
    "prompt_sha256",
    "max_tokens",
    "position",
)
EVENT_KEYS = (*IDENTITY_KEYS, "attempt_id", "attempt", "n_tokens")
DISPATCH_KEYS = ("call_id", "attempt_id", "attempt", "previous_attempt_id", "previous_status")
# The forecast proxy for one solo call. The sealed synthetic runtime evidence holds
# no solo case, and `<task>::short::joint` is its single-document, single-target case.
RUNTIME_SOLO_PROXY_SUFFIX = "::short::joint"


# --- Seal and lock binding -------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return payload


def verify_seal(seal_root: Path) -> dict[str, Any]:
    """Rehash every sealed X1R4 file and return the verified manifest.

    The seal is immutable evidence. A single changed byte stops the screen here,
    before any paid container starts.
    """
    manifest = _load_json(seal_root / "MANIFEST.json")
    if manifest.get("schema_version") != ext.SCHEMA_VERSION:
        raise ValueError("the seal is not an X1R4 seal")
    recomputed = sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if recomputed != manifest.get("manifest_sha256"):
        raise ValueError("the seal manifest does not hash to its own recorded digest")
    if manifest.get("identity_sha256") != ext.disclosure()["identity_sha256"]:
        raise ValueError("the seal identity differs from this code's frozen identity")
    if manifest.get("contains_model_outputs") or manifest.get("contains_dataset_text"):
        raise ValueError("the seal claims to hold model output or dataset text")
    files = manifest["files_sha256"]
    if not isinstance(files, dict) or not files:
        raise ValueError("the seal manifest lists no files")
    changed = [
        name
        for name, digest in sorted(files.items())
        if not (seal_root / name).is_file() or sha256_file(seal_root / name) != digest
    ]
    if changed:
        raise ValueError(f"sealed files changed or are missing: {changed[:4]}")
    return {
        "seal_root": str(seal_root),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "identity_sha256": str(manifest["identity_sha256"]),
        "files_verified": len(files),
        "batch_invariance_plan_sha256": str(manifest["batch_invariance_plan_sha256"]),
        "frames": manifest["frames"],
    }


def verify_runtime_lock(lock_root: Path, seal: dict[str, Any]) -> dict[str, Any]:
    """Rehash the signed runtime/access lock and return the attempted candidates.

    The lock is produced outside this module and is never written here. This
    function only reads it, proves it is intact, and proves it belongs to the
    same X1R4 seal that supplies the call order.
    """
    manifest = _load_json(lock_root / "MANIFEST.json")
    if manifest.get("status") != "sealed":
        raise ValueError("the runtime/access lock is not sealed")
    if manifest.get("x1r4_manifest_identifier") != seal["manifest_sha256"]:
        raise ValueError("the runtime/access lock belongs to a different seal")
    files = manifest["files_sha256"]
    changed = [
        name
        for name, digest in sorted(files.items())
        if not (lock_root / name).is_file() or sha256_file(lock_root / name) != digest
    ]
    if changed:
        raise ValueError(f"runtime/access lock files changed or are missing: {changed[:4]}")
    runtime = _load_json(lock_root / "RUNTIME_LOCK.json")
    access = _load_json(lock_root / "ACCESS_LOCK.json")
    if runtime.get("status") != "passed":
        raise ValueError("the synthetic runtime lock did not pass")
    if runtime.get("scientific_model_outputs") is not False:
        raise ValueError("the runtime lock claims scientific model outputs")
    if runtime.get("sealed_batch_plan_sha256") != seal["batch_invariance_plan_sha256"]:
        raise ValueError("the runtime lock binds a different batch-invariance plan")
    if access.get("runtime_lock_sha256") != files["RUNTIME_LOCK.json"]:
        raise ValueError("the access lock does not bind this runtime lock")
    if access.get("batch_invariance_report_sha256") != runtime.get("batch_report_sha256"):
        raise ValueError("the access lock and runtime lock bind different batch reports")
    decided = {str(row["model_id"]): str(row["candidate_status"]) for row in access["candidates"]}
    if set(decided) != set(ext.EXT_MODEL_IDS):
        raise ValueError("the access lock does not decide every one of the five candidates")
    if any(status not in ext.CANDIDATE_STATUSES for status in decided.values()):
        raise ValueError("the access lock holds a candidate without a terminal status")
    by_model = {str(row["model_id"]): row for row in runtime["candidates"]}
    revisions = {model.repo: model.revision for model in ext.EXT_MODELS}
    for model_id, status in sorted(decided.items()):
        if status != "attempted":
            continue
        row = by_model.get(model_id)
        if row is None:
            raise ValueError(f"{model_id} is attempted but carries no runtime evidence")
        if row.get("revision") != revisions[model_id]:
            raise ValueError(f"{model_id} runtime evidence uses an unpinned revision")
        if row.get("synthetic_runtime_status") != "passed":
            raise ValueError(f"{model_id} did not pass the synthetic runtime check")
        if row.get("batch_invariance_status") != "passed":
            raise ValueError(f"{model_id} did not pass the batch-invariance gate")
        if int(row.get("context_window_tokens", 0)) != ext.ext_policy_for(model_id).max_model_len:
            raise ValueError(f"{model_id} runtime evidence uses a different context window")
    attempted = tuple(
        model_id for model_id in ext.EXT_MODEL_IDS if decided[model_id] == "attempted"
    )
    unavailable = tuple(model_id for model_id in ext.EXT_MODEL_IDS if model_id not in attempted)
    if sorted(access.get("attempted_models", [])) != sorted(attempted) or sorted(
        access.get("unavailable_models", [])
    ) != sorted(unavailable):
        raise ValueError("the access lock's candidate lists disagree with its own statuses")
    if not attempted:
        raise ValueError("no candidate is attempted, so M1 has nothing to run")
    return {
        "lock_root": str(lock_root),
        "manifest_sha256": sha256_json(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        ),
        "runtime_lock_sha256": str(files["RUNTIME_LOCK.json"]),
        "access_lock_sha256": str(files["ACCESS_LOCK.json"]),
        "access_lock": access,
        "candidate_status": decided,
        "attempted_models": list(attempted),
        "unavailable_models": [m for m in ext.EXT_MODEL_IDS if decided[m] != "attempted"],
        "files_verified": len(files),
    }


# --- Pinned checkpoints ----------------------------------------------------


def expected_checkpoint_inventory(
    lock_root: Path, freeze_b_checkpoints: Path
) -> dict[str, dict[str, Any]]:
    """Collect the recorded inventory of every pinned checkpoint from prior evidence.

    Two sources exist. The Freeze B benchmark lock holds per-file hashes for the
    two checkpoints it already used. The X1R4 staging record holds file counts and
    byte totals for the three checkpoints staged later. A record is used only when
    its revision matches the pin in this code.
    """
    pinned = {model.repo: model.revision for model in ext.EXT_MODELS}
    inventory: dict[str, dict[str, Any]] = {}
    if freeze_b_checkpoints.is_file():
        records = json.loads(freeze_b_checkpoints.read_text(encoding="utf-8"))
        for record in records:
            model_id = str(record["model_id"])
            if pinned.get(model_id) != record.get("revision"):
                continue
            files = sorted(record["files"], key=lambda row: str(row["path"]))
            inventory[model_id] = {
                "model_id": model_id,
                "revision": str(record["revision"]),
                "file_count": len(files),
                "total_bytes": sum(int(row["bytes"]) for row in files),
                "files_sha256": sha256_json(files),
                "source": "freeze_b_benchmark_checkpoint_lock",
            }
    staging = _load_json(lock_root / "evidence" / "staging" / "report.json")
    for record in staging["models"]:
        model_id = str(record["model_id"])
        if pinned.get(model_id) != record.get("revision") or model_id in inventory:
            continue
        inventory[model_id] = {
            "model_id": model_id,
            "revision": str(record["revision"]),
            "file_count": int(record["file_count"]),
            "total_bytes": int(record["bytes"]),
            "source": "x1r4_staging_record",
        }
    return inventory


def verify_checkpoint_inventory(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the materialized checkpoint matches its recorded inventory."""
    model_id = str(expected["model_id"])
    if actual.get("model_id") != model_id or actual.get("revision") != expected["revision"]:
        raise ValueError(f"{model_id} materialized a different checkpoint identity")
    for key in ("file_count", "total_bytes"):
        if int(actual.get(key, -1)) != int(expected[key]):
            raise ValueError(f"{model_id} {key} differs from the recorded inventory")
    if "files_sha256" in expected:
        if actual.get("files_sha256") != expected["files_sha256"]:
            raise ValueError(f"{model_id} per-file hashes differ from the recorded inventory")
    return {
        "model_id": model_id,
        "revision": str(expected["revision"]),
        "file_count": int(expected["file_count"]),
        "total_bytes": int(expected["total_bytes"]),
        "per_file_hashes_compared": "files_sha256" in expected,
        "source": str(expected["source"]),
    }


# --- Frames and calls ------------------------------------------------------


def verify_screen_frame(
    work_root: Path, seal: dict[str, Any], task_key: str
) -> list[dict[str, Any]]:
    """Verify the private screen rows against the sealed file hash and row order."""
    if task_key not in ext.EXT_TASK_KEYS:
        raise ValueError(f"{task_key} is not an extension task")
    expected = seal["frames"][task_key]["screen"]
    source = work_root / task_key / "screen.jsonl"
    if sha256_file(source) != expected["source_file_sha256"]:
        raise ValueError(f"{task_key} screen source differs from the X1R4 seal")
    rows = list(read_jsonl(source))
    if len(rows) != int(expected["rows"]) or len(rows) != N_SCREEN:
        raise ValueError(f"{task_key} screen must hold exactly {N_SCREEN} rows")
    ids = [str(row["doc_id"]) for row in rows]
    if sha256_json(ids) != expected["row_ids_sha256"] or len(set(ids)) != len(ids):
        raise ValueError(f"{task_key} screen row order or identity differs from the seal")
    required = {"doc_id", "text", "reference", "reference_normalized", "n_tokens"}
    if any(not required <= set(row) for row in rows):
        raise ValueError(f"{task_key} screen rows lack the required private fields")
    return rows


def build_cell_calls(
    model_id: str, task_id: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rebuild one sealed cell as executable call specs, prompts included.

    The identity of every call comes from the sealed generator. The rendered
    prompt is rebuilt here and must hash to the sealed prompt digest.
    """
    sealed = list(ext.m1_cell_calls(model_id, task_id, rows))
    calls: list[dict[str, Any]] = []
    for record, row in zip(sealed, rows, strict=True):
        doc_id = str(record["doc_id"])
        messages = build_messages(task_id, [(doc_id, str(row["text"]))], [doc_id])
        if sha256_json(messages) != record["prompt_sha256"]:
            raise ValueError(
                f"rendered prompt differs from the sealed hash for {record['call_id']}"
            )
        calls.append(
            {
                **record,
                "messages": messages,
                "n_tokens": int(row["n_tokens"] or 0),
                "attempt": 1,
                "attempt_id": f"{record['call_id']}::attempt::1",
            }
        )
    return calls


def sealed_cell(m1_order: dict[str, Any], model_id: str, task_id: str) -> dict[str, Any]:
    cell_key = f"{model_id}::{task_id}"
    try:
        return next(cell for cell in m1_order["cells"] if cell["cell_key"] == cell_key)
    except StopIteration as exc:
        raise ValueError(f"{cell_key} is not in the sealed M1 order") from exc


def verify_cell_against_seal(
    m1_order: dict[str, Any], model_id: str, task_id: str, calls: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fail closed unless the rebuilt cell reproduces the sealed call and prompt order."""
    cell = sealed_cell(m1_order, model_id, task_id)
    call_ids = [str(call["call_id"]) for call in calls]
    prompts = [str(call["prompt_sha256"]) for call in calls]
    if len(calls) != int(cell["n_calls"]):
        raise ValueError(f"{cell['cell_key']} rebuilt {len(calls)} of {cell['n_calls']} calls")
    if sha256_json(call_ids) != cell["call_id_order_sha256"]:
        raise ValueError(f"{cell['cell_key']} call-ID order differs from the seal")
    if sha256_json(prompts) != cell["prompt_sha256_order_sha256"]:
        raise ValueError(f"{cell['cell_key']} prompt-hash order differs from the seal")
    if len(set(prompts)) != int(cell["distinct_prompt_hashes"]):
        raise ValueError(f"{cell['cell_key']} distinct prompt count differs from the seal")
    return {
        "cell_key": str(cell["cell_key"]),
        "position": int(cell["position"]),
        "n_calls": len(calls),
        "call_id_order_sha256": str(cell["call_id_order_sha256"]),
        "prompt_sha256_order_sha256": str(cell["prompt_sha256_order_sha256"]),
    }


def execution_order(m1_order: dict[str, Any], attempted: list[str]) -> list[dict[str, Any]]:
    """The sealed cell order, keeping only candidates the signed lock attempts."""
    unknown = sorted(set(attempted) - set(ext.EXT_MODEL_IDS))
    if unknown:
        raise ValueError(f"the lock attempts unknown candidates: {unknown}")
    cells = sorted(m1_order["cells"], key=lambda cell: int(cell["position"]))
    return [cell for cell in cells if str(cell["model_id"]) in set(attempted)]


# --- Resume ----------------------------------------------------------------


def pending_calls(
    calls: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return unfinished calls at their next attempt, linked to the previous attempt.

    A retry never replaces a first attempt. It carries a new attempt ID and names
    the attempt it follows, so the immutable first response stays readable.
    """
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        call_id = str(event.get("call_id", ""))
        previous = latest.get(call_id)
        if previous is None or int(event.get("attempt", 0)) >= int(previous.get("attempt", 0)):
            latest[call_id] = event
    pending = _screen_pending_calls(calls, events)
    for spec in pending:
        previous_event = latest.get(str(spec["call_id"]))
        if int(spec["attempt"]) > 1 and previous_event is not None:
            spec["previous_attempt_id"] = str(previous_event["attempt_id"])
            spec["previous_status"] = str(previous_event.get("status", "unknown"))
    return pending


def assert_first_attempts_immutable(
    recorded: dict[str, dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    """Refuse any recovered payload that would rewrite an already recorded attempt."""
    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        prior = recorded.get(attempt_id)
        if prior is None:
            continue
        fields = (*EVENT_KEYS, "status", "first_response")
        differing = [key for key in fields if prior.get(key) != event.get(key)]
        if differing:
            raise ValueError(
                f"recovered payload would change recorded attempt {attempt_id}: {differing}"
            )


# --- Worker provenance and quarantine --------------------------------------


def validate_worker_provenance(
    model_id: str,
    host: object,
    backend: object,
    *,
    policy: RuntimePolicy | None = None,
) -> None:
    """Reject a successful response whose worker differs from the frozen serving policy."""
    if not isinstance(host, dict) or not isinstance(backend, dict):
        raise ValueError(f"{model_id} successful output lacks worker provenance")
    policy = policy or ext.ext_policy_for(model_id)
    expected = {
        "backend": "vllm",
        "model_id": model_id,
        "model_revision": policy.revision,
        "tokenizer_revision": policy.revision,
        "dtype": "bfloat16",
        "max_model_len": policy.max_model_len,
        "max_num_seqs": EXPECTED_MAX_NUM_SEQS,
        "tensor_parallel_size": 1,
        "enable_thinking": policy.enable_thinking,
        "seed": EXPECTED_SEED,
        "batch_invariant": True,
        "batch_invariance_required": True,
    }
    mismatched = [key for key, value in expected.items() if backend.get(key) != value]
    if mismatched:
        raise ValueError(f"{model_id} worker differs from the freeze: {', '.join(mismatched)}")
    if host.get("installed_engine_version_string") != EXPECTED_VLLM_VERSION:
        raise ValueError(f"{model_id} worker vLLM version differs from the freeze")
    if "A100" not in str(host.get("gpu_name") or "").upper():
        raise ValueError(f"{model_id} worker is not an A100: {host.get('gpu_name') or 'unknown'}")


def classify_events(
    specs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    host: object = None,
    backend: object = None,
    recorded_attempt_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split worker output into ledger events and quarantined payloads.

    A quarantined payload is never written into the cell ledger and never enters
    an admission statistic. It is kept with its reason so the loss is auditable.

    Specs are matched by call ID, so a recovered retry is accepted against the
    same planned call that produced its first attempt.
    """
    by_call = {str(spec["call_id"]): spec for spec in specs}
    seen = set(recorded_attempt_ids or set())
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []

    def reject(event: dict[str, Any], reason: str) -> None:
        quarantined.append({**event, "quarantine_reason": reason, "quarantined": True})

    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        spec = by_call.get(str(event.get("call_id", "")))
        if spec is None:
            reject(event, "unknown_call_id")
            continue
        attempt = int(event.get("attempt", 0))
        if attempt < 1 or attempt > MAX_TECHNICAL_ATTEMPTS:
            reject(event, "attempt_outside_technical_limit")
            continue
        if attempt_id != f"{event.get('call_id')}::attempt::{attempt}":
            reject(event, "attempt_id_mismatch")
            continue
        if attempt_id in seen:
            reject(event, "duplicate_attempt_id")
            continue
        differing = [key for key in IDENTITY_KEYS if event.get(key) != spec.get(key)]
        if differing:
            reject(event, f"identity_mismatch:{','.join(sorted(differing))}")
            continue
        status = str(event.get("status", ""))
        if status not in TERMINAL_STATUSES:
            reject(event, "unknown_status")
            continue
        if status == "ok":
            if not isinstance(event.get("first_response"), str):
                reject(event, "missing_first_response")
                continue
            try:
                validate_worker_provenance(
                    str(spec["model_id"]),
                    host if host is not None else event.get("worker_host"),
                    backend if backend is not None else event.get("worker_backend"),
                )
            except ValueError as exc:
                reject(event, f"worker_provenance:{exc}")
                continue
        seen.add(attempt_id)
        retryable = status in RETRYABLE_STATUSES
        accepted.append(
            {
                **event,
                "terminal": True,
                "final_for_call": not retryable or attempt >= MAX_TECHNICAL_ATTEMPTS,
                "status": (
                    "technical_attempts_exhausted"
                    if retryable and attempt >= MAX_TECHNICAL_ATTEMPTS
                    else status
                ),
            }
        )
    return accepted, quarantined


# --- Metering and the first-1,000-attempt gate -----------------------------


def session_delta_seconds(
    session_elapsed: dict[str, float], session_id: str | None, elapsed_seconds: float
) -> float:
    """Meter one worker session by its own high-water mark, never twice.

    A resumed run re-reads the same session totals, so only the new seconds are
    added. Summing these deltas across every worker gives all-worker A100 time.
    """
    if elapsed_seconds < 0:
        raise ValueError("elapsed seconds cannot be negative")
    if session_id is None:
        return float(elapsed_seconds)
    previous = float(session_elapsed.get(session_id, 0.0))
    session_elapsed[session_id] = max(previous, float(elapsed_seconds))
    return max(0.0, float(elapsed_seconds) - previous)


def receipt_identity(payload: dict[str, Any], attempt_ids: list[str]) -> str:
    """A stable identity for one worker payload, independent of when it is absorbed.

    The identity binds the worker session, that session's elapsed reading, and
    the exact attempts the payload covers. A replay of the same committed chunk
    produces the same identity, so it can never be metered twice.
    """
    if not attempt_ids:
        raise ValueError("a meter receipt must cover at least one attempt")
    return sha256_json(
        {
            "worker_session_id": payload.get("worker_session_id"),
            "session_elapsed_seconds": float(payload.get("session_elapsed_seconds", 0.0)),
            "attempt_ids": sorted(str(attempt_id) for attempt_id in attempt_ids),
        }
    )


def empty_meter_state() -> dict[str, Any]:
    """The meter at the extension origin, before any receipt exists."""
    return {
        "cumulative_a100_hours": ext.METER_ORIGIN_HOURS,
        "session_elapsed_seconds": {},
        "receipt_ids": set(),
        "receipts_by_id": {},
        "receipts": 0,
        "metered_a100_seconds": 0.0,
    }


def fold_receipts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the meter from the append-only receipt file, which is authoritative.

    Folding is the only way the meter is restored. A crash between two writes
    therefore cannot double-count and cannot lose a receipt that was written:
    replaying the same receipt identity adds nothing, and every recorded second
    is added exactly once.
    """
    state = empty_meter_state()
    seconds = 0.0
    for row in rows:
        identity = str(row["receipt_id"])
        if identity in state["receipt_ids"]:
            continue
        state["receipt_ids"].add(identity)
        state["receipts_by_id"][identity] = dict(row)
        state["receipts"] += 1
        session_id = row.get("worker_session_id")
        if session_id is not None:
            key = str(session_id)
            elapsed = float(row["session_elapsed_seconds"])
            state["session_elapsed_seconds"][key] = max(
                elapsed, float(state["session_elapsed_seconds"].get(key, 0.0))
            )
        seconds += float(row["metered_a100_seconds"])
    state["metered_a100_seconds"] = seconds
    state["cumulative_a100_hours"] = ext.METER_ORIGIN_HOURS + seconds / 3600.0
    return state


def meter_receipt(
    state: dict[str, Any],
    payload: dict[str, Any],
    *,
    cell_key: str,
    attempt_ids: list[str],
    wall_seconds: float,
    counts: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """Meter one payload once and return its receipt, or None when already metered.

    The caller must append the returned receipt before it appends any scientific
    event. A crash after the receipt keeps the meter correct, and the committed
    worker chunk still supplies the events on the next pass.
    """
    identity = receipt_identity(payload, attempt_ids)
    if identity in state["receipt_ids"]:
        return None
    session_id = payload.get("worker_session_id")
    elapsed = float(payload.get("session_elapsed_seconds", wall_seconds))
    delta = session_delta_seconds(
        state["session_elapsed_seconds"], str(session_id) if session_id else None, elapsed
    )
    state["receipt_ids"].add(identity)
    state["receipts"] += 1
    state["metered_a100_seconds"] = float(state["metered_a100_seconds"]) + delta
    state["cumulative_a100_hours"] = (
        ext.METER_ORIGIN_HOURS + float(state["metered_a100_seconds"]) / 3600.0
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": identity,
        "cell_key": cell_key,
        "worker_session_id": session_id,
        "session_elapsed_seconds": elapsed,
        "worker_load_seconds": payload.get("load_seconds"),
        "metered_a100_seconds": delta,
        "cumulative_a100_hours": state["cumulative_a100_hours"],
        "attempt_ids": sorted(str(attempt_id) for attempt_id in attempt_ids),
        "counts": dict(counts or {}),
    }
    state["receipts_by_id"][identity] = dict(receipt)
    return receipt


def repair_torn_receipt_tail(path: Path) -> int:
    """Drop an incomplete final record so the next append starts on a clean line.

    A crash during an append can leave bytes with no closing newline. Appending
    after them would join two records into one corrupt line. This removes only
    those trailing bytes, and only when the file does not end with a newline.
    Every surviving line must still parse, so a malformed complete line or a
    malformed interior line still fails closed. Returns the bytes dropped.

    Use this on the receipt file only. A scientific ledger is never repaired.
    """
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    cut = data.rfind(b"\n") + 1
    dropped = len(data) - cut
    for line in data[:cut].splitlines():
        if line.strip():
            json.loads(line)  # a malformed kept line must still stop the run
    with path.open("r+b") as handle:
        handle.truncate(cut)
        handle.flush()
        os.fsync(handle.fileno())
    return dropped


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append complete records and flush them to the disk before returning."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_rows(path: Path, *, tolerate_partial: bool = False) -> list[dict[str, Any]]:
    """Read one append-only file. A torn final line is dropped only when allowed.

    A crash during an append can leave bytes with no closing newline. The
    authoritative receipt file tolerates those bytes, because they were never a
    complete record. A scientific ledger never tolerates them.

    Only an unterminated final line is ever dropped. A complete but malformed
    line stops the run, wherever it sits, because complete bytes that do not
    parse are corruption and not a torn write.
    """
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8")
    torn_tail = bool(data) and not data.endswith("\n")
    raw = [line for line in data.splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_partial and torn_tail and index == len(raw) - 1:
                break
            raise
    return rows


def attempt_identity(row: dict[str, Any]) -> str:
    """A stable identity for one actual attempt, whether it is recorded or quarantined.

    A well-formed attempt is named by its own attempt ID. A quarantined payload
    can carry a missing or malformed one, so its content supplies the identity
    instead. The same rejected bytes therefore hash alike on every replay.

    The receipt that timed the attempt is part of the identity. A replayed chunk
    reuses its receipt, so it is one attempt however often it is re-absorbed. A
    call that is run again after a quarantine is timed by a new receipt, so it is
    a second attempt and the gate counts it twice.
    """
    attempt_id = row.get("attempt_id")
    if not (isinstance(attempt_id, str) and attempt_id):
        attempt_id = "unidentified::" + sha256_json(
            {key: row.get(key) for key in (*EVENT_KEYS, "status", "quarantine_reason")}
        )
    return f"{row.get('receipt_id')}::{attempt_id}"


def actual_attempts(
    ledger_rows: list[dict[str, Any]],
    quarantine_rows: list[dict[str, Any]],
    *,
    receipt_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Every unique attempt the screen really ran, in execution order.

    A quarantined payload never reaches an admission statistic, but the call was
    made, the receipt metered it, and the first gate is a gate on attempts. So
    the gate counts it. Each attempt appears once, however many times a resumed
    run re-absorbs the committed chunk that produced it.

    The receipt file is append-only and is written in absorb order, so the rank
    of a row's receipt is its execution order. Rows of one receipt are ordered by
    their sealed position. A row whose receipt is unknown sorts last.
    """
    ranks = {receipt_id: index for index, receipt_id in enumerate(receipt_order or [])}
    last = len(ranks)

    def key(item: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        index, row = item
        rank = ranks.get(str(row.get("receipt_id")), last) if receipt_order is not None else 0
        return (
            rank,
            str(row.get("model_id", "")),
            str(row.get("task_id", "")),
            int(row.get("position", 0) or 0),
            int(row.get("attempt", 0) or 0),
            index,
        )

    seen: set[str] = set()
    unique: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate([*ledger_rows, *quarantine_rows]):
        if row.get("counts_as_attempt") is False:
            continue
        identity = attempt_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((index, row))
    return [row for _, row in sorted(unique, key=key)]


def normalize_worker_events(
    specs: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind one countable result slot to every call actually dispatched.

    The worker is asked to run each supplied spec exactly once. Its payload is
    evidence about those attempts, not authority to invent more attempts. The
    first event with a matching attempt ID fills that spec's slot. A missing
    slot becomes a retryable worker failure, while duplicate and unknown rows
    are retained for audit but explicitly excluded from attempt counts.
    """
    expected = {str(spec["attempt_id"]): spec for spec in specs}
    matched: dict[str, dict[str, Any]] = {}
    surplus: list[dict[str, Any]] = []
    for ordinal, event in enumerate(events):
        attempt_id = str(event.get("attempt_id", ""))
        if attempt_id in expected and attempt_id not in matched:
            matched[attempt_id] = {**event, "counts_as_attempt": True}
        else:
            surplus.append(
                {
                    **event,
                    "payload_event_ordinal": ordinal,
                    "counts_as_attempt": False,
                }
            )

    countable: list[dict[str, Any]] = []
    for spec in specs:
        attempt_id = str(spec["attempt_id"])
        selected = matched.get(attempt_id)
        if selected is None:
            selected = {
                **{key: spec[key] for key in EVENT_KEYS},
                "status": "worker_failed",
                "error": "worker payload omitted the dispatched attempt",
                "counts_as_attempt": True,
            }
        countable.append(selected)
    return [*countable, *surplus]


def dispatch_manifest(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record enough dispatch identity to recover this exact chunk without prompt text."""
    return [
        {key: spec[key] for key in DISPATCH_KEYS if key in spec}
        for spec in specs
    ]


def recover_dispatched_specs(
    calls: list[dict[str, Any]], manifest: object
) -> list[dict[str, Any]]:
    """Rebuild and validate the exact specs named by a committed chunk."""
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("committed chunk lacks its dispatched-attempt manifest")
    by_call = {str(call["call_id"]): call for call in calls}
    recovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in manifest:
        if not isinstance(item, dict):
            raise ValueError("committed chunk has a malformed dispatched attempt")
        call_id = str(item.get("call_id", ""))
        base = by_call.get(call_id)
        if base is None:
            raise ValueError(f"committed chunk names unknown call {call_id}")
        attempt = int(item.get("attempt", 0))
        attempt_id = str(item.get("attempt_id", ""))
        if not 1 <= attempt <= MAX_TECHNICAL_ATTEMPTS:
            raise ValueError("committed chunk has an attempt outside the technical limit")
        if attempt_id != f"{call_id}::attempt::{attempt}":
            raise ValueError("committed chunk has a mismatched attempt ID")
        if attempt_id in seen:
            raise ValueError("committed chunk repeats a dispatched attempt")
        seen.add(attempt_id)
        spec = {**base, "attempt": attempt, "attempt_id": attempt_id}
        if attempt > 1:
            previous_id = str(item.get("previous_attempt_id", ""))
            previous_status = str(item.get("previous_status", ""))
            if previous_id != f"{call_id}::attempt::{attempt - 1}":
                raise ValueError("committed retry does not link to its previous attempt")
            if previous_status not in RETRYABLE_STATUSES:
                raise ValueError("committed retry follows a non-retryable status")
            spec.update(
                {"previous_attempt_id": previous_id, "previous_status": previous_status}
            )
        recovered.append(spec)
    return recovered


def retry_specs(
    specs: list[dict[str, Any]], accepted: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the next attempt for every retryable call, linked to the attempt it follows."""
    by_call = {str(spec["call_id"]): spec for spec in specs}
    retries: list[dict[str, Any]] = []
    for event in accepted:
        if event.get("final_for_call") or str(event.get("status")) not in RETRYABLE_STATUSES:
            continue
        source = by_call[str(event["call_id"])]
        attempt = int(event["attempt"]) + 1
        retries.append(
            {
                **source,
                "attempt": attempt,
                "attempt_id": f"{event['call_id']}::attempt::{attempt}",
                "previous_attempt_id": str(event["attempt_id"]),
                "previous_status": str(event["status"]),
            }
        )
    return retries


def recorded_receipt(
    state: dict[str, Any],
    payload: dict[str, Any],
    attempt_ids: list[str],
    *,
    cell_key: str,
) -> dict[str, Any]:
    """Return the receipt already recorded for this payload, and prove it matches.

    A replay must time its recovered rows from the receipt the first pass wrote.
    The receipt is found by the payload's own identity, so a receipt that does
    not cover the same cell and the same attempts is corrupt evidence and stops
    the run.
    """
    identity = receipt_identity(payload, attempt_ids)
    record = state["receipts_by_id"].get(identity)
    if record is None:
        raise ValueError(f"receipt {identity} is metered but its record is missing")
    if str(record.get("cell_key")) != cell_key:
        raise ValueError(f"receipt {identity} belongs to cell {record.get('cell_key')}")
    if list(record.get("attempt_ids") or []) != sorted(attempt_ids):
        raise ValueError(f"receipt {identity} covers different attempts")
    if float(record.get("metered_a100_seconds", -1.0)) < 0.0:
        raise ValueError(f"receipt {identity} records negative metered time")
    return dict(record)


def absorb_payload(
    *,
    state: dict[str, Any],
    payload: dict[str, Any],
    specs: list[dict[str, Any]],
    cell_key: str,
    wall_seconds: float,
    ledger_path: Path,
    receipts_path: Path,
    quarantine_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Meter, classify, and record one worker payload in a replay-safe order.

    The order is fixed and is the whole point of this function.

    1. Read the ledger and refuse any payload that would change a recorded attempt.
    2. Classify the attempts that are not recorded yet.
    3. Meter the payload once and append its receipt, flushed to the disk.
    4. Append the accepted events, then the quarantined ones.

    The receipt is written before any scientific row, so a crash at any boundary
    leaves the meter correct. The committed worker chunk still holds the events,
    and re-absorbing it adds the rows without metering the payload again.

    A replay still has to time the rows it recovers. The payload's own receipt
    supplies those seconds, so a row written on the second pass carries the same
    time as a row written on the first pass. The total is unchanged, because the
    receipt file remains the only place a second is ever added.
    """
    dispatched_attempt_ids = [str(spec["attempt_id"]) for spec in specs]
    normalized_events = normalize_worker_events(specs, payload["events"])
    recorded = {str(row["attempt_id"]): row for row in read_jsonl_rows(ledger_path)}
    assert_first_attempts_immutable(recorded, normalized_events)
    fresh = [row for row in normalized_events if str(row.get("attempt_id")) not in recorded]
    accepted, quarantined = classify_events(
        specs,
        fresh,
        host=payload.get("host"),
        backend=payload.get("backend"),
        recorded_attempt_ids=set(recorded),
    )
    receipt = meter_receipt(
        state,
        payload,
        cell_key=cell_key,
        attempt_ids=dispatched_attempt_ids,
        wall_seconds=wall_seconds,
        counts={
            "dispatched_attempts": len(specs),
            "payload_events": len(payload["events"]),
            "already_recorded": len(normalized_events) - len(fresh),
            "accepted": len(accepted),
            "quarantined": len(quarantined),
        },
    )
    if receipt is None:
        record = recorded_receipt(state, payload, dispatched_attempt_ids, cell_key=cell_key)
        metered = 0.0
    else:
        repair_torn_receipt_tail(receipts_path)
        append_jsonl(receipts_path, [receipt])
        record = receipt
        metered = float(receipt["metered_a100_seconds"])
    # Every row of one payload carries the same share, on the first pass and on
    # a replay. The divisor is the payload's own attempt count, never the number
    # of rows this pass happens to write, so partial recovery times rows alike.
    per_call = float(record["metered_a100_seconds"]) / max(1, len(record["attempt_ids"]))
    rows = accepted + quarantined
    for row in rows:
        row.update(
            {
                "worker_session_id": payload.get("worker_session_id"),
                "worker_load_seconds": payload.get("load_seconds"),
                "worker_host": payload.get("host"),
                "worker_backend": payload.get("backend"),
                "metered_a100_seconds": per_call,
                "cumulative_a100_hours": record["cumulative_a100_hours"],
                "receipt_id": str(record["receipt_id"]),
                **(extra or {}),
            }
        )
    append_jsonl(ledger_path, accepted)
    append_jsonl(quarantine_path, quarantined)
    return {
        "cell_key": cell_key,
        "accepted": accepted,
        "quarantined": quarantined,
        "already_recorded": len(normalized_events) - len(fresh),
        "receipt": receipt,
        "receipt_id": str(record["receipt_id"]),
        "per_call_a100_seconds": per_call,
        "metered_a100_seconds": metered,
        "retries": retry_specs(specs, accepted),
    }


def m1_base_forecast(runtime_evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Forecast M1 from the sealed synthetic runtime evidence only.

    The per-call proxy is the median measured latency of the single-document
    synthetic case for the same task. The per-model load allowance is the
    container time the measured calls do not explain. This is a planning
    forecast for the budget guard, not a quote.

    The proxy times one call alone, so it ignores request batching and
    over-forecasts a chunked screen. That direction is safe: it can only make
    the start guard stricter. The first-1,000-attempt reforecast replaces it
    with measured chunked time.
    """
    cells: dict[str, float] = {}
    loads: dict[str, float] = {}
    for model_id, evidence in sorted(runtime_evidence.items()):
        measured = [
            row
            for row in evidence["events"]
            if row.get("phase") == "measure" and row.get("status") == "ok"
        ]
        accounted = sum(float(row.get("latency_seconds", 0.0)) for row in evidence["events"])
        loads[model_id] = (
            max(0.0, float(evidence.get("container_gpu_seconds", 0.0)) - accounted) / 3600.0
        )
        for task_id in ext.EXT_TASK_KEYS:
            case = f"{task_id}{RUNTIME_SOLO_PROXY_SUFFIX}"
            latencies = sorted(
                float(row["latency_seconds"]) for row in measured if row.get("case_id") == case
            )
            if not latencies:
                raise ValueError(f"{model_id} runtime evidence lacks the {case} proxy case")
            median = latencies[len(latencies) // 2]
            cells[f"{model_id}::{task_id}"] = median * N_SCREEN / 3600.0
    return {
        "schema_version": SCHEMA_VERSION,
        "proxy_case_suffix": RUNTIME_SOLO_PROXY_SUFFIX,
        "cell_base_a100_hours": cells,
        "model_load_a100_hours": loads,
        "planning_only": True,
    }


def cell_base_hours(forecast: dict[str, Any], cell_key: str, *, include_load: bool) -> float:
    """Base hours for one cell; the first cell of a model also carries its load."""
    model_id = cell_key.split("::", 1)[0]
    hours = float(forecast["cell_base_a100_hours"][cell_key])
    if include_load:
        hours += float(forecast["model_load_a100_hours"][model_id])
    return hours


EXECUTION_SCOPE = (
    "every candidate the signed lock attempts, all four frozen tasks, in the sealed "
    "M1 order; the screen has no partial-panel mode"
)


def verify_full_schedule(
    verified_cells: list[dict[str, Any]], lock: dict[str, Any]
) -> dict[str, Any]:
    """Refuse a run that would screen only part of the signed attempted panel.

    X1R4 runs M1 candidate by candidate in the sealed order, with all four tasks
    for every attempted candidate. A subset run would produce admission evidence
    that no sealed schedule describes, so it is rejected here.
    """
    attempted = list(lock["attempted_models"])
    expected = [f"{model_id}::{task_id}" for model_id in attempted for task_id in ext.EXT_TASK_KEYS]
    keys = [str(cell["cell_key"]) for cell in verified_cells]
    if keys != expected:
        raise ValueError(
            "M1 must run the complete attempted panel in sealed order; "
            f"expected {len(expected)} cells, received {len(keys)}"
        )
    positions = [int(cell["position"]) for cell in verified_cells]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("the M1 cells are not in the sealed position order")
    calls = sum(int(cell["n_calls"]) for cell in verified_cells)
    if calls != len(attempted) * len(ext.EXT_TASK_KEYS) * N_SCREEN:
        raise ValueError("the M1 cells do not hold the sealed call count")
    return {
        "models": attempted,
        "tasks": list(ext.EXT_TASK_KEYS),
        "cell_keys": keys,
        "cell_positions": positions,
        "planned_calls": calls,
        "cell_order_sha256": sha256_json(keys),
        "execution_scope": EXECUTION_SCOPE,
    }


def build_plan(
    *,
    seal: dict[str, Any],
    lock: dict[str, Any],
    m1_order: dict[str, Any],
    verified_cells: list[dict[str, Any]],
    forecast: dict[str, Any],
    chunk_size: int,
    batch_invariant_env: str,
) -> dict[str, Any]:
    """The immutable plan a resumed run must reproduce byte for byte.

    The plan binds the exact schedule that will run: the attempted models, the
    four tasks, every cell key and position, and the total call count. A resume
    under any other schedule changes the plan hash and is refused.
    """
    if not 1 <= chunk_size <= 64:
        raise ValueError("chunk size must be 1 through 64")
    if batch_invariant_env != "1":
        raise ValueError("M1 requires VLLM_BATCH_INVARIANT=1 in the worker environment")
    schedule = verify_full_schedule(verified_cells, lock)
    core = {
        "schema_version": SCHEMA_VERSION,
        "stage": m1_order["stage"],
        "condition": m1_order["condition"],
        "seal_manifest_sha256": seal["manifest_sha256"],
        "sealed_files_verified": seal["files_verified"],
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "access_lock_sha256": lock["access_lock_sha256"],
        "attempted_models": list(lock["attempted_models"]),
        "unavailable_models": list(lock["unavailable_models"]),
        "schedule": schedule,
        "sealed_cells": verified_cells,
        "sealed_calls": int(m1_order["n_calls"]),
        "planned_calls": int(schedule["planned_calls"]),
        "global_call_id_order_sha256": m1_order["global_call_id_order_sha256"],
        "global_prompt_sha256_order_sha256": m1_order["global_prompt_sha256_order_sha256"],
        "chunk_size": chunk_size,
        "batch_invariant_environment": batch_invariant_env,
        "first_gate": GATE_NAME,
        "first_gate_attempts": FIRST_REFORECAST_ATTEMPTS,
        "max_technical_attempts": MAX_TECHNICAL_ATTEMPTS,
        "budget": {
            "meter_origin_a100_hours": ext.METER_ORIGIN_HOURS,
            "stop_a100_hours": ext.STOP_HOURS,
            "reserve_a100_hours": ext.RESERVE_HOURS,
            "start_guard_multiplier": ext.FORECAST_MULTIPLIER,
        },
        "forecast": forecast,
    }
    return {**core, "plan_sha256": sha256_json(core)}


def m1_reforecast(
    events: list[dict[str, Any]],
    *,
    planned_calls: int,
    planned_models: int,
    attempts: int = FIRST_REFORECAST_ATTEMPTS,
) -> dict[str, Any] | None:
    """Rebuild the mandatory first-1,000-attempt forecast from the metered attempts.

    This is one global gate for the whole M1 screen, not one gate per candidate.
    Returns None until the gate has enough attempts. The remaining hours it
    reports are what the budget guard must clear before the rest is released.

    Pass the output of `actual_attempts`. The sample is every attempt the screen
    really ran, so a quarantined payload counts here even though it never enters
    an admission statistic. Its seconds were metered by a receipt like any other
    attempt, and leaving it out would under-time the screen.

    Model-load time is read from the sampled rows only. A model counts as loaded
    when a row of its own carries a worker load reading, never because the runner
    intended to start it.
    """
    if planned_calls < attempts or planned_models < 1:
        raise ValueError("the reforecast needs a plan of at least one model and the gate size")
    if len(events) < attempts:
        return None
    first = events[:attempts]
    loads: dict[str, float] = {}
    loaded_models: set[str] = set()
    for event in first:
        session_id = event.get("worker_session_id")
        load = event.get("worker_load_seconds")
        if session_id is None or load is None:
            continue
        loads[str(session_id)] = max(float(load), loads.get(str(session_id), 0.0))
        loaded_models.add(str(event["model_id"]))
    if not loads:
        raise ValueError("the first gate holds no worker load evidence, so it cannot forecast")
    loading_seconds = sum(loads.values())
    all_in_seconds = sum(float(event.get("metered_a100_seconds", 0.0)) for event in first)
    variable_seconds = max(0.0, all_in_seconds - loading_seconds)
    per_call = variable_seconds / attempts
    mean_load = loading_seconds / len(loads)
    remaining_calls = max(0, planned_calls - len(events))
    remaining_loads = max(0, planned_models - len(loaded_models))
    remaining_base = (per_call * remaining_calls + mean_load * remaining_loads) / 3600.0
    projected_base = (per_call * planned_calls + mean_load * planned_models) / 3600.0
    failures = sum(event.get("status") != "ok" for event in first)
    quarantined = sum(bool(event.get("quarantined")) for event in first)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "attempts": len(first),
        "planned_calls": planned_calls,
        "planned_models": planned_models,
        "models_with_load_evidence": sorted(loaded_models),
        "worker_sessions_with_load_evidence": len(loads),
        "generation_failures": failures,
        "failure_rate": failures / len(first),
        "quarantined_attempts": quarantined,
        "loading_a100_seconds": loading_seconds,
        "all_in_a100_seconds": all_in_seconds,
        "per_call_a100_seconds": per_call,
        "mean_model_load_a100_seconds": mean_load,
        "remaining_calls": remaining_calls,
        "remaining_base_a100_hours": remaining_base,
        "remaining_guarded_a100_hours": remaining_base * ext.FORECAST_MULTIPLIER,
        "projected_screen_base_a100_hours": projected_base,
        "projected_screen_guarded_a100_hours": projected_base * ext.FORECAST_MULTIPLIER,
        "technical_only": True,
    }


def release_decision(
    cumulative_hours: float, reforecast: dict[str, Any], *, plan_sha256: str
) -> dict[str, Any]:
    """Decide whether the remaining screen may be released after the 1,000 gate."""
    if reforecast.get("gate") != GATE_NAME:
        raise ValueError("the release decision needs the first-gate reforecast")
    decision = ext.ext_may_start(cumulative_hours, float(reforecast["remaining_base_a100_hours"]))
    return {
        **decision,
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "plan_sha256": plan_sha256,
        "reforecast_sha256": sha256_json(reforecast),
        "attempts": int(reforecast["attempts"]),
        "release_remaining_work": bool(decision["start"]),
    }


def verify_release_decision(
    decision: dict[str, Any],
    reforecast: dict[str, Any],
    *,
    plan_sha256: str,
    recorded_attempts: int,
    attempts: int = FIRST_REFORECAST_ATTEMPTS,
) -> dict[str, Any]:
    """Re-prove a saved gate decision before a resumed run skips the gate.

    The presence of the file proves nothing. The decision must belong to this
    plan, must bind the saved reforecast byte for byte, must rest on at least
    the full gate, and must actually release the work.

    `recorded_attempts` is the count of unique actual attempts, quarantined ones
    included. Counting ledger rows alone would let a resumed run re-prove a gate
    that the screen had already passed with rejected work.
    """
    if decision.get("schema_version") != SCHEMA_VERSION or decision.get("gate") != GATE_NAME:
        raise ValueError("the saved release decision is not this gate's record")
    if decision.get("plan_sha256") != plan_sha256:
        raise ValueError("the saved release decision belongs to a different plan")
    if decision.get("reforecast_sha256") != sha256_json(reforecast):
        raise ValueError("the saved release decision does not bind the saved reforecast")
    if reforecast.get("gate") != GATE_NAME or int(reforecast.get("attempts", 0)) < attempts:
        raise ValueError("the saved reforecast does not cover the full first gate")
    if int(decision.get("attempts", 0)) != int(reforecast["attempts"]):
        raise ValueError("the saved decision and reforecast disagree on the gate size")
    if recorded_attempts < attempts:
        raise ValueError(
            f"the run holds {recorded_attempts} actual attempts, below the {attempts}-attempt gate"
        )
    if not (decision.get("start") and decision.get("release_remaining_work")):
        raise ValueError("the saved release decision refused the remaining work")
    return {
        "gate": GATE_NAME,
        "verified": True,
        "attempts": int(decision["attempts"]),
        "recorded_attempts": recorded_attempts,
        "plan_sha256": plan_sha256,
        "reforecast_sha256": str(decision["reforecast_sha256"]),
    }


# --- Admission-analysis inputs ---------------------------------------------


def summarize_cell(
    model_id: str,
    task_id: str,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    candidate_status: str = "attempted",
) -> dict[str, Any]:
    """One cell of solo-only admission evidence; no joint output can enter it."""
    if candidate_status not in ext.CANDIDATE_STATUSES:
        raise ValueError(f"unknown candidate status {candidate_status}")
    cell_key = f"{model_id}::{task_id}"
    family = next(model.family for model in ext.EXT_MODELS if model.repo == model_id)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cell_key": cell_key,
        "model_id": model_id,
        "task_id": task_id,
        "family": family,
        "candidate_status": candidate_status,
        "expected_calls": N_SCREEN,
        "selection_evidence": "solo_capability_and_technical_timing_only",
    }
    if candidate_status != "attempted":
        result.update(
            {
                "executed": False,
                "terminal_calls": 0,
                "technical_complete": False,
                "status_counts": {},
                "admitted": False,
                "reasons": ["candidate_technically_unavailable"],
            }
        )
        return result
    calls = build_cell_calls(model_id, task_id, rows)
    expected_ids = [str(call["call_id"]) for call in calls]
    status_counts: dict[str, int] = {}
    attempts_seen: set[str] = set()
    by_call: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        if not attempt_id or attempt_id in attempts_seen:
            raise ValueError(f"duplicate or missing attempt event in {cell_key}")
        attempts_seen.add(attempt_id)
        if event.get("quarantined"):
            raise ValueError(f"a quarantined payload reached the {cell_key} summary")
        by_call.setdefault(str(event.get("call_id", "")), []).append(event)
        key = str(event.get("status", "unknown"))
        status_counts[key] = status_counts.get(key, 0) + 1
    unexpected = sorted(set(by_call) - set(expected_ids))
    if unexpected:
        raise ValueError(f"unexpected call IDs in {cell_key}: {unexpected[:3]}")
    terminal: list[dict[str, Any]] = []
    for call_id in expected_ids:
        final = [
            event
            for event in by_call.get(call_id, [])
            if event.get("terminal") is True and event.get("final_for_call") is True
        ]
        if len(final) > 1:
            raise ValueError(f"multiple final events for {call_id}")
        terminal.extend(final)
    result.update(
        {
            "executed": True,
            "terminal_calls": len(terminal),
            "technical_complete": len(terminal) == N_SCREEN,
            "status_counts": status_counts,
        }
    )
    if len(terminal) != N_SCREEN:
        result.update({"admitted": False, "reasons": ["technical_screen_incomplete"]})
        return result
    final_status_counts: dict[str, int] = {}
    for event in terminal:
        key = str(event.get("status", "unknown"))
        final_status_counts[key] = final_status_counts.get(key, 0) + 1
    result["final_status_counts"] = final_status_counts
    for event in terminal:
        if event.get("status") == "ok":
            validate_worker_provenance(
                model_id, event.get("worker_host"), event.get("worker_backend")
            )
    if any(event.get("status") != "ok" for event in terminal):
        result.update(
            {
                "technical_excluded": True,
                "admitted": False,
                "reasons": ["technical_generation_or_context_exclusion"],
            }
        )
        return result
    final_by_call = {str(event["call_id"]): event for event in terminal}
    predicted = [
        parse_event(final_by_call[call_id], row, task_id)
        for call_id, row in zip(expected_ids, rows, strict=True)
    ]
    references = [normalize_label(row["reference"], task_id) for row in rows]
    result.update(evaluate(task_id, predicted, references))
    result["parsed_calls"] = sum(value is not None for value in predicted)
    return result


def admission_inputs(
    summaries: list[dict[str, Any]],
    lock: dict[str, Any],
    *,
    seal: dict[str, Any],
    meter: dict[str, Any],
) -> dict[str, Any]:
    """Collect the complete solo-only evidence the X1R4 admission lock consumes.

    This is the input to that lock, never the lock itself. Sealing the admission
    decision is a separate, later step.
    """
    expected = {
        f"{model_id}::{task_id}" for model_id in ext.EXT_MODEL_IDS for task_id in ext.EXT_TASK_KEYS
    }
    seen = [str(row["cell_key"]) for row in summaries]
    if len(seen) != len(expected) or set(seen) != expected:
        raise ValueError(f"admission inputs need all {len(expected)} candidate cells")
    status = lock["candidate_status"]
    for row in summaries:
        model_id = str(row["model_id"])
        if str(row["candidate_status"]) != status[model_id]:
            raise ValueError(f"{row['cell_key']} disagrees with the signed access lock")
        if row.get("admitted") and status[model_id] != "attempted":
            raise ValueError(f"{model_id} is unavailable but a cell is admitted")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "M1 blinded solo capability screen",
        "seal_manifest_sha256": seal["manifest_sha256"],
        "runtime_lock_sha256": lock["runtime_lock_sha256"],
        "access_lock_sha256": lock["access_lock_sha256"],
        "attempted_models": list(lock["attempted_models"]),
        "unavailable_models": list(lock["unavailable_models"]),
        "candidate_status": status,
        "summaries": sorted(summaries, key=lambda row: str(row["cell_key"])),
        "solo_error_rates": {
            str(row["cell_key"]): float(row["error_rate"])
            for row in summaries
            if "error_rate" in row
        },
        "admitted_cells": sorted(str(row["cell_key"]) for row in summaries if row.get("admitted")),
        "screen_complete": all(
            row.get("technical_complete") or row["candidate_status"] != "attempted"
            for row in summaries
        ),
        "metered_a100_hours": float(meter["cumulative_a100_hours"]),
        "extension_joint_outputs_seen": False,
        "next_step": (
            "seal the X1R4 admission lock from these solo summaries, the pooled power "
            "table, and the signed access lock"
        ),
    }
