"""Run one released X1R4 A8 model across four B3 datasets on Modal.

The CPU-only default rebuilds and verifies all 137,216 call specifications.
Paid execution requires ``--execute`` and an explicit matching Modal profile.
Datasets run sequentially; the eight sealed execution blocks may run in parallel.
"""

import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, TextIO, cast

import modal

APP_NAME = "ieee-bigdata-extension-b3"
LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/study-120h")
MODEL_CACHE = "/models"
RESULTS_ROOT = "/results"
GPU_TYPE = "A100"
VLLM_VERSION = "0.28.0"
FRAME_ROOT = LOCAL_ROOT / "ext-work/20260913-cpu-draft"
SEAL_ROOT = LOCAL_ROOT / "freeze/20260913T183353Z-X1R4"
FULL_GRID_ROOT = LOCAL_ROOT / "full-grid-lock/20260914T032937Z-X1R4-A8"
RELEASE_ROOT = LOCAL_ROOT / "execution-release/20260914T035244Z-X1R4-A8-model-major"
LOCK_ROOT = LOCAL_ROOT / "runtime-lock/20260913T205646Z-X1R4"
FREEZE_B_CHECKPOINTS = (
    LOCAL_ROOT / "runtime-lock/20260913T004054Z-BRT1/benchmark/checkpoint-manifests.json"
)
TASK_ORDER = ("civil_comments", "multi_nli", "imdb", "ag_news")
N_ITEMS = 8192
CALLS_PER_CELL = 34304
CALLS_PER_MODEL = 137216
MAX_RETRIES = 1
RETRY_STATUSES = frozenset({"generation_failed", "truncated", "worker_failed"})
IRRECOVERABLE_TECHNICAL_STATUSES = frozenset(
    {"generation_failed", "worker_failed", "context_excluded"}
)
RUNTIME_ENV = {
    "HF_HOME": MODEL_CACHE,
    "HF_HUB_CACHE": MODEL_CACHE,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_BATCH_INVARIANT": "1",
}

model_volume = modal.Volume.from_name("ieee-bigdata-freezeb-models", create_if_missing=False)
results_volume = modal.Volume.from_name("ieee-bigdata-extension-b3-results", create_if_missing=True)
support_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface-hub==1.31.0")
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"))
)
runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface-hub==1.31.0",
        "numpy==2.2.6",
        "scipy==1.16.3",
        f"vllm=={VLLM_VERSION}",
    )
    .env(RUNTIME_ENV)
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"))
)
app = modal.App(APP_NAME)


def _local_imports() -> None:
    sys.path.insert(0, str(LOCAL_ROOT / "src"))


def _remote_imports() -> None:
    sys.path.insert(0, str(REMOTE_ROOT / "src"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_manifest(directory: Path) -> dict[str, Any]:
    _local_imports()
    from step0.hashing import sha256_json

    manifest = _load(directory / "MANIFEST.json")
    for relative, expected in manifest["files_sha256"].items():
        if _sha256_file(directory / relative) != expected:
            raise ValueError(f"sealed file changed: {directory / relative}")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if sha256_json(core) != manifest["manifest_sha256"]:
        raise ValueError(f"manifest digest changed: {directory / 'MANIFEST.json'}")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _frame(task_id: str, grid: dict[str, Any]) -> list[dict[str, Any]]:
    _local_imports()
    from step0.hashing import read_jsonl, sha256_json

    if task_id not in TASK_ORDER:
        raise ValueError(f"unknown B3 dataset {task_id}")
    source = FRAME_ROOT / task_id / "confirm.jsonl"
    manifest = _load(FRAME_ROOT / task_id / "manifest.json")
    expected_file = manifest["files"]["confirm.jsonl"]
    locked = grid["datasets"][task_id]
    if _sha256_file(source) != expected_file["sha256"]:
        raise ValueError(f"{task_id} confirmation frame differs from its source manifest")
    if (
        _sha256_file(SEAL_ROOT / "frame-identifiers" / task_id / "confirm.jsonl")
        != locked["identifier_file_sha256"]
    ):
        raise ValueError(f"{task_id} identifier file differs from the A8 lock")
    rows = list(read_jsonl(source))
    ids = [str(row["doc_id"]) for row in rows]
    if len(rows) != N_ITEMS or len(ids) != len(set(ids)):
        raise ValueError(f"{task_id} frame must contain 8,192 unique rows")
    if sha256_json(ids) != locked["ordered_doc_ids_sha256"]:
        raise ValueError(f"{task_id} item ID order differs from the A8 lock")
    required = {"doc_id", "text", "reference", "reference_normalized", "n_chars"}
    if any(not required <= set(row) for row in rows):
        raise ValueError(f"{task_id} frame rows are missing required fields")
    return rows


def _render_cell(model_id: str, task_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _local_imports()
    from freezeb import extension as ext
    from freezeb.assign import Frame, build_calls
    from freezeb.prompts import build_messages
    from step0.hashing import sha256_json

    by_id = {str(row["doc_id"]): row for row in rows}
    frame = Frame(
        tuple(str(row["doc_id"]) for row in rows),
        tuple(int(row["n_chars"]) for row in rows),
    )
    calls = cast(
        list[dict[str, Any]],
        build_calls(
            frame,
            f"{model_id}::{task_id}",
            ext.BATCH_SIZE,
            namespace=ext.ASSIGN_NAMESPACE,
        ),
    )
    assigned_orders = {
        (int(item["assignment"]), int(item["call_index"])): [
            str(value) for value in item["doc_order"]
        ]
        for item in calls
    }
    jobs: list[dict[str, Any]] = []
    for call in ext.b3_calls(model_id, task_id, rows, N_ITEMS):
        condition = str(call["condition"])
        if condition == "solo":
            target_ids = [str(call["call_id"]).rsplit("::", 1)[1]]
            doc_order = target_ids
        else:
            assignment = int(call["assignment"])
            call_index = int(call["call_index"])
            doc_order = assigned_orders[(assignment, call_index)]
            if condition == "joint":
                target_ids = doc_order
            else:
                target_ids = [str(call["call_id"]).rsplit("::", 1)[1]]
        documents = [(doc_id, str(by_id[doc_id]["text"])) for doc_id in doc_order]
        messages = build_messages(task_id, documents, target_ids)
        if sha256_json(doc_order) != call["doc_order_sha256"]:
            raise ValueError("rendered B3 document order differs from the seal")
        if sha256_json(target_ids) != call["target_ids_sha256"]:
            raise ValueError("rendered B3 target IDs differ from the seal")
        if sha256_json(messages) != call["prompt_sha256"]:
            raise ValueError("rendered B3 prompt differs from the seal")
        jobs.append(
            {
                **call,
                "attempt": 1,
                "attempt_id": f"{call['call_id']}::attempt::1",
                "model_id": model_id,
                "task_id": task_id,
                "doc_order": doc_order,
                "target_ids": target_ids,
                "messages": messages,
            }
        )
    return jobs


def _retry(job: dict[str, Any]) -> dict[str, Any]:
    if int(job["attempt"]) != 1:
        raise ValueError("B3 permits one retry of a first attempt only")
    return {
        **job,
        "attempt": 2,
        "attempt_id": f"{job['call_id']}::attempt::2",
        "parent_attempt_id": str(job["attempt_id"]),
    }


def resume_state(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    first = {str(job["attempt_id"]): job for job in jobs}
    seen: dict[str, dict[str, Any]] = {}
    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        call_id = str(event.get("call_id", ""))
        attempt = int(event.get("attempt", 0))
        if attempt_id in seen:
            if event != seen[attempt_id]:
                raise ValueError(f"attempt {attempt_id} has conflicting stored events")
            continue
        if attempt == 1:
            expected = first.get(attempt_id)
            if expected is None or call_id != expected["call_id"]:
                raise ValueError("stored first attempt does not belong to this B3 plan")
        elif attempt == 2:
            expected = first.get(f"{call_id}::attempt::1")
            if (
                expected is None
                or attempt_id != f"{call_id}::attempt::2"
                or event.get("parent_attempt_id") != expected["attempt_id"]
            ):
                raise ValueError("stored retry does not belong to this B3 plan")
        else:
            raise ValueError("B3 permits first attempts and one linked retry only")
        if (
            event.get("model_id") != jobs[0]["model_id"]
            or event.get("task_id") != jobs[0]["task_id"]
        ):
            raise ValueError("stored B3 event names a foreign cell")
        seen[attempt_id] = event
    return {"first": first, "seen": seen}


def pending_work(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = cast(dict[str, dict[str, Any]], resume_state(jobs, events)["seen"])
    retries = [
        _retry(job)
        for job in jobs
        if (event := seen.get(str(job["attempt_id"]))) is not None
        and event.get("status") in RETRY_STATUSES
        and f"{job['call_id']}::attempt::2" not in seen
    ]
    first = [job for job in jobs if str(job["attempt_id"]) not in seen]
    return retries + first


def completeness(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    seen = cast(dict[str, dict[str, Any]], resume_state(jobs, events)["seen"])
    pending = pending_work(jobs, events)
    final_by_call: dict[str, dict[str, Any]] = {}
    for event in seen.values():
        call_id = str(event["call_id"])
        if call_id not in final_by_call or int(event["attempt"]) > int(
            final_by_call[call_id]["attempt"]
        ):
            final_by_call[call_id] = event
    terminal_failures = [
        event
        for event in final_by_call.values()
        if event.get("status") in IRRECOVERABLE_TECHNICAL_STATUSES
    ]
    unresolved = [
        event
        for event in final_by_call.values()
        if event.get("status") == "truncated"
        or event.get("parse_status") in {"malformed", "truncated", "missing"}
    ]
    return {
        "complete": not pending and not terminal_failures,
        "execution_complete": not pending and not terminal_failures,
        "parsed_complete": not pending and not terminal_failures and not unresolved,
        "all_calls_attempted": not pending,
        "expected_first_attempts": len(jobs),
        "first_attempts": sum(int(event["attempt"]) == 1 for event in seen.values()),
        "retry_attempts": sum(int(event["attempt"]) == 2 for event in seen.values()),
        "pending_attempts": len(pending),
        "terminal_failures": len(terminal_failures),
        "terminal_malformed_outputs": sum(
            event.get("status") == "truncated" for event in final_by_call.values()
        ),
        "unresolved_outputs": len(unresolved),
        "status_counts": dict(sorted(Counter(str(e["status"]) for e in seen.values()).items())),
    }


def build_plan(
    model_id: str, chunk_size: int = 32, workers: int = 4
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Rebuild and verify all four sealed cells without contacting Modal."""
    _local_imports()
    from freezeb import extension as ext
    from step0.hashing import sha256_json

    if not 1 <= chunk_size <= 64:
        raise ValueError("chunk_size must be from 1 through 64")
    if not 1 <= workers <= 4:
        raise ValueError("workers must be from 1 through 4")
    _verify_manifest(FULL_GRID_ROOT)
    _verify_manifest(RELEASE_ROOT)
    grid = _load(FULL_GRID_ROOT / "FULL_GRID_LOCK.json")
    release = _load(RELEASE_ROOT / "EXECUTION_RELEASE.json")
    run = next((row for row in release["model_runs"] if row["model_id"] == model_id), None)
    if run is None:
        raise ValueError(f"model is not in the released four-model grid: {model_id}")
    if run["cells"] != [f"{model_id}::{task}" for task in TASK_ORDER]:
        raise ValueError("released model dataset order differs from the required order")
    if int(run["calls"]) != CALLS_PER_MODEL:
        raise ValueError("released model call count differs")

    jobs_by_task: dict[str, list[dict[str, Any]]] = {}
    cells: list[dict[str, Any]] = []
    for task_id in TASK_ORDER:
        rows = _frame(task_id, grid)
        jobs = _render_cell(model_id, task_id, rows)
        digest = ext.b3_plan_digest(model_id, task_id, rows, N_ITEMS)
        locked = grid["selected_schedules"]["cells"][f"{model_id}::{task_id}"]
        if len(jobs) != CALLS_PER_CELL or digest["counts"] != grid["call_totals"]["per_cell"]:
            raise ValueError(f"{task_id} call count differs from the A8 lock")
        for key in ("call_id_order_sha256", "record_stream_sha256"):
            if digest[key] != locked[key]:
                raise ValueError(f"{task_id} {key} differs from the A8 lock")
        if sha256_json(digest) != locked["selected_plan_sha256"]:
            raise ValueError(f"{task_id} selected B3 plan digest differs from the A8 lock")
        jobs_by_task[task_id] = jobs
        cells.append(
            {
                "cell_key": f"{model_id}::{task_id}",
                "calls": len(jobs),
                "call_id_order_sha256": digest["call_id_order_sha256"],
                "prompt_hash_order_sha256": sha256_json([job["prompt_sha256"] for job in jobs]),
                "item_id_order_sha256": grid["datasets"][task_id]["ordered_doc_ids_sha256"],
                "output_path": f"cells/{task_id}/events.jsonl",
            }
        )
    core = {
        "schema_version": "ext-x1r4-a8-b3-model-run-v1",
        "model_id": model_id,
        "model_revision": next(
            row["revision"] for row in grid["models"] if row["model_id"] == model_id
        ),
        "dataset_order": list(TASK_ORDER),
        "cells": cells,
        "calls_per_cell": CALLS_PER_CELL,
        "total_calls": sum(len(jobs) for jobs in jobs_by_task.values()),
        "chunk_size": chunk_size,
        "workers": workers,
        "execution_blocks": 8,
        "batch_invariant_env": RUNTIME_ENV["VLLM_BATCH_INVARIANT"],
        "max_retries": MAX_RETRIES,
        "full_grid_lock_sha256": _sha256_file(FULL_GRID_ROOT / "FULL_GRID_LOCK.json"),
        "execution_release_sha256": _sha256_file(RELEASE_ROOT / "EXECUTION_RELEASE.json"),
    }
    if core["total_calls"] != CALLS_PER_MODEL:
        raise ValueError("model plan must contain exactly 137,216 calls")
    return {**core, "plan_sha256": sha256_json(core)}, jobs_by_task


@app.function(image=support_image, volumes={MODEL_CACHE: model_volume}, timeout=7200)
def inspect_checkpoint(model_id: str, revision: str, excluded: list[str]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            ignore_patterns=excluded,
        )
    )
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    return {
        "model_id": model_id,
        "revision": revision,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


@app.function(image=support_image, volumes={RESULTS_ROOT: results_volume}, timeout=900)
def recover_payloads(namespace: str, task_id: str) -> list[dict[str, Any]]:
    """Return committed chunks so a local crash cannot duplicate model calls."""
    results_volume.reload()
    directory = Path(RESULTS_ROOT) / namespace / task_id
    if not directory.exists():
        return []
    payload_paths = sorted(directory.glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in payload_paths]


@app.cls(
    image=runtime_image,
    volumes={MODEL_CACHE: model_volume, RESULTS_ROOT: results_volume},
    gpu=GPU_TYPE,
    # This is a global ceiling across smoke checks and execution. Without it,
    # warm smoke containers can overlap newly scheduled execution containers.
    max_containers=4,
    scaledown_window=3600,
    startup_timeout=1800,
    timeout=21600,
)
class B3Worker:
    model_id: str = modal.parameter()
    model_revision: str = modal.parameter()
    worker_index: int = modal.parameter()
    worker_count: int = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        _remote_imports()
        from freezeb import batch_invariance as bi
        from freezeb.extension import ext_policy_for
        from step1.backends import VLLMBackend, host_facts

        policy = ext_policy_for(self.model_id)
        if policy.revision != self.model_revision:
            raise ValueError("B3 checkpoint differs from the released revision")
        self.session_started = time.monotonic()
        self.session_id = uuid.uuid4().hex
        self.backend = VLLMBackend(
            self.model_id,
            self.model_revision,
            self.model_revision,
            {
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "max_model_len": policy.max_model_len,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 32,
            },
            bi.DECODING,
            enable_thinking=policy.enable_thinking,
            require_batch_invariance=True,
        )
        self.load_seconds = time.monotonic() - self.session_started
        self.host = host_facts()

    @modal.method()
    def smoke(self) -> dict[str, Any]:
        from freezeb.prompts import build_messages
        from step0.hashing import sha256_text
        from step1.backends import GenRequest

        doc_id = "synthetic-b3-equivalence-item"
        messages = build_messages(
            "civil_comments", [(doc_id, "A short neutral comment.")], [doc_id]
        )
        result = self.backend.generate(
            [GenRequest("b3-equivalence", "synthetic_smoke", (doc_id,), messages, 84)]
        )[0]
        return {
            "worker_index": self.worker_index,
            "worker_session_id": self.session_id,
            "response_sha256": sha256_text(result.text),
            "finish_reason": result.finish_reason,
            "host": self.host,
            "backend": self.backend.describe(),
        }

    @modal.method()
    def run_chunk(
        self,
        namespace: str,
        task_id: str,
        chunk_id: str,
        jobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from freezeb import e2
        from freezeb.extension import ext_policy_for
        from freezeb.runtime_benchmark import context_fit
        from step1.backends import GenRequest

        if not jobs or len(jobs) > 64:
            raise ValueError("a B3 chunk must contain 1 through 64 attempts")
        for job in jobs:
            if job["model_id"] != self.model_id or job["task_id"] != task_id:
                raise ValueError("B3 worker refuses a foreign cell")
            if int(job["execution_block"]) % self.worker_count != self.worker_index:
                raise ValueError("B3 attempt belongs to a different execution-block worker")
        path = Path(RESULTS_ROOT) / namespace / task_id / f"{chunk_id}.json"
        results_volume.reload()
        if path.exists():
            return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

        policy = ext_policy_for(self.model_id)
        requests: list[GenRequest] = []
        request_jobs: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for job in jobs:
            prompt_tokens = self.backend.count_tokens(job["messages"])
            fit = context_fit(prompt_tokens, int(job["max_tokens"]), policy.max_model_len)
            if not fit["fits"]:
                event = e2.event_for_result(
                    job,
                    generation_error="sealed prompt exceeds context",
                    prompt_tokens=prompt_tokens,
                )
                event["status"] = "context_excluded"
                events.append(event)
                continue
            requests.append(
                GenRequest(
                    str(job["attempt_id"]),
                    str(job["condition"]),
                    tuple(str(value) for value in job["target_ids"]),
                    job["messages"],
                    int(job["max_tokens"]),
                )
            )
            request_jobs.append(job)
        started = time.monotonic()
        try:
            generated = self.backend.generate(requests) if requests else []
            by_id = {result.call_id: result for result in generated}
            for job in request_jobs:
                result = by_id.get(str(job["attempt_id"]))
                if result is None:
                    events.append(e2.event_for_result(job, generation_error="missing result"))
                else:
                    events.append(
                        e2.event_for_result(
                            job,
                            raw=result.text,
                            finish_reason=result.finish_reason,
                            prompt_tokens=result.prompt_tokens,
                            output_tokens=result.output_tokens,
                            queue_seconds=result.queue_seconds,
                            prefill_seconds=result.prefill_seconds,
                            decode_seconds=result.decode_seconds,
                        )
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            events.extend(e2.event_for_result(job, generation_error=error) for job in request_jobs)
        order = {str(job["attempt_id"]): index for index, job in enumerate(jobs)}
        events.sort(key=lambda event: order[str(event["attempt_id"])])
        payload = {
            "chunk_id": chunk_id,
            "worker_index": self.worker_index,
            "worker_session_id": self.session_id,
            "session_elapsed_seconds": time.monotonic() - self.session_started,
            "load_seconds": self.load_seconds,
            "chunk_wall_seconds": time.monotonic() - started,
            "host": self.host,
            "backend": self.backend.describe(),
            "events": events,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        results_volume.commit()
        return payload


def _chunk_id(owner: int, jobs: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256("|".join(str(job["attempt_id"]) for job in jobs).encode()).hexdigest()
    return f"w{owner}-{digest[:20]}"


def _append_events(path: Path, events: list[dict[str, Any]]) -> None:
    stored = {str(row["attempt_id"]): row for row in _read_jsonl(path)}
    fresh: list[dict[str, Any]] = []
    for event in events:
        attempt_id = str(event["attempt_id"])
        if attempt_id not in stored:
            stored[attempt_id] = event
            fresh.append(event)
        elif stored[attempt_id] != event:
            raise RuntimeError(f"attempt {attempt_id} has conflicting local events")
    _append_jsonl(path, fresh)


def _verify_worker_payload(payload: dict[str, Any], plan: dict[str, Any]) -> None:
    host = payload.get("host")
    backend = payload.get("backend")
    if not isinstance(host, dict) or not isinstance(backend, dict):
        raise RuntimeError("B3 payload is missing worker provenance")
    expected = {
        "backend": "vllm",
        "model_id": plan["model_id"],
        "model_revision": plan["model_revision"],
        "tokenizer_revision": plan["model_revision"],
        "dtype": "bfloat16",
        "max_num_seqs": 32,
        "tensor_parallel_size": 1,
        "batch_invariant": True,
        "batch_invariance_required": True,
        "seed": 20260912,
    }
    mismatched = [key for key, value in expected.items() if backend.get(key) != value]
    if mismatched:
        raise RuntimeError(f"B3 worker differs from the sealed serving policy: {mismatched}")
    if str(host.get("installed_engine_version_string")) != VLLM_VERSION:
        raise RuntimeError("B3 worker runs a different vLLM version")
    if "A100" not in str(host.get("gpu_name") or "").upper():
        raise RuntimeError("B3 worker is not an A100")


def _acquire_lock(out: Path, profile: str) -> TextIO:
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / ".active-run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another B3 process is using {out}") from None
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "modal_profile": profile}) + "\n")
    handle.flush()
    return handle


def _run(
    plan: dict[str, Any],
    jobs_by_task: dict[str, list[dict[str, Any]]],
    out: Path,
    dataset_count: int,
    namespace: str,
) -> None:
    workers = int(plan["workers"])
    handles = {
        owner: B3Worker(  # type: ignore[call-arg]
            model_id=plan["model_id"],
            model_revision=plan["model_revision"],
            worker_index=owner,
            worker_count=workers,
        )
        for owner in range(workers)
    }
    # Every fresh launch gets a fresh equivalence witness, including resumes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        smoke = list(pool.map(lambda owner: handles[owner].smoke.remote(), range(workers)))
    if len({row["response_sha256"] for row in smoke}) != 1:
        raise RuntimeError("B3 workers produced different equivalence-smoke bytes")
    for row in smoke:
        _verify_worker_payload(row, plan)
    _write_json(out / "worker-equivalence.json", {"status": "passed", "workers": smoke})

    requested_tasks = TASK_ORDER[:dataset_count]
    for task_id in requested_tasks:
        events_path = out / "cells" / task_id / "events.jsonl"
        jobs = jobs_by_task[task_id]
        # Reconcile durable remote chunks before calculating pending work. A
        # crash after the remote commit but before the local append is harmless.
        for payload in recover_payloads.remote(namespace, task_id):
            _append_events(events_path, cast(list[dict[str, Any]], payload["events"]))
        while pending := pending_work(jobs, _read_jsonl(events_path)):
            groups_by_owner: list[list[tuple[int, list[dict[str, Any]]]]] = []
            for owner in range(workers):
                owned = [job for job in pending if int(job["execution_block"]) % workers == owner]
                size = int(plan["chunk_size"])
                groups_by_owner.append(
                    [(owner, owned[i : i + size]) for i in range(0, len(owned), size)]
                )
            # Interleave owners so all GPUs start promptly. Within every sealed
            # execution block, calls are dispatched in exact position order.
            groups = [
                group
                for round_groups in zip_longest(*groups_by_owner)
                for group in round_groups
                if group is not None
            ]
            for owner, group in groups:
                positions = [int(job["position"]) for job in group]
                if positions != sorted(positions):
                    raise RuntimeError(f"worker {owner} B3 call order changed")

            append_lock = threading.Lock()
            stop_lanes = threading.Event()

            def run_owner(
                owner: int,
                owner_groups: list[tuple[int, list[dict[str, Any]]]],
                current_task: str,
                current_events_path: Path,
                current_append_lock: threading.Lock,
                current_stop_lanes: threading.Event,
            ) -> None:
                for group_owner, group in owner_groups:
                    if group_owner != owner or current_stop_lanes.is_set():
                        continue
                    try:
                        payload = handles[owner].run_chunk.remote(
                            namespace,
                            current_task,
                            _chunk_id(owner, group),
                            group,
                        )
                    except Exception as exc:
                        current_stop_lanes.set()
                        raise RuntimeError(
                            f"worker {owner} chunk {_chunk_id(owner, group)} did not commit"
                        ) from exc
                    with current_append_lock:
                        if payload.get("host") is not None:
                            _verify_worker_payload(payload, plan)
                        _append_events(current_events_path, payload["events"])

            # Each owner has one local submission lane. This prevents two
            # chunks for the same sealed worker from running concurrently.
            # Chunk results are appended immediately, and a lost remote call
            # remains pending for a clean resume instead of becoming a fake
            # scientific attempt.
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        run_owner,
                        owner,
                        groups,
                        task_id,
                        events_path,
                        append_lock,
                        stop_lanes,
                    )
                    for owner in range(workers)
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            report = completeness(jobs, _read_jsonl(events_path))
            _write_json(out / "cells" / task_id / "completeness.json", report)
            if report["all_calls_attempted"]:
                break
        report = completeness(jobs, _read_jsonl(events_path))
        _write_json(out / "cells" / task_id / "completeness.json", report)

    reports = [_load(out / "cells" / task / "completeness.json") for task in requested_tasks]
    completed_tasks = [
        task
        for task in TASK_ORDER
        if (out / "cells" / task / "completeness.json").exists()
        and _load(out / "cells" / task / "completeness.json")["complete"]
    ]
    summary = {
        "model_id": plan["model_id"],
        "plan_sha256": plan["plan_sha256"],
        "dataset_order": list(TASK_ORDER),
        "requested_datasets": list(requested_tasks),
        "completed_datasets": completed_tasks,
        "attempted_datasets": [
            task
            for task, report in zip(requested_tasks, reports, strict=True)
            if report["all_calls_attempted"]
        ],
        "incomplete_datasets": [
            task
            for task, report in zip(requested_tasks, reports, strict=True)
            if not report["complete"]
        ],
        "requested_scope_complete": all(report["complete"] for report in reports),
        "full_model_complete": completed_tasks == list(TASK_ORDER),
        "first_attempts": sum(int(report["first_attempts"]) for report in reports),
        "retry_attempts": sum(int(report["retry_attempts"]) for report in reports),
        "terminal_malformed_outputs": sum(
            int(report["terminal_malformed_outputs"]) for report in reports
        ),
        "unresolved_outputs": sum(int(report["unresolved_outputs"]) for report in reports),
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    _write_json(out / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def _verify_modal_environment(plan: dict[str, Any]) -> dict[str, Any]:
    """Verify all released checkpoints and the local sealed frames before GPU work."""
    _local_imports()
    from freezeb import extension_screen as m1

    grid = _load(FULL_GRID_ROOT / "FULL_GRID_LOCK.json")
    expected = m1.expected_checkpoint_inventory(LOCK_ROOT, FREEZE_B_CHECKPOINTS)
    staging = _load(LOCK_ROOT / "evidence/staging/report.json")
    excluded = {
        str(row["model_id"]): list(row["excluded_alternative_files"]) for row in staging["models"]
    }
    checkpoints: list[dict[str, Any]] = []
    for model in grid["models"]:
        model_id = str(model["model_id"])
        wanted = expected.get(model_id)
        if wanted is None:
            raise RuntimeError(f"no sealed checkpoint inventory for {model_id}")
        actual = inspect_checkpoint.remote(
            model_id,
            str(model["revision"]),
            excluded.get(model_id, []),
        )
        for key in ("model_id", "revision", "file_count", "total_bytes"):
            if actual.get(key) != wanted.get(key):
                raise RuntimeError(f"{model_id} checkpoint {key} differs from the sealed inventory")
        checkpoints.append(actual)
    return {
        "workspace_profile": os.environ.get("MODAL_PROFILE"),
        "model_volume": "ieee-bigdata-freezeb-models",
        "results_volume": "ieee-bigdata-extension-b3-results",
        "checkpoints": checkpoints,
        "frames": [
            {
                "task_id": task,
                "path": str(FRAME_ROOT / task / "confirm.jsonl"),
                "rows": N_ITEMS,
                "sha256": _sha256_file(FRAME_ROOT / task / "confirm.jsonl"),
            }
            for task in TASK_ORDER
        ],
        "selected_model": plan["model_id"],
        "verified_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


@app.local_entrypoint()
def main(
    model: str,
    execute: bool = False,
    preflight_only: bool = False,
    modal_profile: str = "",
    chunk_size: int = 32,
    workers: int = 4,
    dataset_count: int = 4,
    execution_namespace: str = "",
) -> None:
    if not 1 <= dataset_count <= len(TASK_ORDER):
        raise ValueError("dataset_count must be from 1 through 4")
    plan, jobs_by_task = build_plan(model, chunk_size, workers)
    print(json.dumps({**plan, "paid_execution": execute}, indent=2, sort_keys=True), flush=True)
    if not execute and not preflight_only:
        print("CPU dry-run complete: no Modal function ran and no model was called", flush=True)
        return
    active = os.environ.get("MODAL_PROFILE", "").strip()
    if not modal_profile.strip() or modal_profile.strip() != active:
        raise ValueError("Modal B3 access requires --modal-profile matching MODAL_PROFILE")
    namespace = execution_namespace.strip() or str(plan["plan_sha256"][:16])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", namespace) is None:
        raise ValueError("execution namespace must be 1-64 safe filename characters")
    if preflight_only:
        print(json.dumps(_verify_modal_environment(plan), indent=2, sort_keys=True), flush=True)
        print("Modal preflight complete: no GPU worker or model was called", flush=True)
        return
    out = LOCAL_ROOT / "extension-b3" / model.replace("/", "--")
    saved = out / "plan.json"
    if saved.exists() and _load(saved).get("plan_sha256") != plan["plan_sha256"]:
        raise RuntimeError("resume refused because the saved B3 plan differs")
    if not saved.exists():
        _write_json(saved, plan)
    lock = _acquire_lock(out, active)
    try:
        preflight = _verify_modal_environment(plan)
        preflight["execution_namespace"] = namespace
        _write_json(out / "modal-preflight.json", preflight)
        _run(plan, jobs_by_task, out, dataset_count, namespace)
    finally:
        lock.close()
