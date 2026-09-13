"""Run one implemented sealed Freeze B E2 cell; plan-only unless --execute is explicit.

Each implemented cell has 768 joint calls, 12,288 exact-neighbor target-only calls,
and 4,096 solo calls. Four independent one-A100 workers may run concurrently after a
synthetic equivalence smoke. Execution block ownership is fixed before output, and
all worker time is summed in the study meter.

Plan mode starts nothing::

    uv run python modal_freezeb_e2.py --cell granite-civil

Paid mode additionally requires an explicit Modal profile::

    MODAL_PROFILE=<profile> modal run modal_freezeb_e2.py --execute \
      --modal-profile <profile> --cell granite-civil
"""

import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, TextIO, cast

import modal

APP_NAME = "ieee-bigdata-freezeb-e2"
DEFAULT_CELL = "qwen-multinli"
LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/study-120h")
MODEL_CACHE = "/models"
RESULTS_ROOT = "/results"
GPU_TYPE = "A100-40GB"
SOURCE_ROOT = LOCAL_ROOT / "freeze-b-work/20260912-cpu-draft"
SEAL_ROOT = LOCAL_ROOT / "freeze/20260912T222618Z-B1"
LOCK_ROOT = LOCAL_ROOT / "freeze/20260913T022442Z-B2"
ADMISSION_LOCK = LOCK_ROOT / "ADMISSION_LOCK.json"
ADMISSION_MANIFEST = LOCK_ROOT / "MANIFEST.json"
RUNTIME_LOCK = LOCAL_ROOT / "runtime-lock/20260913T004054Z-BRT1/LEDGER.json"
CHECKPOINT_LOCK = (
    LOCAL_ROOT / "runtime-lock/20260913T004054Z-BRT1/benchmark/checkpoint-manifests.json"
)

model_volume = modal.Volume.from_name("ieee-bigdata-freezeb-models")
results_volume = modal.Volume.from_name("ieee-bigdata-freezeb-e2-results", create_if_missing=True)
support_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==1.31.0"
)
runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface-hub==1.31.0", "numpy==2.2.6", "scipy==1.16.3", "vllm==0.28.0")
    .env(
        {
            "HF_HOME": MODEL_CACHE,
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
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


def _validated_barrier(path: Path, plan: dict[str, Any]) -> dict[str, Any] | None:
    """Load a continuation barrier only when it authorizes this exact plan."""
    if not path.exists():
        return None
    barrier = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if barrier.get("plan_sha256") != plan.get("plan_sha256"):
        raise RuntimeError("saved reforecast belongs to a different E2 execution plan")
    if barrier.get("cell_key") != plan.get("cell_key"):
        raise RuntimeError("saved reforecast belongs to a different E2 cell")
    if barrier.get("may_continue") is not True:
        raise RuntimeError("saved first-1,000 reforecast does not authorize continuation")
    return barrier


def _session_meter_delta(
    payload: dict[str, Any], session_elapsed: dict[str, float], calibration: float
) -> tuple[str, float, float]:
    """Return newly observed billed time for the payload's original GPU session."""
    session = str(payload["meter_session_id"])
    elapsed = float(payload["meter_session_elapsed_seconds"])
    previous = session_elapsed.get(session, 0.0)
    delta = max(0.0, elapsed - previous) * calibration
    return session, elapsed, delta


@app.function(image=support_image, volumes={MODEL_CACHE: model_volume}, timeout=3600)
def inspect_checkpoint(model_id: str, revision: str) -> dict[str, Any]:
    """Hash the already materialized checkpoint without allocating a GPU."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=model_id, revision=revision, cache_dir=MODEL_CACHE, local_files_only=True
        )
    )
    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {"model_id": model_id, "revision": revision, "files": files}


@app.function(image=support_image, volumes={RESULTS_ROOT: results_volume}, timeout=600)
def list_committed(namespace: str) -> list[str]:
    results_volume.reload()
    directory = Path(RESULTS_ROOT) / namespace
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


@app.function(image=support_image, volumes={RESULTS_ROOT: results_volume}, timeout=600)
def fetch_chunk(namespace: str, chunk_id: str) -> dict[str, Any]:
    results_volume.reload()
    path = Path(RESULTS_ROOT) / namespace / f"{chunk_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"remote E2 chunk {chunk_id} is missing")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


@app.cls(
    image=runtime_image,
    volumes={MODEL_CACHE: model_volume, RESULTS_ROOT: results_volume},
    gpu=GPU_TYPE,
    max_containers=1,
    scaledown_window=60,
    startup_timeout=1800,
    timeout=10800,
)
class E2Worker:
    """One pinned vLLM process that owns a fixed subset of execution blocks."""

    worker_index: int = modal.parameter()
    worker_count: int = modal.parameter()
    model_id: str = modal.parameter()
    task_id: str = modal.parameter()
    model_revision: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        _remote_imports()
        from freezeb.runtime_benchmark import DECODING, policy_for
        from step1.backends import VLLMBackend, host_facts

        policy = policy_for(self.model_id)
        if policy.revision != self.model_revision:
            raise ValueError("selected E2 checkpoint differs from the runtime lock")
        self.session_started = time.monotonic()
        self.session_id = uuid.uuid4().hex
        self.backend = VLLMBackend(
            policy.model_id,
            self.model_revision,
            self.model_revision,
            {
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "max_model_len": policy.max_model_len,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 32,
            },
            DECODING,
            enable_thinking=policy.enable_thinking,
        )
        self.load_seconds = time.monotonic() - self.session_started
        self.host = host_facts()

    @modal.method()
    def smoke(self) -> dict[str, Any]:
        """Generate one synthetic response used only to compare worker equivalence."""
        from freezeb.prompts import build_messages
        from step0.hashing import sha256_text
        from step1.backends import GenRequest

        doc_id = "synthetic-equivalence-item"
        messages = build_messages(
            self.task_id,
            [(doc_id, "A person is outdoors. A person is outside.")],
            [doc_id],
        )
        started = time.monotonic()
        result = self.backend.generate(
            [
                GenRequest(
                    call_id=f"e2-smoke-worker-{self.worker_index}",
                    condition="synthetic_smoke",
                    target_ids=(doc_id,),
                    messages=messages,
                    max_tokens=84,
                )
            ]
        )[0]
        return {
            "worker_index": self.worker_index,
            "worker_count": self.worker_count,
            "worker_session_id": self.session_id,
            "session_elapsed_seconds": time.monotonic() - self.session_started,
            "smoke_wall_seconds": time.monotonic() - started,
            "response_sha256": sha256_text(result.text),
            "finish_reason": result.finish_reason,
            "host": self.host,
            "backend": self.backend.describe(),
        }

    @modal.method()
    def run_chunk(
        self, namespace: str, chunk_id: str, jobs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        from freezeb import e2
        from freezeb.runtime_benchmark import context_fit, policy_for
        from step1.backends import GenRequest

        if not jobs or len(jobs) > 64:
            raise ValueError("an E2 chunk must contain 1 through 64 attempts")
        for job in jobs:
            if job["model_id"] != self.model_id or job["task_id"] != self.task_id:
                raise ValueError("E2 worker refuses a foreign model-task cell")
            if int(job["execution_block"]) % self.worker_count != self.worker_index:
                raise ValueError("an E2 attempt does not belong to this worker")

        path = Path(RESULTS_ROOT) / namespace / f"{chunk_id}.json"
        results_volume.reload()
        if path.exists():
            cached = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
            cached["served_from_cache"] = True
            return cached

        policy = policy_for(self.model_id)
        requests: list[GenRequest] = []
        request_jobs: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for job in jobs:
            prompt_tokens = self.backend.count_tokens(job["messages"])
            fit = context_fit(prompt_tokens, int(job["max_tokens"]), policy.max_model_len)
            if not fit["fits"]:
                event = e2.event_for_result(
                    job,
                    generation_error="prompt and output cap exceed the runtime-locked context",
                    prompt_tokens=prompt_tokens,
                )
                event["status"] = "context_excluded"
                events.append(event)
                continue
            requests.append(
                GenRequest(
                    call_id=str(job["attempt_id"]),
                    condition=str(job["condition"]),
                    target_ids=tuple(str(value) for value in job["target_ids"]),
                    messages=job["messages"],
                    max_tokens=int(job["max_tokens"]),
                )
            )
            request_jobs.append(job)

        started = time.monotonic()
        if requests:
            try:
                generated = self.backend.generate(requests)
                by_id = {result.call_id: result for result in generated}
                for job in request_jobs:
                    result = by_id.get(str(job["attempt_id"]))
                    if result is None:
                        events.append(e2.event_for_result(job, generation_error="missing result"))
                        continue
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
                events.extend(
                    e2.event_for_result(job, generation_error=error) for job in request_jobs
                )

        order = {str(job["attempt_id"]): index for index, job in enumerate(jobs)}
        events.sort(key=lambda event: order[str(event["attempt_id"])])
        for event in events:
            event.update(
                {
                    "worker_index": self.worker_index,
                    "worker_session_id": self.session_id,
                    "worker_load_seconds": self.load_seconds,
                    "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                }
            )
        payload = {
            "chunk_id": chunk_id,
            "worker_index": self.worker_index,
            "worker_session_id": self.session_id,
            "meter_session_id": self.session_id,
            "meter_session_elapsed_seconds": time.monotonic() - self.session_started,
            "load_seconds": self.load_seconds,
            "chunk_wall_seconds": time.monotonic() - started,
            "host": self.host,
            "backend": self.backend.describe(),
            "events": events,
            "served_from_cache": False,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        results_volume.commit()
        return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_events(path: Path, events: list[dict[str, Any]]) -> dict[str, int]:
    stored = {str(row["attempt_id"]): row for row in _read_jsonl(path)}
    fresh: list[dict[str, Any]] = []
    duplicates = 0
    for event in events:
        attempt_id = str(event["attempt_id"])
        prior = stored.get(attempt_id)
        if prior is None:
            stored[attempt_id] = event
            fresh.append(event)
        elif prior == event:
            duplicates += 1
        else:
            raise RuntimeError(f"attempt {attempt_id} has a conflicting local event")
    _append_jsonl(path, fresh)
    return {"appended": len(fresh), "duplicates": duplicates}


def _verify_local_boundary(events: Path, timings: Path, meter_path: Path, state: Path) -> None:
    event_chunks = Counter(str(row["chunk_id"]) for row in _read_jsonl(events))
    timing_chunks = Counter(str(row["chunk_id"]) for row in _read_jsonl(timings))
    meter_chunks = Counter(
        str(row["chunk_id"])
        for row in _read_jsonl(meter_path)
        if row.get("kind") in {"scientific_chunk", "scientific_payload"}
    )
    if event_chunks != timing_chunks:
        raise RuntimeError("local E2 event and timing ledgers disagree after an interrupted write")
    if not set(event_chunks) <= set(meter_chunks) or any(
        meter_chunks[chunk_id] != 1 for chunk_id in event_chunks
    ):
        raise RuntimeError("local E2 chunk and meter ledgers disagree after an interrupted write")
    meters = _read_jsonl(meter_path)
    if meters and not state.exists():
        raise RuntimeError("E2 meter state is missing after metered work")
    if state.exists() and not meters:
        raise RuntimeError("E2 meter state exists without metered work")
    if meters:
        saved = json.loads(state.read_text(encoding="utf-8"))
        if (
            abs(float(saved["cumulative_a100_hours"]) - float(meters[-1]["cumulative_a100_hours"]))
            > 1e-9
        ):
            raise RuntimeError("E2 meter state differs from the final meter record")


def _chunk_id(owner: int, jobs: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(
        "|".join(str(job["attempt_id"]) for job in jobs).encode("utf-8")
    ).hexdigest()
    return f"w{owner}-{digest[:20]}"


def _acquire_run_lock(out: Path, profile: str) -> TextIO:
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / ".active-run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another E2 process is already using {out}") from None
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "modal_profile": profile}) + "\n")
    handle.flush()
    return handle


def _verify_profile(requested: str, environment: str | None) -> str:
    active = (environment or "").strip()
    if not requested.strip() or not active:
        raise ValueError("paid E2 execution requires both --modal-profile and MODAL_PROFILE")
    if requested.strip() != active:
        raise ValueError(f"MODAL_PROFILE is {active!r}, not {requested.strip()!r}")
    return active


def build_plan(chunk_size: int = 32, workers: int = 4, cell: str = DEFAULT_CELL) -> dict[str, Any]:
    """Verify every sealed input and render the complete call plan without Modal."""
    _local_imports()
    from freezeb import e2
    from step0.hashing import sha256_json

    spec = e2.cell_spec(cell)
    if not 1 <= workers <= 4:
        raise ValueError("workers must be from 1 through 4")
    if not 1 <= chunk_size <= 64:
        raise ValueError("chunk_size must be from 1 through 64")
    manifest = json.loads(ADMISSION_MANIFEST.read_text(encoding="utf-8"))
    admission_hash = _sha256_file(ADMISSION_LOCK)
    if admission_hash != manifest["admission_lock_sha256"]:
        raise ValueError("the part-2 admission lock differs from its seal")
    admission = json.loads(ADMISSION_LOCK.read_text(encoding="utf-8"))
    rows = e2.verify_confirm_frame(SOURCE_ROOT, SEAL_ROOT, admission, spec=spec)
    jobs = e2.render_jobs(rows, e2.planned_jobs(rows, spec=spec), spec=spec)
    core = {
        "schema_version": "freezeb-e2-run-plan-v1",
        "cell_key": spec.cell_key,
        "admission_lock_sha256": admission_hash,
        "call_ids_sha256": sha256_json([job["call_id"] for job in jobs]),
        "prompt_hashes_sha256": sha256_json([job["prompt_sha256"] for job in jobs]),
        "calls_by_condition": dict(Counter(str(job["condition"]) for job in jobs)),
        "total_calls": len(jobs),
        "execution_blocks": 8,
        "worker_count": workers,
        "block_owners": {str(block): block % workers for block in range(8)},
        "chunk_size": chunk_size,
        "base_forecast_a100_hours": spec.base_forecast_a100_hours,
        "guarded_forecast_a100_hours": 1.6 * spec.base_forecast_a100_hours,
        "prior_cumulative_a100_hours": spec.prior_cumulative_a100_hours,
        "first_reforecast_attempts": e2.FIRST_REFORECAST_CALLS,
    }
    plan_hash = sha256_json(core)
    return {**core, "plan_sha256": plan_hash, "results_namespace": plan_hash[:16]}


def _jobs(cell: str) -> list[dict[str, Any]]:
    _local_imports()
    from freezeb import e2

    spec = e2.cell_spec(cell)
    admission = json.loads(ADMISSION_LOCK.read_text(encoding="utf-8"))
    rows = e2.verify_confirm_frame(SOURCE_ROOT, SEAL_ROOT, admission, spec=spec)
    return e2.render_jobs(rows, e2.planned_jobs(rows, spec=spec), spec=spec)


@app.local_entrypoint()
def main(
    execute: bool = False,
    modal_profile: str = "",
    chunk_size: int = 32,
    workers: int = 4,
    cell: str = DEFAULT_CELL,
) -> None:
    _local_imports()
    from freezeb import e2, e2_amendment
    from freezeb.meter import may_start, must_stop
    from step1.ledger import append_run, meter

    spec = e2.cell_spec(cell)
    plan = build_plan(chunk_size, workers, cell)
    print(json.dumps({**plan, "paid_execution": execute}, indent=2, sort_keys=True), flush=True)
    if not execute:
        print("plan only: no Modal function ran and no GPU started", flush=True)
        return

    profile = _verify_profile(modal_profile, os.environ.get("MODAL_PROFILE"))
    jobs = _jobs(cell)
    out = LOCAL_ROOT / "e2-out" / spec.output_slug
    saved_plan = out / "plan.json"
    if saved_plan.exists():
        saved = json.loads(saved_plan.read_text(encoding="utf-8"))
        if saved.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("resume refused because the saved E2 plan differs")
    else:
        _write_json(saved_plan, plan)
    run_lock = _acquire_run_lock(out, profile)

    events_path = out / "events.jsonl"
    timings_path = out / "timing.jsonl"
    meter_path = out / "meter.jsonl"
    state_path = out / "meter-state.json"
    barrier_path = out / "reforecast-1000.json"
    _verify_local_boundary(events_path, timings_path, meter_path, state_path)

    cumulative = (
        float(json.loads(state_path.read_text(encoding="utf-8"))["cumulative_a100_hours"])
        if state_path.exists()
        else spec.prior_cumulative_a100_hours
    )
    decision = may_start(cumulative, spec.base_forecast_a100_hours)
    if not decision.start:
        run_lock.close()
        print(json.dumps({"started": False, "reason": decision.reason}, indent=2))
        return

    checkpoint = e2.verify_checkpoint(
        inspect_checkpoint.remote(spec.model_id, spec.model_revision),
        json.loads(CHECKPOINT_LOCK.read_text(encoding="utf-8")),
        spec=spec,
    )
    _write_json(out / "checkpoint-verification.json", checkpoint)
    amendment_path, amendment = e2_amendment.ensure(
        LOCAL_ROOT / "amendments",
        str(plan["admission_lock_sha256"]),
        workers=workers,
        plan_sha256=str(plan["plan_sha256"]),
        amendment_id=f"freezeb-e2-{spec.name}-concurrency",
        scope=f"sealed E2 cell {spec.cell_key} only",
    )

    lock_record = json.loads(RUNTIME_LOCK.read_text(encoding="utf-8"))
    calibration = float(lock_record["measurement_summary"]["meter_calibration_factor"])
    state_mutex = threading.Lock()
    session_elapsed = (
        cast(
            dict[str, float],
            json.loads(state_path.read_text(encoding="utf-8")).get("session_elapsed_seconds", {}),
        )
        if state_path.exists()
        else {}
    )
    workers_by_owner = {
        owner: E2Worker(  # type: ignore[call-arg]
            worker_index=owner,
            worker_count=workers,
            model_id=spec.model_id,
            task_id=spec.task_id,
            model_revision=spec.model_revision,
        )
        for owner in range(workers)
    }
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    def save_state() -> None:
        _write_json(
            state_path,
            {
                "cumulative_a100_hours": cumulative,
                "stop_reached": must_stop(cumulative),
                "max_concurrent_workers": workers,
                "session_elapsed_seconds": session_elapsed,
                "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            },
        )

    # No real E2 event exists yet: compare every worker on one synthetic request before release.
    smoke_passed = amendment.get("equivalence_smoke", {}).get("status") == "passed"
    if not _read_jsonl(events_path) and not smoke_passed:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            smoke_rows = list(
                pool.map(lambda owner: workers_by_owner[owner].smoke.remote(), range(workers))
            )
        for row in smoke_rows:
            session = str(row["worker_session_id"])
            elapsed = float(row["session_elapsed_seconds"])
            delta = max(0.0, elapsed - session_elapsed.get(session, 0.0)) * calibration
            session_elapsed[session] = elapsed
            cumulative += delta / 3600.0
            append_run(
                meter_path,
                {
                    "run_id": run_id,
                    "worker_id": f"e2-worker-{row['worker_index']}",
                    "model_id": spec.model_id,
                    "gpu_name": str((row.get("host") or {}).get("gpu_name") or GPU_TYPE),
                    "gpu_seconds_billed": delta,
                    "chunk_id": f"smoke-{row['worker_index']}-{run_id}",
                    "kind": "equivalence_smoke",
                    "cumulative_a100_hours": cumulative,
                },
            )
            save_state()
        response_hashes = {str(row["response_sha256"]) for row in smoke_rows}
        try:
            for row in smoke_rows:
                e2.verify_worker(row.get("host"), row.get("backend"), spec=spec)
            if len(response_hashes) != 1:
                raise RuntimeError("E2 workers returned different deterministic smoke responses")
        except Exception as exc:
            _write_json(
                out / "quarantine" / f"smoke-{run_id}.json",
                {
                    "reason": f"{type(exc).__name__}: {exc}",
                    "metered_before_quarantine": True,
                    "payloads": smoke_rows,
                },
            )
            run_lock.close()
            raise
        smoke_report = {
            "status": "passed",
            "synthetic_only": True,
            "workers_checked": workers,
            "response_sha256": next(iter(response_hashes)),
            "worker_sessions": [str(row["worker_session_id"]) for row in smoke_rows],
            "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        amendment = e2_amendment.complete_smoke(amendment_path, amendment, smoke_report)
        _write_json(out / "equivalence-smoke.json", smoke_report)
    elif _read_jsonl(events_path) and not smoke_passed:
        run_lock.close()
        raise RuntimeError("E2 output exists but its required worker smoke is not recorded")

    if must_stop(cumulative):
        run_lock.close()
        raise RuntimeError("the cumulative 150-hour stop was reached during E2 setup")

    committed = set(list_committed.remote(str(plan["results_namespace"])))
    timing_rows = _read_jsonl(timings_path)
    metered_chunks = {
        str(row["chunk_id"]): row
        for row in _read_jsonl(meter_path)
        if row.get("kind") in {"scientific_chunk", "scientific_payload"}
    }
    attempt_count = len(timing_rows)
    max_chunk_hours = 0.0
    stopped: str | None = None

    def record(group: list[dict[str, Any]], payload: dict[str, Any], invoked: float) -> None:
        nonlocal cumulative, attempt_count, max_chunk_hours, stopped
        chunk_id = str(payload["chunk_id"])
        returned = cast(list[dict[str, Any]], payload["events"])
        with state_mutex:
            session, elapsed, delta = _session_meter_delta(payload, session_elapsed, calibration)
            session_elapsed[session] = max(elapsed, session_elapsed.get(session, 0.0))
            prior_meter = metered_chunks.get(chunk_id)
            attributed_delta = (
                float(prior_meter["gpu_seconds_billed"]) if prior_meter is not None else delta
            )
            if prior_meter is None:
                cumulative += delta / 3600.0
                meter_row = {
                    "run_id": run_id,
                    "worker_id": f"e2-worker-{payload['worker_index']}",
                    "model_id": spec.model_id,
                    "gpu_name": str((payload.get("host") or {}).get("gpu_name") or GPU_TYPE),
                    "gpu_seconds_billed": delta,
                    "chunk_id": chunk_id,
                    "kind": "scientific_payload",
                    "served_from_cache": bool(payload.get("served_from_cache")),
                    "local_wall_seconds": time.monotonic() - invoked,
                    "cumulative_a100_hours": cumulative,
                }
                append_run(meter_path, meter_row)
                metered_chunks[chunk_id] = meter_row
            max_chunk_hours = max(max_chunk_hours, attributed_delta / 3600.0)
            save_state()
            try:
                e2.verify_worker(payload.get("host"), payload.get("backend"), spec=spec)
                if [row["attempt_id"] for row in returned] != [job["attempt_id"] for job in group]:
                    raise RuntimeError("remote E2 chunk returned a different attempt order")
            except Exception as exc:
                _write_json(
                    out / "quarantine" / f"{chunk_id}-{run_id}.json",
                    {
                        "reason": f"{type(exc).__name__}: {exc}",
                        "metered_before_quarantine": True,
                        "payload": payload,
                    },
                )
                raise
            _append_events(
                events_path, [{**row, "chunk_id": chunk_id, "run_id": run_id} for row in returned]
            )
            records = e2.timing_records(
                group,
                returned,
                chunk_id=chunk_id,
                metered_a100_seconds=attributed_delta,
                cumulative_a100_hours=cumulative,
            )
            _append_jsonl(timings_path, records)
            timing_rows.extend(records)
            attempt_count += len(records)
            committed.add(chunk_id)
            if must_stop(cumulative) or cumulative + max_chunk_hours * workers >= 150.0:
                stopped = "the conservative cumulative 150-hour stop was reached"
        print(f"{attempt_count} attempts; {cumulative:.4f} cumulative A100-hours", flush=True)

    def invoke(owner: int, group: list[dict[str, Any]]) -> None:
        chunk_id = _chunk_id(owner, group)
        invoked = time.monotonic()
        if chunk_id in committed:
            payload = fetch_chunk.remote(str(plan["results_namespace"]), chunk_id)
            payload["served_from_cache"] = True
        else:
            payload = workers_by_owner[owner].run_chunk.remote(
                str(plan["results_namespace"]), chunk_id, group
            )
        record(group, payload, invoked)

    def queue_for_owner(owner: int) -> list[dict[str, Any]]:
        work = e2.pending_work(jobs, _read_jsonl(events_path))
        pending = work["retries"] + work["first_responses"]
        return [job for job in pending if int(job["execution_block"]) % workers == owner]

    def run_owner(owner: int, *, limit: int | None = None) -> None:
        nonlocal stopped
        while stopped is None:
            queue = queue_for_owner(owner)
            if not queue:
                return
            if limit is not None and attempt_count >= limit:
                return
            take = chunk_size if limit is None else min(chunk_size, limit - attempt_count)
            if take <= 0:
                return
            invoke(owner, queue[:take])

    failure: str | None = None
    try:
        barrier = _validated_barrier(barrier_path, plan)
        if barrier is None:
            limit = e2.FIRST_REFORECAST_CALLS
            while stopped is None and not barrier_path.exists():
                run_owner(0, limit=limit)
                remaining = e2.remaining_counts(jobs, _read_jsonl(events_path))
                conditions_measured = set(remaining) <= {
                    str(row["condition"]) for row in timing_rows[:limit]
                }
                if len(timing_rows) >= e2.FIRST_REFORECAST_CALLS and conditions_measured:
                    report = e2.reforecast(
                        timing_rows[:limit],
                        remaining,
                        cumulative_a100_hours=cumulative,
                        workers=1,
                    )
                    report = {
                        **report,
                        "cell_key": plan["cell_key"],
                        "plan_sha256": plan["plan_sha256"],
                    }
                    _write_json(barrier_path, report)
                    _write_json(out / "reforecast-latest.json", report)
                    if not report["may_continue"]:
                        stopped = (
                            "the first-1,000 reforecast refused continuation: "
                            f"{report['decision_reason']}"
                        )
                    break
                if limit >= e2.BARRIER_CALL_CAP:
                    raise RuntimeError(
                        "3,000 E2 attempts did not measure every remaining condition"
                    )
                limit += chunk_size
        if stopped is None and barrier is None and not barrier_path.exists():
            raise RuntimeError(
                "scientific output remains blocked without the first-1,000 reforecast"
            )
        if stopped is None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_owner, owner) for owner in range(workers)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        stopped = stopped or failure

    remaining = e2.remaining_counts(jobs, _read_jsonl(events_path))
    if timing_rows and not (set(remaining) - {str(row["condition"]) for row in timing_rows}):
        _write_json(
            out / "reforecast-final.json",
            e2.reforecast(
                timing_rows,
                remaining,
                cumulative_a100_hours=cumulative,
                workers=workers,
            ),
        )
    complete = e2.completeness(jobs, _read_jsonl(events_path))
    summary = {
        "run_id": run_id,
        "modal_profile": profile,
        "plan_sha256": plan["plan_sha256"],
        "results_namespace": plan["results_namespace"],
        "concurrency_amendment": {
            "path": str(amendment_path),
            "content_hash": amendment["content_hash"],
        },
        "checkpoint_verification": checkpoint,
        "cumulative_a100_hours": cumulative,
        "meter": meter(meter_path),
        "completeness": complete,
        "stopped_early": stopped,
        "failure": failure,
        "written_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    _write_json(out / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    run_lock.close()
    if stopped or not complete["complete"]:
        raise SystemExit(f"E2 cell is incomplete: {stopped or complete}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default=DEFAULT_CELL)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(chunk_size=args.chunk_size, workers=args.workers, cell=args.cell),
            indent=2,
            sort_keys=True,
        )
    )
    print("plan only: no Modal function ran and no GPU started")
