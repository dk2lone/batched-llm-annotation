"""Run the historical Qwen3-8B/Civil Comments Step 2 replication bridge on Modal.

The bridge finishes the sealed Step 0 `qwen_civil` cell: 384 joint calls, 6,144
exact-context calls, and 6,144 filler calls. It reuses the 2,048 Step 1 solo first
responses and never reruns them. It is not Freeze B confirmation work.

Two A100 workers run at the same time. One worker owns one whole execution block, so the
frozen within-block order and the fresh serving session per block are both preserved, and
A100 hours are summed over workers. That concurrency deviates from the sealed sentence
"one NVIDIA A100, one at a time", so the runner records the `step2-bridge-concurrency`
amendment before the first paid call and refuses to run without it.

Plan mode runs locally, starts nothing, and costs nothing::

    python modal_bridge.py

Paid mode needs an explicit flag and an explicit Modal profile::

    MODAL_PROFILE=<profile> modal run modal_bridge.py --execute --modal-profile <profile>

The run is resumable. Every chunk id is deterministic and every committed chunk is
recovered before any generation, so a crash between the remote commit and the local append
never pays for the same call twice.
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
from pathlib import Path
from typing import Any, TextIO, cast

import modal

APP_NAME = "ieee-bigdata-step2-bridge"
LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/study-120h")
MODEL_CACHE = "/models"
RESULTS_ROOT = "/results"
GPU_TYPE = "A100-40GB"
OUTPUT_ROOT = LOCAL_ROOT / "step2-out" / "bridge"

#: The bridge never measures more than this before the barrier reforecast is written.
BARRIER_CALL_CAP = 3000

# The Freeze B volume already holds the pinned Qwen3-8B snapshot in the selected workspace.
# It is never created here: a missing volume means the wrong workspace, and that must fail.
model_volume = modal.Volume.from_name("ieee-bigdata-freezeb-models")
results_volume = modal.Volume.from_name("ieee-bigdata-step2-bridge-results", create_if_missing=True)
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
    .add_local_dir(LOCAL_ROOT / "freeze", remote_path=str(REMOTE_ROOT / "freeze"))
)
app = modal.App(APP_NAME)


def _remote_imports() -> None:
    sys.path.insert(0, str(REMOTE_ROOT / "src"))


def _local_imports() -> None:
    sys.path.insert(0, str(LOCAL_ROOT / "src"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@app.function(
    image=support_image,
    volumes={MODEL_CACHE: model_volume},
    cpu=4,
    memory=8192,
    timeout=3600,
)
def inspect_checkpoint(
    model_id: str, revision: str, hash_weights: bool = True
) -> dict[str, Any]:
    """Describe the pinned snapshot already on the volume. Download nothing, use no GPU."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=model_id, revision=revision, cache_dir=MODEL_CACHE, local_files_only=True
        )
    )
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        name = str(path.relative_to(root))
        entry: dict[str, Any] = {"path": name, "bytes": path.stat().st_size}
        if hash_weights or not name.endswith(".safetensors"):
            entry["sha256"] = _sha256_file(path)
        files.append(entry)
    return {"model_id": model_id, "revision": revision, "files": files}


@app.function(image=support_image, volumes={RESULTS_ROOT: results_volume}, timeout=600)
def list_committed(namespace: str) -> list[str]:
    """Chunk ids this plan already committed remotely, whatever local run produced them."""
    results_volume.reload()
    directory = Path(RESULTS_ROOT) / namespace
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


@app.function(image=support_image, volumes={RESULTS_ROOT: results_volume}, timeout=900)
def fetch_chunk(namespace: str, chunk_id: str) -> dict[str, Any]:
    """Return one committed chunk payload without starting a GPU."""
    results_volume.reload()
    path = Path(RESULTS_ROOT) / namespace / f"{chunk_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{chunk_id} is not committed under namespace {namespace}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


@app.function(
    image=runtime_image, volumes={MODEL_CACHE: model_volume}, cpu=4, memory=16384, timeout=3600
)
def recheck_preflight() -> dict[str, Any]:
    """Recount every frozen prompt of the cell with the pinned tokenizer. Generate nothing."""
    _remote_imports()
    from step1.backends import PinnedTokenizerCounter
    from step1.frozen import load
    from step1.runner import preflight
    from step2 import bridge

    frozen = load()
    bridge.verify_freeze(frozen)
    spec = frozen.protocol["models"][bridge.MODEL_ID]
    counter = PinnedTokenizerCounter(
        bridge.MODEL_ID,
        str(spec["tokenizer_revision"]),
        cache_dir=MODEL_CACHE,
        enable_thinking=bool(spec["enable_thinking"]),
        local_files_only=True,
    )
    return preflight(
        frozen,
        bridge.CELL_ID,
        bridge.FRAME_TASK_ID,
        counter,
        model_id=bridge.MODEL_ID,
        conditions=("joint", "exact_context", "solo", "filler"),
    )


@app.cls(
    image=runtime_image,
    volumes={MODEL_CACHE: model_volume, RESULTS_ROOT: results_volume},
    gpu=GPU_TYPE,
    max_containers=1,
    scaledown_window=30,
    startup_timeout=1800,
    timeout=10800,
)
class BridgeWorker:
    """One engine process per frozen execution block.

    The freeze requires a fresh serving session for every execution block, so the block is a
    class parameter: Modal gives each block its own container and therefore its own engine.
    Two blocks may run at the same time; one block is never split across two workers.
    """

    execution_block: int = modal.parameter()

    @modal.enter()
    def load_model(self) -> None:
        _remote_imports()
        from step1.backends import VLLMBackend, host_facts
        from step1.frozen import load
        from step1.plan import doc_texts
        from step2 import bridge

        started = time.monotonic()
        self.session_started = started
        self.session_id = uuid.uuid4().hex
        self.frozen = load()
        bridge.verify_freeze(self.frozen)
        spec = self.frozen.protocol["models"][bridge.MODEL_ID]
        self.backend = VLLMBackend(
            bridge.MODEL_ID,
            str(spec["revision"]),
            str(spec["tokenizer_revision"]),
            self.frozen.serving,
            self.frozen.decoding,
            enable_thinking=bool(spec["enable_thinking"]),
        )
        self.load_seconds = time.monotonic() - started
        self.host = host_facts()
        self.texts = doc_texts(self.frozen, bridge.FRAME_TASK_ID)
        items = self.frozen.items(bridge.FRAME_TASK_ID)
        self.references = {item.doc_id: item.reference for item in items}
        self.strata = {item.doc_id: item.length_stratum for item in items}

    @modal.method()
    def run_chunk(
        self, namespace: str, run_id: str, chunk_id: str, jobs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate one chunk of frozen calls and return append-ready rows and timing."""
        from step1.backends import GenRequest
        from step1.ledger import RunMeta, rows_for
        from step1.plan import build_call
        from step2 import bridge

        if not jobs:
            raise ValueError("a chunk holds at least one call")
        for job in jobs:
            if str(job["condition"]) not in bridge.BRIDGE_CONDITIONS:
                raise ValueError(f"the bridge refuses condition {job['condition']}")
            if str(job["model_id"]) != bridge.MODEL_ID:
                raise ValueError(f"the bridge refuses model {job['model_id']}")
            if int(job["execution_block"]) != self.execution_block:
                raise ValueError("a call does not belong to this execution block")

        path = Path(RESULTS_ROOT) / namespace / f"{chunk_id}.json"
        results_volume.reload()
        if path.exists():
            cached = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
            # The rows keep the session that produced them. Metering uses the live container,
            # which still loaded the model to answer this call.
            cached["served_from_cache"] = True
            cached["meter_session_id"] = self.session_id
            cached["meter_session_elapsed_seconds"] = time.monotonic() - self.session_started
            return cached

        calls = [build_call(self.frozen, job, bridge.FRAME_TASK_ID, self.texts) for job in jobs]
        for call in calls:
            tokens = self.backend.count_tokens(call.messages)
            budget = self.frozen.prompt_token_budget(len(call.target_ids))
            if tokens > budget:
                raise ValueError(f"{call.call_id} needs {tokens} prompt tokens over {budget}")

        started = time.monotonic()
        failed = False
        failure: str | None = None
        try:
            results = self.backend.generate(
                [
                    GenRequest(
                        call_id=call.call_id,
                        condition=call.condition,
                        target_ids=tuple(call.target_ids),
                        messages=call.messages,
                        max_tokens=call.max_tokens,
                    )
                    for call in calls
                ]
            )
        except Exception as exc:  # one linked retry is the frozen response to this
            results = []
            failed = True
            failure = f"{type(exc).__name__}: {exc}"
        chunk_wall_seconds = time.monotonic() - started
        generated = {result.call_id: result for result in results}

        meta = RunMeta(
            run_id=run_id,
            serving_session=self.session_id,
            engine_version=str(self.host["installed_engine_version_string"]),
            step="step2_bridge",
            chunk_id=chunk_id,
            chunk_size=len(calls),
            chunk_seconds=chunk_wall_seconds,
        )
        rows: list[dict[str, Any]] = []
        measurements: list[dict[str, Any]] = []
        for call in calls:
            result = generated.get(call.call_id)
            rows.extend(
                rows_for(
                    self.frozen,
                    call,
                    result,
                    bridge.FRAME_TASK_ID,
                    self.references,
                    self.strata,
                    meta,
                )
            )
            measurements.append(
                {
                    "call_id": call.call_id,
                    "status": "ok" if result is not None else "generation_failed",
                    "error": failure,
                    "finish_reason": result.finish_reason if result else None,
                    "prompt_tokens": result.prompt_tokens if result else None,
                    "output_tokens": result.output_tokens if result else None,
                    "queue_seconds": result.queue_seconds if result else None,
                    "prefill_seconds": result.prefill_seconds if result else None,
                    "decode_seconds": result.decode_seconds if result else None,
                }
            )
        payload = {
            "chunk_id": chunk_id,
            "execution_block": self.execution_block,
            "serving_session": self.session_id,
            "meter_session_id": self.session_id,
            "meter_session_elapsed_seconds": time.monotonic() - self.session_started,
            "load_seconds": self.load_seconds,
            "chunk_wall_seconds": chunk_wall_seconds,
            "generation_failed": failed,
            "host": self.host,
            "backend": self.backend.describe(),
            "rows": rows,
            "measurements": measurements,
            "served_from_cache": False,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        results_volume.commit()
        return payload


# ------------------------------------------------------------------------------------
# Local side
# ------------------------------------------------------------------------------------


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


def _verify_local_commit_boundary(
    ledger_path: Path, timing_path: Path, meter_path: Path, state_path: Path
) -> None:
    """Refuse a resume after a crash split one locally recorded chunk across files."""
    ledger = _read_jsonl(ledger_path)
    timings = _read_jsonl(timing_path)
    meters = _read_jsonl(meter_path)
    ledger_chunks: dict[str, set[str]] = {}
    timing_chunks: dict[str, set[str]] = {}
    for row in ledger:
        ledger_chunks.setdefault(str(row["chunk_id"]), set()).add(str(row["call_id"]))
    for row in timings:
        timing_chunks.setdefault(str(row["chunk_id"]), set()).add(str(row["call_id"]))
    meter_ids = [str(row["chunk_id"]) for row in meters]
    if ledger_chunks != timing_chunks or set(timing_chunks) != set(meter_ids):
        raise RuntimeError(
            "local bridge ledger, timing, and meter disagree after an interrupted write; "
            "refusing paid calls until the committed remote chunks are reconciled"
        )
    if len(meter_ids) != len(set(meter_ids)):
        raise RuntimeError("a bridge chunk was metered twice; refusing paid calls")
    if meters:
        if not state_path.exists():
            raise RuntimeError("bridge meter state is missing after paid calls")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state_hours = float(state["cumulative_a100_hours"])
        last_hours = float(meters[-1]["cumulative_a100_hours"])
        if abs(state_hours - last_hours) > 1e-9:
            raise RuntimeError("bridge meter state disagrees with the last recorded chunk")
    elif state_path.exists():
        raise RuntimeError("bridge meter state exists without metered chunks")


def _chunk_id(block: int, jobs: list[dict[str, Any]]) -> str:
    """A stable id for one chunk, so a resumed run reuses the paid remote output."""
    digest = hashlib.sha256("|".join(str(job["call_id"]) for job in jobs).encode()).hexdigest()
    return f"b{block}-{digest[:16]}"


def _acquire_run_lock(out: Path, modal_profile: str) -> TextIO:
    """Hold an exclusive lock so two processes never append to one bridge ledger."""
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / ".active-run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another process is already running the bridge in {out}") from None
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "modal_profile": modal_profile,
                "acquired_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def build_plan(chunk_size: int = 32, workers: int = 2) -> dict[str, Any]:
    """Verify the seal, the call ids, the counts, and the reused solo rows. Start nothing."""
    _local_imports()
    from step1.frozen import load
    from step2 import bridge

    frozen = load()
    return bridge.plan(
        frozen,
        solo_ledger=LOCAL_ROOT / bridge.SOLO_LEDGER,
        preflight_report=json.loads(
            (LOCAL_ROOT / bridge.PREFLIGHT_REPORT).read_text(encoding="utf-8")
        ),
        runtime_lock=json.loads((LOCAL_ROOT / bridge.RUNTIME_LOCK).read_text(encoding="utf-8")),
        chunk_size=chunk_size,
        max_concurrent_workers=workers,
    )


def _prior_meter_hours() -> float:
    """The cumulative study meter before the bridge, from the sealed screen state."""
    _local_imports()
    from step2 import bridge

    path = LOCAL_ROOT / bridge.PRIOR_METER_STATE
    if not path.exists():
        raise RuntimeError(f"{path} is missing; the cumulative A100 meter is unknown")
    return float(json.loads(path.read_text(encoding="utf-8"))["cumulative_a100_hours"])


@app.local_entrypoint()
def main(
    execute: bool = False,
    modal_profile: str = "",
    chunk_size: int = 32,
    workers: int = 2,
    blocks: str = "",
    recheck: bool = False,
    hash_weights: bool = True,
) -> None:
    """Print the verified plan, then run only when --execute is supplied."""
    _local_imports()
    from step1.frozen import load
    from step1.ledger import append, append_run, meter
    from step2 import amendment, bridge

    plan = build_plan(chunk_size, workers)
    print(json.dumps({**plan, "paid_execution": execute}, indent=2, sort_keys=True), flush=True)
    if not execute:
        print("plan only: no Modal function ran and no GPU started", flush=True)
        return

    profile = bridge.verify_profile(modal_profile, os.environ.get("MODAL_PROFILE"))
    frozen = load()
    planned = bridge.planned_jobs(frozen)
    namespace = str(plan["execution"]["results_namespace"])
    out = OUTPUT_ROOT
    plan_path = out / "plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("resume refused: the saved plan differs from this one")
        if int(saved["execution"]["chunk_size"]) != chunk_size:
            raise RuntimeError(
                f"resume refused: the saved run uses chunk size "
                f"{saved['execution']['chunk_size']}, and a changed chunk size would change "
                "every chunk id and pay for recovered work twice"
            )
    else:
        _write_json(plan_path, plan)
    run_lock = _acquire_run_lock(out, profile)

    ledger_path = out / "calls.jsonl"
    timing_path = out / "timing.jsonl"
    meter_path = out / "meter.jsonl"
    state_path = out / "meter-state.json"
    barrier_path = out / "reforecast-1000.json"
    n_conflicts = bridge.conflicts_recorded(ledger_path)
    if n_conflicts:
        run_lock.close()
        raise SystemExit(
            f"{n_conflicts} conflicting outputs are already recorded beside {ledger_path}. "
            "Investigate them before any further paid call."
        )
    _verify_local_commit_boundary(ledger_path, timing_path, meter_path, state_path)

    # The deviation from the sealed one-A100-at-a-time sentence is recorded before any call.
    amendment_path, amendment_record = amendment.ensure(frozen, max_concurrent_workers=workers)
    print(f"concurrency amendment: {amendment_path}", flush=True)

    lock_record = json.loads((LOCAL_ROOT / bridge.RUNTIME_LOCK).read_text(encoding="utf-8"))
    calibration = float(lock_record["measurement_summary"]["meter_calibration_factor"])
    cumulative = (
        float(json.loads(state_path.read_text(encoding="utf-8"))["cumulative_a100_hours"])
        if state_path.exists()
        else _prior_meter_hours()
    )
    start_decision = bridge.may_start(
        cumulative, float(plan["forecast"]["adjusted_base_a100_hours"])
    )
    if not start_decision.start:
        print(json.dumps({"started": False, "reason": start_decision.reason}, indent=2))
        run_lock.close()
        return

    checkpoint = bridge.verify_checkpoint(
        inspect_checkpoint.remote(bridge.MODEL_ID, bridge.MODEL_REVISION, hash_weights),
        json.loads((LOCAL_ROOT / bridge.CHECKPOINT_MANIFESTS).read_text(encoding="utf-8")),
    )
    _write_json(out / "checkpoint-verification.json", checkpoint)
    print(json.dumps(checkpoint, indent=2, sort_keys=True), flush=True)
    if recheck:
        report = recheck_preflight.remote()
        _write_json(out / "preflight.json", report)
        bridge.verify_preflight(report)

    selected = sorted(
        {int(part) for part in blocks.split(",") if part}
        if blocks
        else set(range(bridge.N_EXECUTION_BLOCKS))
    )
    if not selected or not set(selected) <= set(range(bridge.N_EXECUTION_BLOCKS)):
        run_lock.close()
        last = bridge.N_EXECUTION_BLOCKS - 1
        raise SystemExit(f"--blocks must name execution blocks 0 to {last}")
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    session_elapsed: dict[str, float] = {}
    timing_rows = _read_jsonl(timing_path)
    real_calls = len(timing_rows)
    max_chunk_hours = 0.0
    stopped: str | None = None
    state_lock = threading.Lock()
    committed = set(list_committed.remote(namespace))
    print(f"{len(committed)} chunks are already committed remotely for this plan", flush=True)

    def remaining_blocks(after: int) -> int:
        return len([block for block in selected if block > after])

    def block_forecast_hours() -> float:
        """A block-sized share of the current best forecast, for the frozen start rule."""
        latest = out / "reforecast-latest.json"
        if latest.exists():
            report = json.loads(latest.read_text(encoding="utf-8"))
            hours = float(report["projected_remaining_base_a100_hours"])
        else:
            hours = float(plan["forecast"]["adjusted_base_a100_hours"])
        return max(hours / bridge.N_EXECUTION_BLOCKS, 1e-6)

    def record(
        block: int, group: list[dict[str, Any]], payload: dict[str, Any], invoked: float
    ) -> list[dict[str, Any]]:
        """Append one chunk under the lock and return its ledger rows."""
        nonlocal cumulative, real_calls, max_chunk_hours, stopped
        bridge.verify_worker(payload.get("host"), payload.get("backend"))
        chunk_id = str(payload["chunk_id"])
        bridge.verify_chunk_payload(payload, group, chunk_id)
        rows = list(payload["rows"])
        metered_session = str(payload["meter_session_id"])
        elapsed = float(payload["meter_session_elapsed_seconds"])
        with state_lock:
            previous = session_elapsed.get(metered_session, 0.0)
            delta = max(0.0, elapsed - previous) * calibration
            session_elapsed[metered_session] = max(elapsed, previous)
            saved_rows = append(ledger_path, rows)
            if saved_rows["conflicts_rejected"]:
                stopped = f"conflicting output rejected for chunk {chunk_id}"
                raise RuntimeError(stopped)
            cumulative += delta / 3600.0
            max_chunk_hours = max(max_chunk_hours, delta / 3600.0)
            records = bridge.timing_records(
                group,
                list(payload["measurements"]),
                run_id=run_id,
                chunk_id=chunk_id,
                serving_session=str(payload["serving_session"]),
                execution_block=block,
                chunk_wall_seconds=float(payload["chunk_wall_seconds"]),
                metered_a100_seconds=delta,
                cumulative_a100_hours=cumulative,
            )
            _append_jsonl(timing_path, records)
            timing_rows.extend(records)
            real_calls += len(records)
            committed.add(chunk_id)
            append_run(
                meter_path,
                {
                    "run_id": run_id,
                    "worker_id": f"bridge-block-{block}",
                    "model_id": bridge.MODEL_ID,
                    "gpu_name": str((payload.get("host") or {}).get("gpu_name") or GPU_TYPE),
                    "gpu_seconds_billed": delta,
                    "chunk_id": chunk_id,
                    "served_from_cache": bool(payload.get("served_from_cache")),
                    "local_wall_seconds": time.monotonic() - invoked,
                    "cumulative_a100_hours": cumulative,
                    "accounting_note": (
                        "calibrated container elapsed time; concurrent workers are summed, "
                        "never converted to wall clock"
                    ),
                },
            )
            _write_json(
                state_path,
                {
                    "cumulative_a100_hours": cumulative,
                    "stop_reached": bridge.must_stop(cumulative),
                    "max_concurrent_workers": workers,
                    "session_elapsed_seconds": session_elapsed,
                    "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                },
            )
            if bridge.stop_now(cumulative, in_flight_hours=max_chunk_hours * workers):
                stopped = "the conservative 150 hour cumulative stop was reached"
        print(
            f"block {block}: {real_calls} real calls, {cumulative:.4f} cumulative A100-hours"
            + (" (recovered)" if payload.get("served_from_cache") else ""),
            flush=True,
        )
        return rows

    def run_block(block: int, stop_at_calls: int | None = None) -> None:
        """Run one whole block on one worker. A block is never split across workers."""
        nonlocal stopped
        worker = BridgeWorker(execution_block=block)  # type: ignore[call-arg]
        while stopped is None:
            with state_lock:
                rows_now = _read_jsonl(ledger_path)
            work = bridge.pending_work(planned, rows_now)
            queue = [job for job in work["first_responses"] if job["execution_block"] == block]
            queue += [job for job in work["retries"] if job["execution_block"] == block]
            if not queue:
                break
            with state_lock:
                decision = bridge.may_start(cumulative, block_forecast_hours())
            if not decision.start:
                stopped = f"the frozen start rule refused block {block}: {decision.reason}"
                break
            reached_limit = False
            while queue and stopped is None:
                if stop_at_calls is not None and real_calls >= stop_at_calls:
                    reached_limit = True
                    break
                take = chunk_size
                if stop_at_calls is not None:
                    take = max(1, min(take, stop_at_calls - real_calls))
                group, queue = queue[:take], queue[take:]
                chunk_id = _chunk_id(block, group)
                invoked = time.monotonic()
                if chunk_id in committed:
                    # Recover paid output before any generation, so a crash between the
                    # remote commit and the local append never buys the same call twice.
                    payload = fetch_chunk.remote(namespace, chunk_id)
                    payload["served_from_cache"] = True
                    payload["recovered"] = True
                    # The original container billed this generation. Charge its measured
                    # chunk time once, under an id that can never be charged again.
                    payload["meter_session_id"] = f"recovered-{chunk_id}"
                    payload["meter_session_elapsed_seconds"] = float(
                        payload.get("chunk_wall_seconds", 0.0)
                    )
                else:
                    try:
                        payload = worker.run_chunk.remote(
                            namespace, run_id, chunk_id, group
                        )
                    except Exception as exc:
                        _append_jsonl(
                            out / "worker-failures.jsonl",
                            [
                                {
                                    "run_id": run_id,
                                    "chunk_id": chunk_id,
                                    "execution_block": block,
                                    "call_ids": [str(job["call_id"]) for job in group],
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "observed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                                }
                            ],
                        )
                        raise RuntimeError(
                            f"chunk {chunk_id} failed on the worker; rerun the same "
                            "command to resume and recover any committed output"
                        ) from exc
                rows = record(block, group, payload, invoked)
                scheduled = bridge.retry_jobs_for(group, rows)
                if scheduled:
                    # The seal permits one linked retry after an infrastructure error or a
                    # truncation. Schedule it now, ahead of the untried planned calls.
                    queue = scheduled + queue
            if reached_limit:
                break

        with state_lock:
            _write_json(
                out / "checkpoints" / f"block-{block}.json",
                bridge.checkpoint_by_condition(timing_rows),
            )
            remaining = bridge.remaining_counts(planned, _read_jsonl(ledger_path))
            measured = list(timing_rows)
        if measured and not bridge.missing_rates(measured, remaining):
            _write_json(
                out / "reforecast-latest.json",
                bridge.reforecast(
                    measured,
                    remaining,
                    cumulative_a100_hours=cumulative,
                    remaining_load_seconds=float(plan["forecast"]["single_load_a100_hours"])
                    * 3600.0
                    * remaining_blocks(block),
                    n_concurrent_workers=workers,
                ),
            )

    def work_is_left() -> bool:
        outstanding = bridge.pending_work(planned, _read_jsonl(ledger_path))
        chosen = set(selected)
        return bool(
            [
                job
                for job in outstanding["first_responses"] + outstanding["retries"]
                if job["execution_block"] in chosen
            ]
        )

    def pass_the_barrier() -> None:
        """Measure, then record a condition-specific reforecast before releasing more calls."""
        nonlocal stopped
        if barrier_path.exists():
            return
        if not work_is_left():
            return  # nothing is left to measure, so there is nothing to release
        limit = bridge.FIRST_REFORECAST_CALLS
        for block in selected:
            while stopped is None:
                run_block(block, stop_at_calls=limit)
                measured = timing_rows[:limit]
                remaining = bridge.remaining_counts(planned, _read_jsonl(ledger_path))
                enough_calls = len(measured) >= bridge.FIRST_REFORECAST_CALLS
                if enough_calls and not bridge.missing_rates(measured, remaining):
                    report = bridge.reforecast(
                        measured,
                        remaining,
                        cumulative_a100_hours=cumulative,
                        remaining_load_seconds=float(
                            plan["forecast"]["single_load_a100_hours"]
                        )
                        * 3600.0
                        * remaining_blocks(block),
                        n_concurrent_workers=1,
                    )
                    _write_json(barrier_path, report)
                    _write_json(out / "reforecast-latest.json", report)
                    if not report["may_continue"]:
                        reason = report["decision_reason"]
                        stopped = f"the barrier reforecast refused the rest: {reason}"
                    return
                if real_calls < limit:
                    break  # this block is finished; measure the rest on the next one
                if limit >= BARRIER_CALL_CAP:
                    raise RuntimeError(
                        f"{limit} real calls did not measure every remaining condition"
                    )
                limit += chunk_size
        if stopped is None:
            raise RuntimeError("the bridge could not measure every condition before the barrier")

    failure: str | None = None
    try:
        pass_the_barrier()
        if stopped is None and work_is_left() and not barrier_path.exists():
            raise RuntimeError(
                "no first-1000-call reforecast was recorded; refusing to release more calls"
            )
        if stopped is None and work_is_left():
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(run_block, block): block for block in selected}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
    except Exception as exc:  # always leave a summary behind after a paid run
        failure = f"{type(exc).__name__}: {exc}"
        stopped = stopped or failure

    final_remaining = bridge.remaining_counts(planned, _read_jsonl(ledger_path))
    if timing_rows and not bridge.missing_rates(timing_rows, final_remaining):
        _write_json(
            out / "reforecast-final.json",
            bridge.reforecast(
                timing_rows,
                final_remaining,
                cumulative_a100_hours=cumulative,
                n_concurrent_workers=workers,
            ),
        )
    final = bridge.completeness(planned, ledger_path)
    summary = {
        "run_id": run_id,
        "modal_profile": profile,
        "plan_sha256": plan["plan_sha256"],
        "results_namespace": namespace,
        "max_concurrent_workers": workers,
        "concurrency_amendment": {
            "path": str(amendment_path),
            "amendment_id": amendment_record["amendment_id"],
            "content_hash": amendment_record["content_hash"],
        },
        "checkpoint_verification": checkpoint,
        "stopped_early": stopped,
        "failure": failure,
        "blocks_selected": selected,
        "cumulative_a100_hours": cumulative,
        "meter": meter(meter_path),
        "timing_by_condition": bridge.checkpoint_by_condition(timing_rows),
        "completeness": final,
        "solo_reuse": plan["solo_reuse"],
        "written_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    _write_json(out / "SUMMARY.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "timing_by_condition"}, indent=2))
    run_lock.close()
    if stopped or not final["complete"]:
        raise SystemExit(
            f"the bridge is incomplete: {final['n_missing_calls']} planned calls have no "
            f"first response ({stopped or 'run ended'})"
        )


if __name__ == "__main__":
    print(json.dumps(build_plan(), indent=2, sort_keys=True))
