"""Run the frozen 36-cell solo screen on Modal; plan-only unless --execute is explicit."""

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

APP_NAME = "ieee-bigdata-freezeb-solo-screen"
LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/study-120h")
MODEL_CACHE = "/models"
RESULTS_ROOT = "/results"
GPU_TYPE = "A100-40GB"
FRAME_ROOT = LOCAL_ROOT / "freeze-b-work/20260912-cpu-draft"
SEAL_ROOT = LOCAL_ROOT / "freeze/20260912T222618Z-B1"
RUNTIME_LOCK = LOCAL_ROOT / "runtime-lock/20260913T004054Z-BRT1/LEDGER.json"
CHECKPOINT_LOCK = (
    LOCAL_ROOT / "runtime-lock/20260913T004054Z-BRT1/benchmark/checkpoint-manifests.json"
)

model_volume = modal.Volume.from_name("ieee-bigdata-freezeb-models", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "ieee-bigdata-freezeb-screen-results", create_if_missing=True
)
download_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==1.31.0"
)
image = (
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


def _remote_imports() -> None:
    sys.path.insert(0, str(REMOTE_ROOT / "src"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@app.function(image=download_image, volumes={MODEL_CACHE: model_volume}, timeout=7200)
def ensure_checkpoint(model_id: str, revision: str) -> dict[str, Any]:
    """Materialize and hash the pinned model in the current Modal account."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=model_id, revision=revision, cache_dir=MODEL_CACHE))
    files = [
        {
            "path": str(path.relative_to(snapshot)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file())
    ]
    model_volume.commit()
    return {"model_id": model_id, "revision": revision, "files": files}


@app.function(image=download_image, volumes={RESULTS_ROOT: results_volume}, timeout=600)
def recover_model(run_id: str, model_id: str) -> list[dict[str, Any]]:
    """Return committed chunks after a local crash in the same Modal account."""
    directory = Path(RESULTS_ROOT) / run_id / model_id.replace("/", "--")
    if not directory.exists():
        return []
    results_volume.reload()
    payloads: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


@app.cls(
    image=image,
    volumes={MODEL_CACHE: model_volume},
    max_containers=1,
    scaledown_window=30,
    startup_timeout=1200,
    timeout=7200,
)
class TokenizerWorker:
    """CPU-only exact-tokenizer gate that runs before any A100 container starts."""

    model_id: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        _remote_imports()
        from freezeb.runtime_benchmark import policy_for
        from step1.backends import PinnedTokenizerCounter

        policy = policy_for(self.model_id)
        self.policy = policy
        self.counter = PinnedTokenizerCounter(
            policy.model_id,
            policy.revision,
            cache_dir=MODEL_CACHE,
            enable_thinking=policy.enable_thinking,
            local_files_only=True,
        )

    @modal.method()
    def check(self, task: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
        from freezeb.runtime_benchmark import context_fit

        rows = []
        for spec in specs:
            prompt_tokens = self.counter.count_tokens(spec["messages"])
            rows.append(
                context_fit(prompt_tokens, int(spec["max_tokens"]), self.policy.max_model_len)
            )
        return {
            "model_id": self.model_id,
            "task_id": task,
            "calls": len(rows),
            "n_over_budget": sum(not row["fits"] for row in rows),
            "max_required_tokens": max(int(row["required_tokens"]) for row in rows),
            "min_headroom_tokens": min(int(row["headroom_tokens"]) for row in rows),
            "max_model_len": self.policy.max_model_len,
            "tokenizer": self.counter.describe(),
        }


@app.cls(
    image=image,
    volumes={MODEL_CACHE: model_volume, RESULTS_ROOT: results_volume},
    gpu=GPU_TYPE,
    max_containers=1,
    scaledown_window=30,
    startup_timeout=1200,
    timeout=7200,
)
class SoloWorker:
    """One pinned model per container; accepts only one-document solo call specs."""

    model_id: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        _remote_imports()
        from freezeb.runtime_benchmark import DECODING, policy_for
        from step1.backends import VLLMBackend, host_facts

        policy = policy_for(self.model_id)
        if not policy.runnable:
            raise RuntimeError(f"model is excluded by the runtime lock: {policy.exclusion_reason}")
        self.session_started = time.monotonic()
        self.session_id = uuid.uuid4().hex
        self.backend = VLLMBackend(
            policy.model_id,
            policy.revision,
            policy.revision,
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
    def run_chunk(self, run_id: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
        from freezeb.runtime_benchmark import context_fit, policy_for
        from step1.backends import GenRequest

        if not specs or len(specs) > 64:
            raise ValueError("a solo chunk must contain 1 through 64 calls")
        if any(spec.get("condition") != "solo" for spec in specs):
            raise ValueError("this worker refuses every non-solo condition")
        chunk_key = hashlib.sha256(
            "|".join(spec["attempt_id"] for spec in specs).encode("utf-8")
        ).hexdigest()[:20]
        path = Path(RESULTS_ROOT) / run_id / self.model_id.replace("/", "--") / f"{chunk_key}.json"
        results_volume.reload()
        if path.exists():
            return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        policy = policy_for(self.model_id)
        started = time.monotonic()
        events: list[dict[str, Any]] = []
        requests: list[Any] = []
        request_specs: list[dict[str, Any]] = []
        for spec in specs:
            if spec["model_id"] != self.model_id:
                raise ValueError("call model does not match worker model")
            if spec["messages"][1]["content"].count("<doc id=") != 1:
                raise ValueError("solo screen requires exactly one rendered document")
            prompt_tokens = self.backend.count_tokens(spec["messages"])
            fit = context_fit(prompt_tokens, int(spec["max_tokens"]), policy.max_model_len)
            if not fit["fits"]:
                events.append(
                    {
                        **{
                            k: spec[k]
                            for k in (
                                "call_id",
                                "attempt_id",
                                "attempt",
                                "model_id",
                                "task_id",
                                "condition",
                                "doc_id",
                                "prompt_sha256",
                                "max_tokens",
                                "n_tokens",
                            )
                        },
                        "terminal": True,
                        "status": "context_excluded",
                        **fit,
                    }
                )
                continue
            requests.append(
                GenRequest(
                    call_id=spec["call_id"],
                    condition="solo",
                    target_ids=(spec["doc_id"],),
                    messages=spec["messages"],
                    max_tokens=int(spec["max_tokens"]),
                )
            )
            request_specs.append(spec)
        if requests:
            try:
                generated = self.backend.generate(requests)
                for spec, result in zip(request_specs, generated, strict=True):
                    events.append(
                        {
                            **{
                                k: spec[k]
                                for k in (
                                    "call_id",
                                    "attempt_id",
                                    "attempt",
                                    "model_id",
                                    "task_id",
                                    "condition",
                                    "doc_id",
                                    "prompt_sha256",
                                    "max_tokens",
                                    "n_tokens",
                                )
                            },
                            "terminal": True,
                            "status": "ok",
                            "first_response": result.text,
                            "prompt_tokens": result.prompt_tokens,
                            "output_tokens": result.output_tokens,
                            "finish_reason": result.finish_reason,
                            "queue_seconds": result.queue_seconds,
                            "prefill_seconds": result.prefill_seconds,
                            "decode_seconds": result.decode_seconds,
                        }
                    )
            except Exception as exc:  # preserve the first failed attempt for every call
                for spec in request_specs:
                    events.append(
                        {
                            **{
                                k: spec[k]
                                for k in (
                                    "call_id",
                                    "attempt_id",
                                    "attempt",
                                    "model_id",
                                    "task_id",
                                    "condition",
                                    "doc_id",
                                    "prompt_sha256",
                                    "max_tokens",
                                    "n_tokens",
                                )
                            },
                            "terminal": True,
                            "status": "generation_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        for event in events:
            event["worker_session_id"] = self.session_id
            event["worker_load_seconds"] = self.load_seconds
        payload = {
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

    @modal.method()
    def run_one_token_probe(self, run_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Time one technical token for the fallback single-call decomposition."""
        from freezeb.runtime_benchmark import context_fit, policy_for
        from step1.backends import GenRequest

        if spec.get("model_id") != self.model_id or spec.get("condition") != "solo":
            raise ValueError("timing probe must match this solo worker")
        probe_key = hashlib.sha256(str(spec["attempt_id"]).encode("utf-8")).hexdigest()[:20]
        path = (
            Path(RESULTS_ROOT)
            / run_id
            / self.model_id.replace("/", "--")
            / f"timing-{probe_key}.json"
        )
        results_volume.reload()
        if path.exists():
            return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        policy = policy_for(self.model_id)
        prompt_tokens = self.backend.count_tokens(spec["messages"])
        if not context_fit(prompt_tokens, 1, policy.max_model_len)["fits"]:
            raise ValueError("timing probe does not fit the frozen context window")
        started = time.monotonic()
        result = self.backend.generate(
            [
                GenRequest(
                    call_id=f"{spec['call_id']}::timing-probe",
                    condition="solo",
                    target_ids=(spec["doc_id"],),
                    messages=spec["messages"],
                    max_tokens=1,
                )
            ]
        )[0]
        payload = {
            "kind": "timing_probe",
            "worker_session_id": self.session_id,
            "session_elapsed_seconds": time.monotonic() - self.session_started,
            "wall_seconds": time.monotonic() - started,
            "first_response": result.text,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "finish_reason": result.finish_reason,
            "host": self.host,
            "backend": self.backend.describe(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        results_volume.commit()
        return payload


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _screen_forecast(lock: dict[str, Any], model: str, task: str) -> float:
    block_id = f"screen::{model}::{task}"
    block = next(row for row in lock["blocks"] if row["block_id"] == block_id)
    if block.get("status") != "forecastable":
        raise ValueError(f"runtime lock cannot forecast {block_id}")
    return float(block["base_a100_hours"])


def _scaled_task_forecast(
    lock: dict[str, Any], model: str, task: str, projected_model_hours: float
) -> float:
    """Scale the locked task mix to a model-level first-1,000-call projection."""
    prefix = f"screen::{model}::"
    model_blocks = [
        row
        for row in lock["blocks"]
        if str(row.get("block_id", "")).startswith(prefix) and row.get("status") == "forecastable"
    ]
    total = sum(float(row["base_a100_hours"]) for row in model_blocks)
    if projected_model_hours <= 0.0 or total <= 0.0:
        raise ValueError("reforecast hours must be positive")
    return _screen_forecast(lock, model, task) * projected_model_hours / total


def _acquire_run_lock(out: Path, modal_profile: str) -> TextIO:
    """Hold an OS-released exclusive lock for one local screen run directory."""
    path = out / ".active-run.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another process is already using screen run {out}") from None
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
    os.fsync(handle.fileno())
    return handle


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@app.local_entrypoint()
def main(
    execute: bool = False,
    output_root: str = "freezeb-screen",
    run_id: str = "",
    models: str = "",
    tasks: str = "",
    chunk_size: int = 32,
    max_parallel_models: int = 6,
) -> None:
    """Validate and print the plan, then run only when --execute is supplied."""
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from freezeb.meter import (
        FORECAST_MULTIPLIER,
        STOP_HOURS,
        may_start,
    )
    from freezeb.screen import (
        MAX_TECHNICAL_ATTEMPTS,
        RUNNABLE_MODELS,
        TASK_KEYS,
        build_calls,
        model_key,
        pending_calls,
        summarize_cell,
        technical_reforecast,
        validate_worker_provenance,
        verify_screen_frame,
    )
    from step0.hashing import sha256_json

    selected_models = tuple(part for part in models.split(",") if part) or RUNNABLE_MODELS
    selected_tasks = tuple(part for part in tasks.split(",") if part) or TASK_KEYS
    if not set(selected_models) <= set(RUNNABLE_MODELS):
        raise ValueError("--models contains a model outside the six runtime-locked policies")
    if not set(selected_tasks) <= set(TASK_KEYS):
        raise ValueError("--tasks contains an unknown task")
    if not 1 <= chunk_size <= 64 or not 1 <= max_parallel_models <= 6:
        raise ValueError("chunk_size must be 1..64 and max_parallel_models must be 1..6")
    frames = {task: verify_screen_frame(FRAME_ROOT, SEAL_ROOT, task) for task in TASK_KEYS}
    lock = json.loads(RUNTIME_LOCK.read_text(encoding="utf-8"))
    base_hours = sum(
        _screen_forecast(lock, model, task) for model in selected_models for task in selected_tasks
    )
    plan_core = {
        "schema_version": "freezeb-solo-screen-plan-v1",
        "models": RUNNABLE_MODELS,
        "tasks": TASK_KEYS,
        "calls": len(RUNNABLE_MODELS) * len(TASK_KEYS) * 2048,
        "chunk_size": chunk_size,
        "frame_sha256": {
            task: sha256_json([row["doc_id"] for row in frames[task]]) for task in TASK_KEYS
        },
        "runtime_lock": str(RUNTIME_LOCK.relative_to(LOCAL_ROOT)),
        "full_base_forecast_a100_hours": sum(
            _screen_forecast(lock, model, task) for model in RUNNABLE_MODELS for task in TASK_KEYS
        ),
        "solo_only": True,
    }
    plan = {**plan_core, "plan_sha256": sha256_json(plan_core)}
    print(
        json.dumps(
            {
                **plan,
                "paid_execution": execute,
                "invocation_models": selected_models,
                "invocation_tasks": selected_tasks,
                "invocation_base_forecast_a100_hours": base_hours,
                "invocation_guarded_forecast_a100_hours": base_hours * FORECAST_MULTIPLIER,
                "max_concurrent_a100s": min(max_parallel_models, len(selected_models)),
            },
            indent=2,
        ),
        flush=True,
    )
    if not execute:
        return

    from freezeb.runtime_benchmark import policy_for

    modal_profile = os.environ.get("MODAL_PROFILE", "").strip()
    if not modal_profile:
        raise RuntimeError("--execute requires an explicit MODAL_PROFILE")
    requested = run_id
    run_id = requested or (
        dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    out = Path(output_root) / run_id
    if out.exists():
        saved = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        if saved["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("resume refused because the saved plan differs")
    else:
        out.mkdir(parents=True)
        _write_json(out / "plan.json", plan)
    run_lock = _acquire_run_lock(out, modal_profile)

    expected_manifests = {
        row["model_id"]: row for row in json.loads(CHECKPOINT_LOCK.read_text(encoding="utf-8"))
    }
    for model_id in selected_models:
        policy = policy_for(model_id)
        actual = ensure_checkpoint.remote(model_id, policy.revision)
        if actual != expected_manifests[model_id]:
            raise RuntimeError(f"{model_id} checkpoint files differ from the runtime lock")

    def preflight_model(model_id: str) -> None:
        worker = TokenizerWorker(model_id=model_id)  # type: ignore[call-arg]
        for task in selected_tasks:
            report_path = out / "preflight" / model_key(model_id) / f"{task}.json"
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = worker.check.remote(task, build_calls(model_id, task, frames[task]))
                _write_json(report_path, report)
            if report["calls"] != 2048 or report["n_over_budget"] != 0:
                raise RuntimeError(f"exact-tokenizer preflight failed for {model_id}::{task}")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_parallel_models, len(selected_models))
    ) as executor:
        futures = [executor.submit(preflight_model, model) for model in selected_models]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    calibration = float(lock["measurement_summary"]["meter_calibration_factor"])
    cumulative = float(lock["budget"]["cumulative_a100_hours_after_benchmark"])
    state_path = out / "meter-state.json"
    session_elapsed: dict[str, float] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        cumulative = float(state["cumulative_a100_hours"])
        session_elapsed = {
            str(key): float(value)
            for key, value in state.get("session_elapsed_seconds", {}).items()
        }
    state_lock = threading.Lock()
    state_changed = threading.Condition(state_lock)
    stop = False
    reserved_guarded_hours = 0.0

    def enrich_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
        events = cast(list[dict[str, Any]], payload["events"])
        per_call_wall = float(payload.get("chunk_wall_seconds", 0.0)) / len(events)
        for event in events:
            event["worker_session_id"] = payload.get("worker_session_id")
            event["worker_load_seconds"] = payload.get("load_seconds")
            event["worker_host"] = payload.get("host")
            event["worker_backend"] = payload.get("backend")
            event["chunk_wall_seconds"] = payload.get("chunk_wall_seconds")
            event["per_call_wall_seconds"] = per_call_wall
            event["modal_profile"] = modal_profile
            retryable = event.get("status") in {"generation_failed", "worker_failed"}
            attempt = int(event.get("attempt", 1))
            event["final_for_call"] = not retryable or attempt >= MAX_TECHNICAL_ATTEMPTS
            if retryable and attempt >= MAX_TECHNICAL_ATTEMPTS:
                event["status"] = "technical_attempts_exhausted"
        return events

    def run_model(model_id: str) -> None:
        nonlocal cumulative, reserved_guarded_hours, stop
        recovered = recover_model.remote(run_id, model_id)
        local_by_task: dict[str, dict[str, dict[str, Any]]] = {}
        recovered_by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in selected_tasks}
        for task in selected_tasks:
            path = out / "cells" / model_key(model_id) / f"{task}.jsonl"
            local_by_task[task] = {row["attempt_id"]: row for row in _read_events(path)}
        with state_lock:
            recovery_state_changed = False
            for payload in sorted(
                recovered,
                key=lambda item: (
                    str(item["worker_session_id"]),
                    float(item["session_elapsed_seconds"]),
                ),
            ):
                sid = str(payload["worker_session_id"])
                elapsed = float(payload["session_elapsed_seconds"])
                previous_elapsed = session_elapsed.get(sid, 0.0)
                delta = max(0.0, elapsed - previous_elapsed) * calibration
                session_elapsed[sid] = max(elapsed, previous_elapsed)
                cumulative += delta / 3600
                recovery_state_changed = recovery_state_changed or elapsed > previous_elapsed
                if payload.get("kind") == "timing_probe":
                    continue
                payload_events = enrich_events(payload)
                if any(row.get("status") == "ok" for row in payload_events):
                    validate_worker_provenance(
                        model_id, payload.get("host"), payload.get("backend")
                    )
                per_call = delta / len(payload_events)
                for row in payload_events:
                    task = str(row["task_id"])
                    if task not in recovered_by_task:
                        continue
                    prior = local_by_task[task].get(row["attempt_id"])
                    if prior is not None:
                        immutable = {
                            k: v
                            for k, v in prior.items()
                            if k
                            not in {
                                "metered_a100_seconds",
                                "cumulative_a100_hours",
                                "timing_probe_method",
                                "probe_prompt_plus_first_token_seconds",
                                "probe_remaining_generation_seconds",
                            }
                        }
                        if immutable != row:
                            raise RuntimeError(f"conflicting recovered event {row['call_id']}")
                        continue
                    row["metered_a100_seconds"] = per_call
                    row["cumulative_a100_hours"] = cumulative
                    recovered_by_task[task].append(row)
            for task, missing in recovered_by_task.items():
                if missing:
                    path = out / "cells" / model_key(model_id) / f"{task}.jsonl"
                    _append(path, missing)
            if recovery_state_changed or any(recovered_by_task.values()):
                _write_json(
                    state_path,
                    {
                        "cumulative_a100_hours": cumulative,
                        "stop_reached": cumulative >= STOP_HOURS,
                        "session_elapsed_seconds": session_elapsed,
                    },
                )
        worker = SoloWorker(model_id=model_id)  # type: ignore[call-arg]
        model_real_calls = sum(
            len(_read_events(out / "cells" / model_key(model_id) / f"{task}.jsonl"))
            for task in TASK_KEYS
        )
        forecast_path = out / "reforecast-1000" / f"{model_key(model_id)}.json"

        def write_reforecast_if_ready() -> None:
            if forecast_path.exists():
                return
            model_events = [
                row
                for task_name in TASK_KEYS
                for row in _read_events(out / "cells" / model_key(model_id) / f"{task_name}.jsonl")
            ]
            reforecast = technical_reforecast(model_id, model_events, calibration)
            if reforecast is not None:
                _write_json(forecast_path, reforecast)

        write_reforecast_if_ready()
        for task in selected_tasks:
            path = out / "cells" / model_key(model_id) / f"{task}.jsonl"
            existing = _read_events(path)
            specs = build_calls(model_id, task, frames[task])
            pending = pending_calls(specs, existing)
            if not pending:
                continue
            forecast = _screen_forecast(lock, model_id, task)
            if forecast_path.exists():
                latest = json.loads(forecast_path.read_text(encoding="utf-8"))
                forecast = _scaled_task_forecast(
                    lock,
                    model_id,
                    task,
                    float(latest["projected_model_screen_base_a100_hours"]),
                )
            guarded = FORECAST_MULTIPLIER * forecast
            with state_changed:
                while True:
                    if stop:
                        return
                    actual_decision = may_start(cumulative, forecast)
                    if not actual_decision.start:
                        stop = True
                        state_changed.notify_all()
                        return
                    concurrent_decision = may_start(cumulative + reserved_guarded_hours, forecast)
                    if concurrent_decision.start:
                        reserved_guarded_hours += guarded
                        break
                    state_changed.wait()
            try:
                queue = list(pending)
                timing_probe_path = out / "timing-probes" / f"{model_key(model_id)}.json"
                while queue:
                    with state_changed:
                        if stop or cumulative >= STOP_HOURS:
                            stop = True
                            state_changed.notify_all()
                            return
                    take = chunk_size
                    if model_real_calls < 1000:
                        take = min(chunk_size, 1000 - model_real_calls)
                    if not timing_probe_path.exists():
                        take = 1
                    group, queue = queue[:take], queue[take:]
                    invoked = time.monotonic()
                    try:
                        result = worker.run_chunk.remote(run_id, group)
                        sid = str(result["worker_session_id"])
                        elapsed = float(result["session_elapsed_seconds"])
                        with state_lock:
                            delta = max(0.0, elapsed - session_elapsed.get(sid, 0.0)) * calibration
                            session_elapsed[sid] = elapsed
                        enrich_events(result)
                    except Exception as exc:
                        result = {
                            "events": [
                                {
                                    **{
                                        k: spec[k]
                                        for k in (
                                            "call_id",
                                            "attempt_id",
                                            "attempt",
                                            "model_id",
                                            "task_id",
                                            "condition",
                                            "doc_id",
                                            "prompt_sha256",
                                            "max_tokens",
                                            "n_tokens",
                                        )
                                    },
                                    "terminal": True,
                                    "status": "worker_failed",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                                for spec in group
                            ],
                            "worker_session_id": None,
                            "load_seconds": None,
                            "chunk_wall_seconds": time.monotonic() - invoked,
                            "host": None,
                            "backend": None,
                        }
                        delta = (time.monotonic() - invoked) * calibration
                        enrich_events(result)
                    successful_probe = next(
                        (event for event in result["events"] if event.get("status") == "ok"),
                        None,
                    )
                    new_timing_probe: dict[str, Any] | None = None
                    timing_error = False
                    if successful_probe is not None:
                        validate_worker_provenance(
                            model_id, result.get("host"), result.get("backend")
                        )
                    if not timing_probe_path.exists() and successful_probe is not None:
                        prefill = successful_probe.get("prefill_seconds")
                        decode = successful_probe.get("decode_seconds")
                        direct_usable = (
                            prefill is not None
                            and decode is not None
                            and float(prefill) > 0.0
                            and float(decode) > 0.0
                        )
                        if direct_usable:
                            method = "vllm_request_metrics_single_call"
                            first_token_seconds = float(prefill)
                            remaining_seconds = float(decode)
                            fallback: dict[str, Any] | None = None
                        else:
                            source = next(
                                spec
                                for spec in group
                                if spec["call_id"] == successful_probe["call_id"]
                            )
                            fallback = worker.run_one_token_probe.remote(run_id, source)
                            validate_worker_provenance(
                                model_id, fallback.get("host"), fallback.get("backend")
                            )
                            fallback_sid = str(fallback["worker_session_id"])
                            fallback_elapsed = float(fallback["session_elapsed_seconds"])
                            with state_lock:
                                fallback_delta = (
                                    max(
                                        0.0,
                                        fallback_elapsed - session_elapsed.get(fallback_sid, 0.0),
                                    )
                                    * calibration
                                )
                                session_elapsed[fallback_sid] = max(
                                    fallback_elapsed,
                                    session_elapsed.get(fallback_sid, 0.0),
                                )
                            delta += fallback_delta
                            first_token_seconds = float(fallback["wall_seconds"])
                            full_wall = float(result.get("chunk_wall_seconds", 0.0))
                            remaining_seconds = full_wall - first_token_seconds
                            method = "one_token_wall_differencing_single_call"
                        usable = first_token_seconds > 0.0 and remaining_seconds > 0.0
                        successful_probe["timing_probe_method"] = method
                        successful_probe["probe_prompt_plus_first_token_seconds"] = (
                            first_token_seconds
                        )
                        successful_probe["probe_remaining_generation_seconds"] = remaining_seconds
                        new_timing_probe = {
                            "model_id": model_id,
                            "attempt_id": successful_probe["attempt_id"],
                            "scope": "single_call_probe_only",
                            "method": method,
                            "prompt_plus_first_token_seconds": first_token_seconds,
                            "remaining_generation_seconds": remaining_seconds,
                            "raw_vllm_prefill_seconds": prefill,
                            "raw_vllm_decode_seconds": decode,
                            "fallback_one_token_call": fallback,
                            "usable": usable,
                        }
                        timing_error = not usable
                    with state_changed:
                        per_call = delta / len(result["events"])
                        retry_specs: list[dict[str, Any]] = []
                        specs_by_id = {str(spec["call_id"]): spec for spec in group}
                        for event in result["events"]:
                            retryable = event.get("status") in {
                                "generation_failed",
                                "worker_failed",
                            }
                            attempt = int(event.get("attempt", 1))
                            if retryable:
                                source = specs_by_id[str(event["call_id"])]
                                retry_specs.append(
                                    {
                                        **source,
                                        "attempt": attempt + 1,
                                        "attempt_id": (
                                            f"{event['call_id']}::attempt::{attempt + 1}"
                                        ),
                                    }
                                )
                            cumulative += per_call / 3600
                            event["metered_a100_seconds"] = per_call
                            event["cumulative_a100_hours"] = cumulative
                        _append(path, result["events"])
                        _append(
                            out / "meter.jsonl",
                            [
                                {
                                    "model_id": model_id,
                                    "task_id": task,
                                    "calls": len(result["events"]),
                                    "metered_a100_seconds": delta,
                                    "cumulative_a100_hours": cumulative,
                                }
                            ],
                        )
                        _write_json(
                            state_path,
                            {
                                "cumulative_a100_hours": cumulative,
                                "stop_reached": cumulative >= STOP_HOURS,
                                "session_elapsed_seconds": session_elapsed,
                            },
                        )
                        if cumulative >= STOP_HOURS:
                            stop = True
                        state_changed.notify_all()
                    if new_timing_probe is not None:
                        destination = timing_probe_path
                        if timing_error:
                            destination = timing_probe_path.with_name(
                                f"{timing_probe_path.stem}-unusable-{time.time_ns()}.json"
                            )
                        _write_json(destination, new_timing_probe)
                    if timing_error:
                        with state_changed:
                            stop = True
                            state_changed.notify_all()
                        raise RuntimeError(
                            f"{model_id} returned no usable single-call timing decomposition"
                        )
                    if retry_specs:
                        # Preserve failed attempts and retry separately before later calls.
                        queue = retry_specs + queue
                    model_real_calls += len(result["events"])
                    if model_real_calls >= 1000:
                        write_reforecast_if_ready()
            finally:
                with state_changed:
                    reserved_guarded_hours -= guarded
                    state_changed.notify_all()
        for task in selected_tasks:
            path = out / "cells" / model_key(model_id) / f"{task}.jsonl"
            events = _read_events(path)
            summary = summarize_cell(model_id, task, frames[task], events)
            _write_json(out / "summaries" / model_key(model_id) / f"{task}.json", summary)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_parallel_models, len(selected_models))
    ) as executor:
        futures = [executor.submit(run_model, model) for model in selected_models]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(
        json.dumps(
            {"output_dir": str(out), "cumulative_a100_hours": cumulative, "stopped": stop}, indent=2
        )
    )
    run_lock.close()
