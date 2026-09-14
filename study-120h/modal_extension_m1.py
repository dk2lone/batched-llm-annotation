"""Run the sealed X1R4 M1 solo screen on Modal; plan-only unless --execute is explicit.

The screen executes the 40,960 sealed solo calls for every candidate the signed
runtime/access lock marks attempted. Every rule it obeys lives in
``freezeb.extension_screen`` and is tested on the CPU. This file only moves data
between that module, the Modal workers, and the run directory.
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

APP_NAME = "ieee-bigdata-extension-m1-screen"
LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/study-120h")
MODEL_CACHE = "/models"
RESULTS_ROOT = "/results"
GPU_TYPE = "A100"
VLLM_VERSION = "0.28.0"
SEAL_ROOT = LOCAL_ROOT / "freeze/20260913T183353Z-X1R4"
LOCK_ROOT = LOCAL_ROOT / "runtime-lock/20260913T205646Z-X1R4"
FRAME_ROOT = LOCAL_ROOT / "ext-work/20260913-cpu-draft"
RUNTIME_EVIDENCE = LOCK_ROOT / "evidence/runtime"
FREEZE_B_CHECKPOINTS = (
    LOCAL_ROOT / "runtime-lock/20260913T004054Z-BRT1/benchmark/checkpoint-manifests.json"
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

model_volume = modal.Volume.from_name("ieee-bigdata-freezeb-models", create_if_missing=True)
results_volume = modal.Volume.from_name("ieee-bigdata-extension-m1-results", create_if_missing=True)
inventory_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==1.31.0"
).add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface-hub==1.31.0", "numpy==2.2.6", "scipy==1.16.3", f"vllm=={VLLM_VERSION}"
    )
    .env(RUNTIME_ENV)
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


@app.function(image=inventory_image, volumes={MODEL_CACHE: model_volume}, timeout=7200)
def inventory_checkpoint(
    model_id: str, revision: str, with_hashes: bool, excluded_alternatives: list[str]
) -> dict[str, Any]:
    """Inventory the pinned checkpoint already staged in the shared model volume.

    The volume is offline, so this can only describe what staging put there. It
    never downloads a weight and never reads a credential.
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            ignore_patterns=excluded_alternatives,
        )
    )
    paths = sorted(item for item in snapshot.rglob("*") if item.is_file())
    files: list[dict[str, Any]] = [
        {
            "path": str(path.relative_to(snapshot)),
            "bytes": path.stat().st_size,
            **({"sha256": _sha256_file(path)} if with_hashes else {}),
        }
        for path in paths
    ]
    result: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
    }
    if with_hashes:
        sys.path.insert(0, str(REMOTE_ROOT / "src"))
        from step0.hashing import sha256_json

        result["files_sha256"] = sha256_json(files)
    return result


@app.function(image=inventory_image, volumes={RESULTS_ROOT: results_volume}, timeout=900)
def recover_model(run_id: str, model_id: str) -> list[dict[str, Any]]:
    """Return the chunk payloads a previous local process already committed."""
    directory = Path(RESULTS_ROOT) / run_id / model_id.replace("/", "--")
    if not directory.exists():
        return []
    results_volume.reload()
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
    ]


@app.cls(
    image=image,
    volumes={MODEL_CACHE: model_volume, RESULTS_ROOT: results_volume},
    gpu=GPU_TYPE,
    max_containers=1,
    scaledown_window=2,
    startup_timeout=1800,
    timeout=10800,
)
class SoloWorker:
    """One pinned, batch-invariant model per container. Solo call specs only."""

    model_id: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        _remote_imports()
        from freezeb import batch_invariance as bi
        from freezeb.extension import ext_policy_for
        from step1.backends import VLLMBackend, host_facts

        policy = ext_policy_for(self.model_id)
        self.session_started = time.monotonic()
        self.session_id = uuid.uuid4().hex
        self.backend = VLLMBackend(
            policy.model_id,
            policy.revision,
            policy.revision,
            {
                "tensor_parallel_size": bi.SERVING["tensor_parallel_size"],
                "dtype": bi.SERVING["dtype"],
                "max_model_len": policy.max_model_len,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": bi.SERVING["max_num_seqs"],
            },
            bi.DECODING,
            enable_thinking=policy.enable_thinking,
            require_batch_invariance=True,
        )
        self.load_seconds = time.monotonic() - self.session_started
        self.host = host_facts()

    @modal.method()
    def run_chunk(self, run_id: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
        from freezeb.extension import ext_policy_for
        from freezeb.extension_screen import EVENT_KEYS, dispatch_manifest
        from freezeb.runtime_benchmark import context_fit
        from step1.backends import GenRequest

        if not specs or len(specs) > 64:
            raise ValueError("a solo chunk must hold 1 through 64 calls")
        if any(spec.get("condition") != "solo" for spec in specs):
            raise ValueError("this worker refuses every non-solo condition")
        chunk_key = hashlib.sha256(
            "|".join(str(spec["attempt_id"]) for spec in specs).encode("utf-8")
        ).hexdigest()[:20]
        path = Path(RESULTS_ROOT) / run_id / self.model_id.replace("/", "--") / f"{chunk_key}.json"
        results_volume.reload()
        if path.exists():
            return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        policy = ext_policy_for(self.model_id)
        started = time.monotonic()
        events: list[dict[str, Any]] = []
        requests: list[Any] = []
        request_specs: list[dict[str, Any]] = []

        def identity(spec: dict[str, Any]) -> dict[str, Any]:
            return {key: spec[key] for key in EVENT_KEYS}

        for spec in specs:
            if spec["model_id"] != self.model_id:
                raise ValueError("call model does not match worker model")
            if spec["messages"][1]["content"].count("<doc id=") != 1:
                raise ValueError("the solo screen requires exactly one rendered document")
            prompt_tokens = self.backend.count_tokens(spec["messages"])
            fit = context_fit(prompt_tokens, int(spec["max_tokens"]), policy.max_model_len)
            if not fit["fits"]:
                events.append({**identity(spec), "status": "context_excluded", **fit})
                continue
            requests.append(
                GenRequest(
                    call_id=str(spec["call_id"]),
                    condition="solo",
                    target_ids=(str(spec["doc_id"]),),
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
                            **identity(spec),
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
            except Exception as exc:  # keep the failed first attempt for every call
                for spec in request_specs:
                    events.append(
                        {
                            **identity(spec),
                            "status": "generation_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        payload = {
            "worker_session_id": self.session_id,
            "session_elapsed_seconds": time.monotonic() - self.session_started,
            "load_seconds": self.load_seconds,
            "chunk_wall_seconds": time.monotonic() - started,
            "host": self.host,
            "backend": self.backend.describe(),
            "dispatched_attempts": dispatch_manifest(specs),
            "events": events,
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


def _acquire_run_lock(out: Path, modal_profile: str) -> TextIO:
    """Hold one exclusive OS lock per local run directory."""
    handle = (out / ".active-run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another process is already using M1 run {out}") from None
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


@app.local_entrypoint()
def main(
    execute: bool = False,
    output_root: str = "extension-m1",
    run_id: str = "",
    chunk_size: int = 32,
    max_parallel_models: int = 5,
) -> None:
    """Validate the sealed plan and print it. Run it only when --execute is supplied.

    The screen has no partial-panel mode. It always runs every candidate the
    signed lock attempts, over all four frozen tasks, in the sealed M1 order.
    """
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from freezeb import extension as ext
    from freezeb import extension_screen as m1

    seal = m1.verify_seal(SEAL_ROOT)
    lock = m1.verify_runtime_lock(LOCK_ROOT, seal)
    m1_order = json.loads((SEAL_ROOT / "M1_CALL_ORDER.json").read_text(encoding="utf-8"))
    attempted = list(lock["attempted_models"])
    if not 1 <= chunk_size <= 64 or not 1 <= max_parallel_models <= len(ext.EXT_MODEL_IDS):
        raise ValueError("chunk_size must be 1..64 and max_parallel_models must be 1..5")

    frames = {task: m1.verify_screen_frame(FRAME_ROOT, seal, task) for task in ext.EXT_TASK_KEYS}
    calls_by_cell: dict[str, list[dict[str, Any]]] = {}
    verified_cells: list[dict[str, Any]] = []
    for cell in m1.execution_order(m1_order, attempted):
        model_id, task_id = str(cell["model_id"]), str(cell["task_id"])
        cell_calls = m1.build_cell_calls(model_id, task_id, frames[task_id])
        verified_cells.append(m1.verify_cell_against_seal(m1_order, model_id, task_id, cell_calls))
        calls_by_cell[str(cell["cell_key"])] = cell_calls

    runtime_evidence = {
        model_id: json.loads(
            (RUNTIME_EVIDENCE / f"{ext.model_key(model_id)}.json").read_text(encoding="utf-8")
        )
        for model_id in attempted
    }
    forecast = m1.m1_base_forecast(runtime_evidence)
    plan = m1.build_plan(
        seal=seal,
        lock=lock,
        m1_order=m1_order,
        verified_cells=verified_cells,
        forecast=forecast,
        chunk_size=chunk_size,
        batch_invariant_env=RUNTIME_ENV["VLLM_BATCH_INVARIANT"],
    )
    base_hours = sum(
        m1.cell_base_hours(forecast, cell["cell_key"], include_load=False)
        for cell in verified_cells
    ) + sum(float(forecast["model_load_a100_hours"][model_id]) for model_id in attempted)
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "seal_manifest_sha256": seal["manifest_sha256"],
                "attempted_models": attempted,
                "unavailable_models": lock["unavailable_models"],
                "execution_scope": plan["schedule"]["execution_scope"],
                "planned_cells": len(verified_cells),
                "planned_calls": plan["planned_calls"],
                "base_forecast_a100_hours": base_hours,
                "guarded_forecast_a100_hours": base_hours * ext.FORECAST_MULTIPLIER,
                "first_gate": plan["first_gate"],
                "first_gate_attempts": plan["first_gate_attempts"],
                "max_concurrent_a100s": min(max_parallel_models, len(attempted)),
                "paid_execution": execute,
            },
            indent=2,
        ),
        flush=True,
    )
    if not execute:
        return

    modal_profile = os.environ.get("MODAL_PROFILE", "").strip()
    if not modal_profile:
        raise RuntimeError("--execute requires an explicit MODAL_PROFILE")
    run_id = run_id or (
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
    try:
        _run(
            plan=plan,
            lock=lock,
            seal=seal,
            out=out,
            run_id=run_id,
            modal_profile=modal_profile,
            attempted=attempted,
            verified_cells=verified_cells,
            calls_by_cell=calls_by_cell,
            frames=frames,
            forecast=forecast,
            chunk_size=chunk_size,
            max_parallel_models=max_parallel_models,
        )
    finally:
        run_lock.close()


def _run(
    *,
    plan: dict[str, Any],
    lock: dict[str, Any],
    seal: dict[str, Any],
    out: Path,
    run_id: str,
    modal_profile: str,
    attempted: list[str],
    verified_cells: list[dict[str, Any]],
    calls_by_cell: dict[str, list[dict[str, Any]]],
    frames: dict[str, list[dict[str, Any]]],
    forecast: dict[str, Any],
    chunk_size: int,
    max_parallel_models: int,
) -> None:
    """Execute the verified plan. The caller owns the run lock."""
    from freezeb import extension as ext
    from freezeb import extension_screen as m1

    expected_inventory = m1.expected_checkpoint_inventory(LOCK_ROOT, FREEZE_B_CHECKPOINTS)
    staging_record = json.loads(
        (LOCK_ROOT / "evidence/staging/report.json").read_text(encoding="utf-8")
    )
    excluded_alternatives = {
        str(row["model_id"]): list(row["excluded_alternative_files"])
        for row in staging_record["models"]
    }
    missing = sorted(set(attempted) - set(expected_inventory))
    if missing:
        raise RuntimeError(f"no recorded checkpoint inventory for {missing}")
    checkpoints = []
    for model_id in attempted:
        expected = expected_inventory[model_id]
        actual = inventory_checkpoint.remote(
            model_id,
            str(expected["revision"]),
            "files_sha256" in expected,
            excluded_alternatives.get(model_id, []),
        )
        checkpoints.append(m1.verify_checkpoint_inventory(expected, actual))
    _write_json(out / "checkpoints.json", checkpoints)

    # The append-only receipt file is the only authority on metered time. The
    # meter is always rebuilt from it, so a crash between any two writes can
    # neither double-count a payload nor lose a recorded second.
    receipts_path = out / "receipts.jsonl"
    state = m1.fold_receipts(m1.read_jsonl_rows(receipts_path, tolerate_partial=True))
    meter_lock = threading.Lock()
    stop = ext.ext_must_stop(float(state["cumulative_a100_hours"]))
    reserved_guarded_hours = 0.0

    def ledger_path(cell_key: str) -> Path:
        model_id, task_id = cell_key.split("::", 1)
        return out / "cells" / ext.model_key(model_id) / f"{task_id}.jsonl"

    quarantine_path = out / "quarantine.jsonl"

    def ledger_events() -> list[dict[str, Any]]:
        return [
            row
            for cell in verified_cells
            for row in m1.read_jsonl_rows(ledger_path(cell["cell_key"]))
        ]

    def attempt_rows() -> list[dict[str, Any]]:
        """Every unique attempt the screen ran, in execution order.

        A quarantined payload cost GPU time and was metered, so the global first
        gate counts it. It still never reaches an admission statistic.
        """
        order = [
            str(row["receipt_id"])
            for row in m1.read_jsonl_rows(receipts_path, tolerate_partial=True)
        ]
        return m1.actual_attempts(
            ledger_events(), m1.read_jsonl_rows(quarantine_path), receipt_order=order
        )

    def recorded_attempts() -> int:
        """Every actual attempt of the whole screen. The first gate is global."""
        return len(attempt_rows())

    def save_derived_state() -> None:
        _write_json(
            out / "meter-state.json",
            {
                "derived_from": "receipts.jsonl",
                "authoritative": False,
                "cumulative_a100_hours": state["cumulative_a100_hours"],
                "metered_a100_seconds": state["metered_a100_seconds"],
                "receipts": state["receipts"],
                "stop_reached": ext.ext_must_stop(float(state["cumulative_a100_hours"])),
                "session_elapsed_seconds": state["session_elapsed_seconds"],
            },
        )

    def absorb(
        cell_key: str, specs: list[dict[str, Any]], payload: dict[str, Any], wall_seconds: float
    ) -> list[dict[str, Any]]:
        """Meter, quarantine, and append one worker payload. Returns retry specs."""
        nonlocal stop
        with meter_lock:
            result = m1.absorb_payload(
                state=state,
                payload=payload,
                specs=specs,
                cell_key=cell_key,
                wall_seconds=wall_seconds,
                ledger_path=ledger_path(cell_key),
                receipts_path=receipts_path,
                quarantine_path=quarantine_path,
                extra={"modal_profile": modal_profile},
            )
            if ext.ext_must_stop(float(state["cumulative_a100_hours"])):
                stop = True
            save_derived_state()
        return cast(list[dict[str, Any]], result["retries"])

    def recover(model_id: str) -> None:
        """Re-absorb committed chunks after a local crash, without double metering."""
        payloads = recover_model.remote(run_id, model_id)
        model_calls = [
            call
            for cell_key, calls in calls_by_cell.items()
            if cell_key.startswith(f"{model_id}::")
            for call in calls
        ]
        for payload in sorted(
            payloads,
            key=lambda item: (
                str(item.get("worker_session_id")),
                float(item.get("session_elapsed_seconds", 0.0)),
            ),
        ):
            specs = m1.recover_dispatched_specs(model_calls, payload.get("dispatched_attempts"))
            cell_keys = {f"{spec['model_id']}::{spec['task_id']}" for spec in specs}
            if len(cell_keys) != 1:
                raise RuntimeError("a committed worker chunk spans multiple cells")
            absorb(cell_keys.pop(), specs, payload, 0.0)

    def run_cell(worker: Any, cell_key: str, gate_limit: int | None) -> None:
        """Execute one sealed cell, retried attempts before later calls."""
        path = ledger_path(cell_key)
        queue = m1.pending_calls(calls_by_cell[cell_key], m1.read_jsonl_rows(path))
        while queue:
            with meter_lock:
                if stop:
                    return
                take = chunk_size
                if gate_limit is not None:
                    remaining = gate_limit - recorded_attempts()
                    if remaining <= 0:
                        return
                    take = min(take, remaining)
            group, queue = queue[:take], queue[take:]
            invoked = time.monotonic()
            try:
                payload = worker.run_chunk.remote(run_id, group)
            except Exception as exc:  # preserve the failed attempt, then retry it
                payload = {
                    "worker_session_id": None,
                    "session_elapsed_seconds": time.monotonic() - invoked,
                    "load_seconds": None,
                    "chunk_wall_seconds": time.monotonic() - invoked,
                    "host": None,
                    "backend": None,
                    "events": [
                        {
                            **{key: spec[key] for key in m1.EVENT_KEYS},
                            "status": "worker_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        for spec in group
                    ],
                }
            queue = absorb(cell_key, group, payload, time.monotonic() - invoked) + queue

    def run_model(model_id: str, gate_limit: int | None = None) -> None:
        """Reserve this model's guarded hours, then run its cells in sealed order."""
        nonlocal stop, reserved_guarded_hours
        recover(model_id)
        cells = [cell for cell in verified_cells if cell["cell_key"].startswith(f"{model_id}::")]
        base = float(forecast["model_load_a100_hours"][model_id]) + sum(
            m1.cell_base_hours(forecast, cell["cell_key"], include_load=False) for cell in cells
        )
        guarded = ext.FORECAST_MULTIPLIER * base
        with meter_lock:
            if stop:
                return
            # Every concurrent worker's guard is held, so two models cannot both
            # fit a block that only one of them has room for.
            decision = ext.ext_may_start(
                float(state["cumulative_a100_hours"]) + reserved_guarded_hours, base
            )
            if not decision["start"]:
                stop = True
                _write_json(out / "stopped.json", {"model_id": model_id, **decision})
                return
            reserved_guarded_hours += guarded
        try:
            worker = SoloWorker(model_id=model_id)  # type: ignore[call-arg]
            for cell in cells:
                if stop:
                    return
                run_cell(worker, cell["cell_key"], gate_limit)
        finally:
            with meter_lock:
                reserved_guarded_hours -= guarded

    release_path = out / "release-decision.json"
    reforecast_path = out / "reforecast-1000.json"
    if release_path.exists() or reforecast_path.exists():
        # A resumed run re-proves the gate. The file's presence proves nothing.
        if not (release_path.exists() and reforecast_path.exists()):
            raise RuntimeError("the first-gate evidence is incomplete; refusing to resume")
        verified_gate = m1.verify_release_decision(
            json.loads(release_path.read_text(encoding="utf-8")),
            json.loads(reforecast_path.read_text(encoding="utf-8")),
            plan_sha256=plan["plan_sha256"],
            recorded_attempts=recorded_attempts(),
        )
        _write_json(out / "release-verified.json", verified_gate)
    else:
        # One global gate for the whole screen: a single worker until the first
        # 1,000 attempts are metered and forecast.
        run_model(attempted[0], gate_limit=m1.FIRST_REFORECAST_ATTEMPTS)
        reforecast = m1.m1_reforecast(
            attempt_rows(),
            planned_calls=int(plan["planned_calls"]),
            planned_models=len(attempted),
        )
        if reforecast is None:
            raise RuntimeError("the first-1,000-attempt gate did not collect enough attempts")
        _write_json(reforecast_path, reforecast)
        decision = m1.release_decision(
            float(state["cumulative_a100_hours"]), reforecast, plan_sha256=plan["plan_sha256"]
        )
        _write_json(release_path, decision)
        if not decision["release_remaining_work"]:
            print(json.dumps({"output_dir": str(out), "stopped_at_gate": decision}, indent=2))
            return

    if not stop:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_parallel_models, len(attempted))
        ) as executor:
            futures = [executor.submit(run_model, model_id) for model_id in attempted]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    summaries = []
    for model_id in ext.EXT_MODEL_IDS:
        status = lock["candidate_status"][model_id]
        for task_id in ext.EXT_TASK_KEYS:
            cell_key = f"{model_id}::{task_id}"
            events = m1.read_jsonl_rows(ledger_path(cell_key)) if cell_key in calls_by_cell else []
            summary = m1.summarize_cell(
                model_id, task_id, frames[task_id], events, candidate_status=status
            )
            _write_json(out / "summaries" / ext.model_key(model_id) / f"{task_id}.json", summary)
            summaries.append(summary)
    inputs = m1.admission_inputs(
        summaries,
        lock,
        seal=seal,
        meter={"cumulative_a100_hours": state["cumulative_a100_hours"]},
    )
    _write_json(out / "admission-inputs.json", inputs)
    _write_json(
        out / "SUMMARY.json",
        {
            "run_id": run_id,
            "output_dir": str(out),
            "plan_sha256": plan["plan_sha256"],
            "cumulative_a100_hours": state["cumulative_a100_hours"],
            "meter_receipts": state["receipts"],
            "stopped": stop,
            "screen_complete": inputs["screen_complete"],
            "admitted_cells": inputs["admitted_cells"],
            "quarantined_payloads": len(m1.read_jsonl_rows(quarantine_path)),
        },
    )
    print(json.dumps(json.loads((out / "SUMMARY.json").read_text(encoding="utf-8")), indent=2))
