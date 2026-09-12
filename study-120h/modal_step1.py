"""Run the frozen Step 1 solo screen on Modal.

Model downloads and exact tokenizer checks run on CPU. The ``screen`` stage
starts one A100 container per model and permits two model containers to run at
the same time. Each model still uses exactly one A100 and tensor parallel size
one.
"""

import concurrent.futures
import datetime as dt
import json
import sys
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ieee-bigdata-step1"
LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/study-120h")
MODEL_CACHE = "/models"
GPU_TYPE = "A100"
CHUNK_SIZE = 64

MODEL_IDS = (
    "Qwen/Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
INITIAL_SCREENS = (
    ("qwen_civil", "civil_comments", MODEL_IDS[0]),
    ("qwen_secondtask", "ag_news", MODEL_IDS[0]),
    ("mistral_civil", "civil_comments", MODEL_IDS[1]),
    ("mistral_secondtask", "ag_news", MODEL_IDS[1]),
)

volume = modal.Volume.from_name("ieee-bigdata-step1-models", create_if_missing=True)
base_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==1.31.0",
    "numpy==2.2.6",
    "scipy==1.16.3",
    "vllm==0.28.0",
)
runtime_image = (
    base_image.env(
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


@app.function(image=base_image, volumes={MODEL_CACHE: volume}, timeout=3600)
def download_model(model_id: str, revision: str) -> dict[str, str]:
    """Download one exact checkpoint without allocating a GPU."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model_id, revision=revision, cache_dir=MODEL_CACHE)
    volume.commit()
    return {"model_id": model_id, "revision": revision, "path": path}


@app.function(
    image=runtime_image,
    volumes={MODEL_CACHE: volume},
    cpu=4,
    memory=16384,
    timeout=3600,
)
def exact_preflight(cell_id: str, frame_task_id: str, model_id: str) -> dict[str, Any]:
    """Count every frozen prompt type with the pinned tokenizer; generate nothing."""
    _remote_imports()
    from step1.backends import PinnedTokenizerCounter
    from step1.frozen import load
    from step1.runner import preflight

    frozen = load()
    spec = frozen.protocol["models"][model_id]
    counter = PinnedTokenizerCounter(
        model_id,
        str(spec["tokenizer_revision"]),
        cache_dir=MODEL_CACHE,
        enable_thinking=bool(spec["enable_thinking"]),
        local_files_only=True,
    )
    return preflight(frozen, cell_id, frame_task_id, counter, model_id=model_id)


@app.cls(
    image=runtime_image,
    volumes={MODEL_CACHE: volume},
    gpu=GPU_TYPE,
    max_containers=1,
    scaledown_window=30,
    startup_timeout=1200,
    timeout=7200,
)
class ModelWorker:
    """One pinned model on one A100 for the life of the container."""

    model_id: str = modal.parameter()

    @modal.enter()
    def load_model(self) -> None:
        _remote_imports()
        from step1.backends import VLLMBackend, host_facts
        from step1.frozen import load

        self.started = time.monotonic()
        self.frozen = load()
        spec = self.frozen.protocol["models"][self.model_id]
        self.backend = VLLMBackend(
            self.model_id,
            str(spec["revision"]),
            str(spec["tokenizer_revision"]),
            self.frozen.serving,
            self.frozen.decoding,
            enable_thinking=bool(spec["enable_thinking"]),
        )
        self.host = host_facts()

    @modal.method()
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate one solo-only chunk and return append-ready rows."""
        from step1.backends import GenRequest
        from step1.ledger import RunMeta, rows_for
        from step1.plan import build_call, doc_texts, retry_job

        jobs = list(payload["jobs"])
        conditions = {str(job["condition"]) for job in jobs}
        if conditions != {"solo"}:
            raise RuntimeError(f"Step 1 refuses non-solo conditions: {sorted(conditions)}")

        frame = str(payload["frame_task_id"])
        items = self.frozen.items(frame)
        references = {item.doc_id: item.reference for item in items}
        strata = {item.doc_id: item.length_stratum for item in items}
        texts = doc_texts(self.frozen, frame)
        calls = [build_call(self.frozen, job, frame, texts) for job in jobs]
        requests = [
            GenRequest(
                call_id=call.call_id,
                condition=call.condition,
                target_ids=tuple(call.target_ids),
                messages=call.messages,
                max_tokens=call.max_tokens,
            )
            for call in calls
        ]

        started = time.perf_counter()
        first_generation_failed = False
        try:
            first_results = self.backend.generate(requests)
        except Exception:  # an infrastructure failure gets one linked retry
            first_results = []
            first_generation_failed = True
        first_seconds = time.perf_counter() - started
        generated = {result.call_id: result for result in first_results}
        first_meta = RunMeta(
            run_id=str(payload["run_id"]),
            serving_session=str(payload["serving_session"]),
            engine_version=str(self.host["installed_engine_version_string"]),
            chunk_id=str(payload["chunk_id"]),
            chunk_size=len(calls),
            chunk_seconds=first_seconds,
        )
        rows: list[dict[str, Any]] = []
        for call in calls:
            rows.extend(
                rows_for(
                    self.frozen,
                    call,
                    generated.get(call.call_id),
                    frame,
                    references,
                    strata,
                    first_meta,
                )
            )

        retry_jobs = []
        for job, call in zip(jobs, calls, strict=True):
            result = generated.get(call.call_id)
            if first_generation_failed or result is None or result.finish_reason == "length":
                retry_jobs.append(retry_job(job, 1))
        retry_calls = [build_call(self.frozen, job, frame, texts) for job in retry_jobs]
        retry_results = []
        retry_seconds = 0.0
        if retry_calls:
            retry_requests = [
                GenRequest(
                    call_id=call.call_id,
                    condition=call.condition,
                    target_ids=tuple(call.target_ids),
                    messages=call.messages,
                    max_tokens=call.max_tokens,
                )
                for call in retry_calls
            ]
            retry_started = time.perf_counter()
            try:
                retry_results = self.backend.generate(retry_requests)
            except Exception:
                retry_results = []
            retry_seconds = time.perf_counter() - retry_started
            retry_generated = {result.call_id: result for result in retry_results}
            retry_meta = RunMeta(
                run_id=str(payload["run_id"]),
                serving_session=str(payload["serving_session"]),
                engine_version=str(self.host["installed_engine_version_string"]),
                chunk_id=f"{payload['chunk_id']}-retry",
                chunk_size=len(retry_calls),
                chunk_seconds=retry_seconds,
            )
            for call in retry_calls:
                rows.extend(
                    rows_for(
                        self.frozen,
                        call,
                        retry_generated.get(call.call_id),
                        frame,
                        references,
                        strata,
                        retry_meta,
                    )
                )

        all_results = first_results + retry_results
        return {
            "rows": rows,
            "chunk": {
                "chunk_id": payload["chunk_id"],
                "n_calls": len(calls) + len(retry_calls),
                "n_first_calls": len(calls),
                "n_retry_calls": len(retry_calls),
                "chunk_seconds": first_seconds + retry_seconds,
                "prompt_tokens_sum": sum(r.prompt_tokens for r in all_results),
                "output_tokens_sum": sum(r.output_tokens for r in all_results),
            },
            "host": self.host,
            "container_seconds": time.monotonic() - self.started,
        }

    @modal.method()
    def status(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "backend": self.backend.describe(),
            "container_seconds": time.monotonic() - self.started,
        }


def _local_imports() -> Any:
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from step1.frozen import load

    return load()


def _write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_amendment(frozen: Any) -> Path:
    from step1.amendment import write
    from step1.paths import AMENDMENTS

    existing = sorted(AMENDMENTS.glob("*-step1-execution-budget.json"))
    if existing:
        return existing[-1]
    path, _ = write(frozen)
    return path


def _prepare(frozen: Any, model_ids: Iterable[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for model_id in model_ids:
        revision = frozen.model_revision(model_id)
        print(f"downloading pinned model on CPU: {model_id}", flush=True)
        records.append(download_model.remote(model_id, revision))
    return records


def _preflight(
    frozen: Any, triples: Iterable[tuple[str, str, str]], paths: Any
) -> dict[tuple[str, str, str], dict[str, Any]]:
    triples = list(triples)
    reports = list(exact_preflight.starmap(triples, order_outputs=True))
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for triple, report in zip(triples, reports, strict=True):
        cell_id, frame, model_id = triple
        path = paths.preflight_report(cell_id, frame, model_id)
        _write_json(path, report)
        indexed[triple] = report
        print(
            f"preflight {cell_id}/{frame}: {report['n_calls_checked']} prompts, "
            f"{report['n_over_budget']} over budget",
            flush=True,
        )
    failed = [key for key, report in indexed.items() if not report["ok"]]
    if failed:
        raise RuntimeError(f"exact tokenizer preflight failed; no GPU started: {failed}")
    return indexed


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _run_model(
    frozen: Any,
    model_id: str,
    cells: list[tuple[str, str]],
    paths: Any,
    run_id: str,
) -> dict[str, Any]:
    from step1.ledger import append, append_run, completed_call_ids
    from step1.plan import solo_jobs

    worker_id = model_id.replace("/", "_")
    failure_path = paths.load_failure(model_id)
    if failure_path.exists():
        return json.loads(failure_path.read_text(encoding="utf-8"))
    worker_run_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
    serving_session = f"{worker_run_id}-{worker_id}"
    payloads: list[dict[str, Any]] = []
    for cell_id, frame in cells:
        ledger_path = paths.ledger(cell_id, frame, model_id)
        done = completed_call_ids(ledger_path)
        jobs = [
            job
            for job in solo_jobs(frozen, cell_id, frame, model_id=model_id)
            if str(job["call_id"]) not in done
        ]
        for index, chunk in enumerate(_chunks(jobs, CHUNK_SIZE)):
            payloads.append(
                {
                    "run_id": worker_run_id,
                    "serving_session": serving_session,
                    "chunk_id": f"{worker_run_id}-{worker_id}-{cell_id}-{index:04d}",
                    "cell_id": cell_id,
                    "frame_task_id": frame,
                    "jobs": chunk,
                }
            )
    if not payloads:
        prior_hosts = sorted(paths.host.glob(f"*-{worker_id}.json"))
        if not prior_hosts:
            raise RuntimeError(
                f"{model_id} has complete ledgers but no prior host record; cannot resume safely"
            )
        host_record = json.loads(prior_hosts[-1].read_text(encoding="utf-8"))
        return {
            "model_id": model_id,
            "n_calls": 0,
            "chunks": [],
            "skipped": True,
            "host": host_record,
            "host_path": str(prior_hosts[-1]),
        }

    worker = ModelWorker(model_id=model_id)
    local_started = time.perf_counter()
    chunks: list[dict[str, Any]] = []
    n_calls = 0
    host: dict[str, Any] | None = None
    try:
        for result in worker.infer.map(payloads, order_outputs=False):
            rows = list(result["rows"])
            if not rows:
                raise RuntimeError("a successful remote chunk returned no rows")
            cell_id = str(rows[0]["cell_id"])
            frame = str(rows[0]["frame_task_id"])
            saved = append(paths.ledger(cell_id, frame, model_id), rows)
            if saved["conflicts_rejected"]:
                raise RuntimeError(f"conflicting outputs rejected for {cell_id}/{frame}")
            chunks.append(dict(result["chunk"]))
            n_calls += int(result["chunk"]["n_calls"])
            host = dict(result["host"])
            print(f"saved {model_id}: {n_calls} solo calls", flush=True)
        remote_status = worker.status.remote()
    except Exception as exc:
        local_wall = time.perf_counter() - local_started
        failure = {
            "model_id": model_id,
            "n_calls": n_calls,
            "chunks": chunks,
            "load_failed": n_calls == 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gpu_seconds_billed": local_wall,
            "observed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "frozen_max_model_len": int(frozen.serving["max_model_len"]),
            "failure_path": str(failure_path),
        }
        if not failure["load_failed"]:
            raise
        _write_json(failure_path, failure)
        append_run(
            paths.runs,
            {
                "run_id": worker_run_id,
                "worker_id": worker_id,
                "model_id": model_id,
                "gpu_name": GPU_TYPE,
                "gpu_seconds_billed": local_wall,
                "local_wall_seconds": local_wall,
                "load_failed": True,
                "error_type": type(exc).__name__,
                "accounting_note": "conservative local wall time for the failed GPU load",
            },
        )
        return failure
    host = dict(remote_status["host"] if host is None else host)
    local_wall = time.perf_counter() - local_started
    container_seconds = float(remote_status["container_seconds"])
    host_path = paths.host_record(f"{worker_run_id}-{worker_id}")
    host_record = {
        **host,
        "worker_id": worker_id,
        "model_id": model_id,
        "run_id": worker_run_id,
        "collected_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    _write_json(host_path, host_record)
    append_run(
        paths.runs,
        {
            "run_id": worker_run_id,
            "worker_id": worker_id,
            "model_id": model_id,
            "gpu_name": host["gpu_name"],
            "gpu_seconds_billed": max(local_wall, container_seconds),
            "container_seconds": container_seconds,
            "local_wall_seconds": local_wall,
            "accounting_note": (
                "conservative local wall time from remote invocation through final status; "
                "it includes queue/startup and is never less than measured container lifetime; "
                "workers are summed, not converted to wall time"
            ),
        },
    )
    return {
        "model_id": model_id,
        "n_calls": n_calls,
        "chunks": chunks,
        "wall_seconds": local_wall,
        "container_seconds": container_seconds,
        "host": host_record,
        "host_path": str(host_path),
    }


def _run_concurrent(
    frozen: Any,
    work: dict[str, list[tuple[str, str]]],
    paths: Any,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, len(work))) as pool:
        futures = {
            pool.submit(_run_model, frozen, model_id, cells, paths, run_id): model_id
            for model_id, cells in work.items()
        }
        results: dict[str, dict[str, Any]] = {}
        for future, model_id in futures.items():
            try:
                results[model_id] = future.result()
            except Exception as exc:
                results[model_id] = {
                    "model_id": model_id,
                    "n_calls": 0,
                    "chunks": [],
                    "load_failed": True,
                    "error_type": type(exc).__name__,
                }
        return results


def _merge_run(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Keep every timed chunk when one model is invoked in more than one phase."""
    return {
        **previous,
        **current,
        "n_calls": int(previous.get("n_calls", 0)) + int(current.get("n_calls", 0)),
        "chunks": list(previous.get("chunks", [])) + list(current.get("chunks", [])),
        "wall_seconds": float(previous.get("wall_seconds", 0.0))
        + float(current.get("wall_seconds", 0.0)),
        "container_seconds": float(previous.get("container_seconds", 0.0))
        + float(current.get("container_seconds", 0.0)),
    }


def _host_for(run: dict[str, Any]) -> dict[str, Any]:
    return dict(run.get("host", {}))


def _measurement_runs(paths: Any) -> dict[str, list[dict[str, Any]]]:
    """Rebuild timed runs from append-only ledgers so report replay is lossless."""
    from step0.hashing import read_jsonl

    timing_by_run = {
        str(row["run_id"]): row for row in read_jsonl(paths.runs) if not row.get("load_failed")
    }
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for ledger_path in sorted(paths.results.glob("*/solo.jsonl")):
        for row in read_jsonl(ledger_path):
            key = (str(row["model_id"]), str(row["run_id"]))
            run = grouped.setdefault(key, {"chunks": {}})
            chunk_id = str(row["chunk_id"])
            chunk = run["chunks"].setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "n_calls": 0,
                    "chunk_seconds": float(row["chunk_seconds"]),
                    "prompt_tokens_sum": 0,
                    "output_tokens_sum": 0,
                },
            )
            chunk["n_calls"] += 1
            chunk["prompt_tokens_sum"] += int(row.get("prompt_tokens") or 0)
            chunk["output_tokens_sum"] += int(row.get("output_tokens") or 0)

    by_model: dict[str, list[dict[str, Any]]] = {}
    for (model_id, run_id), record in grouped.items():
        timing = timing_by_run.get(run_id, {})
        by_model.setdefault(model_id, []).append(
            {
                "run_id": run_id,
                "chunks": list(record["chunks"].values()),
                "wall_seconds": float(
                    timing.get("local_wall_seconds", timing.get("gpu_seconds_billed", 0.0))
                ),
            }
        )
    return by_model


def _screen_one(
    frozen: Any,
    paths: Any,
    preflights: dict[tuple[str, str, str], dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    cell_id: str,
    frame: str,
    model_id: str,
) -> dict[str, Any]:
    from step1.screen import screen_cell

    return screen_cell(
        frozen,
        paths.ledger(cell_id, frame, model_id),
        cell_id,
        frame,
        model_id=model_id,
        preflight_report=preflights[(cell_id, frame, model_id)],
        host=_host_for(runs[model_id]),
        runs_path=paths.runs,
        load_failed=bool(runs[model_id].get("load_failed")),
    )


def _execute_screen(frozen: Any, paths: Any) -> dict[str, Any]:
    from step1.benchmark import forecast_cell, measured, remaining_ledger, render_report
    from step1.ledger import meter
    from step1.screen import decide_replacement, decide_task, render_screen

    preflights = _preflight(frozen, INITIAL_SCREENS, paths)
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    initial_work = {
        MODEL_IDS[0]: [("qwen_civil", "civil_comments"), ("qwen_secondtask", "ag_news")],
        MODEL_IDS[1]: [
            ("mistral_civil", "civil_comments"),
            ("mistral_secondtask", "ag_news"),
        ],
    }
    runs = _run_concurrent(frozen, initial_work, paths, run_id)
    initial_screens = [
        _screen_one(frozen, paths, preflights, runs, cell, frame, model)
        for cell, frame, model in INITIAL_SCREENS
    ]

    replacement_trigger = decide_replacement(frozen, initial_screens)
    replacement = replacement_trigger
    effective_second = MODEL_IDS[1]
    effective_screens = list(initial_screens)
    if replacement_trigger["replace"]:
        effective_second = str(replacement_trigger["replacement_model_id"])
        _prepare(frozen, [effective_second])
        replacement_triples = [
            ("mistral_civil", "civil_comments", effective_second),
            ("mistral_secondtask", "ag_news", effective_second),
        ]
        preflights.update(_preflight(frozen, replacement_triples, paths))
        replacement_run = _run_model(
            frozen,
            effective_second,
            [("mistral_civil", "civil_comments"), ("mistral_secondtask", "ag_news")],
            paths,
            run_id,
        )
        runs[effective_second] = replacement_run
        effective_screens = [s for s in initial_screens if s["model_id"] != MODEL_IDS[1]]
        effective_screens += [
            _screen_one(frozen, paths, preflights, runs, cell, frame, effective_second)
            for cell, frame, _ in replacement_triples
        ]
        replacement_screens = [
            screen for screen in effective_screens if screen["model_id"] == effective_second
        ]
        if replacement_screens and all(screen["gate"]["ok"] for screen in replacement_screens):
            replacement = {
                **replacement_trigger,
                "replacement_passed_load_gate": True,
                "replacements_used": 1,
            }
        else:
            policy = frozen.protocol["second_model_replacement_policy"]
            replacement = {
                **replacement_trigger,
                "replace": False,
                "next_action": None,
                "replacement_passed_load_gate": False,
                "replacements_used": 1,
                "reason": policy["if_no_replacement_passes"],
                "claim_narrowed": True,
            }

    task_decision = decide_task(frozen, effective_screens)
    if task_decision["chosen_second_task"] is None:
        fallback = [
            ("qwen_secondtask", "multi_nli", MODEL_IDS[0]),
            ("mistral_secondtask", "multi_nli", effective_second),
        ]
        preflights.update(_preflight(frozen, fallback, paths))
        fallback_runs = _run_concurrent(
            frozen,
            {
                MODEL_IDS[0]: [("qwen_secondtask", "multi_nli")],
                effective_second: [("mistral_secondtask", "multi_nli")],
            },
            paths,
            run_id,
        )
        for model_id, fallback_run in fallback_runs.items():
            runs[model_id] = _merge_run(runs.get(model_id, {}), fallback_run)
        fallback_screens = [
            _screen_one(frozen, paths, preflights, runs, cell, frame, model)
            for cell, frame, model in fallback
        ]
        effective_screens += fallback_screens
        task_decision = decide_task(frozen, effective_screens)

    screen_record = {
        "freeze_stamp": frozen.stamp,
        "written_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "screens": initial_screens + [s for s in effective_screens if s not in initial_screens],
        "effective_screens": effective_screens,
        "task_decision": task_decision,
        "replacement_decision": replacement,
        "effective_second_model": (
            effective_second if replacement.get("replacement_passed_load_gate", True) else None
        ),
    }
    _write_json(paths.screen / "SCREEN.json", screen_record)
    (paths.screen / "SCREEN.md").write_text(render_screen(screen_record), encoding="utf-8")

    per_model = {
        model: measured(model_runs)
        for model, model_runs in _measurement_runs(paths).items()
        if model_runs
    }
    forecasts: list[dict[str, Any]] = []
    for (_cell, _frame, model), preflight_report in preflights.items():
        if model not in per_model:
            continue
        forecasts.append(forecast_cell(preflight_report, per_model[model]))
    ledger = remaining_ledger(frozen.protocol, solo_done_per_cell=2048)
    meter_record = meter(paths.runs)
    throughput = {
        "freeze_stamp": frozen.stamp,
        "written_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "per_model": per_model,
        "forecasts": forecasts,
        "remaining_ledger": ledger,
        "meter": meter_record,
    }
    _write_json(paths.benchmark / "THROUGHPUT.json", throughput)
    (paths.benchmark / "THROUGHPUT.md").write_text(
        render_report(frozen.stamp, per_model, forecasts, ledger, meter_record),
        encoding="utf-8",
    )
    return screen_record


@app.local_entrypoint()
def main(stage: str = "all") -> None:
    """Run ``prepare``, ``preflight``, ``screen``, or the complete ``all`` sequence."""
    if stage not in {"prepare", "preflight", "screen", "all"}:
        raise SystemExit("stage must be prepare, preflight, screen, or all")
    frozen = _local_imports()
    from step1.paths import Paths
    from step1.report import verify_step0_seal

    paths = Paths()
    seal = verify_step0_seal(frozen)
    if not seal["ok"]:
        raise SystemExit(f"Step 0 seal failed: {seal['problems']}")
    amendment_path = _ensure_amendment(frozen)
    print(f"execution amendment: {amendment_path}", flush=True)
    if stage in {"prepare", "all"}:
        _prepare(frozen, MODEL_IDS)
    if stage == "preflight":
        _preflight(frozen, INITIAL_SCREENS, paths)
    if stage in {"screen", "all"}:
        record = _execute_screen(frozen, paths)
        print(json.dumps(record["task_decision"], indent=2, sort_keys=True))
        print(json.dumps(record["replacement_decision"], indent=2, sort_keys=True))
