"""Run Gate A's frozen primary and conditional control phases on Modal."""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ieee-bigdata-gate-a"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
VLLM_VERSION = "0.29.0"
MODEL_CACHE = "/model-cache"
REMOTE_ROOT = Path("/root/gate-a")
LOCAL_ROOT = Path(__file__).resolve().parent
LOCAL_OUT = LOCAL_ROOT / "artifacts/full"
REMOTE_OUT = REMOTE_ROOT / "artifacts/full"
GPU_TYPE = "L40S"
CHUNK_SIZE = 64
MAX_WORKERS = 4


def download_model() -> None:
    """Bake the exact checkpoint into the image before allocating GPUs."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=MODEL_CACHE,
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface-hub==1.31.0",
        "numpy==2.2.6",
        f"vllm=={VLLM_VERSION}",
    )
    .run_function(download_model)
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
    .add_local_file(
        LOCAL_ROOT / "configs/full.toml",
        remote_path=str(REMOTE_ROOT / "configs/full.toml"),
    )
    .add_local_dir(LOCAL_OUT / "data", remote_path=str(REMOTE_OUT / "data"))
    .add_local_dir(LOCAL_OUT / "plan", remote_path=str(REMOTE_OUT / "plan"))
)

app = modal.App(APP_NAME)


def _load_config(root: Path) -> Any:
    sys.path.insert(0, str(root / "src"))
    from batchgate.config import load_config

    cfg = load_config(root / "configs/full.toml")
    if cfg.model.model_id != MODEL_ID or cfg.model.revision != MODEL_REVISION:
        raise RuntimeError("Modal model constants do not match the frozen config")
    return cfg


def _eligible_jobs(cfg: Any, phase: str) -> list[Any]:
    from batchgate.plan import read_plan

    allowed = {
        "primary": {"solo", "joint"},
        "controls": {"filler"},
    }[phase]
    return [job for job in read_plan(cfg) if job.condition in allowed]


class _TokenizerCounter:
    name = "qwen_tokenizer"

    def __init__(self, tokenizer: Any, enable_thinking: bool) -> None:
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return len(self.tokenizer(rendered)["input_ids"])


@app.function(image=image, cpu=4, memory=8192, timeout=1200)
def tokenizer_preflight(phase: str) -> dict[str, Any]:
    """Count every prompt with the exact pinned tokenizer without a GPU."""
    from transformers import AutoTokenizer

    cfg = _load_config(REMOTE_ROOT)
    from batchgate.runner import load_docs, preflight, verify_plan_file

    verify_plan_file(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=MODEL_CACHE,
        local_files_only=True,
    )
    report = preflight(
        cfg,
        _eligible_jobs(cfg, phase),
        load_docs(cfg),
        _TokenizerCounter(tokenizer, cfg.model.enable_thinking),
    )
    return asdict(report)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    max_containers=MAX_WORKERS,
    scaledown_window=30,
    startup_timeout=900,
    timeout=1800,
)
class GateAModel:
    """One pinned Qwen vLLM instance per L40S, reused across prompt chunks."""

    @modal.enter()
    def load(self) -> None:
        self.cfg = _load_config(REMOTE_ROOT)
        from batchgate.backends import make_backend
        from batchgate.runner import load_docs, verify_plan_file

        verify_plan_file(self.cfg)
        self.docs = load_docs(self.cfg)
        gold = {doc_id: record.gold for doc_id, record in self.docs.items()}
        self.backend = make_backend(self.cfg, gold, "vllm")

    @modal.method()
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        from batchgate.backends import GenRequest
        from batchgate.plan import Job, build_messages
        from batchgate.runner import _rows_for, preflight

        started = time.perf_counter()
        raw_jobs = payload["jobs"]
        try:
            jobs = []
            for raw in raw_jobs:
                item = dict(raw)
                for key in ("doc_order", "target_ids", "target_slots"):
                    item[key] = tuple(item[key])
                jobs.append(Job(**item))

            check = preflight(self.cfg, jobs, self.docs, self.backend)
            if check.over_budget:
                raise RuntimeError(f"exact tokenizer rejected {len(check.over_budget)} jobs")

            requests = [
                GenRequest(
                    job_id=job.job_id,
                    condition=job.condition,
                    assignment=job.assignment,
                    real_batch_id=job.real_batch_id,
                    target_ids=job.target_ids,
                    messages=build_messages(job, self.docs),
                    max_tokens=job.max_tokens,
                )
                for job in jobs
            ]
            generated = {result.job_id: result for result in self.backend.generate(requests)}
            rows: list[dict[str, Any]] = []
            metadata = self.backend.describe()
            for job in jobs:
                rows.extend(
                    _rows_for(
                        self.cfg,
                        job,
                        generated[job.job_id],
                        self.docs,
                        payload["run_id"],
                        metadata,
                        len(jobs),
                    )
                )
            return {
                "records": rows,
                "job_ids": [job.job_id for job in jobs],
                "seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            return {
                "error": {
                    "job_ids": [str(job["job_id"]) for job in raw_jobs],
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            }


def _local_pending(phase: str) -> list[Any]:
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from batchgate.config import load_config
    from batchgate.runner import completed_job_ids, verify_plan_file

    cfg = load_config(LOCAL_ROOT / "configs/full.toml")
    verify_plan_file(cfg)
    done = completed_job_ids(LOCAL_OUT / "raw/predictions.jsonl")
    return [job for job in _eligible_jobs(cfg, phase) if job.job_id not in done]


def _payloads(jobs: list[Any], run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "jobs": [asdict(job) for job in jobs[start : start + CHUNK_SIZE]],
        }
        for start in range(0, len(jobs), CHUNK_SIZE)
    ]


def _save_results(results: Any) -> dict[str, Any]:
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from batchgate.util import append_jsonl, write_json

    out_path = LOCAL_OUT / "raw/predictions.jsonl"
    errors: list[dict[str, Any]] = []
    jobs_written = 0
    rows_written = 0
    remote_seconds = 0.0
    for result in results:
        if "error" in result:
            errors.append(result["error"])
            continue
        append_jsonl(out_path, result["records"])
        jobs_written += len(result["job_ids"])
        rows_written += len(result["records"])
        remote_seconds += float(result["seconds"])
        print(f"saved {jobs_written} jobs", flush=True)

    if errors:
        write_json(LOCAL_OUT / "raw/modal_errors.json", errors)
    return {
        "jobs_written": jobs_written,
        "rows_written": rows_written,
        "remote_inference_seconds": remote_seconds,
        "errors": errors,
    }


def _write_preflight(phase: str, report: dict[str, Any]) -> None:
    path = LOCAL_OUT / "plan" / f"tokenizer_preflight_{phase}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _analyze() -> dict[str, Any]:
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from batchgate.analyze import analyze, write_outputs
    from batchgate.config import load_config

    cfg = load_config(LOCAL_ROOT / "configs/full.toml")
    result = analyze(cfg)
    write_outputs(cfg, result)
    return result["verdict"]


@app.local_entrypoint()
def main(stage: str = "preflight", workers: int = 1) -> None:
    """Run preflight, one-chunk calibration, primary, or controls."""
    if stage == "preflight":
        report = tokenizer_preflight.remote("primary")
        _write_preflight("primary", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["over_budget"]:
            raise SystemExit("exact tokenizer preflight failed; no GPU was started")
        return
    if stage not in {"calibration", "primary", "controls"}:
        raise SystemExit("stage must be preflight, calibration, primary, or controls")
    if not 1 <= workers <= MAX_WORKERS:
        raise SystemExit(f"workers must be between 1 and {MAX_WORKERS}")
    if stage == "calibration" and workers != 1:
        raise SystemExit("calibration requires one worker")

    phase = "controls" if stage == "controls" else "primary"
    jobs = _local_pending(phase)
    if stage == "calibration":
        jobs = jobs[:CHUNK_SIZE]
    if not jobs:
        print(f"no pending {phase} jobs")
        return

    report = tokenizer_preflight.remote(phase)
    _write_preflight(phase, report)
    if report["over_budget"]:
        raise SystemExit("exact tokenizer preflight failed; no GPU was started")

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    payloads = _payloads(jobs, run_id)
    started = time.perf_counter()
    model_cls = GateAModel.with_options(max_containers=workers)
    model = model_cls()
    summary = _save_results(model.infer.map(payloads, order_outputs=False))
    summary.update(
        {
            "stage": stage,
            "workers": workers,
            "gpu": GPU_TYPE,
            "jobs_requested": len(jobs),
            "chunks": len(payloads),
            "wall_seconds": time.perf_counter() - started,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["errors"]:
        raise SystemExit("Modal chunks failed; inspect artifacts/full/raw/modal_errors.json")
    if stage != "calibration":
        print(json.dumps(_analyze(), indent=2, sort_keys=True))
