"""Run Gate K with Mistral Small 3.1 Base on Modal."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ieee-bigdata-gate-k"
MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Base-2503"
MODEL_REVISION = "ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc"
REMOTE_ROOT = Path("/root/gate-k")
REMOTE_OUT = REMOTE_ROOT / "artifacts/mistral-24b-base"
MODEL_CACHE = "/model-cache"
LOCAL_ROOT = Path(__file__).resolve().parent
LOCAL_OUT = LOCAL_ROOT / "artifacts/mistral-24b-base"


def download_model() -> None:
    """Bake the exact model revision into the image before allocating GPUs."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=MODEL_CACHE,
        ignore_patterns=["consolidated.safetensors"],
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.15.0",
        "huggingface-hub==1.31.0",
        "numpy==2.2.6",
        "scikit-learn==1.9.1",
        "scipy==1.17.1",
        "torch==2.14.0",
        "transformers==5.17.0",
    )
    .run_function(download_model)
    .env(
        {
            "HF_HOME": MODEL_CACHE,
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    # Add changing experiment code after the expensive model layer.
    .add_local_dir(LOCAL_ROOT, remote_path=REMOTE_ROOT)
)

app = modal.App(APP_NAME)


def _remote_config() -> Any:
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from gatek.config import load_config

    cfg = load_config(REMOTE_ROOT / "configs/full.toml")
    return replace(
        cfg,
        run=replace(
            cfg.run,
            backend="transformers",
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=MODEL_CACHE,
            hidden_layers=[27],
            max_prompt_tokens=131072,
        ),
    )


@app.function(image=image, cpu=4, memory=8192, timeout=1200)
def preflight() -> dict[str, Any]:
    """Verify every frozen prompt against the exact tokenizer without a GPU."""
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from gatek.planning import tokenizer_preflight

    return tokenizer_preflight(_remote_config(), REMOTE_OUT, allow_download=False)


@app.cls(
    image=image,
    gpu="H100",
    max_containers=10,
    scaledown_window=60,
    startup_timeout=600,
    timeout=300,
)
class GateKModel:
    """One cached Mistral Base instance per H100, reused across frozen jobs."""

    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, str(REMOTE_ROOT / "src"))
        import numpy as np

        from gatek.runner import make_backend

        self.cfg = _remote_config()
        self.npz = np.load(REMOTE_OUT / "synthetic/data.npz")
        self.backend = make_backend(self.cfg)

    @modal.method()
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        import gc

        import torch

        from gatek.planning import Job
        from gatek.runner import execute_job

        try:
            return {"record": execute_job(Job(**payload), self.cfg, self.backend, self.npz)}
        except Exception as exc:  # return the job id so failures remain auditable
            return {
                "error": {
                    "job_id": payload["job_id"],
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            }
        finally:
            gc.collect()
            torch.cuda.empty_cache()


def _local_jobs(stage: str) -> list[dict[str, Any]]:
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from gatek.planning import load_jobs

    jobs = load_jobs(LOCAL_OUT)
    if stage == "anchor":
        jobs = [
            job
            for job in jobs
            if job.arm == "sparse_zero_pad" and job.dim == 2 and job.job_type == "order_random"
        ]
    elif stage == "memory":
        report = json.loads(
            (LOCAL_OUT / "plan/tokenizer_preflight.json").read_text(encoding="utf-8")
        )
        largest_ids = [record["job_id"] for record in report["largest_jobs"]]
        jobs_by_id = {job.job_id: job for job in jobs}
        jobs = [jobs_by_id[job_id] for job_id in largest_ids]
    done_path = LOCAL_OUT / "run/results.jsonl"
    done = set()
    if done_path.exists():
        done = {
            json.loads(line)["job_id"]
            for line in done_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pending = [job.as_dict() for job in jobs if job.job_id not in done]
    return pending[:1] if stage == "memory" else pending


def _save_results(results: Any) -> tuple[int, list[dict[str, str]]]:
    run_dir = LOCAL_OUT / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    errors: list[dict[str, str]] = []
    written = 0
    with results_path.open("a", encoding="utf-8") as output:
        for result in results:
            if "error" in result:
                errors.append(result["error"])
                continue
            output.write(json.dumps(result["record"], sort_keys=True) + "\n")
            output.flush()
            written += 1
    if errors:
        (run_dir / "modal_errors.json").write_text(
            json.dumps(errors, indent=2, sort_keys=True), encoding="utf-8"
        )
    return written, errors


@app.local_entrypoint()
def main(stage: str = "anchor", workers: int = 1) -> None:
    """Run preflight, anchor, memory, or full; full is capped at ten containers."""
    if stage == "preflight":
        result = preflight.remote()
        report_path = LOCAL_OUT / "plan/tokenizer_preflight.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit("tokenizer preflight failed; no GPU was started")
        return
    if stage not in {"anchor", "memory", "full"}:
        raise SystemExit("stage must be preflight, anchor, memory, or full")
    expected_workers = 10 if stage == "full" else 1
    if workers != expected_workers:
        raise SystemExit(f"{stage} requires --workers {expected_workers} for the frozen cost plan")

    jobs = _local_jobs(stage)
    if not jobs:
        print(f"no pending {stage} jobs")
        return
    model_cls = GateKModel.with_options(max_containers=workers)
    model = model_cls()
    written, errors = _save_results(model.infer.map(jobs, order_outputs=False))
    summary = {"stage": stage, "pending": len(jobs), "written": written, "errors": errors}
    print(json.dumps(summary, indent=2))
    if errors:
        path = LOCAL_OUT / "run/modal_errors.json"
        raise SystemExit(f"{len(errors)} Modal jobs failed; inspect {path}")

    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from gatek.analyze import analyze
    from gatek.config import load_config

    analysis = analyze(load_config(LOCAL_ROOT / "configs/full.toml"), LOCAL_OUT)
    print(json.dumps(analysis["gate"], indent=2, sort_keys=True))
