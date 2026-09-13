"""Run the synthetic Freeze B runtime benchmark on Modal.

The local entrypoint is safe by default: it prints the benchmark plan and starts
no remote function. Paid work requires ``--execute``. The Modal workspace's
hard spending limit is the run's cost ceiling.
"""

import concurrent.futures
import datetime as dt
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ieee-bigdata-freezeb-runtime-benchmark"
LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/study-120h")
MODEL_CACHE = "/models"
RESULTS_ROOT = "/results"
GPU_TYPE = "A100"
ENGINE_IMAGE = {
    "python": "3.11",
    "huggingface-hub": "1.31.0",
    "numpy": "2.2.6",
    "scipy": "1.16.3",
    "vllm": "0.28.0",
}
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 20260912}

volume = modal.Volume.from_name("ieee-bigdata-freezeb-models", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "ieee-bigdata-freezeb-benchmark-results", create_if_missing=True
)
base_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    *(f"{package}=={version}" for package, version in ENGINE_IMAGE.items() if package != "python")
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _model_key(model_id: str) -> str:
    return model_id.replace("/", "--") + "-" + _sha256_text(model_id)[:8]


@app.function(
    image=base_image,
    volumes={MODEL_CACHE: volume, RESULTS_ROOT: results_volume},
    timeout=7200,
)
def download_and_hash_model(run_id: str, model_id: str, revision: str) -> dict[str, Any]:
    """Download one pinned snapshot on CPU and hash every checkpoint/tokenizer file."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=model_id, revision=revision, cache_dir=MODEL_CACHE))
    files = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(snapshot)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    volume.commit()
    manifest = {"model_id": model_id, "revision": revision, "files": files}
    target = Path(RESULTS_ROOT) / run_id / _model_key(model_id) / "checkpoint-manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()
    return manifest


@app.cls(
    image=runtime_image,
    volumes={MODEL_CACHE: volume, RESULTS_ROOT: results_volume},
    gpu=GPU_TYPE,
    max_containers=1,
    scaledown_window=30,
    startup_timeout=1200,
    timeout=7200,
)
class ModelWorker:
    """Load one pinned checkpoint once, then run only synthetic benchmark calls."""

    model_id: str = modal.parameter()

    @modal.enter()
    def load_model(self) -> None:
        _remote_imports()
        from freezeb.runtime_benchmark import DECODING, policy_for
        from step1.backends import VLLMBackend, host_facts

        policy = policy_for(self.model_id)
        if not policy.runnable:
            raise RuntimeError(f"model is technically excluded: {policy.exclusion_reason}")
        self.started = time.monotonic()
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
        self.host = host_facts()

    @modal.method()
    def run(self, run_id: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Run all cases and commit each completed case before starting the next."""
        import torch

        from freezeb.runtime_benchmark import context_fit, policy_for
        from step1.backends import GenRequest

        policy = policy_for(self.model_id)
        model_dir = Path(RESULTS_ROOT) / run_id / _model_key(self.model_id)
        case_dir = model_dir / "cases"
        results_volume.reload()
        case_dir.mkdir(parents=True, exist_ok=True)
        events: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats()
        for case in cases:
            case_path = case_dir / f"{_sha256_text(str(case['case_id']))[:16]}.json"
            if case_path.exists():
                saved = json.loads(case_path.read_text(encoding="utf-8"))
                if saved.get("complete") is True:
                    events.extend(saved["events"])
                    continue
            case_events: list[dict[str, Any]] = []
            prompt_tokens = self.backend.count_tokens(case["messages"])
            fit = context_fit(prompt_tokens, int(case["max_tokens"]), policy.max_model_len)
            if not fit["fits"]:
                case_events.append(
                    {
                        "model_id": self.model_id,
                        "case_id": case["case_id"],
                        "phase": "preflight",
                        "status": "context_excluded",
                        **fit,
                    }
                )
            else:
                for phase, repetition in (("warmup", 0), ("measure", 1), ("measure", 2)):
                    request = GenRequest(
                        call_id=f"{case['case_id']}::{phase}::{repetition}",
                        condition=str(case["condition"]),
                        target_ids=(),
                        messages=case["messages"],
                        max_tokens=int(case["max_tokens"]),
                    )
                    started = time.perf_counter()
                    try:
                        result = self.backend.generate([request])[0]
                        event = {
                            "model_id": self.model_id,
                            "case_id": case["case_id"],
                            "phase": phase,
                            "repetition": repetition,
                            "status": "ok",
                            "latency_seconds": time.perf_counter() - started,
                            "prompt_tokens": result.prompt_tokens,
                            "output_tokens": result.output_tokens,
                            "finish_reason": result.finish_reason,
                            "output_sha256": _sha256_text(result.text),
                        }
                    except Exception as exc:
                        event = {
                            "model_id": self.model_id,
                            "case_id": case["case_id"],
                            "phase": phase,
                            "repetition": repetition,
                            "status": "failed",
                            "latency_seconds": time.perf_counter() - started,
                            "prompt_tokens": prompt_tokens,
                            "output_tokens": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    case_events.append(event)
            case_payload = {
                "run_id": run_id,
                "model_id": self.model_id,
                "case_id": case["case_id"],
                "complete": True,
                "events": case_events,
            }
            case_path.write_text(
                json.dumps(case_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            results_volume.commit()
            events.extend(case_events)
        worker = {
            "model_id": self.model_id,
            "revision": policy.revision,
            "backend": self.backend.describe(),
            "host": self.host,
            "events": events,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "container_gpu_seconds": time.monotonic() - self.started,
        }
        (model_dir / "worker.json").write_text(
            json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results_volume.commit()
        return worker


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            )


@app.local_entrypoint()
def main(
    execute: bool = False,
    output_root: str = "freezeb-runtime",
    run_id: str = "",
) -> None:
    """Print the plan, or explicitly run six concurrent model pipelines."""
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from freezeb.runtime_benchmark import plan_payload, summarize_measurements
    from step0.hashing import write_json

    plan = plan_payload()
    if plan["engine_image"] != ENGINE_IMAGE or plan["decoding"] != DECODING:
        raise RuntimeError("Modal wrapper and benchmark plan runtime pins differ")
    runnable = [model for model in plan["models"] if model["runnable"]]
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "synthetic_cases": len(plan["cases"]),
                "runnable_models": [model["model_id"] for model in runnable],
                "paid_execution": execute,
                "concurrent_gpu_workers": len(runnable),
                "cost_ceiling": "Modal workspace hard spending limit",
            },
            indent=2,
        )
    )
    if not execute:
        return

    requested_run_id = run_id
    run_id = requested_run_id or (
        dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output_dir = Path(output_root) / run_id
    if output_dir.exists():
        if not requested_run_id:
            raise FileExistsError(output_dir)
        saved_plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
        if saved_plan.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("cannot resume: saved and current benchmark plans differ")
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "plan.json", plan)
    event_path = output_dir / "events.jsonl"
    manifests: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []

    def execute_model(model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            manifest = download_and_hash_model.remote(
                run_id, model["model_id"], model["revision"]
            )
        except Exception as exc:
            manifest = {
                "model_id": model["model_id"],
                "revision": model["revision"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            return manifest, {
                "model_id": model["model_id"],
                "revision": model["revision"],
                "status": "download_failed",
                "error": manifest["error"],
            }

        invocation_started = time.monotonic()
        try:
            result = ModelWorker(model_id=model["model_id"]).run.remote(run_id, plan["cases"])
            invocation_seconds = time.monotonic() - invocation_started
            result["invocation_wall_seconds"] = invocation_seconds
            result["metered_proxy_seconds"] = max(
                invocation_seconds, float(result["container_gpu_seconds"])
            )
        except Exception as exc:
            invocation_seconds = time.monotonic() - invocation_started
            result = {
                "model_id": model["model_id"],
                "revision": model["revision"],
                "status": "worker_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "metered_proxy_seconds": invocation_seconds,
            }
        return manifest, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runnable)) as executor:
        futures = [executor.submit(execute_model, model) for model in runnable]
        for future in concurrent.futures.as_completed(futures):
            manifest, worker = future.result()
            manifests.append(manifest)
            workers.append(worker)
            events = worker.get("events", [worker])
            _append_jsonl(event_path, events)
            print(
                json.dumps(
                    {
                        "completed_model": worker["model_id"],
                        "status": worker.get("status", "complete"),
                        "events": len(worker.get("events", [])),
                    }
                ),
                flush=True,
            )

    measured = [
        event
        for worker in workers
        for event in worker.get("events", [])
        if event.get("phase") == "measure"
    ]
    report = summarize_measurements(measured)
    aggregate_proxy_seconds = sum(
        float(worker.get("metered_proxy_seconds", 0.0)) for worker in workers
    )
    report.update(
        {
            "run_id": run_id,
            "plan_sha256": plan["plan_sha256"],
            "synthetic_only": True,
            "aggregate_proxy_a100_hours": aggregate_proxy_seconds / 3600.0,
            "concurrent_gpu_workers": len(runnable),
            "cost_ceiling": "Modal workspace hard spending limit",
            "models_attempted": len(workers),
        }
    )
    write_json(output_dir / "checkpoint-manifests.json", manifests)
    write_json(output_dir / "workers.json", workers)
    write_json(output_dir / "report.json", report)
    print(json.dumps({"output_dir": str(output_dir), **report}, indent=2))
