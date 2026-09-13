"""Synthetic runtime/context preflight for the sealed X1R4 candidate panel."""

import datetime as dt
import hashlib
import json
import shutil
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/Users/dz/bigdata/study-120h")
SEAL = ROOT / "freeze/20260913T183353Z-X1R4"
MODEL_CACHE = "/models"
RUNTIME_ENV = {
    "HF_HOME": MODEL_CACHE,
    "HF_HUB_CACHE": MODEL_CACHE,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_BATCH_INVARIANT": "1",
}
app = modal.App("ieee-bigdata-x1r4-synthetic-runtime")
volume = modal.Volume.from_name("ieee-bigdata-freezeb-models")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface-hub==1.31.0", "numpy==2.2.6", "scipy==1.16.3", "vllm==0.28.0")
    .env(RUNTIME_ENV)
    .add_local_dir(ROOT / "src", remote_path="/root/study-120h/src")
)


@app.cls(image=image, volumes={MODEL_CACHE: volume}, gpu="A100", max_containers=1, scaledown_window=2, startup_timeout=1800, timeout=10800)
class RuntimeWorker:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, "/root/study-120h/src")
        from freezeb.batch_invariance import DECODING
        from freezeb.extension import ext_policy_for
        from step1.backends import VLLMBackend, host_facts

        policy = ext_policy_for(self.model_id)
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
            require_batch_invariance=True,
        )
        self.host = host_facts()

    @modal.method()
    def run(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from freezeb.extension import ext_policy_for
        from freezeb.runtime_benchmark import context_fit
        from step1.backends import GenRequest

        policy = ext_policy_for(self.model_id)
        torch.cuda.reset_peak_memory_stats()
        events: list[dict[str, Any]] = []
        for case in cases:
            try:
                prompt_tokens = self.backend.count_tokens(case["messages"])
                fit = context_fit(prompt_tokens, int(case["max_tokens"]), policy.max_model_len)
                events.append({"case_id": case["case_id"], "phase": "fit", "status": "ok" if fit["fits"] else "context_excluded", **fit})
                if not fit["fits"]:
                    continue
                for phase, repetition in (("warmup", 0), ("measure", 1), ("measure", 2)):
                    request = GenRequest(
                        call_id=f"{case['case_id']}::{phase}::{repetition}",
                        condition=case["condition"],
                        target_ids=(),
                        messages=case["messages"],
                        max_tokens=int(case["max_tokens"]),
                    )
                    started = time.perf_counter()
                    try:
                        output = self.backend.generate([request])[0]
                        events.append({
                            "case_id": case["case_id"], "phase": phase, "repetition": repetition,
                            "status": "ok", "latency_seconds": time.perf_counter() - started,
                            "prompt_tokens": output.prompt_tokens, "output_tokens": output.output_tokens,
                            "finish_reason": output.finish_reason,
                            "output_sha256": hashlib.sha256(output.text.encode()).hexdigest(),
                        })
                    except Exception as exc:  # noqa: BLE001 - technical failures are observations
                        events.append({
                            "case_id": case["case_id"], "phase": phase, "repetition": repetition,
                            "status": "failed", "latency_seconds": time.perf_counter() - started,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
            except Exception as exc:  # noqa: BLE001 - preserve failed context checks
                events.append({"case_id": case["case_id"], "phase": "fit", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return {
            "model_id": self.model_id,
            "revision": policy.revision,
            "backend": self.backend.describe(),
            "host": self.host,
            "events": events,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "container_gpu_seconds": time.monotonic() - self.started,
        }


@app.local_entrypoint()
def main(execute: bool = False, output_root: str = "study-120h/extension-runtime") -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from freezeb import batch_invariance as bi
    from freezeb.extension import EXT_MODEL_IDS, EXT_MODELS
    from freezeb.runtime_benchmark import build_cases
    from step0.hashing import sha256_json

    manifest = json.loads((SEAL / "MANIFEST.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        if hashlib.sha256((SEAL / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"X1R4 file hash mismatch: {relative}")
    if manifest["batch_invariance_plan_sha256"] != bi.plan_payload(
        [{"model_id": model.repo, "revision": model.revision} for model in EXT_MODELS]
    )["plan_sha256"]:
        raise ValueError("X1R4 batch-invariance plan drifted")
    task_order = ("civil_comments", "multi_nli", "imdb", "ag_news")
    cases = [asdict(case) for task in task_order for case in build_cases() if case.task_id == task]
    plan = {
        "schema_version": "x1r4-synthetic-runtime-v1",
        "synthetic_only": True,
        "seal_manifest_sha256": manifest["manifest_sha256"],
        "models": [{"model_id": model.repo, "revision": model.revision} for model in EXT_MODELS],
        "cases": cases,
        "warmups_per_case": 1,
        "measured_repetitions_per_case": 2,
        "gpu": "A100",
        "vllm": "0.28.0",
        "batch_invariant": True,
        "serving": {"dtype": "bfloat16", "tensor_parallel_size": 1, "max_num_seqs": 32, "max_model_len": 32768},
        "decoding": bi.DECODING,
        "runtime_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    plan["plan_sha256"] = sha256_json(plan)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "models": list(EXT_MODEL_IDS), "cases_per_model": len(cases), "paid_execution": execute}))
    if not execute:
        return
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(output_root) / run_id
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / "runtime_helper.py")
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    summaries = []
    meter_seconds = 0.0
    for model_id in EXT_MODEL_IDS:
        started = time.monotonic()
        try:
            worker = RuntimeWorker(model_id=model_id).run.remote(cases)  # type: ignore[call-arg]
            worker["invocation_wall_seconds"] = time.monotonic() - started
            worker["metered_proxy_seconds"] = max(worker["container_gpu_seconds"], worker["invocation_wall_seconds"])
            meter_seconds += worker["metered_proxy_seconds"]
            (output / (model_id.replace("/", "--") + ".json")).write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n")
            fits = [e for e in worker["events"] if e["phase"] == "fit"]
            warmups = [e for e in worker["events"] if e["phase"] == "warmup"]
            measures = [e for e in worker["events"] if e["phase"] == "measure"]
            latencies = [e["latency_seconds"] for e in measures if e["status"] == "ok"]
            backend = worker["backend"]
            passed = (
                len(fits) == len(cases)
                and all(e["status"] == "ok" for e in fits)
                and len(warmups) == len(cases)
                and all(e["status"] == "ok" for e in warmups)
                and len(measures) == 2 * len(cases)
                and all(e["status"] == "ok" for e in measures)
                and backend.get("batch_invariant") is True
                and backend.get("batch_invariance_required") is True
                and backend.get("model_id") == model_id
                and backend.get("model_revision") == next(m["revision"] for m in plan["models"] if m["model_id"] == model_id)
                and backend.get("tokenizer_revision") == backend.get("model_revision")
                and backend.get("dtype") == "bfloat16"
                and backend.get("max_num_seqs") == 32
                and backend.get("tensor_parallel_size") == 1
                and backend.get("seed") == 20260912
                and worker["host"].get("installed_engine_version_string") == "0.28.0"
                and "A100" in str(worker["host"].get("gpu_name", ""))
                and worker["revision"] == next(m["revision"] for m in plan["models"] if m["model_id"] == model_id)
            )
            summary = {"model_id": model_id, "status": "passed" if passed else "failed", "fit_cases": sum(e["status"] == "ok" for e in fits), "measured_calls": sum(e["status"] == "ok" for e in measures), "median_measured_latency_seconds": statistics.median(latencies) if latencies else None, "peak_gpu_memory_bytes": worker["peak_gpu_memory_bytes"], "metered_a100_hours": worker["metered_proxy_seconds"] / 3600}
        except Exception as exc:  # noqa: BLE001 - preserve failed model loads
            elapsed = time.monotonic() - started
            meter_seconds += elapsed
            summary = {"model_id": model_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "metered_proxy_a100_hours": elapsed / 3600}
        summaries.append(summary)
        (output / "report.json").write_text(json.dumps({"run_id": run_id, "plan_sha256": plan["plan_sha256"], "models": summaries, "metered_proxy_a100_hours": meter_seconds / 3600, "complete": len(summaries) == len(EXT_MODEL_IDS)}, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary))
    print(json.dumps({"output_dir": str(output), "metered_proxy_a100_hours": meter_seconds / 3600}))
