"""Run the synthetic batch-invariance gate for the five-family extension.

The local entrypoint is dry-run by default. ``--execute`` is required to start
paid A100 workers. Checkpoints must already exist in the shared model volume.
"""

import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ieee-bigdata-extension-batch-invariance"
LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/study-120h")
MODEL_CACHE = "/models"
GPU_TYPE = "A100"
VLLM_VERSION = "0.28.0"
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
runtime_image = (
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


@app.cls(
    image=runtime_image,
    volumes={MODEL_CACHE: model_volume},
    gpu=GPU_TYPE,
    max_containers=1,
    scaledown_window=1,
    startup_timeout=1800,
    timeout=10800,
)
class SmokeWorker:
    """One fresh, batch-invariant vLLM session for one pinned checkpoint."""

    model_id: str = modal.parameter()
    session_index: int = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        _remote_imports()
        from freezeb import batch_invariance as bi
        from freezeb.extension import ext_policy_for
        from step1.backends import VLLMBackend, host_facts

        policy = ext_policy_for(self.model_id)
        self.started = time.monotonic()
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
        self.host = host_facts()

    @modal.method()
    def run(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        from step0.hashing import sha256_text
        from step1.backends import GenRequest

        events = []
        backend = self.backend.describe()
        for case in cases:
            requests = [
                GenRequest(
                    call_id=str(prompt["prompt_id"]),
                    condition="synthetic_batch_invariance",
                    target_ids=(str(prompt["prompt_id"]),),
                    messages=prompt["messages"],
                    max_tokens=int(prompt["max_tokens"]),
                )
                for prompt in case["requests"]
            ]
            started = time.monotonic()
            try:
                outputs = self.backend.generate(requests)
                by_id = {output.call_id: output for output in outputs}
                anchor_id = f"batch-invariance::{case['task']}::00"
                anchor = next(
                    prompt for prompt in case["requests"] if prompt["prompt_id"] == anchor_id
                )
                output = by_id[anchor_id]
                event = {
                    "status": "ok",
                    "anchor_prompt_id": anchor_id,
                    "anchor_prompt_sha256": anchor["prompt_sha256"],
                    "anchor_output_sha256": sha256_text(output.text),
                    "finish_reason": output.finish_reason,
                }
            except Exception as exc:  # noqa: BLE001 - failure is evidence for gate rejection
                event = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            events.append(
                {
                    **event,
                    "model_id": self.model_id,
                    "session_index": self.session_index,
                    "case_id": case["case_id"],
                    "task": case["task"],
                    "batch_size": case["batch_size"],
                    "order": case["order"],
                    "latency_seconds": time.monotonic() - started,
                    "backend": backend,
                    "host": self.host,
                }
            )
        return {
            "model_id": self.model_id,
            "session_index": self.session_index,
            "events": events,
            "container_gpu_seconds": time.monotonic() - self.started,
        }


@app.local_entrypoint()
def main(
    execute: bool = False,
    output_root: str = "extension-batch-invariance",
    model_id: str = "",
) -> None:
    """Print the frozen matrix, or explicitly run it one A100 session at a time."""
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from freezeb import batch_invariance as bi
    from freezeb.extension import EXT_MODEL_IDS, EXT_MODELS

    model_ids = [model_id] if model_id else list(EXT_MODEL_IDS)
    unknown = sorted(set(model_ids) - set(EXT_MODEL_IDS))
    if unknown:
        raise ValueError(f"unknown extension checkpoints: {unknown}")
    revisions = {model.repo: model.revision for model in EXT_MODELS}
    plan = bi.plan_payload(
        [{"model_id": selected, "revision": revisions[selected]} for selected in model_ids]
    )
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "models": model_ids,
                "sessions_per_model": len(bi.SESSION_INDICES),
                "cases_per_session": len(plan["cases"]),
                "paid_execution": execute,
                "maximum_concurrent_a100s": 1,
            },
            indent=2,
        )
    )
    if not execute:
        return

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(output_root) / run_id
    output.mkdir(parents=True)
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    workers = []
    for selected_model in model_ids:
        for session_index in bi.SESSION_INDICES:
            invocation_started = time.monotonic()
            try:
                worker = (
                    # Modal generates this parameterized class constructor at runtime.
                    SmokeWorker(  # type: ignore[call-arg]
                        model_id=selected_model, session_index=session_index
                    ).run.remote(plan["cases"])
                )
            except Exception as exc:  # noqa: BLE001 - persist a failed gate before stopping
                worker = {
                    "model_id": selected_model,
                    "session_index": session_index,
                    "events": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "container_gpu_seconds": 0.0,
                }
            worker["invocation_wall_seconds"] = time.monotonic() - invocation_started
            worker["metered_proxy_seconds"] = max(
                float(worker["container_gpu_seconds"]),
                float(worker["invocation_wall_seconds"]),
            )
            workers.append(worker)
    events = [event for worker in workers for event in worker["events"]]
    failure = None
    try:
        report = bi.validate_events(events, plan["models"])
    except ValueError as exc:
        failure = str(exc)
        report = {
            "schema_version": bi.SCHEMA_VERSION,
            "status": "failed",
            "synthetic_only": True,
            "reason": failure,
        }
    report.update(
        {
            "run_id": run_id,
            "plan_sha256": plan["plan_sha256"],
            "metered_a100_hours": sum(float(worker["metered_proxy_seconds"]) for worker in workers)
            / 3600.0,
        }
    )
    (output / "events.json").write_text(
        json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "workers.json").write_text(
        json.dumps(workers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **report}, indent=2))
    if failure is not None:
        raise RuntimeError(f"batch-invariance gate failed after preserving evidence: {failure}")
