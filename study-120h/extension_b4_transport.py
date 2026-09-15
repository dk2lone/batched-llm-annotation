"""Opt-in Modal transport for four independently metered B4 cell lanes."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import Any

import modal

STUDY = Path(__file__).resolve().parent
REMOTE = "/root/study-120h/src"
ENV = {
    "HF_HOME": "/models",
    "HF_HUB_CACHE": "/models",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_BATCH_INVARIANT": "1",
}
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("vllm==0.28.0", "huggingface-hub==1.31.0", "numpy==2.2.6", "scipy==1.16.3")
    .env(ENV)
    .add_local_dir(STUDY / "src", remote_path=REMOTE)
)
models = modal.Volume.from_name("ieee-bigdata-freezeb-models", create_if_missing=False)
results = modal.Volume.from_name("ieee-bigdata-extension-b4-results", create_if_missing=True)
app = modal.App("ieee-bigdata-extension-b4")


@app.function(image=image, volumes={"/results": results}, timeout=600)
def recover(namespace: str, task: str) -> list[dict[str, Any]]:
    results.reload()
    root = Path("/results") / namespace / task
    return [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]


@app.cls(
    image=image,
    gpu="A100",
    volumes={"/models": models, "/results": results},
    max_containers=1,  # Each model/revision parameter pool has its own limit.
    scaledown_window=1,
    startup_timeout=1800,
    timeout=3600,
)
class Worker:
    model_id: str = modal.parameter()
    revision: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, REMOTE)
        from freezeb import batch_invariance as bi
        from freezeb.extension import ext_policy_for
        from step1.backends import VLLMBackend, host_facts

        if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
            raise RuntimeError("B4 batch invariance must be enabled before importing vLLM")
        policy = ext_policy_for(self.model_id)
        if policy.revision != self.revision:
            raise ValueError("B4 checkpoint differs from runtime seal")
        self.backend = VLLMBackend(
            self.model_id,
            self.revision,
            self.revision,
            {
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "max_model_len": policy.max_model_len,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 32,
            },
            bi.DECODING,
            enable_thinking=policy.enable_thinking,
            require_batch_invariance=True,
        )
        self.host = host_facts()

    @modal.method()
    def chunk(self, namespace: str, chunk_id: str, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        from freezeb import e2
        from freezeb.extension import ext_policy_for
        from freezeb.runtime_benchmark import context_fit
        from step0.hashing import sha256_json
        from step1.backends import GenRequest

        if not 1 <= len(jobs) <= 64:
            raise ValueError("B4 chunk must contain 1 through 64 calls")
        task = jobs[0]["task_id"]
        if any(job["model_id"] != self.model_id or job["task_id"] != task for job in jobs):
            raise ValueError("foreign B4 job")
        if chunk_id != sha256_json([job["attempt_id"] for job in jobs]):
            raise ValueError("B4 chunk identity differs")
        results.reload()
        path = Path("/results") / namespace / task / f"{chunk_id}.json"
        if path.exists():
            return json.loads(path.read_text())  # type: ignore[no-any-return]
        output = []
        requests = []
        ready = []
        for job in jobs:
            tokens = self.backend.count_tokens(job["messages"])
            policy = ext_policy_for(self.model_id)
            if not context_fit(tokens, job["max_tokens"], policy.max_model_len)["fits"]:
                event = e2.event_for_result(job, generation_error="sealed prompt exceeds context")
                event["status"] = "context_excluded"
                output.append(event)
            else:
                ready.append(job)
                requests.append(
                    GenRequest(
                        job["attempt_id"],
                        "joint",
                        tuple(job["target_ids"]),
                        job["messages"],
                        job["max_tokens"],
                    )
                )
        try:
            generated = {result.call_id: result for result in self.backend.generate(requests)}
            for job in ready:
                result = generated.get(job["attempt_id"])
                output.append(
                    e2.event_for_result(job, generation_error="missing result")
                    if result is None
                    else e2.event_for_result(
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
            output.extend(
                e2.event_for_result(job, generation_error=f"{type(exc).__name__}: {exc}")
                for job in ready
            )
        order = {job["attempt_id"]: index for index, job in enumerate(jobs)}
        output.sort(key=lambda event: order[event["attempt_id"]])
        payload = {
            "chunk_id": chunk_id,
            "backend": self.backend.describe(),
            "host": self.host,
            "events": output,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.replace(path)
        results.commit()
        return payload


class Lane:
    def __init__(self, handle: Any, task: str) -> None:
        self.handle = handle
        self.task = task

    def wait(self, call: Any, timeout: float, stop: Event) -> Any:
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or stop.is_set():
                    raise RuntimeError("B4 protected deadline reached or peer lane failed")
                try:
                    return call.get(timeout=min(1.0, remaining))
                except modal.exception.TimeoutError:
                    continue
        except BaseException:
            call.cancel(terminate_containers=True)
            raise

    def recover(self, namespace: str, timeout: float, stop: Event) -> list[dict[str, Any]]:
        if timeout <= 0 or stop.is_set():
            raise RuntimeError("B4 round stopped before recovery")
        return self.wait(recover.spawn(namespace, self.task), timeout, stop)  # type: ignore[no-any-return]

    def run(
        self,
        namespace: str,
        chunk_id: str,
        jobs: list[dict[str, Any]],
        timeout: float,
        stop: Event,
    ) -> dict[str, Any]:
        if timeout <= 0 or stop.is_set():
            raise RuntimeError("B4 round stopped before generation")
        call = self.handle.chunk.spawn(namespace, chunk_id, jobs)
        return self.wait(call, timeout, stop)  # type: ignore[no-any-return]


class ModalTransport:
    def __init__(self, profile: str) -> None:
        if not profile or os.environ.get("MODAL_PROFILE") != profile:
            raise ValueError("active Modal profile does not match the verified budget profile")

    @contextmanager
    def session(self) -> Iterator[None]:
        with app.run():
            yield

    @contextmanager
    def phase(self, cell: str, stream: str, schedule: dict[str, Any]) -> Iterator[Lane]:
        from freezeb.extension import ext_policy_for

        model, task = cell.rsplit("::", 1)
        handle = Worker(  # type: ignore[call-arg]
            model_id=model, revision=ext_policy_for(model).revision
        )
        yield Lane(handle, task)
