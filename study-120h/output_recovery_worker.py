"""Loopback-only structured-output recovery worker for B3 and B4."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def receipt_path(root: Path, namespace: str, model_key: str, task: str, chunk_id: str) -> Path:
    if not re.fullmatch(r"repair-[0-9a-f]{24}", namespace):
        raise ValueError("invalid recovery namespace")
    if not re.fullmatch(r"[a-z0-9-]+", model_key):
        raise ValueError("invalid recovery model key")
    if task not in ("ag_news", "civil_comments", "imdb", "multi_nli"):
        raise ValueError("invalid recovery task")
    if not re.fullmatch(r"[0-9a-f]{64}", chunk_id):
        raise ValueError("invalid recovery chunk identity")
    return root / namespace / model_key / task / f"{chunk_id}.json"


def persist(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Worker:
    def __init__(self, model: str, root: Path) -> None:
        import torch

        from freezeb import batch_invariance as bi
        from freezeb.extension import ext_policy_for
        from step1.backends import VLLMBackend, host_facts

        if sys.version_info[:2] != (3, 11) or os.environ.get("VLLM_BATCH_INVARIANT") != "1":
            raise RuntimeError("recovery runtime or deterministic serving environment differs")
        if version("vllm") != "0.28.0" or torch.cuda.device_count() != 1:
            raise RuntimeError("recovery requires pinned vLLM and exactly one GPU")
        properties = torch.cuda.get_device_properties(0)
        if "A100" not in properties.name.upper() or properties.total_memory < 75 * 1024**3:
            raise RuntimeError("recovery requires one 80GB A100")
        self.model, self.root = model, root
        policy = ext_policy_for(model)
        self.backend: Any = VLLMBackend(
            model,
            policy.revision,
            policy.revision,
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
        self.host.update(gpu_memory_bytes=properties.total_memory, provider="runpod")

    def request(self, body: dict[str, Any]) -> Any:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        from freezeb import e2
        from freezeb.extension import ext_policy_for
        from freezeb.runtime_benchmark import context_fit
        from step0.hashing import sha256_json
        from step1.backends import request_timing

        if body["operation"] == "ready":
            return {"backend": self.backend.describe(), "host": self.host, "structured": True}
        namespace, model_key, task = body["namespace"], body["model_key"], body["task"]
        zero = "0" * 64
        path = receipt_path(self.root, namespace, model_key, task, body.get("chunk_id", zero))
        if body["operation"] == "recover":
            return [json.loads(item.read_text()) for item in sorted(path.parent.glob("*.json"))]
        if body["operation"] != "run":
            raise ValueError("unknown recovery operation")
        jobs = body["jobs"]
        if not 1 <= len(jobs) <= 32 or any(
            job["model_id"] != self.model or job["task_id"] != task for job in jobs
        ):
            raise ValueError("foreign or oversized recovery chunk")
        if body["chunk_id"] != sha256_json([job["attempt_id"] for job in jobs]):
            raise ValueError("recovery chunk identity differs")
        if path.exists():
            return json.loads(path.read_text())
        output, ready, conversations, params = [], [], [], []
        policy = ext_policy_for(self.model)
        for job in jobs:
            tokens = self.backend.count_tokens(job["messages"])
            if not context_fit(tokens, job["max_tokens"], policy.max_model_len)["fits"]:
                event = e2.event_for_result(job, generation_error="recovery prompt exceeds context")
                event["status"] = "context_excluded"
                output.append(event)
                continue
            ready.append(job)
            conversations.append(job["messages"])
            params.append(
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=int(job["max_tokens"]),
                    seed=20260912,
                    structured_outputs=StructuredOutputsParams(
                        json=job["json_schema"],
                        disable_any_whitespace=True,
                        disable_additional_properties=True,
                    ),
                )
            )
        try:
            kwargs = {"chat_template_kwargs": {"enable_thinking": policy.enable_thinking}}
            try:
                generated = self.backend.llm.chat(conversations, params, use_tqdm=False, **kwargs)
            except TypeError:
                generated = self.backend.llm.chat(conversations, params, use_tqdm=False)
            for job, result in zip(ready, generated, strict=True):
                queue, prefill, decode = request_timing(getattr(result, "metrics", None))
                sample = result.outputs[0]
                output.append(
                    e2.event_for_result(
                        job,
                        raw=sample.text,
                        finish_reason=str(sample.finish_reason),
                        prompt_tokens=len(result.prompt_token_ids),
                        output_tokens=len(sample.token_ids),
                        queue_seconds=queue,
                        prefill_seconds=prefill,
                        decode_seconds=decode,
                    )
                )
        except Exception as exc:
            output.extend(
                e2.event_for_result(job, generation_error=f"{type(exc).__name__}: {exc}")
                for job in ready
            )
        order = {job["attempt_id"]: index for index, job in enumerate(jobs)}
        for event in output:
            job = next(job for job in jobs if job["attempt_id"] == event["attempt_id"])
            event.update(
                recovery_id=job["attempt_id"],
                source_attempt_id=job["source_attempt_id"],
                source_event_sha256=job["source_event_sha256"],
                inventory_record_sha256=job["record_sha256"],
                json_schema_sha256=job["json_schema_sha256"],
                structured_output=True,
            )
        output.sort(key=lambda event: order[event["attempt_id"]])
        payload = {
            "chunk_id": body["chunk_id"],
            "backend": self.backend.describe(),
            "host": self.host,
            "structured_output": True,
            "events": output,
        }
        persist(path, payload)
        return payload


def serve(model: str, source: Path, root: Path) -> None:
    os.environ.update(
        HF_HOME="/workspace/hf",
        HF_HUB_CACHE="/workspace/hf",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_BATCH_INVARIANT="1",
    )
    sys.path.insert(0, str(source))
    worker = Worker(model, root)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32 * 1024**2:
                    raise ValueError("invalid request length")
                data = json.dumps(worker.request(json.loads(self.rfile.read(size)))).encode()
                self.send_response(200)
            except Exception as exc:
                print(f"recovery request rejected: {type(exc).__name__}: {exc}", flush=True)
                data = b'{"error":"recovery worker rejected request"}'
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--model", required=True)
    server.add_argument("--source", type=Path, required=True)
    server.add_argument("--root", type=Path, default=Path("/workspace/output-recovery-results"))
    client = sub.add_parser("request")
    client.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    if args.operation == "serve":
        serve(args.model, args.source, args.root)
    else:
        body = sys.stdin.buffer.read()
        request = Request(
            "http://127.0.0.1:8765/", data=body, headers={"Content-Type": "application/json"}
        )
        with urlopen(request, timeout=args.timeout) as response:
            sys.stdout.buffer.write(response.read())


if __name__ == "__main__":
    main()
