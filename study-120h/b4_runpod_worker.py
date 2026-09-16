"""Loopback-only B4 worker. Imported safely on CPU; CUDA loads only in serve()."""

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


def receipt_path(root: Path, namespace: str, task: str, chunk_id: str) -> Path:
    if not re.fullmatch(r"b4-[0-9a-f]{24}", namespace):
        raise ValueError("invalid B4 namespace")
    if task not in ("imdb", "civil_comments", "multi_nli"):
        raise ValueError("invalid B4 task")
    if not re.fullmatch(r"[0-9a-f]{64}", chunk_id):
        raise ValueError("invalid B4 chunk identity")
    return root / namespace / task / f"{chunk_id}.json"


def persist(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
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

        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Runpod B4 runtime must use Python 3.11")
        from freezeb import batch_invariance as bi
        from freezeb.extension import ext_policy_for
        from step1.backends import VLLMBackend, host_facts

        if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
            raise RuntimeError("batch invariance was not enabled before vLLM import")
        for package, expected in {
            "vllm": "0.28.0",
            "huggingface-hub": "1.31.0",
            "numpy": "2.2.6",
            "scipy": "1.16.3",
        }.items():
            if version(package) != expected:
                raise RuntimeError(f"unverified B4 dependency: {package}")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("B4 requires exactly one visible GPU per worker")
        properties = torch.cuda.get_device_properties(0)
        if "A100" not in properties.name.upper() or properties.total_memory < 75 * 1024**3:
            raise RuntimeError("B4 Runpod worker requires an 80GB A100")
        self.model = model
        self.root = root
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
        self.host["gpu_memory_bytes"] = properties.total_memory
        self.host["provider"] = "runpod"

    def request(self, body: dict[str, Any]) -> Any:
        from freezeb import e2
        from freezeb.extension import ext_policy_for
        from freezeb.runtime_benchmark import context_fit
        from step0.hashing import sha256_json
        from step1.backends import GenRequest

        if body["operation"] == "ready":
            return {"backend": self.backend.describe(), "host": self.host}
        namespace, task = body["namespace"], body["task"]
        path = receipt_path(self.root, namespace, task, body.get("chunk_id", "0" * 64))
        if body["operation"] == "recover":
            return [json.loads(item.read_text()) for item in sorted(path.parent.glob("*.json"))]
        if body["operation"] != "run":
            raise ValueError("unknown worker operation")
        jobs = body["jobs"]
        if not 1 <= len(jobs) <= 64 or any(
            job["model_id"] != self.model or job["task_id"] != task for job in jobs
        ):
            raise ValueError("foreign or oversized B4 chunk")
        if body["chunk_id"] != sha256_json([job["attempt_id"] for job in jobs]):
            raise ValueError("B4 chunk identity differs")
        if path.exists():
            return json.loads(path.read_text())
        output, requests, ready = [], [], []
        policy = ext_policy_for(self.model)
        for job in jobs:
            tokens = self.backend.count_tokens(job["messages"])
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
        order = {job["attempt_id"]: i for i, job in enumerate(jobs)}
        output.sort(key=lambda event: order[event["attempt_id"]])
        payload = {
            "chunk_id": body["chunk_id"],
            "backend": self.backend.describe(),
            "host": self.host,
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
                payload = worker.request(json.loads(self.rfile.read(size)))
                data = json.dumps(payload).encode()
                self.send_response(200)
            except Exception as exc:
                print(f"worker request rejected: {type(exc).__name__}", file=sys.stderr, flush=True)
                data = b'{"error":"worker rejected request; inspect persistent worker log"}'
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
    server.add_argument("--root", type=Path, default=Path("/workspace/b4-results"))
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
