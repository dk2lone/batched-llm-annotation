"""Loopback-only B5 worker: structured JSON decoding for every scientific call.

The worker holds the cell frame and rebuilds every prompt with the shared renderer, so
the joint call and its exact-context calls of one group show identical documents in
identical order. The launcher recomputes each prompt hash and refuses any event whose
prompt differs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

PORT = 8771
MAX_REQUEST_BYTES = 64 * 1024**2


def receipt_path(root: Path, namespace: str, model_key: str, task: str, chunk_id: str) -> Path:
    if not re.fullmatch(r"b5-[0-9a-f]{24}", namespace):
        raise ValueError("invalid B5 namespace")
    if not re.fullmatch(r"[a-zA-Z0-9.-]+", model_key):
        raise ValueError("invalid B5 model key")
    if task not in ("ag_news", "civil_comments", "imdb", "multi_nli"):
        raise ValueError("invalid B5 task")
    if not re.fullmatch(r"[0-9a-f]{64}", chunk_id):
        raise ValueError("invalid B5 chunk identity")
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
            raise RuntimeError("B5 runtime or deterministic serving environment differs")
        if version("vllm") != "0.28.0" or torch.cuda.device_count() != 1:
            raise RuntimeError("B5 requires pinned vLLM and exactly one GPU")
        properties = torch.cuda.get_device_properties(0)
        if "A100" not in properties.name.upper() or properties.total_memory < 75 * 1024**3:
            raise RuntimeError("B5 requires one 80GB A100")
        self.model, self.root = model, root
        self.frames: dict[str, dict[str, str]] = {}
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
                "enable_prefix_caching": True,
            },
            bi.DECODING,
            enable_thinking=policy.enable_thinking,
            require_batch_invariance=True,
            require_prefix_caching=True,
        )
        self.host = host_facts()
        self.host.update(gpu_memory_bytes=properties.total_memory, provider="runpod")

    def _load_frame(self, body: dict[str, Any]) -> dict[str, Any]:
        from step0.hashing import sha256_json

        task = str(body["task"])
        texts = {str(key): str(value) for key, value in body["documents"].items()}
        ordered = [str(value) for value in body["doc_order"]]
        if set(ordered) != set(texts) or len(ordered) != len(texts):
            raise ValueError("B5 frame order does not match its documents")
        if sha256_json(ordered) != body["doc_order_sha256"]:
            raise ValueError("B5 frame identifier order differs")
        if sha256_json([texts[doc_id] for doc_id in ordered]) != body["documents_sha256"]:
            raise ValueError("B5 frame text digest differs")
        self.frames[task] = texts
        return {"task": task, "items": len(texts), "loaded": True}

    def _jobs(self, body: dict[str, Any], task: str) -> list[dict[str, Any]]:
        from freezeb import b5
        from step0.hashing import sha256_json

        texts = self.frames.get(task)
        if texts is None:
            raise ValueError("B5 frame was not loaded for this task")
        attempt = int(body.get("attempt", 1))
        if attempt not in (1, b5.MAX_ATTEMPTS):
            raise ValueError("B5 allows one first attempt and one retry")
        selected = body.get("only_call_ids")
        wanted = None if selected is None else {str(value) for value in selected}
        jobs = []
        for group in cast(list[dict[str, Any]], body["groups"]):
            if str(group["cell_key"]).rsplit("::", 1)[0] != self.model:
                raise ValueError("foreign B5 group")
            for call in b5.group_calls(group):
                if wanted is not None and str(call["call_id"]) not in wanted:
                    continue
                messages = b5.render_messages(task, texts, call)
                first = f"{call['call_id']}::attempt::1"
                jobs.append(
                    {
                        **call,
                        "attempt": attempt,
                        "attempt_id": f"{call['call_id']}::attempt::{attempt}",
                        "parent_attempt_id": None if attempt == 1 else first,
                        "messages": messages,
                        "prompt_sha256": sha256_json(messages),
                    }
                )
        if not jobs:
            raise ValueError("B5 chunk selected no call")
        return jobs

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
        if body["operation"] == "frame":
            return self._load_frame(body)
        namespace, model_key, task = body["namespace"], body["model_key"], body["task"]
        path = receipt_path(self.root, namespace, model_key, task, body.get("chunk_id", "0" * 64))
        if body["operation"] == "recover":
            return [json.loads(item.read_text()) for item in sorted(path.parent.glob("*.json"))]
        if body["operation"] != "run":
            raise ValueError("unknown B5 operation")
        jobs = self._jobs(body, str(task))
        if body["chunk_id"] != sha256_json([job["attempt_id"] for job in jobs]):
            raise ValueError("B5 chunk identity differs")
        if path.exists():
            return json.loads(path.read_text())
        policy = ext_policy_for(self.model)
        output: list[dict[str, Any]] = []
        ready: list[dict[str, Any]] = []
        conversations: list[Any] = []
        params: list[Any] = []
        for job in jobs:
            tokens = self.backend.count_tokens(job["messages"])
            if not context_fit(tokens, job["max_tokens"], policy.max_model_len)["fits"]:
                event = e2.event_for_result(job, generation_error="B5 prompt exceeds context")
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
                event = e2.event_for_result(
                    job,
                    raw=sample.text,
                    finish_reason=str(sample.finish_reason),
                    prompt_tokens=len(result.prompt_token_ids),
                    output_tokens=len(sample.token_ids),
                    queue_seconds=queue,
                    prefill_seconds=prefill,
                    decode_seconds=decode,
                )
                event["num_cached_tokens"] = getattr(result, "num_cached_tokens", None)
                event["num_cache_creation_tokens"] = getattr(
                    result, "num_cache_creation_tokens", None
                )
                output.append(event)
        except Exception as exc:  # noqa: BLE001 - record, never silently drop a scientific call
            output.extend(
                e2.event_for_result(job, generation_error=f"{type(exc).__name__}: {exc}")
                for job in ready
            )
        by_id = {str(job["attempt_id"]): job for job in jobs}
        order = {attempt_id: index for index, attempt_id in enumerate(by_id)}
        for event in output:
            job = by_id[str(event["attempt_id"])]
            event.update(
                group_id=job["group_id"],
                call_rank=job["call_rank"],
                json_schema_sha256=job["json_schema_sha256"],
                doc_order_sha256=job["doc_order_sha256"],
                target_ids_sha256=job["target_ids_sha256"],
                structured_output=True,
            )
        output.sort(key=lambda event: order[str(event["attempt_id"])])
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
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid request length")
                data = json.dumps(worker.request(json.loads(self.rfile.read(size)))).encode()
                self.send_response(200)
            except Exception as exc:  # noqa: BLE001 - never echo request content to the log
                print(f"B5 request rejected: {type(exc).__name__}: {exc}", flush=True)
                data = b'{"error":"B5 worker rejected request"}'
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--model", required=True)
    server.add_argument("--source", type=Path, required=True)
    server.add_argument("--root", type=Path, default=Path("/workspace/b5-results"))
    client = sub.add_parser("request")
    client.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    if args.operation == "serve":
        serve(args.model, args.source, args.root)
    else:
        request = Request(
            f"http://127.0.0.1:{PORT}/",
            data=sys.stdin.buffer.read(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=args.timeout) as response:
            sys.stdout.buffer.write(response.read())


if __name__ == "__main__":
    main()
