"""Synthetic loopback requests only; never reads scientific frames or schedules."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from freezeb.prompts import build_messages
from step0.hashing import sha256_json

CELLS = {
    ("ibm-granite/granite-3.3-8b-instruct", "imdb"),
    ("meta-llama/Llama-3.1-8B-Instruct", "civil_comments"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "multi_nli"),
    ("Qwen/Qwen3-8B", "civil_comments"),
}


def request(body: dict[str, Any]) -> Any:
    query = Request(
        "http://127.0.0.1:8765/",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(query, timeout=180) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    model, task = args.model, args.task
    if (model, task) not in CELLS:
        raise ValueError("smoke target is not one of the four sealed B4 cells")
    namespace = "b4-" + uuid.uuid4().hex[:24]
    ids = [f"synthetic-smoke-{index:02d}" for index in range(16)]
    messages = build_messages(
        task,
        [(item, "This is synthetic neutral text for a technical smoke test.") for item in ids],
        ids,
    )
    job = {
        "schema_version": "supervised-synthetic-smoke-v1",
        "cell_key": f"{model}::{task}",
        "call_id": f"synthetic-smoke::{namespace}",
        "attempt_id": f"synthetic-smoke::{namespace}::attempt0",
        "attempt": 0,
        "model_id": model,
        "task_id": task,
        "condition": "joint",
        "assignment": "synthetic-smoke",
        "call_index": 0,
        "execution_block": "synthetic-smoke",
        "execution_order_key": "synthetic-smoke",
        "batch_size": 16,
        "doc_order": ids,
        "target_ids": ids,
        "messages": messages,
        "prompt_sha256": sha256_json(messages),
        "max_tokens": 512,
    }
    ready = request({"operation": "ready"})
    body = {
        "operation": "run",
        "namespace": namespace,
        "task": task,
        "chunk_id": sha256_json([job["attempt_id"]]),
        "jobs": [job],
    }
    result = request(body)
    assert len(result["events"]) == 1 and result["events"][0]["status"] == "ok", result
    assert request(body) == result, "receipt replay changed the accepted response"
    recovered = request({**body, "operation": "recover"})
    assert recovered == [result], "durable receipt recovery differed"
    print(
        json.dumps(
            {
                "status": "passed",
                "synthetic_only": True,
                "ready": ready,
                "receipt_replay_verified": True,
                "recovery_verified": True,
                "result": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
