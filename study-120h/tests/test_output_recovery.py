from __future__ import annotations

from pathlib import Path

import pytest

from freezeb.output_recovery import (
    final_events,
    inventory_record,
    label_schema,
    recovery_max_tokens,
    summarize,
)
from step0.hashing import sha256_text


def event(call_id: str, attempt: int = 1) -> dict[str, object]:
    raw = '{"labels":{"x":9}}'
    return {
        "call_id": call_id,
        "attempt": attempt,
        "attempt_id": f"{call_id}::attempt::{attempt}",
        "parent_attempt_id": f"{call_id}::attempt::1" if attempt == 2 else None,
        "model_id": "model",
        "task_id": "imdb",
        "condition": "joint",
        "status": "ok",
        "parse_status": "malformed",
        "first_response": raw,
        "output_sha256": sha256_text(raw),
        "prompt_sha256": "a" * 64,
        "target_ids": ["x"],
        "doc_order": ["x"],
        "max_tokens": 84,
    }


def test_schema_requires_exact_ids_and_legal_labels() -> None:
    schema = label_schema("multi_nli", ["a", "b"])
    labels = schema["properties"]["labels"]
    assert labels["required"] == ["a", "b"]
    assert labels["additionalProperties"] is False
    assert labels["properties"]["a"]["enum"] == [0, 1, 2]
    with pytest.raises(ValueError, match="unique"):
        label_schema("imdb", ["a", "a"])


def test_final_events_validates_retry_linkage() -> None:
    first, retry = event("call"), event("call", 2)
    assert final_events([first, retry])["call"] == retry
    retry["parent_attempt_id"] = "wrong"
    with pytest.raises(ValueError, match="linkage"):
        final_events([first, retry])


def test_inventory_is_content_addressed_and_unique(tmp_path: Path) -> None:
    ledger = tmp_path / "events.jsonl"
    ledger.write_text("{}\n")
    first = inventory_record("b4", ledger, event("call"))
    assert first["failure_class"] == "malformed"
    assert first["recovery_max_tokens"] == 128
    assert recovery_max_tokens([str(i) for i in range(16)]) == 512
    assert summarize([first])["calls"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        summarize([first, first])
