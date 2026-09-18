from __future__ import annotations

import copy

import pytest

from freezeb.b3_repaired import repaired_final_events
from step0.hashing import sha256_json


def fixtures() -> tuple[dict, dict, dict]:
    call_id = "cell::call"
    expected = {
        call_id: {
            "call_id": call_id,
            "cell_key": "model::task",
            "condition": "solo",
            "assignment": None,
            "call_index": 0,
            "execution_block": 1,
            "batch_size": 1,
            "prompt_sha256": "prompt",
            "doc_order_sha256": sha256_json(["doc"]),
            "target_ids_sha256": sha256_json(["doc"]),
            "target_ids": ["doc"],
        }
    }
    original = {
        "call_id": call_id,
        "attempt_id": f"{call_id}::attempt::1",
        "attempt": 1,
        "parent_attempt_id": None,
        "model_id": "model",
        "task_id": "task",
        "cell_key": "model::task",
        "condition": "solo",
        "assignment": None,
        "call_index": 0,
        "execution_block": 1,
        "batch_size": 1,
        "prompt_sha256": "prompt",
        "doc_order": ["doc"],
        "target_ids": ["doc"],
        "status": "ok",
        "parse_status": "malformed",
        "predictions": None,
    }
    repaired = {
        **original,
        "attempt_id": f"repair-v1::{call_id}",
        "status": "ok",
        "parse_status": "ok",
        "predictions": {"doc": 1},
    }
    return expected, original, repaired


def test_exact_repair_replaces_terminal_unresolved_call() -> None:
    expected, original, repaired = fixtures()
    call_id = original["call_id"]
    record = {
        "source_attempt_id": original["attempt_id"],
        "source_event_sha256": sha256_json(original),
        "prompt_sha256": "prompt",
        "target_ids": ["doc"],
        "doc_order_sha256": sha256_json(["doc"]),
    }
    merged, counts = repaired_final_events(
        expected, [original], {call_id: record}, {call_id: repaired}
    )
    assert merged == [repaired]
    assert counts["repaired_calls"] == 1
    assert counts["revised_unresolved_calls"] == 0


def test_repair_of_resolved_original_is_rejected() -> None:
    expected, original, repaired = fixtures()
    original["parse_status"] = "ok"
    original["predictions"] = {"doc": 0}
    call_id = original["call_id"]
    with pytest.raises(ValueError, match="originally resolved"):
        repaired_final_events(expected, [original], {call_id: {}}, {call_id: repaired})


def test_wrong_source_link_is_rejected() -> None:
    expected, original, repaired = fixtures()
    call_id = original["call_id"]
    record = {
        "source_attempt_id": "wrong",
        "source_event_sha256": sha256_json(original),
        "prompt_sha256": "prompt",
        "target_ids": ["doc"],
        "doc_order_sha256": sha256_json(["doc"]),
    }
    with pytest.raises(ValueError, match="final original attempt"):
        repaired_final_events(expected, [original], {call_id: record}, {call_id: repaired})


def test_repair_metadata_mismatch_is_rejected() -> None:
    expected, original, repaired = fixtures()
    call_id = original["call_id"]
    record = {
        "source_attempt_id": original["attempt_id"],
        "source_event_sha256": sha256_json(original),
        "prompt_sha256": "prompt",
        "target_ids": ["doc"],
        "doc_order_sha256": sha256_json(["doc"]),
    }
    damaged = copy.deepcopy(repaired)
    damaged["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="metadata differs"):
        repaired_final_events(expected, [original], {call_id: record}, {call_id: damaged})
