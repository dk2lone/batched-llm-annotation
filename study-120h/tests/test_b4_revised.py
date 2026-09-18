from __future__ import annotations

import json

import pytest

from freezeb import e2
from freezeb import extension_b4 as b4
from freezeb.b4_revised import revised_stream_errors
from step0.hashing import sha256_json


def rows() -> list[dict[str, object]]:
    return [
        {"doc_id": str(index), "reference_normalized": index % 2, "text": f"text {index}"}
        for index in range(512)
    ]


def recovery(job: dict[str, object], source: dict[str, object]) -> tuple[dict, dict]:
    raw = json.dumps({"labels": {doc_id: 0 for doc_id in job["target_ids"]}})
    repaired = e2.event_for_result(
        {
            **job,
            "schema_version": "ext-output-recovery-v1",
            "attempt_id": f"repair-v1::{job['call_id']}",
            "attempt": 1,
            "parent_attempt_id": source["attempt_id"],
            "max_tokens": 512,
        },
        raw=raw,
        finish_reason="stop",
    )
    repaired.update(
        recovery_id=repaired["attempt_id"],
        source_attempt_id=source["attempt_id"],
        source_event_sha256=sha256_json(source),
    )
    record = {
        "source_attempt_id": source["attempt_id"],
        "source_event_sha256": sha256_json(source),
        "prompt_sha256": job["prompt_sha256"],
        "target_ids": job["target_ids"],
    }
    return record, repaired


def test_exact_repair_completes_an_unresolved_stream() -> None:
    data = rows()
    planned = list(b4.jobs(b4.CELLS[-1], data, "reference", 1, render=False))
    stored = []
    for job in planned:
        raw = json.dumps({"labels": {doc_id: 0 for doc_id in job["target_ids"]}})
        stored.append(e2.event_for_result(job, raw=raw, finish_reason="stop"))
    failed = e2.event_for_result(planned[0], raw="not json", finish_reason="stop")
    stored[0] = failed
    record, repaired = recovery(planned[0], failed)
    array, report = revised_stream_errors(
        planned,
        stored,
        data,
        {planned[0]["call_id"]: record},
        {planned[0]["call_id"]: repaired},
    )
    assert array.shape == (1, 32, 16)
    assert report["original_resolved_calls"] == 31
    assert report["repaired_calls"] == 1
    assert report["parsed_complete"] is True


def test_repair_of_an_originally_resolved_call_is_rejected() -> None:
    data = rows()
    planned = list(b4.jobs(b4.CELLS[-1], data, "reference", 1, render=False))
    job = planned[0]
    stored = [
        e2.event_for_result(
            item,
            raw=json.dumps({"labels": {doc_id: 0 for doc_id in item["target_ids"]}}),
            finish_reason="stop",
        )
        for item in planned
    ]
    record, repaired = recovery(job, stored[0])
    with pytest.raises(ValueError, match="originally resolved"):
        revised_stream_errors(
            planned, stored, data, {job["call_id"]: record}, {job["call_id"]: repaired}
        )
