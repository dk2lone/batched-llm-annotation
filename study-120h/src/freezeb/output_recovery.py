"""Post-output recovery inventory for B3 and B4 terminal unresolved calls."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from step0.hashing import sha256_file, sha256_json, sha256_text

SCHEMA_VERSION = "ext-output-recovery-v1"
LABEL_VALUES = {
    "civil_comments": (0, 1),
    "imdb": (0, 1),
    "multi_nli": (0, 1, 2),
    "ag_news": (0, 1, 2, 3),
}


def final_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return one validated final event per call without changing either source ledger."""
    by_call: dict[str, list[dict[str, Any]]] = {}
    attempts: dict[str, dict[str, Any]] = {}
    for event in events:
        call_id = str(event.get("call_id", ""))
        attempt_id = str(event.get("attempt_id", ""))
        if not call_id or not attempt_id:
            raise ValueError("recovery source event lacks call identity")
        if attempt_id in attempts and attempts[attempt_id] != event:
            raise ValueError("recovery source contains conflicting attempts")
        attempts[attempt_id] = event
        by_call.setdefault(call_id, []).append(event)
    final: dict[str, dict[str, Any]] = {}
    for call_id, rows in by_call.items():
        rows.sort(key=lambda row: int(row.get("attempt", 0)))
        sequence = [int(row.get("attempt", 0)) for row in rows]
        if sequence not in ([1], [1, 2]):
            raise ValueError(f"invalid source attempt sequence: {call_id}")
        if rows[0]["attempt_id"] != f"{call_id}::attempt::1":
            raise ValueError(f"invalid source first-attempt identity: {call_id}")
        if len(rows) == 2 and (
            rows[1]["attempt_id"] != f"{call_id}::attempt::2"
            or rows[1].get("parent_attempt_id") != rows[0]["attempt_id"]
        ):
            raise ValueError(f"invalid source retry linkage: {call_id}")
        final[call_id] = rows[-1]
    return final


def is_unresolved(event: dict[str, Any]) -> bool:
    return event.get("status") != "ok" or event.get("parse_status") != "ok"


def label_schema(task_id: str, target_ids: list[str]) -> dict[str, Any]:
    """Require exactly the requested IDs and legal scalar labels."""
    if task_id not in LABEL_VALUES:
        raise ValueError(f"unsupported recovery task: {task_id}")
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("recovery target IDs must be nonempty and unique")
    label = {"type": "integer", "enum": list(LABEL_VALUES[task_id])}
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "object",
                "properties": {doc_id: label for doc_id in target_ids},
                "required": target_ids,
                "additionalProperties": False,
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def recovery_max_tokens(target_ids: list[str]) -> int:
    if len(target_ids) == 1:
        return 128
    if len(target_ids) == 16:
        return 512
    raise ValueError("recovery supports the observed one- or sixteen-target calls only")


def inventory_record(
    study: str,
    ledger: Path,
    event: dict[str, Any],
    ledger_sha256: str | None = None,
) -> dict[str, Any]:
    if study not in {"b3", "b4"} or not is_unresolved(event):
        raise ValueError("only terminal unresolved B3/B4 events enter recovery")
    raw = event.get("first_response")
    if not isinstance(raw, str) or event.get("output_sha256") != sha256_text(raw):
        raise ValueError("recovery source response hash differs")
    target_ids = [str(value) for value in event.get("target_ids", [])]
    schema = label_schema(str(event["task_id"]), target_ids)
    failure = "truncated" if event.get("status") == "truncated" else "malformed"
    core = {
        "study": study,
        "call_id": str(event["call_id"]),
        "recovery_id": f"repair-v1::{event['call_id']}",
        "model_id": str(event["model_id"]),
        "task_id": str(event["task_id"]),
        "condition": str(event.get("condition", "joint")),
        "source_attempt_id": str(event["attempt_id"]),
        "source_attempt": int(event["attempt"]),
        "source_status": str(event["status"]),
        "source_parse_status": str(event["parse_status"]),
        "failure_class": failure,
        "source_event_sha256": sha256_json(event),
        "source_ledger": str(ledger),
        "source_ledger_sha256": ledger_sha256 or sha256_file(ledger),
        "prompt_sha256": str(event["prompt_sha256"]),
        "target_ids": target_ids,
        "target_ids_sha256": sha256_json(target_ids),
        "doc_order_sha256": sha256_json(event["doc_order"]),
        "original_max_tokens": int(event["max_tokens"]),
        "recovery_max_tokens": recovery_max_tokens(target_ids),
        "json_schema": schema,
        "json_schema_sha256": sha256_json(schema),
    }
    return {**core, "record_sha256": sha256_json(core)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in records).items()))

    identities = [str(row["recovery_id"]) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("recovery inventory contains duplicate calls")
    return {
        "calls": len(records),
        "by_study": count("study"),
        "by_model": count("model_id"),
        "by_task": count("task_id"),
        "by_condition": count("condition"),
        "by_failure_class": count("failure_class"),
    }
