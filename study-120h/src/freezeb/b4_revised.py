"""Post-execution B4 coverage analysis using exact linked output repairs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from step0.hashing import read_jsonl, sha256_file, sha256_json, sha256_text

from . import e2
from . import extension as ext
from . import extension_b4 as b4
from .labels import disagreement

SCHEMA_VERSION = "ext-x1r4-a11-b4-repaired-analysis-v1"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def recovery_evidence(
    base_run: Path, recovery_dir: Path, inventory_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Authenticate the complete recovery and return its B4 records/events by call ID."""
    inventory = load(inventory_path)
    core = {key: value for key, value in inventory.items() if key != "manifest_sha256"}
    if inventory.get("manifest_sha256") != sha256_json(core):
        raise ValueError("recovery inventory manifest hash differs")
    report = load(recovery_dir / "FULL_REPORT.json")
    if (
        report.get("inventory_sha256") != sha256_file(inventory_path)
        or report.get("complete") is not True
        or report.get("passed") is not True
        or report.get("completed_calls") != 4097
        or report.get("recovered_calls") != 4097
        or report.get("unresolved_calls") != 0
    ):
        raise ValueError("recovery completion report differs")
    events = list(read_jsonl(recovery_dir / "events.jsonl"))
    by_recovery_id = {str(event.get("recovery_id", "")): event for event in events}
    if len(events) != 4097 or len(by_recovery_id) != 4097:
        raise ValueError("recovery event ledger is incomplete or duplicated")
    receipt_events: dict[str, dict[str, Any]] = {}
    for path in sorted((recovery_dir / "receipts").glob("*.json")):
        payload = load(path)
        if payload.get("chunk_id") != path.stem:
            raise ValueError("recovery receipt filename differs from its chunk")
        for event in payload.get("events", []):
            identity = str(event.get("recovery_id", ""))
            if identity in receipt_events:
                raise ValueError("duplicate recovery event across receipts")
            receipt_events[identity] = event
    if {identity: sha256_json(event) for identity, event in receipt_events.items()} != {
        identity: sha256_json(event) for identity, event in by_recovery_id.items()
    }:
        raise ValueError("recovery receipts do not reproduce the event ledger")

    records: dict[str, dict[str, Any]] = {}
    repaired: dict[str, dict[str, Any]] = {}
    for record in inventory.get("records", []):
        if record.get("study") != "b4":
            continue
        call_id = str(record["call_id"])
        if call_id in records:
            raise ValueError("duplicate B4 call in recovery inventory")
        ledger = Path(str(record["source_ledger"]))
        if not ledger.resolve().is_relative_to(base_run.resolve()):
            raise ValueError("B4 recovery source ledger is outside the selected base run")
        if sha256_file(ledger) != record["source_ledger_sha256"]:
            raise ValueError("B4 recovery source ledger hash differs")
        event = by_recovery_id.get(str(record["recovery_id"]))
        if event is None:
            raise ValueError("B4 recovery event is missing")
        raw = event.get("first_response")
        if not isinstance(raw, str) or event.get("output_sha256") != sha256_text(raw):
            raise ValueError("B4 recovery response hash differs")
        parsed = e2.parse_result(
            raw,
            [str(value) for value in record["target_ids"]],
            event.get("finish_reason"),
            task_id=str(record["task_id"]),
        )
        if any(event.get(key) != parsed[key] for key in ("status", "parse_status", "predictions")):
            raise ValueError("B4 recovery stored parse differs from response bytes")
        expected = {
            "recovery_id": record["recovery_id"],
            "source_attempt_id": record["source_attempt_id"],
            "source_event_sha256": record["source_event_sha256"],
            "inventory_record_sha256": record["record_sha256"],
            "json_schema_sha256": record["json_schema_sha256"],
            "model_id": record["model_id"],
            "task_id": record["task_id"],
            "target_ids": record["target_ids"],
            "prompt_sha256": record["prompt_sha256"],
        }
        if any(event.get(key) != value for key, value in expected.items()):
            raise ValueError("B4 recovery identity differs from its inventory record")
        if event["status"] != "ok" or event["parse_status"] != "ok":
            raise ValueError("B4 recovery contains an unresolved output")
        records[call_id], repaired[call_id] = record, event
    if len(records) != 1095:
        raise ValueError("B4 recovery inventory must contain exactly 1,095 calls")
    return (
        records,
        repaired,
        {
            "inventory_sha256": sha256_file(inventory_path),
            "recovery_events_sha256": sha256_file(recovery_dir / "events.jsonl"),
            "recovery_report_sha256": sha256_file(recovery_dir / "FULL_REPORT.json"),
            "recovery_receipts": len(list((recovery_dir / "receipts").glob("*.json"))),
        },
    )


def revised_stream_errors(
    planned: list[dict[str, Any]],
    stored: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    repaired: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use original valid outputs and exact repairs for terminal unresolved calls."""
    seen = b4.validate_events(planned, stored)
    references = {str(row["doc_id"]): row["reference_normalized"] for row in rows}
    errors = np.full((len(planned) // 32, 32, 16), np.nan)
    original_resolved = 0
    replaced = 0
    call_ids = {str(job["call_id"]) for job in planned}
    foreign = set(repaired) & call_ids - set(records)
    if foreign:
        raise ValueError("repaired B4 stream contains a call absent from its inventory")
    for job in planned:
        call_id = str(job["call_id"])
        original = seen.get(f"{call_id}::attempt::2", seen.get(str(job["attempt_id"])))
        resolved = (
            original is not None
            and original.get("status") == "ok"
            and original.get("parse_status") == "ok"
        )
        if resolved:
            if call_id in repaired or call_id in records:
                raise ValueError("recovery attempted an originally resolved B4 call")
            event = original
            original_resolved += 1
        else:
            record, event = records.get(call_id), repaired.get(call_id)
            if record is None or event is None or original is None:
                raise ValueError("terminal unresolved B4 call lacks an exact repair")
            if (
                record["source_attempt_id"] != original["attempt_id"]
                or record["source_event_sha256"] != sha256_json(original)
                or record["prompt_sha256"] != job["prompt_sha256"]
                or record["target_ids"] != job["target_ids"]
            ):
                raise ValueError("B4 repair is not linked to the final original attempt")
            replaced += 1
        assert event is not None
        values = [
            disagreement(event["predictions"][doc_id], references[doc_id], job["task_id"])
            for doc_id in job["target_ids"]
        ]
        errors[int(job["replicate"]), int(job["call_index"])] = values
    expected_repairs = set(records) & call_ids
    if replaced != len(expected_repairs) or not np.isfinite(errors).all():
        raise ValueError("revised B4 stream is not complete")
    return errors, {
        "expected_calls": len(planned),
        "stored_original_attempts": len(seen),
        "original_resolved_calls": original_resolved,
        "repaired_calls": replaced,
        "revised_resolved_calls": original_resolved + replaced,
        "revised_unresolved_calls": 0,
        "parsed_complete": True,
        "original_status_counts": dict(
            sorted(Counter(str(event["status"]) for event in seen.values()).items())
        ),
    }


def analyze(
    study: Path, base_run: Path, recovery_dir: Path, inventory_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the unchanged original analysis and the separately revised analysis."""
    study = study.resolve()
    base_run = base_run.resolve()
    recovery_dir = recovery_dir.resolve()
    inventory_path = inventory_path.resolve()
    original = b4.analyze(study, base_run)
    if any(cell.get("status") != "coverage_unavailable" for cell in original["cells"]):
        raise ValueError("selected B4 base run does not reproduce the original unavailable result")
    records, repaired, recovery_hashes = recovery_evidence(base_run, recovery_dir, inventory_path)
    stored_plan = b4.load(base_run / "PLAN-400.json")
    cells = []
    coverage = []
    used_repairs: set[str] = set()
    inputs: dict[str, str] = {
        "base_plan": sha256_file(base_run / "PLAN-400.json"),
        **recovery_hashes,
    }
    for cell in b4.CELLS:
        rows = b4.frame(study, cell.rsplit("::", 1)[1])
        arrays: dict[str, np.ndarray] = {}
        reports: dict[str, dict[str, Any]] = {}
        for stream, count in (("reference", 250), ("evaluation", 400)):
            path = b4.stream_path(base_run, cell, stream)
            planned = list(b4.jobs(cell, rows, stream, count, render=False))
            stream_ids = {str(job["call_id"]) for job in planned}
            stream_records = {key: value for key, value in records.items() if key in stream_ids}
            stream_repairs = {key: value for key, value in repaired.items() if key in stream_ids}
            arrays[stream], reports[stream] = revised_stream_errors(
                planned, b4.events(path), rows, stream_records, stream_repairs
            )
            used_repairs.update(stream_repairs)
            inputs[str(path.relative_to(study))] = sha256_file(path)
        measured = ext.ext_e5_analyze(
            cell,
            arrays["reference"],
            arrays["evaluation"],
            n_frame=8192,
            bootstrap_draws=ext.E5_BOOTSTRAP_DRAWS,
        )
        cells.append(
            {
                "cell_key": cell,
                "status": "analyzed_after_exact_repair",
                "streams": reports,
                "analysis": measured,
            }
        )
        coverage.append(measured)
    if used_repairs != set(repaired):
        raise ValueError("some B4 recovery calls were not consumed by the revised analysis")
    revised = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": stored_plan["plan_sha256"],
        "evaluation_prefix": 400,
        "post_execution_technical_amendment": True,
        "original_result": "coverage_unavailable_original",
        "replacement_rule": "valid original output else exact linked repair-v1 output",
        "repair_calls_used": len(used_repairs),
        "cells": cells,
        "coverage_gate": ext.c2_coverage_gate(coverage, list(b4.CELLS)),
        "input_sha256": dict(sorted(inputs.items())),
        "gpu_work_launched": False,
    }
    return original, revised


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Revised B4 repeated-sampling coverage",
        "",
        "Post-execution technical amendment using exact linked repair-v1 outputs.",
        "The original prespecified result remains coverage_unavailable_original.",
        "",
        f"Repairs used: {result['repair_calls_used']}",
        "",
    ]
    for cell in result["cells"]:
        analysis = cell["analysis"]
        lines.extend(
            [
                f"## {cell['cell_key']}",
                "",
                f"Repaired calls: {sum(s['repaired_calls'] for s in cell['streams'].values())}",
                f"Reference target error: {analysis['reference']['target']:.6f}",
                f"IID Wald coverage: {analysis['coverage']:.4f}",
                "Call-aware t coverage: "
                f"{analysis['intervals']['call_aware_t_coverage']['point']:.4f}",
                "Whole-call bootstrap coverage: "
                f"{analysis['intervals']['whole_call_bootstrap_coverage']['point']:.4f}",
                f"Reference precise: {analysis['reference_precise']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Revised C2 gate",
            "",
            "```json",
            json.dumps(result["coverage_gate"], indent=2, sort_keys=True),
            "```",
            "",
            "These are reference-relative results, not verified true semantic accuracy.",
            "",
        ]
    )
    return "\n".join(lines)
