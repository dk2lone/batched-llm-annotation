"""B4 execution identities, append-only recovery, and frozen coverage analysis."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np

from step0.hashing import read_jsonl, sha256_file, sha256_json, sha256_text

from . import e2
from . import extension as ext
from .labels import disagreement
from .prompts import build_messages, estimated_response_tokens
from .repeated import replicate_calls

SEAL = "freeze/20260913T183353Z-X1R4"
ADMISSION = "admission-lock/20260914T012453Z-X1R4-A7"
RUNTIME = "runtime-lock/20260913T205646Z-X1R4"
FRAME = "ext-work/20260913-cpu-draft"
SCHEMA = "ext-x1r4-a11-b4-run-v1"
RETRYABLE = frozenset({"generation_failed", "truncated", "worker_failed"})
CELLS = (
    "ibm-granite/granite-3.3-8b-instruct::imdb",
    "meta-llama/Llama-3.1-8B-Instruct::civil_comments",
    "mistralai/Mistral-7B-Instruct-v0.3::multi_nli",
    "Qwen/Qwen3-8B::civil_comments",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_manifest(directory: Path) -> None:
    manifest = load(directory / "MANIFEST.json")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if "manifest_sha256" in manifest and sha256_json(core) != manifest["manifest_sha256"]:
        raise ValueError(f"manifest digest differs: {directory}")
    for relative, digest in manifest["files_sha256"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"sealed file differs: {relative}")


def frame(study: Path, task: str) -> list[dict[str, Any]]:
    source = study / FRAME / task / "confirm.jsonl"
    sealed = study / SEAL / "frame-identifiers" / task
    expected = load(sealed / "source-manifest.json")["files"]["confirm.jsonl"]
    if sha256_file(source) != expected["sha256"]:
        raise ValueError(f"private confirmation frame differs: {task}")
    rows = list(read_jsonl(source))
    ids = [str(row["doc_id"]) for row in rows]
    if (
        len(ids) != 8192
        or len(set(ids)) != 8192
        or ids != [str(row["doc_id"]) for row in read_jsonl(sealed / "confirm.jsonl")]
        or sha256_json(ids) != expected["row_ids_sha256"]
    ):
        raise ValueError(f"confirmation ID order differs: {task}")
    if any(not {"text", "reference_normalized", "n_chars"} <= row.keys() for row in rows):
        raise ValueError(f"private confirmation fields missing: {task}")
    return rows


def jobs(
    cell: str,
    rows: list[dict[str, Any]],
    stream: str,
    replicates: int,
    *,
    render: bool = True,
) -> Iterator[dict[str, Any]]:
    model, task = cell.rsplit("::", 1)
    by_id = {str(row["doc_id"]): row for row in rows}
    ids = tuple(by_id)
    for replicate in range(replicates):
        for call in replicate_calls(
            ids,
            cell,
            stream,
            replicate,
            namespace=ext.E5_NAMESPACE,
            call_id_prefix=f"{ext.CALL_PREFIX}::b4::",
        ):
            order = [str(value) for value in cast(list[str], call["doc_order"])]
            documents = [(doc_id, str(by_id[doc_id]["text"])) for doc_id in order]
            messages = build_messages(task, documents, order)
            position = replicate * 32 + int(cast(int, call["call_index"])) + 1
            yield {
                **call,
                "schema_version": ext.B4_SCHEMA,
                "position": position,
                "model_id": model,
                "task_id": task,
                "condition": "joint",
                "assignment": None,
                "execution_block": 0,
                "execution_order_key": f"{stream}:{position:08d}",
                "batch_size": 16,
                "target_ids": order,
                "doc_order_sha256": sha256_json(order),
                "attempt": 1,
                "attempt_id": f"{call['call_id']}::attempt::1",
                **({"messages": messages} if render else {}),
                "prompt_sha256": sha256_json(messages),
                "max_tokens": estimated_response_tokens(
                    task, 16, [len(text.split()) for _, text in documents]
                ),
            }


def build_plan(study: Path, prefix: int = 400) -> dict[str, Any]:
    if prefix not in ext.E5_EVALUATION_PREFIXES:
        raise ValueError("evaluation prefix must be 400, 1000, or 2500")
    for relative in (SEAL, ADMISSION, RUNTIME):
        verify_manifest(study / relative)
    admission = load(study / ADMISSION / "ADMISSION_LOCK.json")
    if admission["ordered_cells"]["e5_cells"] != list(CELLS):
        raise ValueError("prespecified B4 family selection/order differs")
    schedules: dict[str, Any] = {}
    frames: dict[str, str] = {}
    for cell in CELLS:
        model, task = cell.rsplit("::", 1)
        rows = frame(study, task)
        digest = ext.b4_schedule_digest(model, task, tuple(str(row["doc_id"]) for row in rows))
        locked = admission["selected_schedules"]["b4"][cell]
        if sha256_json(digest) != locked["selected_schedule_sha256"]:
            raise ValueError(f"B4 sampled order differs from A7: {cell}")
        schedules[cell] = digest
        frames[task] = sha256_file(study / FRAME / task / "confirm.jsonl")
    core = {
        "schema_version": SCHEMA,
        "cells": list(CELLS),
        "evaluation_prefix": prefix,
        "reference_replicates": 250,
        "calls_per_replicate": 32,
        "batch_size": 16,
        "first_calls": len(CELLS) * (250 + prefix) * 32,
        "schedules": schedules,
        "frame_sha256": frames,
        "source_sha256": {
            relative: sha256_file(study / relative / "MANIFEST.json")
            for relative in (SEAL, ADMISSION, RUNTIME)
        },
        "implementation_sha256": {
            relative: sha256_file(study / relative)
            for relative in (
                "modal_extension_b4.py",
                "extension_b4_transport.py",
                "src/freezeb/extension_b4.py",
                "src/freezeb/extension.py",
                "src/freezeb/prompts.py",
                "src/freezeb/labels.py",
                "src/step0/stats.py",
            )
        },
        "serving_policy": {
            cell.rsplit("::", 1)[0]: ext.ext_policy_for(cell.rsplit("::", 1)[0]).revision
            for cell in CELLS
        },
        "execution_order": "all_reference_cells_then_equal_family_evaluation_prefixes",
        "max_concurrent_a100s": 4,
        "parallelism": "one_single_gpu_lane_per_cell_with_round_barriers",
        "max_retries": 1,
        "unresolved_policy": "block_coverage_without_imputation",
        "stop_a100_hours": 80,
        "reserve_a100_hours": 10,
        "forecast_multiplier": 1.6,
    }
    return {**core, "plan_sha256": sha256_json(core)}


def events(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path)) if path.exists() else []


def append(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_events(
    planned: list[dict[str, Any]], stored: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    expected = {str(job["call_id"]): job for job in planned}
    seen: dict[str, dict[str, Any]] = {}
    for event in stored:
        call_id = str(event.get("call_id", ""))
        job = expected.get(call_id)
        if job is None:
            raise ValueError(f"foreign B4 call: {call_id}")
        attempt = event.get("attempt")
        if attempt not in (1, 2) or event.get("attempt_id") != f"{call_id}::attempt::{attempt}":
            raise ValueError("invalid B4 attempt identity")
        for key in (
            "schema_version",
            "cell_key",
            "model_id",
            "task_id",
            "condition",
            "assignment",
            "call_index",
            "execution_block",
            "execution_order_key",
            "batch_size",
            "doc_order",
            "target_ids",
            "prompt_sha256",
            "max_tokens",
        ):
            if event.get(key) != job[key]:
                raise ValueError(f"B4 event {key} differs: {call_id}")
        raw = event.get("first_response")
        if not isinstance(raw, str) or event.get("output_sha256") != sha256_text(raw):
            raise ValueError("B4 response bytes/hash differ")
        parsed = (
            {"status": "generation_failed", "parse_status": "missing", "predictions": None}
            if event.get("generation_error") is not None
            else e2.parse_result(
                raw, job["target_ids"], event.get("finish_reason"), task_id=str(job["task_id"])
            )
        )
        if event.get("status") == "context_excluded" and event.get("generation_error"):
            parsed["status"] = "context_excluded"
        if any(event.get(key) != parsed[key] for key in ("status", "parse_status", "predictions")):
            raise ValueError("B4 stored parse differs from response bytes")
        identity = str(event["attempt_id"])
        if identity in seen and seen[identity] != event:
            raise ValueError("conflicting B4 duplicate attempt")
        seen[identity] = event
    for event in seen.values():
        call_id = str(event["call_id"])
        if event["attempt"] == 2:
            parent = seen.get(f"{call_id}::attempt::1")
            if (
                parent is None
                or parent["status"] not in RETRYABLE
                or event.get("parent_attempt_id") != parent["attempt_id"]
            ):
                raise ValueError("unpermitted or unlinked B4 retry")
        elif event.get("parent_attempt_id") is not None:
            raise ValueError("first attempt cannot have a parent")
    return seen


def pending(planned: list[dict[str, Any]], stored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = validate_events(planned, stored)
    work = []
    for job in planned:
        first = seen.get(str(job["attempt_id"]))
        if first is None:
            work.append(job)
        elif first["status"] in RETRYABLE and f"{job['call_id']}::attempt::2" not in seen:
            work.append(e2.retry_job(job))
    return work


def stream_errors(
    planned: list[dict[str, Any]], stored: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    seen = validate_events(planned, stored)
    references = {str(row["doc_id"]): row["reference_normalized"] for row in rows}
    errors = np.full((len(planned) // 32, 32, 16), np.nan)
    resolved = 0
    for job in planned:
        event = seen.get(f"{job['call_id']}::attempt::2", seen.get(str(job["attempt_id"])))
        if event is not None and event["status"] == "ok" and event["parse_status"] == "ok":
            values = [
                disagreement(event["predictions"][doc_id], references[doc_id], job["task_id"])
                for doc_id in job["target_ids"]
            ]
            errors[int(job["replicate"]), int(job["call_index"])] = values
            resolved += 1
    n_pending = len(pending(planned, stored))
    return errors, {
        "expected_calls": len(planned),
        "stored_attempts": len(seen),
        "retry_attempts": sum(event["attempt"] == 2 for event in seen.values()),
        "pending_attempts": n_pending,
        "resolved_calls": resolved,
        "unresolved_calls": len(planned) - resolved,
        "execution_complete": n_pending == 0,
        "parsed_complete": n_pending == 0 and resolved == len(planned),
        "status_counts": dict(sorted(Counter(str(e["status"]) for e in seen.values()).items())),
    }


def stream_path(out: Path, cell: str, stream: str) -> Path:
    return out / "cells" / cell.replace("/", "--").replace("::", "--") / stream / "events.jsonl"


def verify_budget(record: dict[str, Any], plan: dict[str, Any], profile: str) -> None:
    core = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != sha256_json(core):
        raise ValueError("budget record hash differs")
    if record.get("plan_sha256") != plan["plan_sha256"] or record.get("profile") != profile:
        raise ValueError("budget record is not bound to this plan/profile")
    if record.get("workspace_enabled") is not True or record.get("exclusive_gpu_work") is not True:
        raise ValueError("fresh workspace availability/exclusive metering must be confirmed")
    concurrency = record.get("a100_concurrency_limit")
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < int(plan.get("max_concurrent_a100s", 4))
    ):
        raise ValueError("verified workspace capacity must permit four concurrent A100s")
    verified = dt.datetime.fromisoformat(str(record["verified_at_utc"]))
    if (
        verified.tzinfo is None
        or not 0 <= (dt.datetime.now(dt.UTC) - verified).total_seconds() <= 3600
    ):
        raise ValueError("budget verification must be timezone-aware and less than one hour old")
    for key in (
        "prior_extension_a100_hours",
        "spend_limit_usd",
        "spend_used_usd",
        "a100_usd_per_hour",
    ):
        value = record.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid budget value: {key}")
        if value < 0:
            raise ValueError(f"negative budget value: {key}")
    if record["a100_usd_per_hour"] <= 0 or record["spend_limit_usd"] <= record["spend_used_usd"]:
        raise ValueError("no verified paid capacity")
    forecasts = record.get("base_forecast_hours", {})
    for cell in CELLS:
        for stream in ("reference", "evaluation"):
            value = forecasts.get(f"{cell}|{stream}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("every complete B4 stream needs a base forecast")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("forecasts must be finite and positive")


def verify_serving(payload: dict[str, Any], plan: dict[str, Any], cell: str) -> None:
    model = cell.rsplit("::", 1)[0]
    backend = payload.get("backend", {})
    expected = {
        "backend": "vllm",
        "model_id": model,
        "model_revision": plan["serving_policy"][model],
        "tokenizer_revision": plan["serving_policy"][model],
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_num_seqs": 32,
        "seed": 20260912,
        "batch_invariant": True,
        "batch_invariance_required": True,
    }
    if any(backend.get(key) != value for key, value in expected.items()):
        raise ValueError("B4 worker differs from sealed serving policy")
    host = payload.get("host", {})
    if (
        host.get("installed_engine_version_string") != "0.28.0"
        or "A100" not in str(host.get("gpu_name", "")).upper()
    ):
        raise ValueError("B4 worker runtime/GPU differs")


def analyze(study: Path, out: Path, prefix: int = 400) -> dict[str, Any]:
    plan = build_plan(study, prefix)
    stored_plan = load(out / f"PLAN-{prefix}.json")
    if stored_plan != plan:
        raise ValueError("analysis input plan differs from rebuilt sealed B4 plan")
    cells = []
    coverage = []
    hashes: dict[str, str | None] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for path in sorted((out / "receipts").glob("*.json")):
        payload = load(path)
        if path.stem != sha256_json(payload):
            raise ValueError("B4 receipt content hash differs")
        for event in payload["events"]:
            cell = event["cell_key"]
            if cell not in CELLS:
                raise ValueError("foreign B4 receipt cell")
            verify_serving(payload, plan, cell)
            identity = event["attempt_id"]
            if identity in receipts and receipts[identity] != event:
                raise ValueError("conflicting B4 receipt attempt")
            receipts[identity] = event
        hashes[str(path.relative_to(out))] = sha256_file(path)
    for cell in CELLS:
        rows = frame(study, cell.rsplit("::", 1)[1])
        arrays = {}
        reports = {}
        for stream, count in (("reference", 250), ("evaluation", prefix)):
            path = stream_path(out, cell, stream)
            planned = list(jobs(cell, rows, stream, count, render=False))
            stored = events(path)
            for event in stored:
                if receipts.get(event["attempt_id"]) != event:
                    raise ValueError("B4 ledger event lacks a matching verified worker receipt")
            arrays[stream], reports[stream] = stream_errors(planned, stored, rows)
            hashes[str(path.relative_to(out))] = sha256_file(path) if path.exists() else None
        result: dict[str, Any] = {"cell_key": cell, "streams": reports}
        if all(report["parsed_complete"] for report in reports.values()):
            measured = ext.ext_e5_analyze(
                cell,
                arrays["reference"],
                arrays["evaluation"],
                n_frame=8192,
                bootstrap_draws=ext.E5_BOOTSTRAP_DRAWS,
            )
            result.update(status="analyzed", analysis=measured)
            coverage.append(measured)
        else:
            result.update(status="coverage_unavailable", reason="required fresh labels unresolved")
        cells.append(result)
    return {
        "schema_version": "ext-x1r4-a11-b4-analysis-v1",
        "plan_sha256": plan["plan_sha256"],
        "evaluation_prefix": prefix,
        "cells": cells,
        "input_sha256": hashes,
        "coverage_gate": ext.c2_coverage_gate(coverage, list(CELLS)),
        "unresolved_policy": "no imputation, discarded replicates, or reused outputs",
        "gpu_work_launched": False,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# B4 repeated-sampling coverage",
        "",
        f"Evaluation prefix: {result['evaluation_prefix']}",
        "",
    ]
    for cell in result["cells"]:
        lines.extend([f"## {cell['cell_key']}", "", f"Status: {cell['status']}", ""])
        if cell["status"] == "analyzed":
            analysis = cell["analysis"]
            lines.extend(
                [
                    f"Reference precision passes: {analysis['reference_precise']}",
                    "",
                    "```json",
                    json.dumps(analysis, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    cell["reason"],
                    "",
                    "```json",
                    json.dumps(cell["streams"], indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "## Frozen C2 gate",
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
