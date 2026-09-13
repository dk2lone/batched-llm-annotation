"""Freeze B solo-screen planning, validation, summaries, and part-2 lock payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from step0.hashing import derived_seed, read_jsonl, sha256_file, sha256_json

from .admission import evaluate
from .frames import N_SCREEN, TASKS
from .labels import ParseError, normalize_label, parse_first_response
from .meter import FORECAST_MULTIPLIER
from .models import MODELS
from .power import choose_n
from .prompts import build_messages, estimated_response_tokens
from .runtime_benchmark import MODEL_POLICIES, RuntimePolicy, policy_for
from .selection import Cell, panels

SCHEMA_VERSION = "freezeb-solo-screen-v1"
RUNNABLE_MODELS = tuple(policy.model_id for policy in MODEL_POLICIES if policy.runnable)
TASK_KEYS = tuple(task.key for task in TASKS)
FIRST_REFORECAST_CALLS = 1000
MAX_TECHNICAL_ATTEMPTS = 3
EXPECTED_VLLM_VERSION = "0.28.0"


def model_key(model_id: str) -> str:
    return model_id.replace("/", "--")


def call_id(model_id: str, task: str, index: int, doc_id: str) -> str:
    return f"screen::{model_key(model_id)}::{task}::{index:04d}::{doc_id}"


def verify_screen_frame(
    source_root: Path,
    seal_root: Path,
    task: str,
    *,
    task_keys: tuple[str, ...] = TASK_KEYS,
) -> list[dict[str, Any]]:
    """Verify full private rows against both the part-1 source and identifier seals."""
    if task not in task_keys:
        raise ValueError(f"unknown task {task}")
    source = source_root / task / "screen.jsonl"
    manifest_path = seal_root / "frame-identifiers" / task / "source-manifest.json"
    identifiers_path = seal_root / "frame-identifiers" / task / "screen.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files"]["screen.jsonl"]
    if sha256_file(source) != expected["sha256"]:
        raise ValueError(f"{task} source screen hash differs from the part-1 seal")
    rows = list(read_jsonl(source))
    identifiers = list(read_jsonl(identifiers_path))
    ids = [str(row["doc_id"]) for row in rows]
    if len(rows) != N_SCREEN or len(identifiers) != N_SCREEN:
        raise ValueError(f"{task} screen must contain exactly {N_SCREEN} rows")
    if ids != [str(row["doc_id"]) for row in identifiers]:
        raise ValueError(f"{task} source row order differs from the part-1 identifier seal")
    if sha256_json(ids) != expected["row_ids_sha256"] or len(ids) != len(set(ids)):
        raise ValueError(f"{task} screen IDs fail the part-1 seal")
    required = {"doc_id", "text", "reference", "reference_normalized", "n_tokens"}
    if any(not required <= set(row) for row in rows):
        raise ValueError(f"{task} screen rows are missing required private fields")
    return rows


def build_calls(
    model_id: str,
    task: str,
    rows: list[dict[str, Any]],
    *,
    policy: RuntimePolicy | None = None,
    task_keys: tuple[str, ...] = TASK_KEYS,
) -> list[dict[str, Any]]:
    """Build only one-document, one-target calls; joint calls are impossible here."""
    policy = policy or policy_for(model_id)
    if not policy.runnable or task not in task_keys or len(rows) != N_SCREEN:
        raise ValueError("solo-screen model, task, or frame is invalid")
    calls: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        doc_id = str(row["doc_id"])
        n_tokens = int(row["n_tokens"] or len(str(row["text"]).split()))
        messages = build_messages(task, [(doc_id, str(row["text"]))], [doc_id])
        calls.append(
            {
                "schema_version": SCHEMA_VERSION,
                "call_id": call_id(model_id, task, index, doc_id),
                "attempt_id": f"{call_id(model_id, task, index, doc_id)}::attempt::1",
                "attempt": 1,
                "model_id": model_id,
                "task_id": task,
                "condition": "solo",
                "doc_id": doc_id,
                "messages": messages,
                "prompt_sha256": sha256_json(messages),
                "max_tokens": estimated_response_tokens(task, 1, [n_tokens]),
                "n_tokens": n_tokens,
            }
        )
    return calls


def chunks(calls: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    # Preserve an exact first-1,000-call checkpoint even when size does not divide 1,000.
    boundaries = list(range(0, min(1000, len(calls)), size)) + [min(1000, len(calls))]
    boundaries += list(range(min(1000, len(calls)) + size, len(calls), size)) + [len(calls)]
    points = sorted(set(boundaries))
    return [calls[a:b] for a, b in zip(points, points[1:], strict=False) if a < b]


def completed_call_ids(events: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(row["call_id"])
        for row in events
        if row.get("terminal") is True and row.get("final_for_call") is True
    }


def pending_calls(
    calls: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resume unfinished planned calls at the next preserved technical attempt."""
    done = completed_call_ids(events)
    attempts: dict[str, int] = {}
    for event in events:
        cid = str(event.get("call_id", ""))
        attempts[cid] = max(attempts.get(cid, 0), int(event.get("attempt", 0)))
    pending: list[dict[str, Any]] = []
    for call in calls:
        cid = str(call["call_id"])
        if cid in done:
            continue
        attempt = attempts.get(cid, 0) + 1
        if attempt > MAX_TECHNICAL_ATTEMPTS:
            raise ValueError(f"unfinished call exceeded its technical attempt limit: {cid}")
        pending.append(
            {
                **call,
                "attempt": attempt,
                "attempt_id": f"{cid}::attempt::{attempt}",
            }
        )
    return pending


def validate_worker_provenance(
    model_id: str,
    host: object,
    backend: object,
    *,
    policy: RuntimePolicy | None = None,
) -> None:
    """Reject successful outputs that do not match the frozen runtime policy."""
    if not isinstance(host, dict) or not isinstance(backend, dict):
        raise ValueError(f"{model_id} successful output lacks worker provenance")
    policy = policy or policy_for(model_id)
    expected_backend = {
        "backend": "vllm",
        "model_id": model_id,
        "model_revision": policy.revision,
        "tokenizer_revision": policy.revision,
        "dtype": "bfloat16",
        "max_model_len": policy.max_model_len,
        "max_num_seqs": 32,
        "tensor_parallel_size": 1,
    }
    mismatched = [key for key, expected in expected_backend.items() if backend.get(key) != expected]
    if mismatched:
        raise ValueError(
            f"{model_id} worker backend differs from the freeze: {', '.join(mismatched)}"
        )
    if host.get("installed_engine_version_string") != EXPECTED_VLLM_VERSION:
        raise ValueError(f"{model_id} worker vLLM version differs from the freeze")
    gpu_name = str(host.get("gpu_name") or "")
    if "A100" not in gpu_name.upper():
        raise ValueError(f"{model_id} worker is not an A100: {gpu_name or 'missing GPU name'}")


def technical_reforecast(
    model_id: str,
    events: list[dict[str, Any]],
    calibration: float,
    *,
    n_tasks: int = len(TASK_KEYS),
) -> dict[str, Any] | None:
    """Rebuild the mandatory first-1,000-call technical forecast from the ledger."""
    if calibration <= 0:
        raise ValueError("meter calibration must be positive")
    if len(events) < FIRST_REFORECAST_CALLS:
        return None
    first = events[:FIRST_REFORECAST_CALLS]
    successful = [row for row in first if row.get("status") == "ok"]
    probe = next(
        (
            row
            for row in successful
            if row.get("timing_probe_method")
            and float(row.get("probe_prompt_plus_first_token_seconds") or 0.0) > 0.0
            and float(row.get("probe_remaining_generation_seconds") or 0.0) > 0.0
        ),
        None,
    )
    if successful and probe is None:
        raise RuntimeError(f"{model_id} lacks a usable single-call timing probe")
    worker_loads: dict[str, float] = {}
    for row in first:
        session_id = row.get("worker_session_id")
        load = row.get("worker_load_seconds")
        if session_id is not None and load is not None:
            key = str(session_id)
            worker_loads[key] = max(float(load), worker_loads.get(key, 0.0))
    loading_seconds = sum(worker_loads.values()) * calibration
    all_in_seconds = sum(float(row.get("metered_a100_seconds", 0.0)) for row in first)
    variable_seconds = max(0.0, all_in_seconds - loading_seconds)
    full_model_calls = n_tasks * N_SCREEN
    projected_base = (
        loading_seconds + variable_seconds * full_model_calls / FIRST_REFORECAST_CALLS
    ) / 3600
    failures = sum(row.get("status") != "ok" for row in first)
    return {
        "model_id": model_id,
        "real_calls": len(first),
        "generation_failures": failures,
        "failure_rate": failures / len(first),
        "loading_a100_seconds": loading_seconds,
        "all_in_a100_seconds": all_in_seconds,
        "component_timing": (
            {
                "scope": "single_call_probe_only",
                "method": probe["timing_probe_method"],
                "attempt_id": probe["attempt_id"],
                "prompt_plus_first_token_seconds": probe["probe_prompt_plus_first_token_seconds"],
                "remaining_generation_seconds": probe["probe_remaining_generation_seconds"],
            }
            if probe is not None
            else None
        ),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in successful),
        "output_tokens": sum(int(row["output_tokens"]) for row in successful),
        "projected_model_screen_base_a100_hours": projected_base,
        "projected_model_screen_guarded_a100_hours": (projected_base * FORECAST_MULTIPLIER),
        "technical_only": True,
    }


def parse_event(event: dict[str, Any], row: dict[str, Any], task: str) -> Any | None:
    if event.get("status") != "ok":
        return None
    try:
        return parse_first_response(
            str(event["first_response"]),
            [str(row["doc_id"])],
            task,
            token_counts={str(row["doc_id"]): int(row["n_tokens"] or 0)},
        )[str(row["doc_id"])]
    except ParseError:
        return None


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _bootstrap_intervals(
    task: str,
    predicted: list[Any | None],
    references: list[Any],
    cell_key: str,
    draws: int = 1000,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(derived_seed("freezeb-screen-bootstrap-v1", cell_key))
    values: dict[str, list[float]] = {}
    n = len(predicted)
    for _ in range(draws):
        indexes = rng.integers(0, n, n)
        result = evaluate(task, [predicted[i] for i in indexes], [references[i] for i in indexes])
        for key, value in result.items():
            if key not in {"admitted", "reasons"} and isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))
    return {
        key: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
        for key, v in sorted(values.items())
        if v
    }


def summarize_cell(
    model_id: str,
    task: str,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    bootstrap_draws: int = 1000,
    policy: RuntimePolicy | None = None,
    task_keys: tuple[str, ...] = TASK_KEYS,
) -> dict[str, Any]:
    expected = build_calls(model_id, task, rows, policy=policy, task_keys=task_keys)
    attempts_seen: set[str] = set()
    by_id: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        cid = str(event.get("call_id", ""))
        attempt_id = str(event.get("attempt_id", ""))
        if not attempt_id or attempt_id in attempts_seen:
            raise ValueError(f"duplicate or missing attempt event {attempt_id or cid}")
        attempts_seen.add(attempt_id)
        by_id.setdefault(cid, []).append(event)
    expected_ids = [call["call_id"] for call in expected]
    unexpected = sorted(set(by_id) - set(expected_ids))
    if unexpected:
        raise ValueError(f"unexpected call IDs in {model_id}::{task}")
    terminal: list[dict[str, Any]] = []
    for cid in expected_ids:
        final = [
            event
            for event in by_id.get(cid, [])
            if event.get("terminal") is True and event.get("final_for_call") is True
        ]
        if len(final) > 1:
            raise ValueError(f"multiple final events for {cid}")
        terminal.extend(final)
    status_counts: dict[str, int] = {}
    for event in events:
        key = str(event.get("status", "unknown"))
        status_counts[key] = status_counts.get(key, 0) + 1
    result: dict[str, Any] = {
        "cell_key": f"{model_id}::{task}",
        "model_id": model_id,
        "task_id": task,
        "expected_calls": N_SCREEN,
        "terminal_calls": len(terminal),
        "technical_complete": len(terminal) == N_SCREEN,
        "status_counts": status_counts,
    }
    if len(terminal) != N_SCREEN:
        result.update({"admitted": False, "reasons": ["technical_screen_incomplete"]})
        return result
    final_status_counts: dict[str, int] = {}
    for event in terminal:
        status = str(event.get("status", "unknown"))
        final_status_counts[status] = final_status_counts.get(status, 0) + 1
    result["final_status_counts"] = final_status_counts
    for event in terminal:
        if event.get("status") == "ok":
            validate_worker_provenance(
                model_id, event.get("worker_host"), event.get("worker_backend"), policy=policy
            )
    if any(event.get("status") != "ok" for event in terminal):
        result.update(
            {
                "technical_excluded": True,
                "admitted": False,
                "reasons": ["technical_generation_or_context_exclusion"],
            }
        )
        return result
    final_by_id = {str(event["call_id"]): event for event in terminal}
    predicted = [
        parse_event(final_by_id[cid], row, task)
        for cid, row in zip(expected_ids, rows, strict=True)
    ]
    references = [
        normalize_label(
            row["reference"],
            task,
            n_tokens=int(row["n_tokens"] or 0) if task == "multinerd" else None,
        )
        for row in rows
    ]
    metrics = evaluate(task, predicted, references)
    parsed = sum(value is not None for value in predicted)
    correct = sum(p is not None and p == r for p, r in zip(predicted, references, strict=True))
    intervals = _bootstrap_intervals(
        task, predicted, references, result["cell_key"], bootstrap_draws
    )
    intervals["parse_rate_wilson"] = _wilson(parsed, N_SCREEN) or [0.0, 0.0]
    intervals["accuracy_wilson_parsed"] = _wilson(correct, parsed) or [0.0, 0.0]
    result.update(metrics)
    result["uncertainty_95"] = intervals
    return result


def build_part2_payload(
    summaries: list[dict[str, Any]],
    power_rows: list[dict[str, float]],
) -> dict[str, Any]:
    """Create the complete admission-lock content; the caller seals it immutably."""
    expected = {f"{m}::{t}" for m in RUNNABLE_MODELS for t in TASK_KEYS}
    if {row["cell_key"] for row in summaries} != expected or len(summaries) != len(expected):
        raise ValueError("part 2 requires exactly all 36 candidate summaries")
    admitted_cells: list[Cell] = []
    item_counts: dict[str, Any] = {}
    for row in summaries:
        if not row.get("technical_complete"):
            raise ValueError("part 2 requires every candidate to reach a terminal technical status")
        if row.get("admitted"):
            model = next(model for model in MODELS if model.repo == row["model_id"])
            task_order = TASK_KEYS.index(str(row["task_id"])) + 1
            admitted_cells.append(
                Cell(model.repo, model.family, str(row["task_id"]), model.order, task_order)
            )
            item_counts[row["cell_key"]] = choose_n(float(row["error_rate"]), power_rows)
    panel = panels(admitted_cells)
    primary, held = panel["primary"], panel["held_out"]
    model_family = {model.repo: model.family for model in MODELS}
    primary_families = sorted({model_family[key.split("::", 1)[0]] for key in primary})
    primary_tasks = sorted({key.split("::", 1)[1] for key in primary})
    if len(primary_families) >= 4 and len(primary_tasks) >= 4:
        recurrence_scope = "full_four_family_four_task"
    elif len(primary_families) >= 3 and len(primary_tasks) >= 3:
        recurrence_scope = "narrowed_three_family_three_task"
    else:
        recurrence_scope = "recurrence_claim_not_supported"
    queue: list[dict[str, Any]] = [
        {"stage": "replication_bridge", "cells": ["historical_qwen_civil"]}
    ]
    queue += [{"stage": "e2_primary", "cells": [key]} for key in primary]
    queue.append({"stage": "e5_first_two_to_400", "cells": primary[:2]})
    queue.append({"stage": "e3_primary_first_six", "cells": primary[:6]})
    queue.append(
        {
            "stage": "e5_primary_remaining_prefixes",
            "cells": primary[:6],
            "prefixes": [1000, 2500, 5000],
        }
    )
    queue.append({"stage": "e4_discovery", "cells": primary[:4]})
    queue.append({"stage": "e4_held_out", "cells": held[:4]})
    queue += [{"stage": "e2_held_out", "cells": [key]} for key in held]
    queue.append(
        {"stage": "e5_held_out_prefixes", "cells": held[:6], "prefixes": [400, 1000, 2500, 5000]}
    )
    queue.append({"stage": "e6_reused_output", "cells": primary})
    queue.append(
        {
            "stage": "e5_capacity_fill",
            "cells": primary[:6],
            "cycle_evaluations": 1000,
            "final_prefix_step": 250,
        }
    )
    return {
        "schema_version": "freezeb-admission-lock-part2-v1",
        "attempted_candidates": sorted(expected),
        "technical_status": {
            row["cell_key"]: {
                "complete": row["technical_complete"],
                "status_counts": row["status_counts"],
            }
            for row in summaries
        },
        "solo_metrics": {row["cell_key"]: row for row in summaries},
        "ordered_cells": panel,
        "claim_scope": {
            "recurrence_scope": recurrence_scope,
            "primary_model_families": primary_families,
            "primary_tasks": primary_tasks,
            "distinct_primary_model_families": len(primary_families),
            "distinct_primary_tasks": len(primary_tasks),
        },
        "item_counts": item_counts,
        "complete_block_queue": queue,
        "joint_outputs_seen": False,
    }
