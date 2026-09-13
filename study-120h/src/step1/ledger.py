"""Append-only call ledger, duplicate and conflict rejection, and the A100-hour meter.

One row per requested label, exactly as the frozen call-lineage schema requires. A row is
therefore identified by its call and its target, not by its call alone: a joint call that
requests 16 labels stores 16 rows that share one `call_id` and one `output_sha256`.

A row is appended once and never edited. A repeated `(call_id, doc_id)` with the same output
is a duplicate and is skipped. A call offered with a different output than the stored one is
a conflict: the whole call is rejected, recorded, and never mixed into the ledger beside the
original. A retry is a new row linked by `parent_call_id`; it never replaces a first response.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from step0.hashing import canonical_json, read_jsonl, sha256_text

from .backends import GenResult
from .frozen import Frozen
from .plan import Call
from .prompts import parse_labels, target_status

#: The frozen call-lineage schema of protocol section 7. Every field is mandatory.
ROW_FIELDS: tuple[str, ...] = (
    "call_id",
    "parent_call_id",
    "retry_index",
    "first_response_status",
    "cell_id",
    "model_id",
    "model_revision",
    "task_id",
    "dataset_revision",
    "condition",
    "assignment",
    "call_index",
    "batch_id",
    "doc_order",
    "doc_id",
    "slot",
    "length_stratum",
    "execution_block",
    "serving_session",
    "concurrency_window",
    "prompt_sha256",
    "prompt_tokens",
    "output_tokens",
    "max_tokens",
    "latency_seconds",
    "gpu_seconds",
    "raw_text",
    "output_sha256",
    "parse_status",
    "parse_notes",
    "pred",
    "reference",
    "error",
    "created_at",
    "run_id",
    "engine_version",
)

#: Step 1 additions. They record how a call was executed and timed; they add no design field.
EXTRA_FIELDS: tuple[str, ...] = (
    "frame_task_id",
    "step",
    "serving_concurrency",
    "chunk_id",
    "chunk_size",
    "chunk_seconds",
    "finish_reason",
)

#: The frozen first-response state machine. `refused` is never assigned automatically.
FIRST_RESPONSE_STATES = frozenset(
    {"valid_first_response", "malformed", "truncated", "refused", "missing", "retry_valid"}
)


class LedgerError(RuntimeError):
    """Raised when a row breaks the schema or a conflicting result is offered."""


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    serving_session: str
    engine_version: str
    step: str = "step1"
    chunk_id: str | None = None
    chunk_size: int | None = None
    chunk_seconds: float | None = None
    concurrency_window: str | None = None


def first_response_status(result: GenResult | None, parse_status: str, retry_index: int) -> str:
    """Classify one response. A refusal is never guessed; it needs a human label."""
    if result is None:
        return "missing"
    if result.finish_reason == "length":
        return "truncated"
    if parse_status == "ok":
        return "retry_valid" if retry_index > 0 else "valid_first_response"
    return "malformed"


def rows_for(
    frozen: Frozen,
    call: Call,
    result: GenResult | None,
    frame_task_id: str,
    references: dict[str, int],
    strata: dict[str, int],
    meta: RunMeta,
) -> list[dict[str, Any]]:
    """One row per requested label of one call, in frozen schema order."""
    job = call.job
    permitted = frozen.permitted_labels(frame_task_id)
    raw_text = result.text if result else ""
    parsed = parse_labels(raw_text, call.target_ids, permitted) if result else None
    doc_order = [str(d) for d in job["doc_order"]]
    n_calls = max(1, int(meta.chunk_size or 1))
    share = float(meta.chunk_seconds or 0.0) / n_calls
    created_at = dt.datetime.now(dt.UTC).isoformat()

    rows: list[dict[str, Any]] = []
    for target_id, slot in zip(call.target_ids, job["target_slots"], strict=True):
        status = target_status(parsed, target_id) if parsed else "missing"
        pred = parsed.labels.get(target_id) if parsed and status == "ok" else None
        reference = references[target_id]
        row = {
            "call_id": str(job["call_id"]),
            "parent_call_id": job["parent_call_id"],
            "retry_index": int(job["retry_index"]),
            "first_response_status": first_response_status(
                result, parsed.status if parsed else "missing", int(job["retry_index"])
            ),
            "cell_id": str(job["cell_id"]),
            "model_id": str(job["model_id"]),
            "model_revision": frozen.model_revision(str(job["model_id"])),
            "task_id": str(job["task_id"]),
            "dataset_revision": frozen.dataset_revision(frame_task_id),
            "condition": str(job["condition"]),
            "assignment": int(job["assignment"]),
            "call_index": int(job["call_index"]),
            "batch_id": str(job["batch_id"]),
            "doc_order": doc_order,
            "doc_id": str(target_id),
            "slot": int(slot),
            "length_stratum": strata[target_id],
            "execution_block": int(job["execution_block"]),
            "serving_session": meta.serving_session,
            "concurrency_window": meta.concurrency_window,
            "prompt_sha256": call.prompt_sha256,
            "prompt_tokens": result.prompt_tokens if result else None,
            "output_tokens": result.output_tokens if result else None,
            "max_tokens": call.max_tokens,
            "latency_seconds": share,
            "gpu_seconds": share,
            "raw_text": raw_text,
            "output_sha256": sha256_text(raw_text),
            "parse_status": status,
            "parse_notes": list(parsed.notes) if parsed else ["no_response"],
            "pred": pred,
            "reference": reference,
            "error": None if pred is None else int(pred != reference),
            "created_at": created_at,
            "run_id": meta.run_id,
            "engine_version": meta.engine_version,
            "frame_task_id": frame_task_id,
            "step": meta.step,
            "serving_concurrency": int(job.get("serving_concurrency", 0)),
            "chunk_id": meta.chunk_id,
            "chunk_size": meta.chunk_size,
            "chunk_seconds": meta.chunk_seconds,
            "finish_reason": result.finish_reason if result else None,
        }
        check_row(row)
        rows.append(row)
    return rows


def check_row(row: dict[str, Any]) -> None:
    """Reject a row that breaks the frozen schema before it can reach the file."""
    missing = [field for field in ROW_FIELDS if field not in row]
    if missing:
        raise LedgerError(f"row is missing mandatory schema fields: {missing}")
    unexpected = sorted(set(row) - set(ROW_FIELDS) - set(EXTRA_FIELDS))
    if unexpected:
        raise LedgerError(f"row carries undeclared fields: {unexpected}")
    if row["first_response_status"] not in FIRST_RESPONSE_STATES:
        raise LedgerError(f"unknown first_response_status {row['first_response_status']!r}")
    if int(row["retry_index"]) > 0 and not row["parent_call_id"]:
        raise LedgerError("a retry must name the call it retries in parent_call_id")
    if int(row["retry_index"]) == 0 and row["parent_call_id"]:
        raise LedgerError("a first response must have a null parent_call_id")


def rows_by_call(path: Path) -> dict[str, dict[str, str]]:
    """call_id to {doc_id: output_sha256} for every stored row. Empty when the file is absent.

    This is the true row identity of the frozen schema. A solo call holds one entry; a joint
    call that requests 16 labels holds 16, one per target.
    """
    index: dict[str, dict[str, str]] = {}
    if not path.exists():
        return index
    for row in read_jsonl(path):
        index.setdefault(str(row["call_id"]), {})[str(row["doc_id"])] = str(row["output_sha256"])
    return index


def index_of(path: Path) -> dict[str, str]:
    """call_id to output_sha256 for every call already stored, whatever its target count."""
    return {
        call_id: _one_digest(call_id, targets)
        for call_id, targets in rows_by_call(path).items()
    }


def _one_digest(call_id: str, targets: dict[str, str]) -> str:
    """Every row of one call carries the hash of that call's single unedited output."""
    digests = set(targets.values())
    if len(digests) != 1:
        raise LedgerError(f"{call_id} is stored with more than one output hash")
    return digests.pop()


def completed_call_ids(path: Path) -> set[str]:
    """Calls that hold at least one stored row. Completeness per target is checked separately."""
    return set(rows_by_call(path))


def _group_offered(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    offered: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        offered.setdefault(str(row["call_id"]), []).append(row)
    for call_id, call_rows in offered.items():
        if len({str(row["output_sha256"]) for row in call_rows}) != 1:
            raise LedgerError(f"{call_id} was offered with more than one output in one append")
        doc_ids = [str(row["doc_id"]) for row in call_rows]
        if len(set(doc_ids)) != len(doc_ids):
            raise LedgerError(f"{call_id} was offered the same target twice in one append")
    return offered


def append(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Append new rows. Skip duplicates. Reject conflicts. Never edit a stored row.

    The unit of decision is the call, and the unit of storage is the requested label. A call
    whose output differs from the stored one is rejected whole, so a ledger never holds a
    mixture of two outputs for one call. A call that is stored with the same output but with
    missing target rows is completed, because every requested label needs its own row.
    """
    for row in rows:
        check_row(row)
    stored = rows_by_call(path)
    fresh: list[dict[str, Any]] = []
    duplicate_rows = 0
    repaired: list[str] = []
    conflicts: list[dict[str, Any]] = []

    for call_id, call_rows in _group_offered(rows).items():
        digest = str(call_rows[0]["output_sha256"])
        known = stored.get(call_id)
        if known is None:
            stored[call_id] = {str(row["doc_id"]): digest for row in call_rows}
            fresh.extend(call_rows)
            continue
        if _one_digest(call_id, known) != digest:
            conflicts.append(
                {
                    "call_id": call_id,
                    "stored_output_sha256": _one_digest(call_id, known),
                    "offered_output_sha256": digest,
                    "n_rows_rejected": len(call_rows),
                    "rejected_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "reason": "the same call_id already has a different stored output",
                }
            )
            continue
        missing = [row for row in call_rows if str(row["doc_id"]) not in known]
        duplicate_rows += len(call_rows) - len(missing)
        if missing:
            repaired.append(call_id)
            for row in missing:
                known[str(row["doc_id"])] = digest
            fresh.extend(missing)

    if conflicts:
        _append_lines(path.with_suffix(path.suffix + ".conflicts.jsonl"), conflicts)
    if fresh:
        _append_lines(path, fresh)
    return {
        "path": str(path),
        "appended": len(fresh),
        "duplicates_skipped": duplicate_rows,
        "calls_completed": sorted(repaired),
        "conflicts_rejected": len(conflicts),
        "conflicts": conflicts,
    }


def _append_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    """One write, then one fsync, so an interrupted run leaves whole lines only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(canonical_json(row) + "\n" for row in rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())


def validate(
    path: Path, expected_jobs: list[dict[str, Any]], *, rows_per_call: int | None = None
) -> dict[str, Any]:
    """Completeness and consistency report. `ok` is the gate that lets evaluation run."""
    problems: list[str] = []
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "problems": [f"{path} does not exist; no call has been executed"],
            "n_rows": 0,
            "n_calls": 0,
            "n_expected_calls": len(expected_jobs),
            "n_missing_calls": len(expected_jobs),
        }

    rows = list(read_jsonl(path))
    expected = {str(job["call_id"]) for job in expected_jobs}
    first_rows = [r for r in rows if int(r["retry_index"]) == 0]
    retry_rows = [r for r in rows if int(r["retry_index"]) > 0]
    present = {str(r["call_id"]) for r in first_rows}
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)

    per_call: dict[str, int] = {}
    digests: dict[str, set[str]] = {}
    for row in rows:
        call_id = str(row["call_id"])
        per_call[call_id] = per_call.get(call_id, 0) + 1
        digests.setdefault(call_id, set()).add(str(row["output_sha256"]))

    if missing:
        problems.append(f"{len(missing)} planned calls have no first response")
    if unexpected:
        problems.append(f"{len(unexpected)} stored calls are not in the frozen plan")
    split = sorted(call_id for call_id, values in digests.items() if len(values) > 1)
    if split:
        problems.append(f"{len(split)} call ids carry more than one output")
    if rows_per_call is not None:
        wrong = sorted(c for c, n in per_call.items() if n != rows_per_call)
        if wrong:
            problems.append(f"{len(wrong)} calls do not hold {rows_per_call} rows")
    orphans = sorted(
        str(r["call_id"]) for r in retry_rows if str(r["parent_call_id"]) not in per_call
    )
    if orphans:
        problems.append(f"{len(orphans)} retries name a parent call that is not stored")
    conflicts_path = path.with_suffix(path.suffix + ".conflicts.jsonl")
    n_conflicts = sum(1 for _ in read_jsonl(conflicts_path)) if conflicts_path.exists() else 0
    if n_conflicts:
        problems.append(f"{n_conflicts} conflicting results were rejected; investigate before use")

    return {
        "ok": not problems,
        "path": str(path),
        "n_rows": len(rows),
        "n_calls": len(per_call),
        "n_first_response_calls": len(present),
        "n_retry_rows": len(retry_rows),
        "n_expected_calls": len(expected),
        "n_missing_calls": len(missing),
        "missing_call_ids_sample": missing[:10],
        "unexpected_call_ids_sample": unexpected[:10],
        "n_rejected_conflicts": n_conflicts,
        "problems": problems,
    }


# ------------------------------------------------------------------------------------
# A100 meter
# ------------------------------------------------------------------------------------


def append_run(path: Path, record: dict[str, Any]) -> None:
    """Record one metered GPU run. Concurrent workers each append their own record."""
    required = {"run_id", "worker_id", "model_id", "gpu_name", "gpu_seconds_billed"}
    missing = sorted(required - set(record))
    if missing:
        raise LedgerError(f"a run record is missing {missing}")
    _append_lines(path, [record])


def meter(path: Path) -> dict[str, Any]:
    """Total metered A100 hours. Concurrent workers are summed, never counted as wall clock.

    Two workers that each run one A100 for one hour cost two A100-hours, even though the wall
    clock advanced one hour. Concurrency shortens the calendar, never the budget.
    """
    runs = list(read_jsonl(path)) if path.exists() else []
    by_worker: dict[str, float] = {}
    for run in runs:
        worker = str(run["worker_id"])
        by_worker[worker] = by_worker.get(worker, 0.0) + float(run["gpu_seconds_billed"])
    total = sum(by_worker.values())
    wall = max(by_worker.values()) if by_worker else 0.0
    return {
        "n_runs": len(runs),
        "a100_seconds_total": total,
        "a100_hours_total": total / 3600.0,
        "a100_hours_by_worker": {w: s / 3600.0 for w, s in sorted(by_worker.items())},
        "longest_single_worker_hours": wall / 3600.0,
        "accounting_rule": (
            "metered A100 hours are summed over workers. Two concurrent single-A100 workers "
            "cost the same total hours as one worker running twice as long."
        ),
    }
