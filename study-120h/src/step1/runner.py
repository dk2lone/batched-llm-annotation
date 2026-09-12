"""Exact-tokenizer preflight and the solo-only run loop.

The preflight builds every frozen prompt type of a cell and counts it with the pinned
tokenizer. It generates nothing. It is the gate that must pass before a GPU starts, because
the frozen context-fit criterion covers the joint, exact-context, solo, and filler prompts,
not only the prompts Step 1 executes.

`run_solo` executes the solo condition only. It refuses any other condition.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from .backends import Backend, GenRequest, GenResult, TokenCounter
from .frozen import Frozen
from .ledger import RunMeta, append
from .plan import PREFLIGHT_CONDITIONS, Call, build_call, doc_texts, job_rows, solo_jobs

#: Histogram edges for the frozen prompt-length distribution, in tokens.
HISTOGRAM_EDGES = (0, 128, 256, 512, 1024, 2048, 4096, 6144, 7808, 8192)


class SoloOnlyError(RuntimeError):
    """Raised when anything but a solo call reaches the Step 1 run loop."""


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return float(ordered[index])

    return {"p50": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": float(ordered[-1])}


def _histogram(values: list[int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        for low, high in zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:], strict=False):
            if low <= value < high:
                counts[f"{low}-{high}"] += 1
                break
        else:
            counts[f"{HISTOGRAM_EDGES[-1]}+"] += 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+"))))


def preflight(
    frozen: Frozen,
    cell_id: str,
    frame_task_id: str,
    counter: TokenCounter,
    *,
    model_id: str | None = None,
    conditions: tuple[str, ...] = PREFLIGHT_CONDITIONS,
) -> dict[str, Any]:
    """Count every frozen prompt of one cell with the pinned tokenizer. Generate nothing.

    The returned `ok` field is the frozen context-fit criterion: zero prompts over budget.
    `by_condition` carries the frozen prompt-length distribution the throughput forecast uses.
    """
    exact = bool(getattr(counter, "exact", False))
    texts = doc_texts(frozen, frame_task_id)
    started = time.perf_counter()
    per_condition: dict[str, dict[str, Any]] = {}
    over_budget: list[dict[str, Any]] = []
    n_calls = 0
    all_jobs = job_rows(
        frozen,
        cell_id,
        frame_task_id,
        model_id=model_id,
        conditions=conditions,
    )

    for condition in conditions:
        jobs = [job for job in all_jobs if job["condition"] == condition]
        if not jobs:
            continue
        prompt_tokens: list[int] = []
        for job in jobs:
            call = build_call(frozen, job, frame_task_id, texts)
            tokens = counter.count_tokens(call.messages)
            prompt_tokens.append(tokens)
            budget = frozen.prompt_token_budget(len(call.target_ids))
            if tokens > budget:
                over_budget.append(
                    {
                        "call_id": call.call_id,
                        "condition": condition,
                        "prompt_tokens": tokens,
                        "budget": budget,
                        "max_tokens": call.max_tokens,
                    }
                )
        n_calls += len(jobs)
        n_requested = int(jobs[0]["n_requested"])
        per_condition[condition] = {
            "n_calls": len(jobs),
            "n_requested_labels_per_call": n_requested,
            "max_tokens": frozen.max_tokens(n_requested),
            "prompt_token_budget": frozen.prompt_token_budget(n_requested),
            "prompt_tokens_sum": sum(prompt_tokens),
            "prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
            **_percentiles(prompt_tokens),
            "histogram": _histogram(prompt_tokens),
        }

    return {
        "ok": exact and not over_budget,
        "exact_tokenizer": exact,
        "counter": getattr(counter, "name", type(counter).__name__),
        "cell_id": cell_id,
        "model_id": model_id or str(frozen.cell(cell_id)["model_id"]),
        "frame_task_id": frame_task_id,
        "max_model_len": int(frozen.serving["max_model_len"]),
        "conditions_checked": sorted(per_condition),
        "n_calls_checked": n_calls,
        "n_over_budget": len(over_budget),
        "over_budget": over_budget[:50],
        "by_condition": per_condition,
        "seconds": time.perf_counter() - started,
        "generated_any_output": False,
        "note": (
            "the context-fit criterion needs an exact pinned-tokenizer count. A coarse counter "
            "can never set ok to true."
        ),
        "checked_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def _requests(calls: list[Call]) -> list[GenRequest]:
    return [
        GenRequest(
            call_id=call.call_id,
            condition=call.condition,
            target_ids=tuple(call.target_ids),
            messages=call.messages,
            max_tokens=call.max_tokens,
        )
        for call in calls
    ]


def run_solo(
    frozen: Frozen,
    cell_id: str,
    frame_task_id: str,
    backend: Backend,
    out_path: Path,
    *,
    model_id: str | None = None,
    run_id: str | None = None,
    serving_session: str | None = None,
    engine_version: str = "unknown",
    chunk_size: int = 64,
    limit: int | None = None,
    jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute pending solo calls in frozen order and append them to the ledger.

    Resumable: a call already in the ledger is skipped. Append-safe: each chunk is one append.
    Solo-only: any other condition raises before a single call is sent.
    """
    from .ledger import completed_call_ids, rows_for  # noqa: PLC0415 - avoids a cycle at import

    planned = (
        jobs if jobs is not None else solo_jobs(frozen, cell_id, frame_task_id, model_id=model_id)
    )
    offenders = sorted({str(j["condition"]) for j in planned} - {"solo"})
    if offenders:
        raise SoloOnlyError(f"Step 1 generates solo responses only, got {offenders}")

    items = frozen.items(frame_task_id)
    references = {i.doc_id: i.reference for i in items}
    strata = {i.doc_id: i.length_stratum for i in items}
    texts = doc_texts(frozen, frame_task_id)

    done = completed_call_ids(out_path)
    pending = [job for job in planned if str(job["call_id"]) not in done]
    if limit is not None:
        pending = pending[:limit]

    run = run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    session = serving_session or uuid.uuid4().hex[:16]
    started = time.perf_counter()
    appended = duplicates = conflicts = 0
    chunk_records: list[dict[str, Any]] = []

    for start in range(0, len(pending), chunk_size):
        chunk = [
            build_call(frozen, job, frame_task_id, texts)
            for job in pending[start : start + chunk_size]
        ]
        chunk_id = f"{run}-{start // chunk_size:05d}"
        chunk_started = time.perf_counter()
        results: dict[str, GenResult] = {
            result.call_id: result for result in backend.generate(_requests(chunk))
        }
        chunk_seconds = time.perf_counter() - chunk_started
        meta = RunMeta(
            run_id=run,
            serving_session=session,
            engine_version=engine_version,
            chunk_id=chunk_id,
            chunk_size=len(chunk),
            chunk_seconds=chunk_seconds,
        )
        rows: list[dict[str, Any]] = []
        for call in chunk:
            rows.extend(
                rows_for(
                    frozen, call, results.get(call.call_id), frame_task_id, references, strata, meta
                )
            )
        report = append(out_path, rows)
        appended += int(report["appended"])
        duplicates += int(report["duplicates_skipped"])
        conflicts += int(report["conflicts_rejected"])
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "n_calls": len(chunk),
                "chunk_seconds": chunk_seconds,
                "prompt_tokens_sum": sum(
                    r.prompt_tokens for r in results.values() if r.prompt_tokens
                ),
                "output_tokens_sum": sum(
                    r.output_tokens for r in results.values() if r.output_tokens
                ),
            }
        )

    return {
        "cell_id": cell_id,
        "frame_task_id": frame_task_id,
        "model_id": model_id or str(frozen.cell(cell_id)["model_id"]),
        "run_id": run,
        "serving_session": session,
        "backend": backend.describe(),
        "n_planned": len(planned),
        "n_already_done": len(done),
        "n_attempted": len(pending),
        "rows_appended": appended,
        "duplicates_skipped": duplicates,
        "conflicts_rejected": conflicts,
        "generation_seconds": sum(float(c["chunk_seconds"]) for c in chunk_records),
        "wall_seconds": time.perf_counter() - started,
        "chunks": chunk_records,
        "out_path": str(out_path),
    }
