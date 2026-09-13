"""Frozen E2 planning for admitted Freeze B model-task cells."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from step0.hashing import read_jsonl, sha256_file, sha256_json, sha256_text

from .assign import N_BLOCKS, Frame, build_calls
from .labels import ParseError, parse_first_response
from .meter import may_start, must_stop
from .prompts import build_messages, estimated_response_tokens
from .runtime_benchmark import DECODING, policy_for
from .screen import model_key

SCHEMA_VERSION = "freezeb-e2-v1"


@dataclass(frozen=True)
class CellSpec:
    name: str
    queue_index: int
    cell_key: str
    model_id: str
    task_id: str
    model_revision: str
    output_slug: str
    base_forecast_a100_hours: float
    prior_cumulative_a100_hours: float


FIRST_CELL_SPEC = CellSpec(
    name="qwen-multinli",
    queue_index=0,
    cell_key="Qwen/Qwen3-8B::multi_nli",
    model_id="Qwen/Qwen3-8B",
    task_id="multi_nli",
    model_revision="b968826d9c46dd6066d109eabc6255188de91218",
    output_slug="qwen3-8b--multi_nli",
    base_forecast_a100_hours=1.306764,
    prior_cumulative_a100_hours=4.012538759696318,
)
SECOND_CELL_SPEC = CellSpec(
    name="granite-civil",
    queue_index=1,
    cell_key="ibm-granite/granite-3.3-8b-instruct::civil_comments",
    model_id="ibm-granite/granite-3.3-8b-instruct",
    task_id="civil_comments",
    model_revision="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    output_slug="granite-3.3-8b-instruct--civil_comments",
    base_forecast_a100_hours=0.8081930165939837,
    prior_cumulative_a100_hours=4.824000183954456,
)
CELL_SPECS = {spec.name: spec for spec in (FIRST_CELL_SPEC, SECOND_CELL_SPEC)}

# Preserve the first-cell API used by its completed runner and analysis.
FIRST_CELL = FIRST_CELL_SPEC.cell_key
MODEL_ID = FIRST_CELL_SPEC.model_id
TASK_ID = FIRST_CELL_SPEC.task_id
BATCH_SIZE = 16
N_ITEMS = 4096
N_ASSIGNMENTS = 3
JOINT_CALLS = N_ITEMS // BATCH_SIZE * N_ASSIGNMENTS
EXACT_CALLS = N_ITEMS * N_ASSIGNMENTS
SOLO_CALLS = N_ITEMS
TOTAL_CALLS = JOINT_CALLS + EXACT_CALLS + SOLO_CALLS
CONDITIONS = ("joint", "exact_context", "solo")
FIRST_REFORECAST_CALLS = 1000
BARRIER_CALL_CAP = 3000
MAX_RETRIES = 1
RETRY_STATUSES = frozenset({"generation_failed", "truncated"})
MODEL_REVISION = FIRST_CELL_SPEC.model_revision
ENGINE_VERSION = "0.28.0"
BASE_FORECAST_A100_HOURS = FIRST_CELL_SPEC.base_forecast_a100_hours
PRIOR_CUMULATIVE_A100_HOURS = FIRST_CELL_SPEC.prior_cumulative_a100_hours


def cell_spec(name: str) -> CellSpec:
    try:
        return CELL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown E2 cell {name!r}; choose from {sorted(CELL_SPECS)}") from exc


def verify_cell(admission_lock: dict[str, Any], spec: CellSpec) -> None:
    primary = admission_lock.get("ordered_cells", {}).get("primary", [])
    if len(primary) <= spec.queue_index or primary[spec.queue_index] != spec.cell_key:
        raise ValueError(f"sealed E2 cell {spec.queue_index + 1} is not {spec.cell_key}")
    cell = admission_lock.get("item_counts", {}).get(spec.cell_key, {})
    if cell.get("n_items") != N_ITEMS or cell.get("power_target_met") is not True:
        raise ValueError("the E2 cell's sealed item-count decision differs")
    queue = [
        row
        for row in admission_lock.get("complete_block_queue", [])
        if row.get("stage") == "e2_primary"
    ]
    if len(queue) <= spec.queue_index or queue[spec.queue_index].get("cells") != [spec.cell_key]:
        raise ValueError("the sealed complete-block queue differs")


def verify_first_cell(admission_lock: dict[str, Any]) -> None:
    verify_cell(admission_lock, FIRST_CELL_SPEC)


def verify_confirm_frame(
    source_root: Path,
    seal_root: Path,
    admission_lock: dict[str, Any],
    *,
    spec: CellSpec = FIRST_CELL_SPEC,
) -> list[dict[str, Any]]:
    """Verify private confirmation rows against the complete part-1 identifier seal."""
    verify_cell(admission_lock, spec)
    source = source_root / spec.task_id / "confirm.jsonl"
    identifiers_path = seal_root / "frame-identifiers" / spec.task_id / "confirm.jsonl"
    manifest_path = seal_root / "frame-identifiers" / spec.task_id / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files"]["confirm.jsonl"]
    if sha256_file(source) != expected["sha256"]:
        raise ValueError(f"{spec.task_id} private confirmation frame differs from the part-1 seal")
    rows = list(read_jsonl(source))
    identifiers = list(read_jsonl(identifiers_path))
    ids = [str(row["doc_id"]) for row in rows]
    sealed_ids = [str(row["doc_id"]) for row in identifiers]
    if ids != sealed_ids or sha256_json(ids) != expected["row_ids_sha256"]:
        raise ValueError(f"{spec.task_id} confirmation ID order differs from the part-1 seal")
    if len(rows) != 8192 or len(ids) != len(set(ids)):
        raise ValueError(f"{spec.task_id} confirmation frame must contain 8,192 unique rows")
    required = {"doc_id", "text", "reference", "reference_normalized", "n_chars"}
    if any(not required <= set(row) for row in rows):
        raise ValueError(f"{spec.task_id} confirmation rows are missing private fields")
    return rows[:N_ITEMS]


def _job(
    spec: CellSpec,
    condition: str,
    call_id: str,
    block: int,
    doc_order: list[str],
    target_ids: list[str],
    *,
    assignment: int | None,
    call_index: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_key": spec.cell_key,
        "call_id": call_id,
        "attempt_id": f"{call_id}::attempt::1",
        "attempt": 1,
        "model_id": spec.model_id,
        "task_id": spec.task_id,
        "condition": condition,
        "assignment": assignment,
        "call_index": call_index,
        "execution_block": block,
        "batch_size": len(doc_order),
        "doc_order": doc_order,
        "target_ids": target_ids,
    }


def planned_jobs(
    rows: list[dict[str, Any]], *, spec: CellSpec = FIRST_CELL_SPEC
) -> list[dict[str, Any]]:
    """Create and deterministically interleave one complete sealed E2 cell."""
    if len(rows) != N_ITEMS:
        raise ValueError(f"the first E2 cell requires exactly {N_ITEMS} rows")
    ids = [str(row["doc_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("E2 item IDs must be unique")
    frame = Frame(tuple(ids), tuple(int(row["n_chars"]) for row in rows))
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {
        block: {condition: [] for condition in CONDITIONS} for block in range(N_BLOCKS)
    }
    for call in build_calls(frame, spec.cell_key, BATCH_SIZE):
        assignment = cast(int, call["assignment"])
        call_index = cast(int, call["call_index"])
        block = cast(int, call["execution_block"])
        doc_order = [str(doc_id) for doc_id in cast(list[str], call["doc_order"])]
        prefix = (
            f"e2::{model_key(spec.model_id)}::{spec.task_id}::k{BATCH_SIZE}::"
            f"a{assignment}::b{call_index:04d}"
        )
        grouped[block]["joint"].append(
            _job(
                spec,
                "joint",
                f"{prefix}::joint",
                block,
                doc_order,
                doc_order,
                assignment=assignment,
                call_index=call_index,
            )
        )
        for target_id in doc_order:
            grouped[block]["exact_context"].append(
                _job(
                    spec,
                    "exact_context",
                    f"{prefix}::exact::{target_id}",
                    block,
                    doc_order,
                    [target_id],
                    assignment=assignment,
                    call_index=call_index,
                )
            )
    for index, doc_id in enumerate(ids):
        block = index % N_BLOCKS
        grouped[block]["solo"].append(
            _job(
                spec,
                "solo",
                f"e2::{model_key(spec.model_id)}::{spec.task_id}::solo::{index:04d}::{doc_id}",
                block,
                [doc_id],
                [doc_id],
                assignment=None,
                call_index=index,
            )
        )

    ordered: list[dict[str, Any]] = []
    for block in range(N_BLOCKS):
        lanes = grouped[block]
        condition_order = (
            CONDITIONS[block % len(CONDITIONS) :] + CONDITIONS[: block % len(CONDITIONS)]
        )
        rounds = max(len(lanes[condition]) for condition in CONDITIONS)
        for round_index in range(rounds):
            for rank, condition in enumerate(condition_order):
                if round_index < len(lanes[condition]):
                    job = lanes[condition][round_index]
                    job["execution_order_key"] = f"{block:02d}:{round_index:05d}:{rank}"
                    ordered.append(job)
    validate_plan(ordered, spec=spec)
    return ordered


def validate_plan(jobs: list[dict[str, Any]], *, spec: CellSpec | None = None) -> None:
    if len(jobs) != TOTAL_CALLS or len({job["call_id"] for job in jobs}) != TOTAL_CALLS:
        raise ValueError("E2 plan call count or call IDs differ")
    counts = Counter(str(job["condition"]) for job in jobs)
    if counts != Counter(joint=JOINT_CALLS, exact_context=EXACT_CALLS, solo=SOLO_CALLS):
        raise ValueError("E2 condition counts differ")
    if spec is not None and any(
        job.get("cell_key") != spec.cell_key
        or job.get("model_id") != spec.model_id
        or job.get("task_id") != spec.task_id
        for job in jobs
    ):
        raise ValueError("E2 plan contains a foreign model-task cell")
    keys = [(int(job["execution_block"]), str(job["execution_order_key"])) for job in jobs]
    if keys != sorted(keys):
        raise ValueError("E2 execution order is not block-monotone")
    appearances: Counter[tuple[str, str]] = Counter()
    for job in jobs:
        condition = str(job["condition"])
        for target_id in job["target_ids"]:
            appearances[(condition, str(target_id))] += 1
        if condition == "exact_context" and len(job["target_ids"]) != 1:
            raise ValueError("an exact-context call must request one target")
        if condition == "joint" and job["target_ids"] != job["doc_order"]:
            raise ValueError("a joint call must request every shown document")
    ids = {str(target_id) for job in jobs for target_id in job["target_ids"]}
    for doc_id in ids:
        if appearances[("joint", doc_id)] != N_ASSIGNMENTS:
            raise ValueError("each item must have three joint appearances")
        if appearances[("exact_context", doc_id)] != N_ASSIGNMENTS:
            raise ValueError("each item must have three exact-context appearances")
        if appearances[("solo", doc_id)] != 1:
            raise ValueError("each item must have one solo appearance")


def render_jobs(
    rows: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    spec: CellSpec = FIRST_CELL_SPEC,
) -> list[dict[str, Any]]:
    """Attach frozen prompts and output caps after validating the call-only plan."""
    validate_plan(jobs, spec=spec)
    policy = policy_for(spec.model_id)
    if not policy.runnable:
        raise ValueError("the first E2 model is not runnable under the runtime lock")
    by_id = {str(row["doc_id"]): row for row in rows}
    rendered: list[dict[str, Any]] = []
    for job in jobs:
        docs = [(doc_id, str(by_id[doc_id]["text"])) for doc_id in job["doc_order"]]
        target_ids = [str(doc_id) for doc_id in job["target_ids"]]
        messages = build_messages(spec.task_id, docs, target_ids)
        token_counts = [len(text.split()) for _, text in docs]
        rendered.append(
            {
                **job,
                "messages": messages,
                "prompt_sha256": sha256_json(messages),
                "max_tokens": estimated_response_tokens(
                    spec.task_id, len(target_ids), token_counts
                ),
            }
        )
    return rendered


def retry_job(job: dict[str, Any]) -> dict[str, Any]:
    """Create the single linked retry without changing the scientific call identity."""
    if int(job["attempt"]) != 1:
        raise ValueError("only a first attempt can receive the frozen single retry")
    return {
        **job,
        "attempt": 2,
        "attempt_id": f"{job['call_id']}::attempt::2",
        "parent_attempt_id": str(job["attempt_id"]),
    }


def _expected_attempts(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(job["attempt_id"]): job for job in jobs}


def resume_state(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the append-only attempt ledger before scheduling more work."""
    validate_plan(jobs)
    first = _expected_attempts(jobs)
    expected_model = str(jobs[0]["model_id"])
    expected_task = str(jobs[0]["task_id"])
    seen: dict[str, dict[str, Any]] = {}
    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        call_id = str(event.get("call_id", ""))
        attempt = int(event.get("attempt", 0))
        if attempt_id in seen:
            if sha256_json(event) != sha256_json(seen[attempt_id]):
                raise ValueError(f"attempt {attempt_id} has conflicting stored events")
            continue
        if attempt == 1:
            expected = first.get(attempt_id)
            if expected is None or call_id != str(expected["call_id"]):
                raise ValueError("stored first attempts do not belong to this E2 plan")
        elif attempt == 2:
            expected_first = first.get(f"{call_id}::attempt::1")
            if (
                expected_first is None
                or attempt_id != f"{call_id}::attempt::2"
                or event.get("parent_attempt_id") != expected_first["attempt_id"]
            ):
                raise ValueError("stored retries do not belong to this E2 plan")
        else:
            raise ValueError("E2 permits only first attempts and one linked retry")
        if event.get("model_id") != expected_model or event.get("task_id") != expected_task:
            raise ValueError("stored events name a foreign model or task")
        seen[attempt_id] = event
    return {"seen": seen, "first": first}


def pending_work(
    jobs: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    state = resume_state(jobs, events)
    seen = cast(dict[str, dict[str, Any]], state["seen"])
    first_responses = [job for job in jobs if str(job["attempt_id"]) not in seen]
    retries: list[dict[str, Any]] = []
    for job in jobs:
        event = seen.get(str(job["attempt_id"]))
        if event is None or event.get("status") not in RETRY_STATUSES:
            continue
        retry = retry_job(job)
        if str(retry["attempt_id"]) not in seen:
            retries.append(retry)
    return {"first_responses": first_responses, "retries": retries}


def completeness(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    work = pending_work(jobs, events)
    state = resume_state(jobs, events)
    seen = cast(dict[str, dict[str, Any]], state["seen"])
    first_seen = [event for event in seen.values() if int(event["attempt"]) == 1]
    retry_seen = [event for event in seen.values() if int(event["attempt"]) == 2]
    status_counts = Counter(str(event["status"]) for event in seen.values())
    return {
        "complete": not work["first_responses"] and not work["retries"],
        "n_expected_first_attempts": len(jobs),
        "n_first_attempts": len(first_seen),
        "n_retry_attempts": len(retry_seen),
        "n_pending_first_attempts": len(work["first_responses"]),
        "n_pending_retries": len(work["retries"]),
        "missing_call_ids_sample": [str(job["call_id"]) for job in work["first_responses"][:10]],
        "status_counts": dict(sorted(status_counts.items())),
    }


def parse_result(
    raw: str,
    target_ids: list[str],
    finish_reason: str | None,
    *,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    """Keep the raw first response and add a deterministic parse classification."""
    if finish_reason == "length":
        return {"status": "truncated", "parse_status": "truncated", "predictions": None}
    try:
        predictions = parse_first_response(raw, target_ids, task_id)
    except ParseError as exc:
        return {
            "status": "ok",
            "parse_status": "malformed",
            "parse_error": str(exc),
            "predictions": None,
        }
    return {"status": "ok", "parse_status": "ok", "predictions": predictions}


def event_for_result(
    job: dict[str, Any],
    *,
    raw: str = "",
    finish_reason: str | None = None,
    generation_error: str | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    queue_seconds: float | None = None,
    prefill_seconds: float | None = None,
    decode_seconds: float | None = None,
) -> dict[str, Any]:
    """Build one append-only event for one model invocation."""
    parsed = (
        {
            "status": "generation_failed",
            "parse_status": "missing",
            "predictions": None,
        }
        if generation_error is not None
        else parse_result(
            raw,
            [str(value) for value in job["target_ids"]],
            finish_reason,
            task_id=str(job["task_id"]),
        )
    )
    return {
        **{
            key: job[key]
            for key in (
                "schema_version",
                "cell_key",
                "call_id",
                "attempt_id",
                "attempt",
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
            )
        },
        "parent_attempt_id": job.get("parent_attempt_id"),
        "terminal": True,
        **parsed,
        "first_response": raw,
        "output_sha256": sha256_text(raw),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "queue_seconds": queue_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "generation_error": generation_error,
    }


def verify_worker(
    host: dict[str, Any] | None,
    backend: dict[str, Any] | None,
    *,
    spec: CellSpec = FIRST_CELL_SPEC,
) -> None:
    """Refuse output from a changed checkpoint, engine, serving policy, or GPU class."""
    if not isinstance(host, dict) or not isinstance(backend, dict):
        raise ValueError("a successful E2 chunk carries no worker provenance")
    policy = policy_for(spec.model_id)
    expected = {
        "backend": "vllm",
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "tokenizer_revision": spec.model_revision,
        "dtype": "bfloat16",
        "max_model_len": policy.max_model_len,
        "max_num_seqs": 32,
        "tensor_parallel_size": 1,
        "enable_thinking": False,
        "seed": int(DECODING["seed"]),
    }
    mismatched = sorted(key for key, value in expected.items() if backend.get(key) != value)
    if mismatched:
        raise ValueError(f"E2 worker backend differs from the runtime lock: {mismatched}")
    if str(host.get("installed_engine_version_string")) != ENGINE_VERSION:
        raise ValueError("E2 worker runs a different vLLM version")
    if "A100" not in str(host.get("gpu_name") or "").upper():
        raise ValueError("E2 worker is not an A100")


def verify_checkpoint(
    actual: dict[str, Any],
    manifests: list[dict[str, Any]],
    *,
    spec: CellSpec = FIRST_CELL_SPEC,
) -> dict[str, Any]:
    """Check a materialized cell checkpoint against its sealed runtime manifest."""
    expected = next((row for row in manifests if str(row.get("model_id")) == spec.model_id), None)
    if expected is None:
        raise ValueError(f"the runtime lock records no checkpoint manifest for {spec.model_id}")
    if str(expected["revision"]) != spec.model_revision:
        raise ValueError("the runtime-lock manifest pins a different model revision")
    if (
        str(actual.get("model_id")) != spec.model_id
        or str(actual.get("revision")) != spec.model_revision
    ):
        raise ValueError("the materialized checkpoint is not the selected pinned model revision")
    checked = list(actual.get("files", []))
    wanted = {str(row["path"]): row for row in expected["files"]}
    seen = {str(row["path"]): row for row in checked}
    missing = sorted(set(wanted) - set(seen))
    if missing:
        raise ValueError(f"the volume is missing {len(missing)} pinned files: {missing[:3]}")
    hashed = 0
    for path, row in sorted(wanted.items()):
        found = seen[path]
        if int(found["bytes"]) != int(row["bytes"]):
            raise ValueError(f"{path} differs in size from the runtime-lock manifest")
        if found.get("sha256") is None:
            continue
        if str(found["sha256"]) != str(row["sha256"]):
            raise ValueError(f"{path} differs in content from the runtime-lock manifest")
        hashed += 1
    return {
        "model_id": spec.model_id,
        "revision": spec.model_revision,
        "volume": "ieee-bigdata-freezeb-models",
        "n_files": len(wanted),
        "n_files_hash_verified": hashed,
        "n_files_size_verified_only": len(wanted) - hashed,
        "source": "runtime-lock/20260913T004054Z-BRT1 checkpoint manifest",
    }


def attribute_seconds(total: float, events: list[dict[str, Any]]) -> tuple[list[float], str]:
    components: list[float | None] = []
    for event in events:
        values = [event.get(key) for key in ("queue_seconds", "prefill_seconds", "decode_seconds")]
        components.append(
            sum(float(value) for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )
    if components and all(value is not None and value > 0 for value in components):
        denominator = sum(cast(float, value) for value in components)
        return [total * cast(float, value) / denominator for value in components], "engine_timing"
    return [total / len(events)] * len(events), "equal_share_within_chunk"


def timing_records(
    jobs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    chunk_id: str,
    metered_a100_seconds: float,
    cumulative_a100_hours: float,
) -> list[dict[str, Any]]:
    if [job["attempt_id"] for job in jobs] != [event["attempt_id"] for event in events]:
        raise ValueError("E2 chunk events differ from their requested attempt order")
    shares, method = attribute_seconds(metered_a100_seconds, events)
    return [
        {
            "attempt_id": event["attempt_id"],
            "call_id": event["call_id"],
            "condition": event["condition"],
            "execution_block": event["execution_block"],
            "chunk_id": chunk_id,
            "status": event["status"],
            "parse_status": event["parse_status"],
            "metered_a100_seconds": share,
            "chunk_metered_a100_seconds": metered_a100_seconds,
            "attribution_method": method,
            "prompt_tokens": event.get("prompt_tokens"),
            "output_tokens": event.get("output_tokens"),
            "queue_seconds": event.get("queue_seconds"),
            "prefill_seconds": event.get("prefill_seconds"),
            "decode_seconds": event.get("decode_seconds"),
            "cumulative_a100_hours": cumulative_a100_hours,
        }
        for event, share in zip(events, shares, strict=True)
    ]


def remaining_counts(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, int]:
    work = pending_work(jobs, events)
    return dict(Counter(str(job["condition"]) for job in work["retries"] + work["first_responses"]))


def reforecast(
    timings: list[dict[str, Any]],
    remaining: dict[str, int],
    *,
    cumulative_a100_hours: float,
    remaining_load_seconds: float = 0.0,
    workers: int = 1,
) -> dict[str, Any]:
    rates: dict[str, float] = {}
    for condition in CONDITIONS:
        values = [
            float(row["metered_a100_seconds"]) for row in timings if row["condition"] == condition
        ]
        if values:
            rates[condition] = statistics.fmean(values)
    missing = sorted(
        condition for condition, count in remaining.items() if count and condition not in rates
    )
    if missing:
        raise ValueError(f"no measured E2 rate yet for {missing}")
    base_seconds = remaining_load_seconds + sum(
        rates[condition] * count for condition, count in remaining.items()
    )
    base_hours = base_seconds / 3600.0
    decision = may_start(cumulative_a100_hours, base_hours)
    return {
        "technical_only": True,
        "real_attempts_used": len(timings),
        "seconds_per_attempt_by_condition": rates,
        "remaining_attempts_by_condition": dict(sorted(remaining.items())),
        "remaining_load_a100_seconds": remaining_load_seconds,
        "projected_remaining_base_a100_hours": base_hours,
        "projected_remaining_guarded_a100_hours": 1.6 * base_hours,
        "cumulative_a100_hours": cumulative_a100_hours,
        "projected_total_a100_hours": decision.projected_total_hours,
        "may_continue": decision.start,
        "decision_reason": decision.reason,
        "must_stop_now": must_stop(cumulative_a100_hours),
        "n_concurrent_workers_when_measured": workers,
    }
