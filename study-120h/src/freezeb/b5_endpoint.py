"""B5 batch-size dose-response: nesting, schedule, output contract, audit, forecast.

Nothing here has seen a B5 model output. The module is the single source of the
frozen schedule: the same code builds the smoke selection and the full schedule,
so the two differ only in which scheduled work is selected.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np

from step0.hashing import derived_seed, sha256_json

from .prompts import build_messages

SCHEMA_VERSION = "b5-endpoint-pilot-v1"
CONTRACT_VERSION = "b5-structured-output-v1"
CALL_PREFIX = "b5-endpoint-pilot-v1"
ASSIGN_NAMESPACE = "b5-endpoint-pilot-v1"
POWER_NAMESPACE = "b5-power-v1"
BOOTSTRAP_NAMESPACE = "b5-endpoint-pilot-v1-bootstrap"

SIZES = (2, 32)
N_ASSIGNMENTS = 1
N_ITEMS = 8192
CONDITIONS = ("joint", "exact_context")

# Model-major execution order. One task per model, as in the sealed B4 panel.
CELLS = (
    "Qwen/Qwen3-8B::civil_comments",
    "ibm-granite/granite-3.3-8b-instruct::imdb",
    "meta-llama/Llama-3.1-8B-Instruct::civil_comments",
    "mistralai/Mistral-7B-Instruct-v0.3::multi_nli",
)

EXPECTED_JOINT = 17_408
EXPECTED_EXACT = 65_536
EXPECTED_TOTAL = 82_944

LABEL_VALUES: dict[str, tuple[int, ...]] = {
    "civil_comments": (0, 1),
    "imdb": (0, 1),
    "multi_nli": (0, 1, 2),
    "ag_news": (0, 1, 2, 3),
}

MAX_MODEL_LEN = 32_768
# A byte-level BPE or SentencePiece token always covers at least one UTF-8 byte, so the
# serialized byte length bounds the token count. The allowance covers the end-of-sequence
# token and any leading byte the chat template adds before the JSON object.
OUTPUT_TOKEN_ALLOWANCE = 16

RETRYABLE = frozenset({"generation_failed", "truncated", "worker_failed"})
MAX_ATTEMPTS = 2
CHUNK_CALLS = 512


def split_cell(cell_key: str) -> tuple[str, str]:
    model_id, task_id = cell_key.rsplit("::", 1)
    if cell_key not in CELLS:
        raise ValueError(f"{cell_key} is not a B5 cell")
    return model_id, task_id


def model_key(model_id: str) -> str:
    return model_id.replace("/", "--")


# --- nesting -----------------------------------------------------------------


def canonical_permutation(cell_key: str, assignment: int, n_items: int = N_ITEMS) -> np.ndarray:
    """One fixed permutation per assignment; every size reads the same order.

    Every size-specific grouping is a contiguous slice of this permutation, so a
    size-2k group is exactly the union of two adjacent size-k groups.
    """
    if not 1 <= assignment <= N_ASSIGNMENTS:
        raise ValueError("assignment must be 1")
    split_cell(cell_key)
    rng = np.random.default_rng(derived_seed(ASSIGN_NAMESPACE, cell_key, "permutation", assignment))
    return cast(np.ndarray, rng.permutation(n_items))


def group_positions(permutation: np.ndarray, k: int) -> list[tuple[int, ...]]:
    """Contiguous slices of the canonical permutation, in permutation order."""
    if k not in SIZES:
        raise ValueError(f"the frozen B5 sizes are {SIZES}")
    if permutation.size % k:
        raise ValueError("the frame must divide evenly into groups")
    return [
        tuple(int(value) for value in permutation[start : start + k])
        for start in range(0, permutation.size, k)
    ]


def nesting_report(cell_key: str, n_items: int = N_ITEMS) -> dict[str, Any]:
    """Verify the nesting rule for every assignment and every adjacent size pair."""
    checked = 0
    for assignment in range(1, N_ASSIGNMENTS + 1):
        permutation = canonical_permutation(cell_key, assignment, n_items)
        smaller = group_positions(permutation, SIZES[0])
        for k in SIZES[1:]:
            larger = group_positions(permutation, k)
            for index, group in enumerate(larger):
                if group != tuple(
                    value for sub in smaller[16 * index : 16 * index + 16] for value in sub
                ):
                    raise ValueError(f"{cell_key} size {k} group {index} is not nested")
                checked += 1
            smaller = larger
        seen = sorted(value for group in smaller for value in group)
        if seen != list(range(n_items)):
            raise ValueError(f"{cell_key} assignment {assignment} does not cover the frame once")
    return {
        "cell_key": cell_key,
        "rule": (
            "one canonical permutation; every size-32 group is the union of "
            "16 adjacent size-2 groups"
        ),
        "nested_pairs_checked": checked,
        "sizes": list(SIZES),
        "assignments": N_ASSIGNMENTS,
    }


# --- output contract ---------------------------------------------------------


def label_schema(task_id: str, target_ids: Sequence[str]) -> dict[str, Any]:
    """Require exactly the requested IDs and only that task's legal integer labels."""
    if task_id not in LABEL_VALUES:
        raise ValueError(f"unsupported B5 task: {task_id}")
    ids = list(target_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("B5 target IDs must be nonempty and unique")
    label = {"type": "integer", "enum": list(LABEL_VALUES[task_id])}
    return {
        "$schema_contract": CONTRACT_VERSION,
        "type": "object",
        "properties": {
            "labels": {
                "type": "object",
                "properties": dict.fromkeys(ids, label),
                "required": ids,
                "additionalProperties": False,
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def max_output_tokens(task_id: str, target_ids: Sequence[str]) -> int:
    """Serialized bound on the longest valid response, plus a fixed allowance.

    Structured decoding runs with `disable_any_whitespace`, so the longest legal
    response is the compact object carrying every requested ID and the widest legal
    label. One token never covers less than one UTF-8 byte.
    """
    if task_id not in LABEL_VALUES:
        raise ValueError(f"unsupported B5 task: {task_id}")
    ids = list(target_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("B5 target IDs must be nonempty and unique")
    widest = max(LABEL_VALUES[task_id])
    body = ",".join(f"{json.dumps(doc_id)}:{widest}" for doc_id in ids)
    payload = '{"labels":{' + body + "}}"
    return len(payload.encode("utf-8")) + OUTPUT_TOKEN_ALLOWANCE


# --- schedule ----------------------------------------------------------------


def group_tasks(cell_key: str, doc_ids: Sequence[str]) -> Iterator[dict[str, Any]]:
    """Every dispatch unit of one cell, in model-major then size-major order.

    A group is the unit so that a joint call and its exact-context calls always reach
    the engine together: they show the same documents in the same order.
    """
    model_id, task_id = split_cell(cell_key)
    ids = list(doc_ids)
    if len(ids) != N_ITEMS or len(set(ids)) != N_ITEMS:
        raise ValueError(f"{cell_key} needs {N_ITEMS} unique frame IDs")
    stem = f"{CALL_PREFIX}::{model_key(model_id)}::{task_id}"
    index = 0
    for k in SIZES:
        for assignment in range(1, N_ASSIGNMENTS + 1):
            permutation = canonical_permutation(cell_key, assignment, N_ITEMS)
            for group_index, positions in enumerate(group_positions(permutation, k)):
                order = [ids[position] for position in positions]
                yield {
                    "schema_version": SCHEMA_VERSION,
                    "group_id": f"{stem}::k{k:02d}::a{assignment}::g{group_index:05d}",
                    "cell_key": cell_key,
                    "model_id": model_id,
                    "task_id": task_id,
                    "batch_size": k,
                    "assignment": assignment,
                    "group_index": group_index,
                    "execution_index": index,
                    "execution_order_key": f"{k:02d}:{assignment}:{group_index:05d}",
                    "doc_order": order,
                    "doc_order_sha256": sha256_json(order),
                }
                index += 1


def group_calls(task: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The joint call and then one exact-context call per target, in document order."""
    order = [str(value) for value in cast(list[str], task["doc_order"])]
    task_id = str(task["task_id"])
    common = {
        key: task[key]
        for key in (
            "schema_version",
            "group_id",
            "cell_key",
            "model_id",
            "task_id",
            "batch_size",
            "assignment",
            "group_index",
            "execution_index",
            "execution_order_key",
            "doc_order",
            "doc_order_sha256",
        )
    }
    for rank, (condition, targets) in enumerate(
        [("joint", order)] + [("exact_context", [doc_id]) for doc_id in order]
    ):
        schema = label_schema(task_id, targets)
        suffix = "joint" if condition == "joint" else f"exact::{targets[0]}"
        yield {
            **common,
            "call_id": f"{task['group_id']}::{suffix}",
            "condition": condition,
            "call_rank": rank,
            "call_index": int(task["group_index"]),
            # One serial A100 lane runs the whole schedule, so there is no interleaving
            # block to rotate; the field stays for the shared event builder.
            "execution_block": 0,
            "n_targets": len(targets),
            "target_ids": targets,
            "target_ids_sha256": sha256_json(targets),
            "json_schema": schema,
            "json_schema_sha256": sha256_json(schema),
            "max_tokens": max_output_tokens(task_id, targets),
            "attempt": 1,
            "attempt_id": f"{task['group_id']}::{suffix}::attempt::1",
        }


def cell_calls(cell_key: str, doc_ids: Sequence[str]) -> Iterator[dict[str, Any]]:
    for task in group_tasks(cell_key, doc_ids):
        yield from group_calls(task)


def expected_counts() -> dict[str, int]:
    joint = N_ASSIGNMENTS * sum(N_ITEMS // k for k in SIZES)
    exact = N_ASSIGNMENTS * N_ITEMS * len(SIZES)
    return {
        "joint_per_cell": joint,
        "exact_context_per_cell": exact,
        "total_per_cell": joint + exact,
        "cells": len(CELLS),
        "joint": joint * len(CELLS),
        "exact_context": exact * len(CELLS),
        "total": (joint + exact) * len(CELLS),
    }


def verify_counts(counts: dict[str, int]) -> None:
    """Fail closed when the generated schedule differs from the frozen totals."""
    expected = {
        "joint": EXPECTED_JOINT,
        "exact_context": EXPECTED_EXACT,
        "total": EXPECTED_TOTAL,
    }
    for key, value in expected.items():
        if counts.get(key) != value:
            raise ValueError(f"B5 schedule {key} is {counts.get(key)}, expected {value}")


def render_messages(
    task_id: str, texts: dict[str, str], call: dict[str, Any]
) -> list[dict[str, str]]:
    order = [str(value) for value in cast(list[str], call["doc_order"])]
    documents = [(doc_id, texts[doc_id]) for doc_id in order]
    return build_messages(task_id, documents, [str(v) for v in cast(list[str], call["target_ids"])])


def cell_digest(cell_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze one cell by the ordered call stream, including every prompt hash."""
    doc_ids = [str(row["doc_id"]) for row in rows]
    texts = {str(row["doc_id"]): str(row["text"]) for row in rows}
    task_id = split_cell(cell_key)[1]
    counts = {condition: 0 for condition in CONDITIONS}
    call_ids: list[str] = []
    prompt_hashes: list[str] = []
    records: list[str] = []
    by_size: dict[int, dict[str, int]] = {k: dict.fromkeys(CONDITIONS, 0) for k in SIZES}
    by_assignment: dict[int, int] = dict.fromkeys(range(1, N_ASSIGNMENTS + 1), 0)
    for call in cell_calls(cell_key, doc_ids):
        condition = str(call["condition"])
        counts[condition] += 1
        by_size[int(call["batch_size"])][condition] += 1
        by_assignment[int(call["assignment"])] += 1
        call_ids.append(str(call["call_id"]))
        prompt_hashes.append(sha256_json(render_messages(task_id, texts, call)))
        records.append(
            sha256_json(
                {
                    "call_id": call["call_id"],
                    "condition": condition,
                    "batch_size": call["batch_size"],
                    "assignment": call["assignment"],
                    "group_index": call["group_index"],
                    "execution_order_key": call["execution_order_key"],
                    "doc_order_sha256": call["doc_order_sha256"],
                    "target_ids_sha256": call["target_ids_sha256"],
                    "json_schema_sha256": call["json_schema_sha256"],
                    "max_tokens": call["max_tokens"],
                    "prompt_sha256": prompt_hashes[-1],
                }
            )
        )
    per_cell = expected_counts()
    if (
        counts["joint"] != per_cell["joint_per_cell"]
        or counts["exact_context"] != per_cell["exact_context_per_cell"]
        or len(set(call_ids)) != per_cell["total_per_cell"]
    ):
        raise ValueError(f"{cell_key} call stream differs from the frozen design")
    return {
        "cell_key": cell_key,
        "counts": {**counts, "total": len(call_ids)},
        "counts_by_size": {str(k): by_size[k] for k in SIZES},
        "calls_by_assignment": {str(a): by_assignment[a] for a in by_assignment},
        "call_id_order_sha256": sha256_json(call_ids),
        "prompt_sha256_order_sha256": sha256_json(prompt_hashes),
        "record_stream_sha256": sha256_json(records),
        "first_call_ids": call_ids[:3],
        "last_call_ids": call_ids[-3:],
    }


def chunks(
    cell_key: str, doc_ids: Sequence[str], chunk_calls: int = CHUNK_CALLS
) -> Iterator[dict[str, Any]]:
    """Consecutive groups, never split, bounded by a call budget.

    The smoke schedule and the full schedule both read this generator; a smoke run
    selects whole chunks and changes nothing else.
    """
    if chunk_calls < max(SIZES) + 1:
        raise ValueError("a chunk must hold at least one complete largest group")
    batch: list[dict[str, Any]] = []
    size = 0
    index = 0

    def emit(rows: list[dict[str, Any]], position: int) -> dict[str, Any]:
        call_ids = [str(call["call_id"]) for task in rows for call in group_calls(task)]
        return {
            "chunk_index": position,
            "cell_key": cell_key,
            "batch_size": int(rows[0]["batch_size"]),
            "assignment": int(rows[0]["assignment"]),
            "chunk_id": sha256_json(call_ids),
            "groups": rows,
            "calls": len(call_ids),
        }

    for task in group_tasks(cell_key, doc_ids):
        cost = int(task["batch_size"]) + 1
        same = not batch or (
            batch[0]["batch_size"] == task["batch_size"]
            and batch[0]["assignment"] == task["assignment"]
        )
        if batch and (size + cost > chunk_calls or not same):
            yield emit(batch, index)
            index += 1
            batch, size = [], 0
        batch.append(task)
        size += cost
    if batch:
        yield emit(batch, index)


# --- context audit -----------------------------------------------------------


def context_audit(
    cell_key: str, doc_ids: Sequence[str], evidence: dict[str, Any]
) -> dict[str, Any]:
    """Bound every scheduled prompt with measured pinned-tokenizer counts.

    `solo_prompt_tokens` are the sealed B3 single-document prompt lengths measured by
    each model's pinned tokenizer. Writing `solo_j = S + t_j` for a per-model constant
    `S >= 0` and a per-document cost `t_j`, an exact-context prompt over group `G` is
    `E + sum t_j = K + (16 - k) S + sum solo_j`, where `K` is the measured
    `exact - sum solo` offset. `S >= 0` makes `K_max + sum solo_j` an upper bound at
    every size. A joint prompt is `A (1 - k) + sum solo_j` with `A > 0`, so `sum solo_j`
    bounds it as well.
    """
    task_id = split_cell(cell_key)[1]
    ids = list(doc_ids)
    solo = {
        str(key): int(value)
        for key, value in cast(dict[str, Any], evidence["solo_prompt_tokens"]).items()
    }
    if set(solo) != set(ids):
        raise ValueError(f"{cell_key} token evidence does not cover the frame")
    offset = int(evidence["exact_offset_max"])
    worst: dict[str, Any] = {"required_tokens": -1}
    rows: list[dict[str, Any]] = []
    for k in SIZES:
        peak = 0
        peak_group = ""
        for assignment in range(1, N_ASSIGNMENTS + 1):
            permutation = canonical_permutation(cell_key, assignment, len(ids))
            for group_index, positions in enumerate(group_positions(permutation, k)):
                order = [ids[position] for position in positions]
                total = sum(solo[doc_id] for doc_id in order)
                required = max(
                    offset + total + max_output_tokens(task_id, [order[0]]),
                    total + max_output_tokens(task_id, order),
                )
                if required > peak:
                    peak = required
                    peak_group = f"k{k:02d}::a{assignment}::g{group_index:05d}"
        rows.append(
            {
                "batch_size": k,
                "max_required_tokens": peak,
                "max_required_group": peak_group,
                "headroom_tokens": MAX_MODEL_LEN - peak,
                "fits": peak <= MAX_MODEL_LEN,
            }
        )
        if peak > int(worst["required_tokens"]):
            worst = {"required_tokens": peak, "batch_size": k, "group": peak_group}
    return {
        "cell_key": cell_key,
        "max_model_len": MAX_MODEL_LEN,
        "method": "measured pinned-tokenizer solo prompt lengths, upper-bounded per size",
        "by_size": rows,
        "worst_case": worst,
        "fits": all(bool(row["fits"]) for row in rows),
    }


# --- runtime and cost forecast ----------------------------------------------


def runtime_coefficients(anchors: list[dict[str, Any]]) -> dict[str, float]:
    """Least squares on the completed B4 A100 PCIe lanes: prefill, decode, dispatch.

    Each anchor is one completed B4 cell: its measured metered hours, its call count,
    and its measured mean prompt and output tokens. The three coefficients are seconds
    per uncached prompt token, seconds per generated token, and seconds per dispatched
    chunk of the B4 chunk size.
    """
    if len(anchors) < 3:
        raise ValueError("the runtime fit needs at least three completed lanes")
    design = np.array(
        [
            [
                float(row["mean_prompt_tokens"]) * float(row["calls"]),
                float(row["mean_output_tokens"]) * float(row["calls"]),
                float(row["calls"]) / float(row["chunk_calls"]),
            ]
            for row in anchors
        ]
    )
    observed = np.array([float(row["metered_hours"]) * 3600.0 for row in anchors])
    solution = np.linalg.lstsq(design, observed, rcond=None)[0]
    prefill, decode, dispatch = (float(value) for value in solution)
    if min(prefill, decode, dispatch) <= 0:
        raise ValueError("the runtime fit produced a non-positive coefficient")
    residual = design @ solution - observed
    return {
        "seconds_per_prompt_token": prefill,
        "seconds_per_output_token": decode,
        "seconds_per_chunk": dispatch,
        "max_absolute_relative_error": float(np.max(np.abs(residual / observed))),
    }


def block_workload(evidence: dict[str, Any], k: int, *, prefix_reuse: bool) -> dict[str, float]:
    """Prompt and generated tokens of one cell at one size, under a stated reuse rule.

    Measured per-cell constants: `A` the joint template, `E` the exact-context
    template, `x` one serialized target, `c` the joint response frame, `o1` one
    exact-context response, `T` the total document cost of the frame.

    With `prefix_reuse` the k exact-context calls of a group follow the group's joint
    call in one dispatched chunk, so the group's documents are prefilled once for the
    exact-context arm; without it every call prefills its own documents.
    """
    if k not in SIZES:
        raise ValueError(f"the frozen B5 sizes are {SIZES}")
    a_joint = float(evidence["joint_template_tokens"])
    e_exact = float(evidence["exact_template_tokens"])
    per_target = float(evidence["target_tokens"])
    frame_tokens = float(evidence["frame_document_tokens"])
    joint_frame = float(evidence["joint_response_frame_tokens"])
    exact_response = float(evidence["exact_response_tokens"])
    exact_calls = N_ASSIGNMENTS * N_ITEMS
    joint_calls = N_ASSIGNMENTS * N_ITEMS // k
    if prefix_reuse:
        prompt = N_ASSIGNMENTS * (frame_tokens + N_ITEMS * e_exact)
    else:
        prompt = exact_calls * (e_exact + k * frame_tokens / N_ITEMS)
    prompt += joint_calls * (a_joint + k * per_target + k * frame_tokens / N_ITEMS)
    return {
        "prompt_tokens": prompt,
        "output_tokens": exact_calls * exact_response
        + joint_calls * (joint_frame + k * per_target),
        "calls": float(exact_calls + joint_calls),
    }


def cell_workload(evidence: dict[str, Any], *, prefix_reuse: bool) -> dict[str, float]:
    rows = [block_workload(evidence, k, prefix_reuse=prefix_reuse) for k in SIZES]
    return {key: float(sum(row[key] for row in rows)) for key in rows[0]}


def forecast(
    token_evidence: dict[str, dict[str, Any]],
    runtime_evidence: dict[str, Any],
    *,
    chunk_calls: int = CHUNK_CALLS,
    usd_per_hour: float,
    boot_hours_per_model: float,
    retry_fraction: float,
) -> dict[str, Any]:
    """Metered A100 hours, wall time and cost under both prefix-reuse scenarios."""
    if not 0.0 <= retry_fraction < 1.0 or boot_hours_per_model < 0 or usd_per_hour <= 0:
        raise ValueError("invalid forecast allowance")
    coefficients = runtime_coefficients(cast(list[dict[str, Any]], runtime_evidence["b4_anchors"]))

    def hours(work: dict[str, float]) -> float:
        return (
            coefficients["seconds_per_prompt_token"] * work["prompt_tokens"]
            + coefficients["seconds_per_output_token"] * work["output_tokens"]
            + coefficients["seconds_per_chunk"] * work["calls"] / chunk_calls
        ) / 3600.0

    scenarios: dict[str, Any] = {}
    for name, reuse in (("prefix_reuse", True), ("no_prefix_reuse", False)):
        cells: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        for cell_key in CELLS:
            evidence = token_evidence[cell_key]
            for k in SIZES:
                block = block_workload(evidence, k, prefix_reuse=reuse)
                blocks.append(
                    {
                        "cell_key": cell_key,
                        "batch_size": k,
                        "calls": int(block["calls"]),
                        "generation_hours": hours(block),
                    }
                )
            work = cell_workload(evidence, prefix_reuse=reuse)
            generation = hours(work)
            cells.append(
                {
                    "cell_key": cell_key,
                    "calls": int(work["calls"]),
                    "prompt_tokens": work["prompt_tokens"],
                    "output_tokens": work["output_tokens"],
                    "generation_hours": generation,
                    "retry_hours": generation * retry_fraction,
                    "boot_hours": boot_hours_per_model,
                    "metered_hours": generation * (1.0 + retry_fraction) + boot_hours_per_model,
                }
            )
        total = float(sum(float(row["metered_hours"]) for row in cells))
        scenarios[name] = {
            "cells": cells,
            "blocks": blocks,
            "metered_a100_hours": total,
            "wall_hours_one_a100": total,
            "estimated_usd": total * usd_per_hour,
        }
    point = float(scenarios["prefix_reuse"]["metered_a100_hours"])
    bound = float(scenarios["no_prefix_reuse"]["metered_a100_hours"])
    return {
        "schema_version": "b5-forecast-v1",
        "chunk_calls": chunk_calls,
        "usd_per_hour": usd_per_hour,
        "boot_hours_per_model": boot_hours_per_model,
        "retry_fraction": retry_fraction,
        "coefficients": coefficients,
        "scenarios": scenarios,
        "point_metered_a100_hours": point,
        "upper_metered_a100_hours": bound,
        "point_estimated_usd": point * usd_per_hour,
        "upper_estimated_usd": bound * usd_per_hour,
        "assumptions": [
            "coefficients come from the four completed B4 A100 PCIe 80GB lanes only",
            "token counts come from the sealed B3 ledgers of the same four cells",
            (
                "the point estimate assumes the engine reuses a group's document prefix "
                "across that group's exact-context calls, which the chunk order enforces"
            ),
            (
                "the upper bound assumes no prefix reuse at all and exceeds the ceiling, so "
                "the run carries a live meter and a projection gate rather than a multiplier"
            ),
            "one A100 PCIe 80GB runs at a time, so wall hours equal metered hours",
        ],
    }


def required_balance_usd(
    forecast_record: dict[str, Any], ceiling_hours: float, reserve_usd: float
) -> float:
    """Fund the complete estimate under the ceiling, plus an explicit reserve."""
    if ceiling_hours <= 0 or reserve_usd < 0:
        raise ValueError("invalid balance parameters")
    hours = min(float(forecast_record["point_metered_a100_hours"]), ceiling_hours)
    return hours * float(forecast_record["usd_per_hour"]) + reserve_usd


def safe_float(value: float) -> float | None:
    return None if not math.isfinite(value) else float(value)
