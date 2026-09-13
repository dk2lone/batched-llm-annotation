"""Cross-family extension (X1R4): candidates, frames, schedules, budget, and claim gates.

This module is a separate wrapper over the Freeze B primitives. Freeze B module
defaults are never changed, so every sealed Freeze B artifact still reproduces.
The extension is not part of the sealed Freeze B claim family. Qwen and Granite
joint results are already open, so this panel is disclosed as a new extension.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import binomtest

from step0 import stats
from step0.hashing import canonical_json, read_jsonl, sha256_json, sha256_text
from step0.stats import holm

from . import frames as fb_frames
from . import repeated as fb_repeated
from .assign import Frame, build_calls
from .claims import CellData, cell_delta
from .frames import N_CONFIRM, N_FILLER, N_SCREEN, Task
from .models import Model
from .prompts import build_messages, estimated_response_tokens
from .runtime_benchmark import RuntimePolicy
from .selection import Cell, balanced_order

SCHEMA_VERSION = "ext-x1r4-v1"
SUPERSEDES = "study-120h/freeze/20260913T182841Z-X1R3"
CALL_PREFIX = "extx1"
M1_SCHEMA = "ext-x1r2-m1-call-order-v1"
B3_SCHEMA = "ext-x1r2-b3-plan-v1"
B4_SCHEMA = "ext-x1r2-b4-schedule-v1"
E5_ANALYSIS_SCHEMA = "ext-x1r2-e5-analysis-v1"

# --- Frozen identity -------------------------------------------------------

EXT_SEED = 20260913
ASSIGN_NAMESPACE = "bigdata2026-ext-v1"
E5_NAMESPACE = "bigdata2026-ext-e5-v1"
POWER_NAMESPACE = "ext-power-v1"
OPENED_JOINT_FAMILIES = ("granite", "qwen")

EXT_MODELS = (
    Model("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "qwen", 1),
    Model(
        "ibm-granite/granite-3.3-8b-instruct",
        "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "granite",
        2,
    ),
    Model(
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
        "llama",
        3,
    ),
    Model(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "mistral",
        4,
    ),
    Model("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083", "phi", 5),
)

# Extension IMDb reads the train split. The test split cannot supply a frame that
# is disjoint from Freeze B: 23,310 eligible rows less 14,336 already used leaves
# 8,974, below the 14,336 this extension needs. The train split is label sorted,
# so scan_limit must cover all 25,000 rows or the pool is negative reviews only.
EXT_TASKS = (
    Task(
        "civil_comments",
        "google/civil_comments",
        "f2970eb3a55777454c94069077cc8d9b5866312d",
        "validation",
        97320,
        40,
        1000,
        "cc",
    ),
    Task(
        "multi_nli",
        "nyu-mll/multi_nli",
        "da70db2af9d09693783c3320c4249840212ee221",
        "train",
        392702,
        40,
        1000,
        "mn",
    ),
    Task(
        "imdb",
        "stanfordnlp/imdb",
        "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
        "train",
        25000,
        100,
        3000,
        "im",
    ),
    Task(
        "ag_news",
        "fancyzhx/ag_news",
        "eb185aade064a813bc0b7f42de02595523103ca4",
        "train",
        120000,
        100,
        1000,
        "ag",
    ),
)
EXT_TASK_KEYS = tuple(task.key for task in EXT_TASKS)
EXT_MODEL_IDS = tuple(model.repo for model in EXT_MODELS)

FREEZE_B1 = "study-120h/freeze/20260912T222618Z-B1"
FRAME_ROLES = ("screen", "confirm", "filler")

# --- Frozen budget ---------------------------------------------------------

METER_ORIGIN_HOURS = 0.0
STOP_HOURS = 80.0
RESERVE_HOURS = 10.0
FORECAST_MULTIPLIER = 1.6

# --- Frozen statistics -----------------------------------------------------

MAX_PRIMARY_CELLS = 20
N_DRAWS = 4_999
ALPHA = 0.05
PRACTICAL_FLOOR = 0.002
PLANNING_ALPHA = ALPHA / MAX_PRIMARY_CELLS
MIN_POSITIVE_FAMILIES = 3
E5_N_REFERENCE = 250
E5_EVALUATION_PREFIXES = (400, 1000, 2500)
E5_MIN_EVALUATION = 400
E5_MAX_COVERAGE = 0.90
E5_MIN_SUPPORTED_CELLS = 2
BATCH_SIZE = 16
N_ASSIGNMENTS = 3


def ext_runtime_policy(model: Model) -> RuntimePolicy:
    """Immutable serving policy. Every candidate is attempted in M1.

    `runnable` is True for every candidate on purpose. Access is not a property of
    the frozen protocol: it is an observation that can change without a code change.
    A checkpoint leaves the panel only through a signed runtime and access lock, and
    the M1 order always lists all five.
    """
    return RuntimePolicy(
        model_id=model.repo,
        revision=model.revision,
        family=model.family,
        max_model_len=32768,
        enable_thinking=False,
        runnable=True,
        exclusion_reason=None,
    )


EXT_MODEL_POLICIES = tuple(ext_runtime_policy(model) for model in EXT_MODELS)
ACCESS_STATUSES = ("verified", "gated_pending_access", "technically_unavailable")
CANDIDATE_STATUSES = ("attempted", "technically_unavailable")


def ext_policy_for(model_id: str) -> RuntimePolicy:
    try:
        return next(p for p in EXT_MODEL_POLICIES if p.model_id == model_id)
    except StopIteration as exc:
        raise ValueError(f"unknown extension model {model_id}") from exc


@dataclass(frozen=True)
class AccessRecord:
    """A mutable observation about one candidate, kept outside the frozen policy."""

    model_id: str
    access_verified_now: bool
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.model_id not in EXT_MODEL_IDS:
            raise ValueError(f"{self.model_id} is not an extension candidate")
        if self.status not in ACCESS_STATUSES:
            raise ValueError(f"unknown access status {self.status}")
        if self.access_verified_now != (self.status == "verified"):
            raise ValueError("access_verified_now must agree with the status")


def runtime_access_lock(
    records: list[AccessRecord],
    *,
    signed_by: str,
    runtime_lock_sha256: str,
    batch_invariance_report_sha256: str,
) -> dict[str, Any]:
    """The signed gate that decides which candidates M1 actually executes.

    This is produced after the synthetic runtime lock, never inside the protocol
    seal. Changing it needs a new signed lock, not an edit to this module.
    """
    if {r.model_id for r in records} != set(EXT_MODEL_IDS):
        raise ValueError("the access lock must decide every one of the five candidates")
    if len(records) != len(EXT_MODEL_IDS):
        raise ValueError("duplicate access record")
    if not signed_by or len(runtime_lock_sha256) != 64 or len(batch_invariance_report_sha256) != 64:
        raise ValueError(
            "a signed access lock needs a signer, runtime-lock digest, and "
            "batch-invariance report digest"
        )
    by_id = {r.model_id: r for r in records}
    candidates = [
        {
            "model_id": model_id,
            "access_verified_now": by_id[model_id].access_verified_now,
            "access_status": by_id[model_id].status,
            "detail": by_id[model_id].detail,
            "candidate_status": (
                "attempted" if by_id[model_id].access_verified_now else "technically_unavailable"
            ),
        }
        for model_id in EXT_MODEL_IDS
    ]
    attempted = [c["model_id"] for c in candidates if c["candidate_status"] == "attempted"]
    return {
        "schema_version": "ext-x1r4-runtime-access-lock-v1",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "signed_by": signed_by,
        "runtime_lock_sha256": runtime_lock_sha256,
        "batch_invariance_report_sha256": batch_invariance_report_sha256,
        "candidates": candidates,
        "attempted_models": attempted,
        "unavailable_models": [
            c["model_id"] for c in candidates if c["candidate_status"] != "attempted"
        ],
        "n_attempted": len(attempted),
        "rule": (
            "a candidate is never silently omitted; every one of the five carries a "
            "terminal attempted or technically_unavailable status"
        ),
    }


# --- Power table ------------------------------------------------------------

POWER_REPLICATE_NAMESPACES = (
    POWER_NAMESPACE,
    f"{POWER_NAMESPACE}-b",
    f"{POWER_NAMESPACE}-c",
)
POWER_SIMS_PER_REPLICATE = 2000


def ext_power_table(
    simulate: Callable[..., dict[str, Any]],
    error_grid: Iterable[float],
    sizes: Iterable[int],
) -> list[dict[str, Any]]:
    """Pool independent simulation batches so the item count cannot flip on noise.

    One 2,000-simulation batch leaves the 0.35 grid rate on the 0.80 power target,
    where the selected item count moved between 4,096 and 8,192 with the RNG
    namespace alone. Three independent batches are pooled into one 6,000-simulation
    rate, and the Wilson lower bound is taken on the pooled count. The batches are
    independent, so this is variance reduction, not a repeated test.
    """
    rows: list[dict[str, Any]] = []
    for p_error in error_grid:
        for n_items in sizes:
            batches = [
                simulate(
                    p_error,
                    n_items,
                    sims=POWER_SIMS_PER_REPLICATE,
                    alpha=PLANNING_ALPHA,
                    namespace=namespace,
                )
                for namespace in POWER_REPLICATE_NAMESPACES
            ]
            powers = [float(b["power"]) for b in batches]
            pooled = float(np.mean(powers))
            rows.append(
                {
                    "p": float(p_error),
                    "n": float(n_items),
                    "alpha": PLANNING_ALPHA,
                    "sims": float(POWER_SIMS_PER_REPLICATE * len(batches)),
                    "power": pooled,
                    "power_batches": powers,
                    "power_batch_min": min(powers),
                    "power_batch_max": max(powers),
                    "power_batch_spread": max(powers) - min(powers),
                    "namespaces": list(POWER_REPLICATE_NAMESPACES),
                    "delta": float(batches[0]["delta"]),
                    "critical": float(np.mean([b["critical"] for b in batches])),
                    "null_sd": float(np.mean([b["null_sd"] for b in batches])),
                    "pooling_rule": (
                        "three independent batches pooled into one rate; the Wilson lower "
                        "bound is taken on the pooled simulation count"
                    ),
                }
            )
    return rows


# --- Frames ----------------------------------------------------------------


def freeze_b_exclusions(root: Path, task_key: str) -> set[str]:
    """Every document ID already materialized in the Freeze B part-1 seal."""
    seal = root / FREEZE_B1 / "frame-identifiers" / task_key
    if not seal.is_dir():
        raise FileNotFoundError(f"the Freeze B part-1 identifier seal is missing: {seal}")
    out: set[str] = set()
    for role in FRAME_ROLES:
        out.update(str(row["doc_id"]) for row in read_jsonl(seal / f"{role}.jsonl"))
    return out


def ext_exclusions(root: Path, task: Task) -> dict[str, set[str]]:
    """Gate A, sealed Step 0, and Freeze B identifiers, reported by source."""
    return {
        "historical": fb_frames.historical_exclusions(root, task),
        "freeze_b1": freeze_b_exclusions(root, task.key),
    }


def build_ext_task(root: Path, task: Task, out_dir: Path) -> dict[str, Any]:
    """Materialize one extension task frame; existing frames are never replaced."""
    if task.key not in EXT_TASK_KEYS:
        raise ValueError(f"{task.key} is not an extension task")
    sources = ext_exclusions(root, task)
    manifest = fb_frames.build_task(
        root,
        task,
        out_dir,
        seed=EXT_SEED,
        extra_excluded=sources["freeze_b1"],
    )
    manifest["extension"] = {
        "schema_version": SCHEMA_VERSION,
        "excluded_historical_ids": len(sources["historical"]),
        "excluded_freeze_b1_ids": len(sources["freeze_b1"]),
        "excluded_total_ids": len(sources["historical"] | sources["freeze_b1"]),
        "freeze_b1_seal": FREEZE_B1,
        "disjoint_from": ["gate_a", "step0_20260912T041633Z", "freeze_b_20260912T222618Z-B1"],
    }
    return manifest


# --- Deterministic run order ----------------------------------------------


def model_key(model_id: str) -> str:
    return model_id.replace("/", "--")


def stream_digest(records: Iterable[dict[str, Any]], *, keep: int = 3) -> dict[str, Any]:
    """Rolling SHA256 over canonical JSON records, with first and last samples.

    The digest is computed without holding the schedule in memory, so a
    1.7-million-call plan can be frozen by its hash and audit records alone.
    """
    if keep < 1:
        raise ValueError("keep at least one audit record")
    digest = hashlib.sha256()
    first: list[dict[str, Any]] = []
    last: list[dict[str, Any]] = []
    count = 0
    for record in records:
        line = canonical_json(record)
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
        count += 1
        if len(first) < keep:
            first.append(record)
        last.append(record)
        if len(last) > keep:
            last.pop(0)
    return {
        "count": count,
        "sha256": digest.hexdigest(),
        "first_records": first,
        "last_records": last,
    }


# --- M1: the exact 40,960 solo call order ----------------------------------


def m1_cell_calls(
    model_id: str, task_id: str, rows: list[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Every individual screen call for one cell, in the exact frozen order.

    The record carries the call ID, the document ID, and the prompt hash. It never
    carries the document text, so the seal holds no dataset text.
    """
    if model_id not in EXT_MODEL_IDS or task_id not in EXT_TASK_KEYS:
        raise ValueError(f"{model_id}::{task_id} is not an extension cell")
    if len(rows) != N_SCREEN:
        raise ValueError(f"a screen frame holds exactly {N_SCREEN} rows")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        doc_id = str(row["doc_id"])
        if doc_id in seen:
            raise ValueError(f"duplicate screen row {doc_id}")
        seen.add(doc_id)
        n_tokens = int(row["n_tokens"] or len(str(row["text"]).split()))
        messages = build_messages(task_id, [(doc_id, str(row["text"]))], [doc_id])
        yield {
            "schema_version": M1_SCHEMA,
            "position": index + 1,
            "call_id": (
                f"{CALL_PREFIX}::m1::{model_key(model_id)}::{task_id}::{index:04d}::{doc_id}"
            ),
            "model_id": model_id,
            "task_id": task_id,
            "condition": "solo",
            "doc_id": doc_id,
            "prompt_sha256": sha256_json(messages),
            "max_tokens": estimated_response_tokens(task_id, 1, [n_tokens]),
        }


def m1_cell_digest(model_id: str, task_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze one cell by its call-ID order digest and its prompt-hash order digest."""
    calls = list(m1_cell_calls(model_id, task_id, rows))
    call_ids = [c["call_id"] for c in calls]
    prompts = [c["prompt_sha256"] for c in calls]
    if len(set(call_ids)) != len(call_ids):
        raise ValueError("duplicate M1 call ID")
    return {
        "cell_key": f"{model_id}::{task_id}",
        "model_id": model_id,
        "task_id": task_id,
        "n_calls": len(calls),
        "call_id_order_sha256": sha256_json(call_ids),
        "prompt_sha256_order_sha256": sha256_json(prompts),
        "record_stream": stream_digest(calls),
        "distinct_prompt_hashes": len(set(prompts)),
    }


def m1_call_order(frames_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The complete frozen M1 schedule: 5 candidates x 4 tasks x 2,048 calls.

    Order is candidate by candidate in the frozen model order, then task in the
    frozen task order, then the sealed screen-frame row order. All five candidates
    are listed; a signed access lock decides which ones actually execute.
    """
    if set(frames_by_task) != set(EXT_TASK_KEYS):
        raise ValueError("a screen frame is required for every extension task")
    cells: list[dict[str, Any]] = []
    all_call_ids: list[str] = []
    all_prompts: list[str] = []
    for model in EXT_MODELS:
        for task_key in EXT_TASK_KEYS:
            rows = frames_by_task[task_key]
            calls = list(m1_cell_calls(model.repo, task_key, rows))
            digest = m1_cell_digest(model.repo, task_key, rows)
            digest["position"] = len(cells) + 1
            digest["family"] = model.family
            digest["model_order"] = model.order
            digest["task_order"] = EXT_TASK_KEYS.index(task_key) + 1
            cells.append(digest)
            all_call_ids.extend(c["call_id"] for c in calls)
            all_prompts.extend(c["prompt_sha256"] for c in calls)
    if len(set(all_call_ids)) != len(all_call_ids):
        raise ValueError("M1 call IDs are not globally unique")
    expected = len(EXT_MODELS) * len(EXT_TASK_KEYS) * N_SCREEN
    if len(all_call_ids) != expected:
        raise ValueError(f"the M1 schedule must hold exactly {expected} calls")
    return {
        "schema_version": M1_SCHEMA,
        "stage": "M1 blinded solo capability screen",
        "condition": "solo only; one document and one target per call",
        "call_id_prefix": f"{CALL_PREFIX}::m1::",
        "n_cells": len(cells),
        "n_calls": len(all_call_ids),
        "global_call_id_order_sha256": sha256_json(all_call_ids),
        "global_prompt_sha256_order_sha256": sha256_json(all_prompts),
        "global_distinct_prompt_hashes": len(set(all_prompts)),
        "cells": cells,
        "candidates_listed": list(EXT_MODEL_IDS),
        "execution_rule": (
            "all five candidates are frozen here; a signed runtime and access lock "
            "decides which are attempted and which are technically unavailable"
        ),
        "admission_rules": {
            "min_parse_rate": 0.98,
            "error_rate_band": [0.05, 0.80],
            "accuracy_floor": {
                "civil_comments": 0.70,
                "multi_nli": 0.55,
                "imdb": 0.70,
                "ag_news": 0.70,
            },
            "civil_balanced_accuracy": 0.60,
            "civil_toxic_recall": 0.25,
        },
    }


# --- B3: the exact within-cell joint, exact-context, and solo interleaving ---

B3_SIZES = (2048, 4096, 8192)
B3_CONDITIONS = ("joint", "exact_context", "solo")
B3_BLOCKS = 8


def b3_calls(
    model_id: str, task_id: str, rows: list[dict[str, Any]], n_items: int
) -> Iterator[dict[str, Any]]:
    """One complete B3 cell, deterministically interleaved across execution blocks.

    Each block round-robins joint, exact-context, and solo with a block-dependent
    rotation, so no condition is systematically served first. The assignment draw
    uses the extension namespace, so it never reuses a Freeze B permutation.
    """
    if model_id not in EXT_MODEL_IDS or task_id not in EXT_TASK_KEYS:
        raise ValueError(f"{model_id}::{task_id} is not an extension cell")
    if n_items not in B3_SIZES:
        raise ValueError(f"the frozen B3 sizes are {B3_SIZES}")
    if len(rows) < n_items:
        raise ValueError(f"the confirmation frame must hold at least {n_items} rows")
    chosen = rows[:n_items]
    ids = [str(row["doc_id"]) for row in chosen]
    if len(set(ids)) != n_items:
        raise ValueError("B3 item IDs must be unique")
    cell_key = f"{model_id}::{task_id}"
    by_id = {str(row["doc_id"]): row for row in chosen}
    frame = Frame(tuple(ids), tuple(int(row["n_chars"]) for row in chosen))
    stem = f"{CALL_PREFIX}::b3::{model_key(model_id)}::{task_id}::n{n_items}"
    lanes: dict[int, dict[str, list[dict[str, Any]]]] = {
        block: {condition: [] for condition in B3_CONDITIONS} for block in range(B3_BLOCKS)
    }
    for call in build_calls(frame, cell_key, BATCH_SIZE, namespace=ASSIGN_NAMESPACE):
        assignment = int(cast(int, call["assignment"]))
        call_index = int(cast(int, call["call_index"]))
        block = int(cast(int, call["execution_block"]))
        doc_order = [str(doc_id) for doc_id in cast(list[str], call["doc_order"])]
        documents = [(doc_id, str(by_id[doc_id]["text"])) for doc_id in doc_order]
        token_counts = {
            doc_id: int(by_id[doc_id]["n_tokens"] or len(str(by_id[doc_id]["text"]).split()))
            for doc_id in doc_order
        }
        prefix = f"{stem}::k{BATCH_SIZE}::a{assignment}::b{call_index:04d}"
        lanes[block]["joint"].append(
            {
                "call_id": f"{prefix}::joint",
                "condition": "joint",
                "assignment": assignment,
                "call_index": call_index,
                "execution_block": block,
                "batch_size": BATCH_SIZE,
                "n_targets": BATCH_SIZE,
                "doc_order_sha256": sha256_json(doc_order),
                "target_ids_sha256": sha256_json(doc_order),
                "prompt_sha256": sha256_json(build_messages(task_id, documents, doc_order)),
                "max_tokens": estimated_response_tokens(
                    task_id, BATCH_SIZE, list(token_counts.values())
                ),
            }
        )
        for target_id in doc_order:
            lanes[block]["exact_context"].append(
                {
                    "call_id": f"{prefix}::exact::{target_id}",
                    "condition": "exact_context",
                    "assignment": assignment,
                    "call_index": call_index,
                    "execution_block": block,
                    "batch_size": BATCH_SIZE,
                    "n_targets": 1,
                    "doc_order_sha256": sha256_json(doc_order),
                    "target_ids_sha256": sha256_json([target_id]),
                    "prompt_sha256": sha256_json(build_messages(task_id, documents, [target_id])),
                    "max_tokens": estimated_response_tokens(task_id, 1, [token_counts[target_id]]),
                }
            )
    for index, doc_id in enumerate(ids):
        block = index % B3_BLOCKS
        row = by_id[doc_id]
        n_tokens = int(row["n_tokens"] or len(str(row["text"]).split()))
        lanes[block]["solo"].append(
            {
                "call_id": f"{stem}::solo::{index:05d}::{doc_id}",
                "condition": "solo",
                "assignment": None,
                "call_index": index,
                "execution_block": block,
                "batch_size": 1,
                "n_targets": 1,
                "doc_order_sha256": sha256_json([doc_id]),
                "target_ids_sha256": sha256_json([doc_id]),
                "prompt_sha256": sha256_json(
                    build_messages(task_id, [(doc_id, str(row["text"]))], [doc_id])
                ),
                "max_tokens": estimated_response_tokens(task_id, 1, [n_tokens]),
            }
        )
    position = 0
    for block in range(B3_BLOCKS):
        rotation = block % len(B3_CONDITIONS)
        order = B3_CONDITIONS[rotation:] + B3_CONDITIONS[:rotation]
        rounds = max(len(lanes[block][condition]) for condition in B3_CONDITIONS)
        for round_index in range(rounds):
            for rank, condition in enumerate(order):
                bucket = lanes[block][condition]
                if round_index < len(bucket):
                    position += 1
                    yield {
                        "schema_version": B3_SCHEMA,
                        "position": position,
                        "cell_key": cell_key,
                        "n_items": n_items,
                        "execution_order_key": f"{block:02d}:{round_index:05d}:{rank}",
                        **bucket[round_index],
                    }


def b3_expected_counts(n_items: int) -> dict[str, int]:
    if n_items not in B3_SIZES:
        raise ValueError(f"the frozen B3 sizes are {B3_SIZES}")
    joint = n_items // BATCH_SIZE * N_ASSIGNMENTS
    exact = n_items * N_ASSIGNMENTS
    return {
        "joint": joint,
        "exact_context": exact,
        "solo": n_items,
        "total": joint + exact + n_items,
    }


def b3_plan_digest(
    model_id: str, task_id: str, rows: list[dict[str, Any]], n_items: int
) -> dict[str, Any]:
    """Freeze one candidate and size by its interleaved call-ID order digest."""
    counts: dict[str, int] = {condition: 0 for condition in B3_CONDITIONS}
    ids: list[str] = []

    def tracked() -> Iterator[dict[str, Any]]:
        for record in b3_calls(model_id, task_id, rows, n_items):
            counts[str(record["condition"])] += 1
            ids.append(str(record["call_id"]))
            yield record

    stream = stream_digest(tracked())
    expected = b3_expected_counts(n_items)
    if stream["count"] != expected["total"] or len(set(ids)) != expected["total"]:
        raise ValueError("the B3 plan count or call IDs differ from the frozen design")
    for condition in B3_CONDITIONS:
        if counts[condition] != expected[condition]:
            raise ValueError(f"B3 {condition} count differs from the frozen design")
    return {
        "cell_key": f"{model_id}::{task_id}",
        "model_id": model_id,
        "task_id": task_id,
        "n_items": n_items,
        "batch_size": BATCH_SIZE,
        "assignments": N_ASSIGNMENTS,
        "assignment_namespace": ASSIGN_NAMESPACE,
        "call_id_prefix": f"{CALL_PREFIX}::b3::",
        "counts": {**counts, "total": stream["count"]},
        "call_id_order_sha256": sha256_json(ids),
        "record_stream_sha256": stream["sha256"],
        "first_call_ids": [r["call_id"] for r in stream["first_records"]],
        "last_call_ids": [r["call_id"] for r in stream["last_records"]],
        "first_records": stream["first_records"],
        "last_records": stream["last_records"],
    }


# --- B4: the exact reference and evaluation replicate schedules -------------

B4_STREAMS = ("reference", "evaluation")


def b4_calls(
    model_id: str, task_id: str, doc_ids: tuple[str, ...], stream: str, replicates: int
) -> Iterator[dict[str, Any]]:
    """Every fresh size-16 call of one E5 stream, in replicate then call order."""
    if model_id not in EXT_MODEL_IDS or task_id not in EXT_TASK_KEYS:
        raise ValueError(f"{model_id}::{task_id} is not an extension cell")
    if stream not in B4_STREAMS:
        raise ValueError(f"the frozen E5 streams are {B4_STREAMS}")
    if replicates < 1:
        raise ValueError("a stream holds at least one replicate")
    cell_key = f"{model_id}::{task_id}"
    position = 0
    for replicate in range(replicates):
        for call in fb_repeated.replicate_calls(
            doc_ids,
            cell_key,
            stream,
            replicate,
            namespace=E5_NAMESPACE,
            call_id_prefix=f"{CALL_PREFIX}::b4::",
        ):
            position += 1
            yield {
                "schema_version": B4_SCHEMA,
                "position": position,
                "cell_key": cell_key,
                "stream": stream,
                "replicate": replicate,
                "call_index": call["call_index"],
                "call_id": call["call_id"],
                "batch_size": fb_repeated.K,
                "doc_order_sha256": sha256_json(call["doc_order"]),
            }


def b4_schedule_digest(model_id: str, task_id: str, doc_ids: tuple[str, ...]) -> dict[str, Any]:
    """Freeze the reference stream and every frozen evaluation prefix by digest.

    The evaluation prefixes are nested, so the 400-replicate schedule is the exact
    prefix of the 1,000-replicate schedule, which is the prefix of 2,500. Expanding
    a prefix can never change a call that already ran.
    """
    reference_ids: list[str] = []

    def tracked_reference() -> Iterator[dict[str, Any]]:
        for record in b4_calls(model_id, task_id, doc_ids, "reference", E5_N_REFERENCE):
            reference_ids.append(str(record["call_id"]))
            yield record

    ref_stream = stream_digest(tracked_reference())
    evaluation_ids: list[str] = []
    evaluation_doc_orders: list[str] = []
    eval_first: list[dict[str, Any]] = []
    eval_last: list[dict[str, Any]] = []
    largest = max(E5_EVALUATION_PREFIXES)
    for record in b4_calls(model_id, task_id, doc_ids, "evaluation", largest):
        evaluation_ids.append(str(record["call_id"]))
        evaluation_doc_orders.append(str(record["doc_order_sha256"]))
        if len(eval_first) < 3:
            eval_first.append(record)
        eval_last.append(record)
        if len(eval_last) > 3:
            eval_last.pop(0)
    per_replicate = fb_repeated.CALLS_PER_REPLICATE
    prefixes = {}
    for replicates in E5_EVALUATION_PREFIXES:
        cut = replicates * per_replicate
        prefixes[str(replicates)] = {
            "replicates": replicates,
            "n_calls": cut,
            "call_id_order_sha256": sha256_json(evaluation_ids[:cut]),
            "doc_order_sha256_order_sha256": sha256_json(evaluation_doc_orders[:cut]),
            "first_call_id": evaluation_ids[0],
            "last_call_id": evaluation_ids[cut - 1],
        }
    if len(set(reference_ids) & set(evaluation_ids)):
        raise ValueError("the reference and evaluation streams share a call ID")
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("duplicate E5 evaluation call ID")
    return {
        "cell_key": f"{model_id}::{task_id}",
        "model_id": model_id,
        "task_id": task_id,
        "repeated_namespace": E5_NAMESPACE,
        "call_id_prefix": f"{CALL_PREFIX}::b4::",
        "appearances_per_replicate": fb_repeated.N_APPEARANCES,
        "calls_per_replicate": per_replicate,
        "batch_size": fb_repeated.K,
        "reference": {
            "replicates": E5_N_REFERENCE,
            "n_calls": ref_stream["count"],
            "call_id_order_sha256": sha256_json(reference_ids),
            "record_stream_sha256": ref_stream["sha256"],
            "first_call_ids": [r["call_id"] for r in ref_stream["first_records"]],
            "last_call_ids": [r["call_id"] for r in ref_stream["last_records"]],
            "first_records": ref_stream["first_records"],
            "last_records": ref_stream["last_records"],
        },
        "evaluation": {
            "max_replicates": largest,
            "n_calls": len(evaluation_ids),
            "call_id_order_sha256": sha256_json(evaluation_ids),
            "doc_order_sha256_order_sha256": sha256_json(evaluation_doc_orders),
            "prefixes_are_nested": True,
            "prefixes": prefixes,
            "first_records": eval_first,
            "last_records": eval_last,
        },
    }


def admitted_cells(summaries: list[dict[str, Any]]) -> list[Cell]:
    """Build Cell records from solo-only screen summaries."""
    family = {m.repo: m for m in EXT_MODELS}
    out: list[Cell] = []
    for row in summaries:
        if not row.get("admitted"):
            continue
        model = family[str(row["model_id"])]
        task_key = str(row["task_id"])
        out.append(
            Cell(model.repo, model.family, task_key, model.order, EXT_TASK_KEYS.index(task_key) + 1)
        )
    return out


def ext_panels(cells: list[Cell]) -> dict[str, Any]:
    """Balanced order, then the B3 family core and one B4 cell per family.

    `family_core` is the first admitted cell of each family in balanced order and
    is the B3 milestone-2 block. `e5_cells` uses the frozen task order instead,
    because B4 selection is made from solo evidence before any B3 outcome.
    """
    if len(cells) > MAX_PRIMARY_CELLS:
        raise ValueError(f"the extension panel holds at most {MAX_PRIMARY_CELLS} cells")
    ordered = balanced_order(cells)
    core: list[str] = []
    seen: set[str] = set()
    for cell in ordered:
        if cell.family not in seen:
            seen.add(cell.family)
            core.append(cell.key)
    e5: list[str] = []
    for fam in sorted(seen):
        first = min(
            (c for c in ordered if c.family == fam),
            key=lambda c: (c.task_order, c.model_order, c.key),
        )
        e5.append(first.key)
    return {
        "primary": [c.key for c in ordered],
        "family_core": core,
        "e5_cells": e5,
        "families": sorted(seen),
        "tasks": sorted({c.task for c in ordered}),
    }


def b3_b4_order(panel: dict[str, Any]) -> list[dict[str, Any]]:
    """Conditional milestone order; each stage names its own start condition."""
    primary = list(panel["primary"])
    core = list(panel["family_core"])
    rest = [key for key in primary if key not in core]
    e5 = list(panel["e5_cells"])
    return [
        {
            "stage": "M2_b3_family_core",
            "cells": core,
            "condition": "every sealed family core cell must complete before any C1 reading",
        },
        {
            "stage": "M3_b3_remaining",
            "cells": rest,
            "condition": "start each cell only if its 1.6x guard plus the 10-hour reserve fits 80",
        },
        {
            "stage": "M4_b4_reference",
            "cells": e5,
            "replicates": E5_N_REFERENCE,
            "condition": "reference stream runs before any evaluation replicate",
        },
        {
            "stage": "M4_b4_evaluation",
            "cells": e5,
            "replicates": E5_EVALUATION_PREFIXES[0],
            "condition": "reference precision must pass before C2 is read",
        },
        {
            "stage": "M5_b4_prefixes",
            "cells": e5,
            "prefixes": list(E5_EVALUATION_PREFIXES[1:]),
            "condition": "fixed round-robin technical expansion only; never chosen from effects",
        },
    ]


# --- Budget ----------------------------------------------------------------


def ext_may_start(cumulative_hours: float, forecast_base_hours: float) -> dict[str, Any]:
    """Extension block admission against its own 80-hour stop and 10-hour reserve."""
    if cumulative_hours < 0 or forecast_base_hours < 0:
        raise ValueError("hours cannot be negative")
    guarded = FORECAST_MULTIPLIER * forecast_base_hours
    projected = cumulative_hours + guarded + RESERVE_HOURS
    if cumulative_hours >= STOP_HOURS:
        start, reason = False, "extension_stop_reached"
    elif projected > STOP_HOURS:
        start, reason = False, "guarded_complete_block_exceeds_extension_stop"
    else:
        start, reason = True, "complete_block_fits_extension_ceiling"
    return {
        "start": start,
        "reason": reason,
        "cumulative_hours": cumulative_hours,
        "forecast_base_hours": forecast_base_hours,
        "forecast_guarded_hours": guarded,
        "reserve_hours": RESERVE_HOURS,
        "stop_hours": STOP_HOURS,
        "projected_total_hours": projected,
    }


def ext_must_stop(cumulative_hours: float) -> bool:
    if cumulative_hours < 0:
        raise ValueError("hours cannot be negative")
    return cumulative_hours >= STOP_HOURS


# --- Claim gates -----------------------------------------------------------


def _family_equal_mean(values: dict[str, float], family_of: dict[str, str]) -> float:
    if not values:
        raise ValueError("a family-equal mean needs at least one cell")
    by_family: dict[str, list[float]] = {}
    for key, value in values.items():
        by_family.setdefault(family_of[key], []).append(value)
    return float(np.mean([float(np.mean(v)) for _, v in sorted(by_family.items())]))


def c1_panel_gate(
    data: list[CellData],
    primary: list[Cell],
    *,
    n_draws: int = N_DRAWS,
) -> dict[str, Any]:
    """C1 recurrence over a variable 1-to-20-cell family, weighted family-equal.

    The primary statistic gives every family one vote, so a family with four
    admitted cells cannot outweigh a family with one. The cell-equal mean is
    reported beside it as a sensitivity, never as the gate.
    """
    if not 1 <= len(primary) <= MAX_PRIMARY_CELLS:
        raise ValueError(f"the primary family holds 1 to {MAX_PRIMARY_CELLS} cells")
    if len({c.key for c in primary}) != len(primary):
        raise ValueError("the primary family contains a duplicate cell")
    if n_draws < 20 * len(primary):
        raise ValueError("too few draws to resolve this Holm family")
    by_key = {d.cell.key: d for d in data}
    if len(by_key) != len(data):
        raise ValueError("duplicate cell results")
    if not set(by_key) <= {c.key for c in primary}:
        raise ValueError("results do not match the locked primary panel")
    family_of = {c.key: c.family for c in primary}
    task_of = {c.key: c.task for c in primary}
    if len(by_key) != len(primary):
        return {
            "passed": False,
            "reason": "incomplete_primary_panel",
            "completed": len(by_key),
            "required": len(primary),
        }
    ordered = [by_key[c.key] for c in primary]
    keys = [d.cell.key for d in ordered]
    observed = np.asarray([cell_delta(d, d.observed) for d in ordered])
    null = np.asarray([[cell_delta(d, d.replay(draw)) for d in ordered] for draw in range(n_draws)])

    observed_map = dict(zip(keys, (float(v) for v in observed), strict=True))
    global_observed = _family_equal_mean(observed_map, family_of)
    null_global = np.asarray(
        [
            _family_equal_mean(dict(zip(keys, (float(v) for v in row), strict=True)), family_of)
            for row in null
        ]
    )
    global_p = float((1 + np.sum(null_global >= global_observed)) / (1 + n_draws))
    p_cell = {
        key: float((1 + np.sum(null[:, i] >= observed[i])) / (1 + n_draws))
        for i, key in enumerate(keys)
    }
    adjusted = holm(p_cell, ALPHA)["adjusted_p"]
    positive = [
        key
        for i, key in enumerate(keys)
        if adjusted[key] <= ALPHA and observed[i] >= PRACTICAL_FLOOR
    ]
    families = sorted({family_of[key] for key in positive})
    tasks = sorted({task_of[key] for key in positive})
    family_loo = {
        fam: _family_equal_mean(
            {k: v for k, v in observed_map.items() if family_of[k] != fam}, family_of
        )
        for fam in sorted({family_of[k] for k in keys})
        if len({family_of[k] for k in keys} - {fam}) >= 1
    }
    task_loo = {
        task: _family_equal_mean(
            {k: v for k, v in observed_map.items() if task_of[k] != task}, family_of
        )
        for task in sorted({task_of[k] for k in keys})
        if any(task_of[k] != task for k in keys)
    }
    passed = bool(
        global_observed > 0
        and global_p <= ALPHA
        and len(families) >= MIN_POSITIVE_FAMILIES
        and bool(family_loo)
        and min(family_loo.values()) > 0
    )
    return {
        "passed": passed,
        "n_cells": len(primary),
        "n_families": len({family_of[k] for k in keys}),
        "global_family_equal_delta": global_observed,
        "global_p": global_p,
        "global_cell_equal_delta_sensitivity": float(observed.mean()),
        "cell_delta": observed_map,
        "cell_p": p_cell,
        "holm_adjusted_p": adjusted,
        "positive_cells": positive,
        "positive_families": families,
        "positive_tasks": tasks,
        "leave_one_family_out": family_loo,
        "leave_one_task_out": task_loo,
        "practical_floor": PRACTICAL_FLOOR,
        "min_positive_families": MIN_POSITIVE_FAMILIES,
        "n_draws": n_draws,
    }


def c2_coverage_gate(coverage_rows: list[dict[str, Any]], e5_cells: list[str]) -> dict[str, Any]:
    """C2 undercoverage over one prespecified E5 cell per admitted family."""
    if not e5_cells or len(set(e5_cells)) != len(e5_cells):
        raise ValueError("the E5 family must be a nonempty set of distinct cells")
    by_key = {str(r["cell_key"]): r for r in coverage_rows}
    if len(by_key) != len(coverage_rows):
        raise ValueError("duplicate coverage rows")
    if not set(by_key) <= set(e5_cells):
        raise ValueError("coverage rows do not match the locked E5 cells")
    p_values: dict[str, float] = {}
    for key in e5_cells:
        row = by_key.get(key)
        if (
            row is None
            or int(row["n_evaluation"]) < E5_MIN_EVALUATION
            or not row["reference_precise"]
        ):
            p_values[key] = 1.0
        else:
            p_values[key] = float(
                binomtest(
                    int(row["n_covered"]), int(row["n_evaluation"]), 0.95, alternative="less"
                ).pvalue
            )
    adjusted = holm(p_values, ALPHA)["adjusted_p"]
    supported = [
        key
        for key in e5_cells
        if key in by_key
        and int(by_key[key]["n_evaluation"]) >= E5_MIN_EVALUATION
        and by_key[key]["reference_precise"]
        and float(by_key[key]["coverage"]) <= E5_MAX_COVERAGE
        and float(by_key[key]["worst_reference_upper_coverage"]) < 0.95
        and adjusted[key] <= ALPHA
    ]
    return {
        "passed": len(supported) >= E5_MIN_SUPPORTED_CELLS,
        "supported_cells": supported,
        "holm_adjusted_p": adjusted,
        "family": list(e5_cells),
        "min_supported_cells": E5_MIN_SUPPORTED_CELLS,
    }


# --- E5 analysis composition ------------------------------------------------

E5_BOOTSTRAP_DRAWS = 2000
E5_VARIANCE_RATIO_FLOOR = 1.25


def _replicate_intervals(errors: np.ndarray) -> dict[str, Any]:
    """Both frozen intervals plus the whole-call bootstrap for one replicate.

    `errors` is (32 calls, 16 item appearances) of 0/1 reference-relative errors.
    The i.i.d. Wald interval is the one a practitioner would report by ignoring
    call membership. The 32-call t interval is the call-aware comparison.
    """
    values = np.asarray(errors, dtype=float)
    if values.ndim != 2 or values.shape[0] != fb_repeated.CALLS_PER_REPLICATE:
        raise ValueError(f"a replicate holds exactly {fb_repeated.CALLS_PER_REPLICATE} calls")
    if values.shape[1] != fb_repeated.K:
        raise ValueError(f"every call holds exactly {fb_repeated.K} appearances")
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("E5 inputs must be resolved binary reference-relative errors")
    flat = values.ravel()
    call_means = values.mean(axis=1)
    wald = stats.wald_interval(flat)
    t_interval = stats.call_t_interval(call_means)
    return {
        "mean": float(flat.mean()),
        "iid_wald": wald.as_dict(),
        "call_aware_t": t_interval.as_dict(),
        "nominal_iid_variance": float(max(flat.mean() * (1 - flat.mean()), 0.0) / flat.size),
        "nominal_half_width": float(wald.high - wald.point),
        "call_means": call_means,
    }


def ext_e5_analyze(
    cell_key: str,
    reference_errors: np.ndarray,
    evaluation_errors: np.ndarray,
    *,
    n_frame: int,
    bootstrap_draws: int = E5_BOOTSTRAP_DRAWS,
    seed_namespace: str = E5_NAMESPACE,
) -> dict[str, Any]:
    """End-to-end frozen E5 calculation for one cell, on CPU, from error arrays.

    `reference_errors` is (250, 32, 16) and `evaluation_errors` is (n, 32, 16) of
    reference-relative item errors. The return value carries every frozen E5
    quantity and, at the top level, exactly the row that `c2_coverage_gate` reads.
    """
    reference = np.asarray(reference_errors, dtype=float)
    evaluation = np.asarray(evaluation_errors, dtype=float)
    if reference.ndim != 3 or evaluation.ndim != 3:
        raise ValueError("both streams must be (replicates, calls, appearances)")
    if reference.shape[0] != E5_N_REFERENCE:
        raise ValueError(f"the frozen reference stream has {E5_N_REFERENCE} replicates")
    if evaluation.shape[0] not in E5_EVALUATION_PREFIXES:
        raise ValueError(f"evaluation must use a frozen prefix {E5_EVALUATION_PREFIXES}")
    if bootstrap_draws != E5_BOOTSTRAP_DRAWS:
        raise ValueError(f"the frozen whole-call bootstrap uses {E5_BOOTSTRAP_DRAWS} draws")
    if n_frame < evaluation.shape[2] * evaluation.shape[1]:
        raise ValueError("the frame cannot be smaller than one replicate")

    reference_rows = [_replicate_intervals(block) for block in reference]
    reference_means = np.asarray([row["mean"] for row in reference_rows])
    evaluation_rows = [_replicate_intervals(block) for block in evaluation]
    evaluation_means = np.asarray([row["mean"] for row in evaluation_rows])

    mean_half_width = float(np.mean([row["nominal_half_width"] for row in evaluation_rows]))
    precision = stats.reference_precision_check(reference_means, mean_half_width)
    target = float(reference_means.mean())
    target_se = float(precision["target_monte_carlo_se"])

    rng = np.random.default_rng(
        int(sha256_text(f"{seed_namespace}|{cell_key}|e5-bootstrap")[:16], 16)
    )
    bootstrap = [
        stats.whole_call_bootstrap(row["call_means"], bootstrap_draws, rng)
        for row in evaluation_rows
    ]

    def bounds(key: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([row[key]["low"] for row in evaluation_rows]),
            np.asarray([row[key]["high"] for row in evaluation_rows]),
        )

    iid_low, iid_high = bounds("iid_wald")
    t_low, t_high = bounds("call_aware_t")
    boot_low = np.asarray([b.low for b in bootstrap])
    boot_high = np.asarray([b.high for b in bootstrap])

    iid_coverage = stats.coverage(iid_low, iid_high, target)
    with_reference = stats.coverage_with_reference_uncertainty(iid_low, iid_high, target, target_se)
    correction = stats.fpc(evaluation.shape[1] * evaluation.shape[2], n_frame)
    ratio = stats.variance_ratio(
        evaluation_means,
        np.asarray([row["nominal_iid_variance"] for row in evaluation_rows]),
        bootstrap_draws,
        rng,
    )
    return {
        "schema_version": E5_ANALYSIS_SCHEMA,
        "cell_key": cell_key,
        "n_evaluation": int(evaluation.shape[0]),
        "n_covered": int(iid_coverage["n_covered"]),
        "coverage": float(iid_coverage["point"]),
        "worst_reference_upper_coverage": float(with_reference["least_favourable_upper_bound"]),
        "reference_precise": bool(precision["precise_enough_for_a_coverage_claim"]),
        "reference": {
            "n_replicates": int(reference.shape[0]),
            "target": target,
            "target_monte_carlo_se": target_se,
            "precision_check": precision,
        },
        "intervals": {
            "iid_wald_coverage": iid_coverage,
            "call_aware_t_coverage": stats.coverage(t_low, t_high, target),
            "whole_call_bootstrap_coverage": stats.coverage(boot_low, boot_high, target),
            "mean_width_iid_wald": float(np.mean(iid_high - iid_low)),
            "mean_width_call_aware_t": float(np.mean(t_high - t_low)),
            "mean_width_whole_call_bootstrap": float(np.mean(boot_high - boot_low)),
        },
        "coverage_with_reference_uncertainty": with_reference,
        "variance_ratio": ratio,
        "variance_ratio_floor": E5_VARIANCE_RATIO_FLOOR,
        "finite_population_correction": {
            "factor": correction,
            "applied_to_iid_wald": False,
            "applied_to_call_aware_t": False,
            "role": "reported sensitivity factor only; primary intervals use no correction",
        },
        "sensitivities": [
            "call-aware t interval",
            "whole-call bootstrap",
            "finite-population correction",
            "variance ratio",
        ],
        "claim_note": (
            "if reference precision fails, report the variance ratio and the widths "
            "and make no measured-coverage claim"
        ),
    }


# --- Admission lock payload ------------------------------------------------


def build_x1_admission_payload(
    summaries: list[dict[str, Any]],
    power_rows: list[dict[str, Any]],
    access_lock: dict[str, Any],
    *,
    choose_n: Callable[[float, list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the extension admission lock from solo-only screen evidence.

    Every one of the five candidates must appear with a terminal status. A
    technically unavailable candidate still contributes its four cells, each
    marked and not admitted, so no candidate can be silently dropped.
    """
    expected = {f"{model_id}::{task}" for model_id in EXT_MODEL_IDS for task in EXT_TASK_KEYS}
    seen = {str(row["cell_key"]) for row in summaries}
    missing = sorted(expected - seen)
    if missing or len(summaries) != len(expected):
        raise ValueError(
            f"the extension lock needs all {len(expected)} candidate cells; missing {missing[:4]}"
        )
    status_by_model = {
        str(row["model_id"]): str(row["candidate_status"]) for row in access_lock["candidates"]
    }
    if set(status_by_model) != set(EXT_MODEL_IDS):
        raise ValueError("the access lock does not decide every candidate")
    for row in summaries:
        model_id = str(row["model_id"])
        status = status_by_model[model_id]
        if status not in CANDIDATE_STATUSES:
            raise ValueError(f"{model_id} has no terminal candidate status")
        if status == "technically_unavailable":
            if row.get("admitted"):
                raise ValueError(f"{model_id} is unavailable but a cell is admitted")
        elif not row.get("technical_complete"):
            raise ValueError(f"{row['cell_key']} did not reach a terminal technical status")
    cells = admitted_cells(summaries)
    panel = ext_panels(cells)
    item_counts = {
        c.key: choose_n(
            float(next(r for r in summaries if r["cell_key"] == c.key)["error_rate"]), power_rows
        )
        for c in cells
    }
    families = panel["families"]
    tasks = panel["tasks"]
    if len(families) >= 5 and len(tasks) >= 4:
        scope = "full_five_family_four_task"
    elif len(families) >= MIN_POSITIVE_FAMILIES:
        scope = f"narrowed_{len(families)}_family_{len(tasks)}_task"
    else:
        scope = "cross_family_recurrence_claim_not_supported"
    return {
        "schema_version": "ext-x1r2-admission-lock-v1",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "attempted_candidates": sorted(expected),
        "candidate_status": {
            model_id: {
                "candidate_status": status_by_model[model_id],
                "access_status": next(
                    str(c["access_status"])
                    for c in access_lock["candidates"]
                    if c["model_id"] == model_id
                ),
                "family": next(m.family for m in EXT_MODELS if m.repo == model_id),
                "cells": sorted(f"{model_id}::{task}" for task in EXT_TASK_KEYS),
            }
            for model_id in EXT_MODEL_IDS
        },
        "unavailable_models": list(access_lock["unavailable_models"]),
        "runtime_access_lock_sha256": sha256_json(access_lock),
        "technical_status": {
            str(row["cell_key"]): {
                "complete": row.get("technical_complete", False),
                "candidate_status": status_by_model[str(row["model_id"])],
                "status_counts": row.get("status_counts", {}),
            }
            for row in summaries
        },
        "solo_metrics": {str(row["cell_key"]): row for row in summaries},
        "ordered_cells": panel,
        "claim_scope": {
            "recurrence_scope": scope,
            "primary_model_families": families,
            "primary_tasks": tasks,
            "distinct_primary_model_families": len(families),
            "distinct_primary_tasks": len(tasks),
        },
        "item_counts": item_counts,
        "complete_block_queue": b3_b4_order(panel),
        "selection_evidence": "solo_capability_and_technical_timing_only",
        "prior_opened_joint_families": list(OPENED_JOINT_FAMILIES),
        "extension_joint_outputs_seen": False,
    }


def disclosure() -> dict[str, Any]:
    """The standing honesty record carried into every extension artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "separate_from_freeze_b": True,
        "prior_opened_joint_families": list(OPENED_JOINT_FAMILIES),
        "prior_opened_note": (
            "Qwen and Granite joint results are already open from Freeze B. This extension "
            "uses new untouched frames and does not rewrite the Freeze B claim family."
        ),
        "selection_rule": "admission uses solo capability and technical timing only",
        "supersedes": SUPERSEDES,
        "supersedes_reason": (
            "X1R3 pinned checkpoint revisions in its preflight plan but did not validate the "
            "reported worker revision, anchor ID, and request layout against those pins"
        ),
        "candidate_access_rule": (
            "the immutable runtime policy attempts all five candidates; a signed runtime "
            "and access lock binds the passing batch-invariance report and decides attempted "
            "against technically unavailable"
        ),
        "call_id_prefix": CALL_PREFIX,
        "substitution_rule": (
            "a candidate may be replaced only before any extension joint output exists"
        ),
        "budget": {
            "meter_origin_a100_hours": METER_ORIGIN_HOURS,
            "stop_a100_hours": STOP_HOURS,
            "reserve_a100_hours": RESERVE_HOURS,
            "start_guard_multiplier": FORECAST_MULTIPLIER,
            "shares_freeze_b_allowance": False,
        },
        "frame_sizes": {"screen": N_SCREEN, "confirm": N_CONFIRM, "filler": N_FILLER},
        "seeds": {
            "frame_selection": EXT_SEED,
            "assignment_namespace": ASSIGN_NAMESPACE,
            "repeated_namespace": E5_NAMESPACE,
            "power_namespace": POWER_NAMESPACE,
        },
        "statistics": {
            "n_draws": N_DRAWS,
            "alpha": ALPHA,
            "planning_alpha": PLANNING_ALPHA,
            "practical_floor": PRACTICAL_FLOOR,
            "max_primary_cells": MAX_PRIMARY_CELLS,
            "primary_contrast": "family_equal_joint_minus_exact_covariance",
            "cell_equal_contrast": "reported_as_sensitivity_only",
            "batch_size": BATCH_SIZE,
            "assignments": N_ASSIGNMENTS,
            "e5_reference_replicates": E5_N_REFERENCE,
            "e5_evaluation_prefixes": list(E5_EVALUATION_PREFIXES),
            "e5_intervals": [
                "iid Wald",
                "32-call t",
                "whole-call bootstrap",
                "Clopper-Pearson coverage",
                "reference uncertainty",
                "variance ratio",
            ],
            "c1_gate": (
                "family-equal global contrast above zero; global replay p at most 0.05; "
                "at least three families with a Holm-positive cell at effect at least "
                "0.002; every leave-one-family-out family-equal mean above zero"
            ),
            "c2_gate": (
                "at least two prespecified E5 cells with at least 400 evaluation "
                "replicates, passing reference precision, coverage at most 0.90, worst "
                "reference upper coverage below 0.95, and Holm-adjusted p at most 0.05"
            ),
        },
        "identity_sha256": sha256_json(
            {
                "models": [[m.repo, m.revision, m.family, m.order] for m in EXT_MODELS],
                "tasks": [[t.key, t.dataset, t.revision, t.split] for t in EXT_TASKS],
                "seed": EXT_SEED,
                "namespaces": [ASSIGN_NAMESPACE, E5_NAMESPACE, POWER_NAMESPACE],
            }
        ),
    }
