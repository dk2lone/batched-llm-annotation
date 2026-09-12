"""Deterministic operational assignment generator.

One policy only: the operational policy of `plan.md` section 7.3. Items are
partitioned at random subject to the frozen length strata, and output positions are
balanced exactly. The balanced mechanism design of that same section is deliberately
out of scope for this bounded paper.

Every draw is reproducible from its key tuple alone, so a randomization-inference
replay reproduces the same partitions byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import derived_seed, sha256_file, sha256_json, write_json, write_jsonl
from .lock import ASSIGNMENT_GENERATOR, CONDITIONS, PANEL

NAMESPACE = str(ASSIGNMENT_GENERATOR["seeds"]["namespace"])
K = int(PANEL["batch_size_k"])
N_STRATA = int(ASSIGNMENT_GENERATOR["length_strata"]["count"])
N_BLOCKS = int(CONDITIONS["interleaving"]["n_execution_blocks_per_cell"])
FILLER_CANDIDATE_DRAW = int(CONDITIONS["filler"]["filler_candidate_draw"])
FILLER_TOLERANCE = float(CONDITIONS["filler"]["length_match_tolerance"])


@dataclass(frozen=True)
class Frame:
    """The frozen item frame, indexed by stratum."""

    doc_ids: tuple[str, ...]
    strata: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.doc_ids) != len(self.strata):
            raise ValueError("doc_ids and strata must have the same length")
        counts = np.bincount(np.asarray(self.strata), minlength=N_STRATA)
        if counts.size != N_STRATA or len(set(counts.tolist())) != 1:
            raise ValueError(f"every stratum must hold the same number of items, got {counts}")

    @property
    def n_items(self) -> int:
        return len(self.doc_ids)

    @property
    def n_calls(self) -> int:
        return self.n_items // K

    def stratum_members(self) -> list[list[int]]:
        """Item indices per stratum, ordered by doc_id so the order is data independent."""
        out: list[list[int]] = [[] for _ in range(N_STRATA)]
        for index in sorted(range(self.n_items), key=lambda i: self.doc_ids[i]):
            out[self.strata[index]].append(index)
        return out


def rng_for(*key: object) -> np.random.Generator:
    return np.random.default_rng(derived_seed(NAMESPACE, *key))


def _partition(frame: Frame, cell_id: str, assignment: int, draw: int) -> np.ndarray:
    """Return an (n_calls, N_STRATA) matrix of item indices.

    Column s of row b holds the item that call b takes from stratum s. `draw` is -1
    for the observed assignment and 0, 1, 2, ... for randomization-inference replays.
    """
    members = frame.stratum_members()
    columns = []
    for stratum, ids in enumerate(members):
        key = (
            (cell_id, assignment, stratum)
            if draw < 0
            else (cell_id, "replay", assignment, draw, stratum)
        )
        rng = rng_for(*key)
        columns.append(rng.permutation(np.asarray(ids, dtype=np.int64)))
    return np.stack(columns, axis=1)


def observed_partition(frame: Frame, cell_id: str, assignment: int) -> np.ndarray:
    return _partition(frame, cell_id, assignment, draw=-1)


def replay_partition(frame: Frame, cell_id: str, assignment: int, draw: int) -> np.ndarray:
    if draw < 0:
        raise ValueError("a replay draw index must be 0 or greater")
    observed = observed_partition(frame, cell_id, assignment)
    replayed = observed.copy()
    for stratum in range(N_STRATA):
        for block in range(N_BLOCKS):
            call_indices = np.arange(block, frame.n_calls, N_BLOCKS)
            rng = rng_for(cell_id, "replay", assignment, draw, stratum, block)
            replayed[call_indices, stratum] = rng.permutation(
                observed[call_indices, stratum]
            )
    return replayed


def observed_stratum_matrices(frame: Frame, cell_id: str, n_assignments: int) -> list[np.ndarray]:
    """Per assignment, an (n_strata, n_calls) matrix of item indices, as actually run."""
    return [
        observed_partition(frame, cell_id, a).T.copy() for a in range(1, n_assignments + 1)
    ]


def replay_stratum_matrices(
    frame: Frame, cell_id: str, n_assignments: int, draw: int
) -> list[np.ndarray]:
    """The same shape as `observed_stratum_matrices`, for randomization-inference draw `draw`.

    This is the frozen generator itself, not a separate shuffler. Each item remains in its
    observed execution block while peers change within that block. Call index order,
    length strata, and the one-item-per-stratum call rule are preserved.
    """
    return [
        replay_partition(frame, cell_id, a, draw).T.copy() for a in range(1, n_assignments + 1)
    ]


def _match_slots(
    call_items: list[int], used: dict[int, set[int]], rng: np.random.Generator
) -> list[int]:
    """Assign the K items of one call to K distinct slots, avoiding each item's used slots.

    A deterministic augmenting-path matching. After two assignments an item has at
    least K-2 free slots, so Hall's condition holds and a perfect matching exists.
    """
    order = rng.permutation(K)
    candidates = [
        [int(s) for s in order if int(s) not in used.get(item, set())] for item in call_items
    ]
    slot_owner: dict[int, int] = {}

    def augment(position: int, seen: set[int]) -> bool:
        for slot in candidates[position]:
            if slot in seen:
                continue
            seen.add(slot)
            if slot not in slot_owner or augment(slot_owner[slot], seen):
                slot_owner[slot] = position
                return True
        return False

    for position in range(len(call_items)):
        if not augment(position, set()):
            raise RuntimeError("slot matching failed; the frozen constraints are infeasible")
    slot_of = [0] * len(call_items)
    for slot, position in slot_owner.items():
        slot_of[position] = slot
    return slot_of


def build_cell(frame: Frame, cell_id: str, n_assignments: int) -> list[dict[str, Any]]:
    """Build every call of every assignment for one cell, with exact slot balance."""
    used: dict[int, set[int]] = {i: set() for i in range(frame.n_items)}
    calls: list[dict[str, Any]] = []
    for assignment in range(1, n_assignments + 1):
        matrix = observed_partition(frame, cell_id, assignment)
        for call_index in range(frame.n_calls):
            items = [int(x) for x in matrix[call_index]]
            rng = rng_for(cell_id, assignment, "slots", call_index)
            slots = _match_slots(items, used, rng)
            for item, slot in zip(items, slots, strict=True):
                used[item].add(slot)
            ordered = [item for _, item in sorted(zip(slots, items, strict=True))]
            calls.append(
                {
                    "cell_id": cell_id,
                    "assignment": assignment,
                    "call_index": call_index,
                    "execution_block": call_index % N_BLOCKS,
                    "doc_order": [frame.doc_ids[i] for i in ordered],
                    "item_indices": ordered,
                    "length_strata": [frame.strata[i] for i in ordered],
                }
            )
    return calls


def balance_report(
    frame: Frame, calls: list[dict[str, Any]], references: list[int]
) -> dict[str, Any]:
    """Realized balance diagnostics. Class composition is a diagnostic, never a constraint."""
    n_assignments = max(int(c["assignment"]) for c in calls)
    index_of = {doc: i for i, doc in enumerate(frame.doc_ids)}
    appearances = np.zeros(frame.n_items, dtype=int)
    slot_by_item: dict[int, list[int]] = {i: [] for i in range(frame.n_items)}
    slot_counts = np.zeros((n_assignments + 1, K), dtype=int)
    stratum_slot = np.zeros((N_STRATA, K), dtype=int)
    per_call_class: list[list[int]] = []
    per_call_stratum_sum: list[int] = []

    for call in calls:
        assignment = int(call["assignment"])
        for slot, doc in enumerate(call["doc_order"]):
            item = index_of[doc]
            appearances[item] += 1
            slot_by_item[item].append(slot)
            slot_counts[assignment, slot] += 1
            stratum_slot[frame.strata[item], slot] += 1
        per_call_class.append(
            [sum(1 for d in call["doc_order"] if references[index_of[d]] == k)
             for k in sorted(set(references))]
        )
        per_call_stratum_sum.append(sum(frame.strata[index_of[d]] for d in call["doc_order"]))

    distinct_slots = [len(set(v)) for v in slot_by_item.values()]
    class_matrix = np.asarray(per_call_class, dtype=float)
    return {
        "n_assignments": n_assignments,
        "n_calls_total": len(calls),
        "n_calls_per_assignment": len(calls) // n_assignments,
        "appearances_per_item": {
            "min": int(appearances.min()),
            "max": int(appearances.max()),
            "expected": n_assignments,
            "all_equal": bool(appearances.min() == appearances.max() == n_assignments),
        },
        "distinct_slots_per_item": {
            "min": int(min(distinct_slots)),
            "max": int(max(distinct_slots)),
            "expected": n_assignments,
            "all_distinct": bool(min(distinct_slots) == n_assignments),
        },
        "slot_usage_per_assignment": {
            "min": int(slot_counts[1:].min()),
            "max": int(slot_counts[1:].max()),
            "expected": frame.n_calls,
            "uniform": bool(slot_counts[1:].min() == slot_counts[1:].max() == frame.n_calls),
        },
        "stratum_by_slot": {
            "min": int(stratum_slot.min()),
            "max": int(stratum_slot.max()),
            "expected": n_assignments * frame.n_calls // K,
        },
        "one_item_per_stratum_per_call": bool(
            all(sorted(c["length_strata"]) == list(range(N_STRATA)) for c in calls)
        ),
        "class_composition_per_call": {
            "mean": class_matrix.mean(axis=0).tolist(),
            "sd": class_matrix.std(axis=0, ddof=1).tolist(),
            "note": (
                "a diagnostic only. The operational policy does not balance class composition; "
                "plan.md section 7.3 reserves that for the balanced estimand, which this paper "
                "does not run."
            ),
        },
        "length_stratum_sum_per_call": {
            "min": int(min(per_call_stratum_sum)),
            "max": int(max(per_call_stratum_sum)),
            "expected": N_STRATA * (N_STRATA - 1) // 2,
            "constant_by_construction": bool(
                min(per_call_stratum_sum) == max(per_call_stratum_sum)
            ),
        },
    }


def choose_filler(
    pool_chars: list[int], want_chars: int, n: int, rng: np.random.Generator, draw: int
) -> list[int]:
    """Greedy nearest-fit draw without replacement, so the prompt holds no duplicate.

    The same rule as the pilot's `batchgate.plan.choose_filler`: each step samples
    `draw` candidates and takes the one closest to the remaining ideal length, so the
    peer set totals about `want_chars` characters.
    """
    size = len(pool_chars)
    if n > size:
        raise ValueError(f"filler pool holds {size} records, {n} peers requested")
    used: set[int] = set()
    chosen: list[int] = []
    remaining = float(want_chars)
    for step in range(n):
        ideal = remaining / (n - step)
        sample = rng.choice(size, size=min(draw, size), replace=False)
        candidates = [int(i) for i in sample if int(i) not in used]
        if not candidates:
            candidates = [i for i in range(size) if i not in used][:draw]
        best = min(candidates, key=lambda i: abs(pool_chars[i] - ideal))
        used.add(best)
        chosen.append(best)
        remaining -= pool_chars[best]
    return chosen


def job_skeletons(
    cell: dict[str, Any],
    calls: list[dict[str, Any]],
    chars_by_doc: dict[str, int],
    filler_pool: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    """One row per planned call. No prompt text, no output, no result."""
    cell_id = str(cell["cell_id"])
    out: list[dict[str, Any]] = []

    def emit(
        condition: str,
        assignment: int,
        call_index: int,
        doc_order: list[str],
        targets: list[str],
        slots: list[int],
        block: int,
        serving_concurrency: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        spec = {
            "cell_id": cell_id,
            "model_id": cell["model_id"],
            "task_id": cell["task_id"],
            "condition": condition,
            "assignment": assignment,
            "call_index": call_index,
            "doc_order": doc_order,
            "targets": targets,
        }
        call_id = sha256_json(spec)[:16]
        execution_order_key = sha256_json(
            {
                "namespace": NAMESPACE,
                "purpose": "within-block-execution-order",
                "cell_id": cell_id,
                "execution_block": block,
                "call_id": call_id,
            }
        )
        out.append(
            {
                "call_id": call_id,
                "parent_call_id": None,
                "retry_index": 0,
                "batch_id": call_id,
                "cell_id": cell_id,
                "model_id": cell["model_id"],
                "task_id": cell["task_id"],
                "condition": condition,
                "assignment": assignment,
                "call_index": call_index,
                "execution_block": block,
                "execution_order_key": execution_order_key,
                "doc_order": doc_order,
                "target_ids": targets,
                "target_slots": slots,
                "n_requested": len(targets),
                "serving_concurrency": serving_concurrency,
                **(extra or {}),
            }
        )

    seen_solo: set[str] = set()
    filler_spec = CONDITIONS["filler"]["cells"].get(cell_id)
    serial_spec = CONDITIONS["serial_serving"]["cells"].get(cell_id)
    serial_calls = int(serial_spec["joint_calls"]) if serial_spec else 0
    filler_assignments = int(filler_spec["assignments"]) if filler_spec else 0
    filler_calls_budget = int(filler_spec["calls"]) if filler_spec else 0
    filler_emitted = 0

    for call in calls:
        assignment = int(call["assignment"])
        call_index = int(call["call_index"])
        block = int(call["execution_block"])
        docs = [str(d) for d in call["doc_order"]]
        emit("joint", assignment, call_index, docs, docs, list(range(K)), block, 0)
        for slot, doc in enumerate(docs):
            emit("exact_context", assignment, call_index, docs, [doc], [slot], block, 0)
            if assignment == 1 and doc not in seen_solo:
                seen_solo.add(doc)
        if (
            filler_spec
            and assignment <= filler_assignments
            and filler_emitted < filler_calls_budget
        ):
            for slot, doc in enumerate(docs):
                if filler_emitted >= filler_calls_budget:
                    break
                want = sum(chars_by_doc[d] for d in docs if d != doc)
                rng = rng_for(cell_id, assignment, "filler", call_index, slot)
                picks = choose_filler(
                    [n for _, n in filler_pool], want, K - 1, rng, FILLER_CANDIDATE_DRAW
                )
                peers = [filler_pool[i][0] for i in picks]
                emit(
                    "filler",
                    assignment,
                    call_index,
                    peers[:slot] + [doc] + peers[slot:],
                    [doc],
                    [slot],
                    block,
                    0,
                    {
                        "match_chars_target": want,
                        "match_chars_actual": sum(filler_pool[i][1] for i in picks),
                    },
                )
                filler_emitted += 1
        if serial_spec and assignment == 1 and call_index < serial_calls:
            emit("serial_joint", assignment, call_index, docs, docs, list(range(K)), block, 1)
            for slot, doc in enumerate(docs):
                emit("serial_exact", assignment, call_index, docs, [doc], [slot], block, 1)

    for index, doc in enumerate(sorted(seen_solo)):
        emit("solo", 0, -1, [doc], [doc], [0], index % N_BLOCKS, 0)
    return out


def filler_match_report(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """How close the drawn filler peers come to the real peer character total."""
    rows = [j for j in jobs if j["condition"] == "filler"]
    if not rows:
        return {"n_filler_calls": 0, "note": "this cell runs no filler condition"}
    relative = np.asarray(
        [
            abs(float(j["match_chars_actual"]) - float(j["match_chars_target"]))
            / max(float(j["match_chars_target"]), 1.0)
            for j in rows
        ]
    )
    return {
        "n_filler_calls": len(rows),
        "tolerance": FILLER_TOLERANCE,
        "mean_relative_gap": float(relative.mean()),
        "max_relative_gap": float(relative.max()),
        "fraction_within_tolerance": float((relative <= FILLER_TOLERANCE).mean()),
        "note": (
            "a diagnostic of the frozen greedy nearest-fit draw. The tolerance is the pilot's "
            "reporting threshold, not a filter: no call is removed for missing it."
        ),
    }


def write_cell(
    out_dir: Path,
    cell: dict[str, Any],
    frame: Frame,
    references: list[int],
    n_assignments: int,
    chars: list[int],
    filler_pool: list[tuple[str, int]],
) -> dict[str, Any]:
    cell_dir = out_dir / str(cell["cell_id"])
    calls = build_cell(frame, str(cell["cell_id"]), n_assignments)
    chars_by_doc = dict(zip(frame.doc_ids, chars, strict=True))
    jobs = job_skeletons(cell, calls, chars_by_doc, filler_pool)
    matrix_path = cell_dir / "assignment_matrix.jsonl"
    jobs_path = cell_dir / "job_skeletons.jsonl"
    write_jsonl(matrix_path, calls)
    write_jsonl(jobs_path, jobs)
    balance = balance_report(frame, calls, references)
    write_json(cell_dir / "balance.json", balance)

    condition_counts: dict[str, int] = {}
    block_counts: dict[str, dict[str, int]] = {}
    for job in jobs:
        condition = str(job["condition"])
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
        block_counts.setdefault(condition, {})
        key = str(job["execution_block"])
        block_counts[condition][key] = block_counts[condition].get(key, 0) + 1
    meta = {
        "cell": cell,
        "namespace": NAMESPACE,
        "master_seed": ASSIGNMENT_GENERATOR["seeds"]["master_seed"],
        "policy": ASSIGNMENT_GENERATOR["policy"],
        "n_items": frame.n_items,
        "batch_size_k": K,
        "n_strata": N_STRATA,
        "n_assignments": n_assignments,
        "n_calls_per_assignment": frame.n_calls,
        "n_execution_blocks": N_BLOCKS,
        "within_block_order": "ascending execution_order_key",
        "planned_calls_by_condition": dict(sorted(condition_counts.items())),
        "planned_calls_by_condition_and_execution_block": {
            condition: dict(sorted(blocks.items(), key=lambda kv: int(kv[0])))
            for condition, blocks in sorted(block_counts.items())
        },
        "every_arm_appears_in_every_block": all(
            len(blocks) == N_BLOCKS for blocks in block_counts.values()
        ),
        "planned_calls_total": len(jobs),
        "filler_length_match": filler_match_report(jobs),
        "fake_batch_calls": 0,
        "fake_batch_note": (
            "the fake-batch condition creates no call. It regroups exact-context labels with "
            "this same generator and recomputes the identical statistic."
        ),
        "unique_call_ids": len({j["call_id"] for j in jobs}),
        "files": {
            "assignment_matrix.jsonl": {"sha256": sha256_file(matrix_path), "rows": len(calls)},
            "job_skeletons.jsonl": {"sha256": sha256_file(jobs_path), "rows": len(jobs)},
        },
    }
    write_json(cell_dir / "meta.json", meta)
    return meta
