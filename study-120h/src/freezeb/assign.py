"""Reproducible length-balanced assignments for all five batch sizes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from step0.hashing import derived_seed

BATCH_SIZES = (2, 4, 8, 16, 32)
N_ASSIGNMENTS = 3
N_BLOCKS = 8
NAMESPACE = "bigdata2026-freezeb-v1"


@dataclass(frozen=True)
class Frame:
    doc_ids: tuple[str, ...]
    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.doc_ids) != len(self.lengths) or len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("frame IDs must be unique and match lengths")

    def strata(self, k: int) -> list[list[int]]:
        if k not in BATCH_SIZES or len(self.doc_ids) % k:
            raise ValueError("unsupported batch size or indivisible frame")
        ordered = sorted(range(len(self.doc_ids)), key=lambda i: (self.lengths[i], self.doc_ids[i]))
        per = len(ordered) // k
        return [ordered[s * per:(s + 1) * per] for s in range(k)]


def partition(
    frame: Frame, cell_key: str, k: int, assignment: int, *, namespace: str = NAMESPACE
) -> np.ndarray:
    if not 1 <= assignment <= N_ASSIGNMENTS:
        raise ValueError("assignment must be 1, 2, or 3")
    columns = []
    for stratum, members in enumerate(frame.strata(k)):
        rng = np.random.default_rng(derived_seed(namespace, cell_key, k, assignment, stratum))
        columns.append(rng.permutation(members))
    return np.stack(columns, axis=1)


def replay(
    frame: Frame,
    cell_key: str,
    k: int,
    assignment: int,
    draw: int,
    *,
    namespace: str = NAMESPACE,
) -> np.ndarray:
    """Regroup observed item outputs inside stratum and execution-block cells."""
    if draw < 0:
        raise ValueError("draw must be nonnegative")
    observed = partition(frame, cell_key, k, assignment, namespace=namespace)
    result = observed.copy()
    for stratum in range(k):
        for block in range(N_BLOCKS):
            call_indices = np.arange(block, len(observed), N_BLOCKS)
            rng = np.random.default_rng(
                derived_seed(namespace, cell_key, k, assignment, "replay", draw, stratum, block)
            )
            result[call_indices, stratum] = rng.permutation(observed[call_indices, stratum])
    return result


def observed_stat_matrices(
    frame: Frame, cell_key: str, k: int, *, namespace: str = NAMESPACE
) -> list[np.ndarray]:
    """Step 0 stats expects (strata, calls), unlike the call-ledger matrix."""
    return [partition(frame, cell_key, k, a, namespace=namespace).T.copy()
            for a in range(1, N_ASSIGNMENTS + 1)]


def replay_stat_matrices(
    frame: Frame, cell_key: str, k: int, draw: int, *, namespace: str = NAMESPACE
) -> list[np.ndarray]:
    return [replay(frame, cell_key, k, a, draw, namespace=namespace).T.copy()
            for a in range(1, N_ASSIGNMENTS + 1)]


def _slots(items: list[int], seen: dict[int, dict[int, int]], seed: int) -> list[int]:
    k = len(items)
    rng = np.random.default_rng(seed)
    slot_order = rng.permutation(k).tolist()
    rank = {slot: i for i, slot in enumerate(slot_order)}
    ordered = [sorted(range(k), key=lambda s: (seen[item].get(s, 0), rank[s])) for item in items]

    def match(candidates: list[list[int]]) -> dict[int, int] | None:
        owner: dict[int, int] = {}

        def augment(position: int, visited: set[int]) -> bool:
            for slot in candidates[position]:
                if slot in visited:
                    continue
                visited.add(slot)
                if slot not in owner or augment(owner[slot], visited):
                    owner[slot] = position
                    return True
            return False

        for position in range(k):
            if not augment(position, set()):
                return None
        return owner

    unseen = [[s for s in candidates if seen[item].get(s, 0) == 0]
              for item, candidates in zip(items, ordered, strict=True)]
    owner = match(unseen) if k >= 3 else None
    if owner is None:
        owner = match(ordered)
    if owner is None:
        raise RuntimeError("slot matching failed")
    slots = [0] * k
    for slot, position in owner.items():
        slots[position] = slot
    return slots


def build_calls(
    frame: Frame, cell_key: str, k: int, *, namespace: str = NAMESPACE
) -> list[dict[str, object]]:
    seen: dict[int, dict[int, int]] = {i: {} for i in range(len(frame.doc_ids))}
    calls: list[dict[str, object]] = []
    for assignment in range(1, N_ASSIGNMENTS + 1):
        matrix = partition(frame, cell_key, k, assignment, namespace=namespace)
        for call_index, row in enumerate(matrix):
            items = [int(i) for i in row]
            if k <= 4:
                offset = derived_seed(namespace, cell_key, k, "small-slot-offset") % k
                slots = [(stratum + offset + assignment - 1) % k for stratum in range(k)]
            else:
                slots = _slots(
                    items,
                    seen,
                    derived_seed(namespace, cell_key, k, assignment, "slots", call_index),
                )
            for item, slot in zip(items, slots, strict=True):
                seen[item][slot] = seen[item].get(slot, 0) + 1
            doc_order = [frame.doc_ids[item] for _, item in sorted(zip(slots, items, strict=True))]
            calls.append({
                "call_id": f"{cell_key}:k{k}:a{assignment}:b{call_index}",
                "assignment": assignment,
                "call_index": call_index,
                "execution_block": call_index % N_BLOCKS,
                "batch_size": k,
                "doc_order": doc_order,
            })
    return calls
