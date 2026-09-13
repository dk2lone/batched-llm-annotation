"""Fixed-reference composition and position arms; no model outcome is used."""

from __future__ import annotations

from typing import Any

import numpy as np

from step0.hashing import derived_seed, order_key

N_PER_GROUP = 256
K = 16
ASSIGNMENTS = 3
ROTATIONS = (0, 4, 8, 12)


def reference_group(task: str, reference: Any) -> int:
    """Binary, prespecified reference category for the composition contrast."""
    if task in {"civil_comments", "imdb"}:
        return int(reference == 1)
    if task in {"ag_news", "multi_nli"}:
        return int(reference == 0)
    if task == "go_emotions":
        return int(tuple(reference) == (27,))
    if task == "multinerd":
        return int(any(span[2] == "PER" for span in reference))
    raise ValueError(f"unknown task {task}")


def select_items(task: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len({r["doc_id"] for r in rows}) != len(rows):
        raise ValueError("confirmation frame contains duplicate IDs")
    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        grouped[reference_group(task, row["reference_normalized"])].append(row)
    if any(len(grouped[g]) < N_PER_GROUP for g in (0, 1)):
        raise ValueError("fewer than 256 reference rows in a mechanism group")
    return [row for group in (0, 1) for row in sorted(
        grouped[group], key=lambda r: (order_key(derived_seed("e4", task), r["doc_id"]),
                                     r["doc_id"])
    )[:N_PER_GROUP]]


def build_arms(task: str, rows: list[dict[str, Any]], cell_key: str) -> list[dict[str, Any]]:
    """Use the same 512 items in every arm and assignment."""
    chosen = select_items(task, rows)
    groups = {g: [r["doc_id"] for r in chosen
                  if reference_group(task, r["reference_normalized"]) == g]
              for g in (0, 1)}
    out = []
    for assignment in range(1, ASSIGNMENTS + 1):
        shuffled = {}
        for group in (0, 1):
            rng = np.random.default_rng(derived_seed("freezeb-e4-v1", cell_key,
                                                     assignment, group))
            shuffled[group] = rng.permutation(groups[group]).tolist()
        mixed = [shuffled[0][i * 8:(i + 1) * 8]
                 + shuffled[1][i * 8:(i + 1) * 8] for i in range(32)]
        homogeneous = ([shuffled[0][i * K:(i + 1) * K] for i in range(16)]
                       + [shuffled[1][i * K:(i + 1) * K] for i in range(16)])
        for arm, batches, rotations in (("mixed", mixed, ROTATIONS),
                                        ("homogeneous", homogeneous, (0,))):
            for call_index, batch in enumerate(batches):
                for rotation in rotations:
                    order = batch[rotation:] + batch[:rotation]
                    out.append({
                        "call_id": (f"{cell_key}:e4:a{assignment}:{arm}:"
                                    f"b{call_index}:r{rotation}"),
                        "arm": arm, "assignment": assignment,
                        "call_index": call_index, "rotation": rotation,
                        "execution_block": call_index % 8, "doc_order": order,
                    })
    return out
