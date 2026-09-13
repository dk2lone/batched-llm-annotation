"""Independent 512-appearance, 32-call replicate schedule for E5."""

from __future__ import annotations

import numpy as np

from step0.hashing import derived_seed

N_APPEARANCES = 512
K = 16
CALLS_PER_REPLICATE = N_APPEARANCES // K
N_REFERENCE = 250
EVALUATION_PREFIXES = (400, 1000, 2500, 5000)
NAMESPACE = "bigdata2026-freezeb-e5-v1"


def replicate_calls(
    doc_ids: tuple[str, ...],
    cell_key: str,
    stream: str,
    replicate: int,
    *,
    namespace: str = NAMESPACE,
    call_id_prefix: str = "",
) -> list[dict[str, object]]:
    """Every call is fresh; item reuse across replicates never reuses an output."""
    if len(doc_ids) < N_APPEARANCES or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("the frame needs at least 512 distinct IDs")
    if stream not in {"reference", "evaluation", "capacity"} or replicate < 0:
        raise ValueError("invalid stream or replicate index")
    rng = np.random.default_rng(derived_seed(namespace, cell_key, stream, replicate))
    chosen = rng.choice(len(doc_ids), size=N_APPEARANCES, replace=False)
    rng.shuffle(chosen)
    out: list[dict[str, object]] = []
    for call_index, group in enumerate(chosen.reshape(CALLS_PER_REPLICATE, K)):
        out.append({
            "call_id": (
                f"{call_id_prefix}{cell_key}:e5:{stream}:r{replicate}:b{call_index}"
            ),
            "cell_key": cell_key,
            "stream": stream,
            "replicate": replicate,
            "call_index": call_index,
            "doc_order": [doc_ids[int(i)] for i in group],
        })
    return out


def prefix_calls(
    n_cells: int,
    evaluation_replicates: int,
    *,
    prefixes: tuple[int, ...] = EVALUATION_PREFIXES,
) -> int:
    if n_cells <= 0 or evaluation_replicates not in prefixes:
        raise ValueError("unfrozen repeated-sampling prefix")
    return n_cells * (N_REFERENCE + evaluation_replicates) * CALLS_PER_REPLICATE
