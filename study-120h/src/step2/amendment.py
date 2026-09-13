"""The operational amendment that permits two concurrent A100 workers for the bridge.

The sealed Step 0 protocol records `panel.serving.hardware` as "one NVIDIA A100, one at a
time". Daniel authorised two concurrent A100 workers for this bridge before any scientific
output existed. That is an operational deviation from a recorded serving statement, so it is
written down here, in full, before the first paid call.

The deviation is about how many independent engine processes run at the same time. It is not
about how a call is served. Every call still runs on exactly one A100, in one model process,
with `tensor_parallel_size` 1 and the frozen serving fields. The freeze itself is never
edited; a change to a design field would need a new freeze directory instead.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from step0.hashing import sha256_json, write_json, write_text
from step1.frozen import Frozen
from step1.paths import AMENDMENTS

AMENDMENT_ID = "step2-bridge-concurrency"

#: The exact sealed sentence this amendment deviates from.
DEVIATION_FIELD = "panel.serving.hardware"

CLAUSES: tuple[dict[str, str], ...] = (
    {
        "id": "B1",
        "topic": "recorded deviation",
        "clause": (
            "the sealed protocol records the serving hardware as 'one NVIDIA A100, one at a "
            "time'. This bridge runs at most two A100 workers at the same time. That is an "
            "operational deviation from that sentence and is recorded here before any bridge "
            "output exists."
        ),
    },
    {
        "id": "B2",
        "topic": "one A100 per model process",
        "clause": (
            "each worker owns exactly one A100 and one model process. Tensor parallelism stays "
            "forbidden and tensor_parallel_size stays 1. No call is ever split across GPUs."
        ),
    },
    {
        "id": "B3",
        "topic": "block ownership",
        "clause": (
            "a frozen execution block is never split across workers. One worker owns one block "
            "for the whole life of that block, so the fresh serving session per block and the "
            "ascending execution_order_key inside the block are both preserved."
        ),
    },
    {
        "id": "B4",
        "topic": "budget accounting",
        "clause": (
            "concurrency shortens the calendar and never the budget. Metered A100 hours are "
            "summed over workers, so two concurrent single-A100 workers cost two A100 hours per "
            "wall-clock hour. The 150-hour stop is applied to that sum and is checked "
            "conservatively, counting the work both workers may still bill."
        ),
    },
    {
        "id": "B5",
        "topic": "timing evidence",
        "clause": (
            "throughput measured under two concurrent workers is not the single-worker rate of "
            "the sealed forecast. Every reforecast written by this bridge records the worker "
            "count that produced it, and per-condition rates are reported as measured, never "
            "rescaled to a one-worker equivalent."
        ),
    },
    {
        "id": "B6",
        "topic": "scientific separation",
        "clause": (
            "no scientific field changes. The worker count is not a treatment, is not recorded "
            "as a condition, and never enters an estimand, a test, or a claim gate. The frozen "
            "serial-serving control keeps its own definition and is not affected."
        ),
    },
)

UNCHANGED = (
    "the sealed Step 0 freeze, which stays byte-identical and read-only",
    "every sample frame, item id, filler pool, and dataset revision",
    "every prompt template, response grammar, and parser rule",
    "every call id, assignment, execution block, and within-block execution order",
    "the model and tokenizer revision, and the vllm==0.28.0 engine pin",
    "every serving field: max_model_len 8192, bfloat16, tensor_parallel_size 1, "
    "gpu_memory_utilization 0.9, max_num_seqs 64, greedy decoding at seed 0",
    "the first-response policy, the one-retry rule, and the retry triggers",
    "every estimand, test, threshold, and claim gate",
)


def build(
    frozen: Frozen, stamp: str | None = None, *, max_concurrent_workers: int = 2
) -> dict[str, Any]:
    """The machine-readable amendment record."""
    if max_concurrent_workers < 1:
        raise ValueError("a worker count starts at 1")
    created = stamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "amendment_id": AMENDMENT_ID,
        "stamp_utc": created,
        "scope": "worker concurrency for the historical replication bridge only",
        "authorised_before_any_bridge_output": True,
        "amends_freeze_stamp": frozen.stamp,
        "amends_protocol_content_hash": frozen.content_hash,
        "freeze_is_unchanged": True,
        "freeze_change_rule": (
            "the sealed freeze is never edited. A change to a design field would require a new "
            "freeze directory, not this amendment."
        ),
        "deviation": {
            "field": DEVIATION_FIELD,
            "sealed_text": str(frozen.serving["hardware"]),
            "operational_practice": (
                f"at most {max_concurrent_workers} A100 workers run at the same time, one model "
                "process on one A100 each"
            ),
            "is_a_deviation": True,
            "affects_a_scientific_field": False,
        },
        "clauses": [dict(clause) for clause in CLAUSES],
        "max_concurrent_workers": max_concurrent_workers,
        "gpus_per_worker": 1,
        "models_per_worker": 1,
        "tensor_parallel_size": int(frozen.serving["tensor_parallel_size"]),
        "max_num_seqs": int(frozen.serving["max_num_seqs"]),
        "blocks_are_never_split_across_workers": True,
        "fresh_serving_session_per_block": bool(
            frozen.protocol["conditions"]["interleaving"]["fresh_serving_session_per_block"]
        ),
        "overall_ceiling_a100_hours": 150,
        "unchanged_by_this_amendment": list(UNCHANGED),
        "no_scientific_field_is_changed": True,
    }
    payload["content_hash"] = sha256_json(payload)
    return payload


def render(payload: dict[str, Any]) -> str:
    deviation = payload["deviation"]
    lines = [
        "# Protocol amendment: two concurrent A100 workers for the replication bridge",
        "",
        f"- Amendment id: `{payload['amendment_id']}`",
        f"- Stamp (UTC): `{payload['stamp_utc']}`",
        f"- Amends freeze: `{payload['amends_freeze_stamp']}`",
        f"- Protocol content hash: `{payload['amends_protocol_content_hash']}`",
        f"- Content hash: `{payload['content_hash']}`",
        "",
        "## The deviation",
        "",
        f"- Sealed field: `{deviation['field']}`",
        f"- Sealed text: \"{deviation['sealed_text']}\"",
        f"- Operational practice: {deviation['operational_practice']}",
        "",
        "This is a deviation from a recorded serving statement. It is written before any bridge",
        "output exists. It changes no scientific field.",
        "",
        "## Clauses",
        "",
        "| id | topic | clause |",
        "|---|---|---|",
    ]
    for clause in payload["clauses"]:
        lines.append(f"| {clause['id']} | {clause['topic']} | {clause['clause']} |")
    lines += ["", "## What this amendment does not change", ""]
    lines += [f"- {text}" for text in payload["unchanged_by_this_amendment"]]
    lines += [
        "",
        "## Reading the timing evidence",
        "",
        "Measured throughput under two workers is a two-worker rate. Summed A100 hours are the",
        "budget quantity and are unaffected by the worker count. Elapsed wall-clock time is",
        "roughly halved and is never reported as a cost saving.",
        "",
    ]
    return "\n".join(lines)


def write(
    frozen: Frozen,
    stamp: str | None = None,
    *,
    out_dir: Path = AMENDMENTS,
    max_concurrent_workers: int = 2,
) -> tuple[Path, dict[str, Any]]:
    """Write the amendment. A stamp is never reused, so an earlier amendment is preserved."""
    payload = build(frozen, stamp, max_concurrent_workers=max_concurrent_workers)
    base = f"{payload['stamp_utc']}-{AMENDMENT_ID}"
    write_json(out_dir / f"{base}.json", payload)
    write_text(out_dir / f"{base}.md", render(payload))
    return out_dir / f"{base}.md", payload


def existing(out_dir: Path = AMENDMENTS) -> Path | None:
    """The amendment already on disk, if the bridge recorded one in an earlier invocation."""
    found = sorted(out_dir.glob(f"*-{AMENDMENT_ID}.json"))
    return found[-1] if found else None


def ensure(
    frozen: Frozen, *, out_dir: Path = AMENDMENTS, max_concurrent_workers: int = 2
) -> tuple[Path, dict[str, Any]]:
    """Record the deviation once, then reuse it. The bridge refuses to run without it."""
    import json  # noqa: PLC0415 - only needed on the resume path

    found = existing(out_dir)
    if found is None:
        return write(frozen, out_dir=out_dir, max_concurrent_workers=max_concurrent_workers)
    payload: dict[str, Any] = json.loads(found.read_text(encoding="utf-8"))
    if payload["amends_protocol_content_hash"] != frozen.content_hash:
        raise RuntimeError(f"{found} amends a different protocol; refusing to reuse it")
    if int(payload["max_concurrent_workers"]) < max_concurrent_workers:
        raise RuntimeError(
            f"{found} permits {payload['max_concurrent_workers']} workers, "
            f"not {max_concurrent_workers}"
        )
    return found, payload
