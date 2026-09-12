"""The timestamped execution-budget amendment, written outside the sealed freeze.

The amendment records an authorisation the user gave before any model output existed. It
changes the execution budget and the worker layout only. It changes no sample frame, no
prompt, no threshold, no estimand, no test, and no claim gate. The Step 0 freeze stays
byte-identical and sealed; a new decision about the design would need a new freeze instead.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from step0.hashing import sha256_json, write_json, write_text

from .frozen import Frozen
from .paths import AMENDMENTS

AMENDMENT_ID = "step1-execution-budget"

CLAUSES: tuple[dict[str, str], ...] = (
    {
        "id": "A1",
        "topic": "planning target",
        "clause": "120 metered A100 hours is a planning target, not the hard limit of the study.",
    },
    {
        "id": "A2",
        "topic": "overall ceiling",
        "clause": "150 metered A100 hours is the overall ceiling. Work stops at that total.",
    },
    {
        "id": "A3",
        "topic": "Step 1 allocation",
        "clause": (
            "the 8-hour Step 1 allocation is a checkpoint, not a hard cap. Reaching it requires a "
            "report and a decision, not an automatic stop."
        ),
    },
    {
        "id": "A4",
        "topic": "concurrent workers",
        "clause": (
            "Step 1 may run independent workers concurrently, one model on one A100 per worker. "
            "Tensor parallelism stays forbidden and tensor_parallel_size stays 1."
        ),
    },
    {
        "id": "A5",
        "topic": "budget accounting",
        "clause": (
            "concurrency shortens the calendar and never the budget. Metered A100 hours are summed "
            "over workers, so two concurrent single-A100 workers cost the same total as one worker "
            "running twice as long."
        ),
    },
)

UNCHANGED = (
    "every sample frame, item id, and dataset revision",
    "every prompt template, response grammar, and parser rule",
    "every model and tokenizer revision, and the vllm==0.28.0 engine pin",
    "every serving field: max_model_len 8192, bfloat16, tensor_parallel_size 1, "
    "gpu_memory_utilization 0.9, max_num_seqs 64, greedy decoding",
    "every screen criterion, threshold, floor, estimand, test, and claim gate",
    "the discovery and confirmation separation, and the replacement and fallback policies",
)


def build(frozen: Frozen, stamp: str | None = None) -> dict[str, Any]:
    """The machine-readable amendment record."""
    created = stamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "amendment_id": AMENDMENT_ID,
        "stamp_utc": created,
        "scope": "execution budget and worker layout only",
        "authorised_before_any_model_output": True,
        "amends_freeze_stamp": frozen.stamp,
        "amends_protocol_content_hash": frozen.content_hash,
        "freeze_is_unchanged": True,
        "freeze_change_rule": (
            "the sealed freeze is never edited. A change to a design field would require a new "
            "freeze directory, not this amendment."
        ),
        "clauses": [dict(clause) for clause in CLAUSES],
        "frozen_ceilings_for_reference": dict(
            frozen.protocol["execution_ledger"]["a100_ceilings_hours"]
        ),
        "planning_target_a100_hours": 120,
        "overall_ceiling_a100_hours": 150,
        "step_1_checkpoint_a100_hours": 8,
        "step_1_checkpoint_is_a_hard_cap": False,
        "max_concurrent_workers": 2,
        "gpus_per_worker": 1,
        "models_per_worker": 1,
        "tensor_parallel_size": int(frozen.serving["tensor_parallel_size"]),
        "unchanged_by_this_amendment": list(UNCHANGED),
        "no_scientific_field_is_changed": True,
    }
    payload["content_hash"] = sha256_json(payload)
    return payload


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol amendment: Step 1 execution budget and worker layout",
        "",
        f"- Amendment id: `{payload['amendment_id']}`",
        f"- Stamp (UTC): `{payload['stamp_utc']}`",
        f"- Amends freeze: `{payload['amends_freeze_stamp']}`",
        f"- Protocol content hash: `{payload['amends_protocol_content_hash']}`",
        f"- Content hash: `{payload['content_hash']}`",
        "",
        "## Status",
        "",
        "This amendment records an authorisation the user gave before any model output existed.",
        "It changes the execution budget and the worker layout only. The Step 0 freeze is",
        "unchanged and stays sealed. No sample frame, prompt, threshold, estimand, test, or claim",
        "gate is touched. A change to any of those would require a new freeze directory, never an",
        "edit of the sealed one.",
        "",
        "## Clauses",
        "",
        "| id | topic | clause |",
        "|---|---|---|",
    ]
    for clause in payload["clauses"]:
        lines.append(f"| {clause['id']} | {clause['topic']} | {clause['clause']} |")
    lines += [
        "",
        "## What this amendment does not change",
        "",
    ]
    lines += [f"- {text}" for text in payload["unchanged_by_this_amendment"]]
    lines += [
        "",
        "## Frozen stage ceilings, for reference",
        "",
        "| step | frozen ceiling in A100 hours |",
        "|---|---:|",
    ]
    for step, hours in payload["frozen_ceilings_for_reference"].items():
        lines.append(f"| {step} | {hours} |")
    lines += [
        "",
        "The frozen table above is unchanged. Clauses A1 to A3 place it inside a 150-hour overall",
        "ceiling and make the Step 1 row a reporting checkpoint. Clause A5 keeps the accounting",
        "honest: two concurrent single-A100 workers cost two A100 hours per wall-clock hour.",
        "",
    ]
    return "\n".join(lines)


def write(
    frozen: Frozen, stamp: str | None = None, out_dir: Path = AMENDMENTS
) -> tuple[Path, dict[str, Any]]:
    """Write the amendment. A stamp is never reused, so an earlier amendment is preserved."""
    payload = build(frozen, stamp)
    base = f"{payload['stamp_utc']}-{AMENDMENT_ID}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"
    write_json(json_path, payload)
    write_text(md_path, render(payload))
    return md_path, payload
