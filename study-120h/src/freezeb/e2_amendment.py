"""Pre-output record for parallel workers in one fresh Freeze B E2 cell."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, cast

from step0.hashing import sha256_json, write_json, write_text

AMENDMENT_ID = "freezeb-e2-qwen-multinli-concurrency"


def build(
    admission_lock_sha256: str,
    *,
    workers: int,
    plan_sha256: str,
    amendment_id: str = AMENDMENT_ID,
    scope: str = "first sealed E2 cell Qwen/Qwen3-8B::multi_nli only",
    stamp: str | None = None,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("Freeze B permits one through four concurrent A100 workers")
    payload: dict[str, Any] = {
        "amendment_id": amendment_id,
        "stamp_utc": stamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        "scope": scope,
        "admission_lock_sha256": admission_lock_sha256,
        "plan_sha256": plan_sha256,
        "recorded_before_any_e2_model_output": True,
        "max_concurrent_workers": workers,
        "gpus_per_worker": 1,
        "tensor_parallel_size": 1,
        "block_owner_rule": "execution_block modulo worker_count",
        "blocks_are_never_split_between_workers": True,
        "within_block_order_is_preserved": True,
        "all_worker_a100_seconds_are_summed": True,
        "equivalence_smoke_required_before_scientific_output": True,
        "equivalence_smoke": {
            "status": "pending",
            "synthetic_only": True,
            "required_checks": [
                "same image and vLLM version",
                "same checkpoint and tokenizer revisions",
                "same deterministic serving and decoding settings",
                "same response to one synthetic request",
            ],
        },
        "scientific_design_changes": [],
    }
    payload["content_hash"] = sha256_json(payload)
    return payload


def render(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Operational amendment: Freeze B E2 worker concurrency",
            "",
            f"Date: `{payload['stamp_utc']}`",
            f"Amendment: `{payload['amendment_id']}`",
            f"Admission lock SHA256: `{payload['admission_lock_sha256']}`",
            f"Plan SHA256: `{payload['plan_sha256']}`",
            "",
            f"Scope: `{payload['scope']}`",
            "",
            "This record is written before any output for the named E2 cell. It may use",
            f"at most {payload['max_concurrent_workers']} independent one-A100 workers. Execution",
            "block ownership is fixed by block number modulo worker count; blocks are never split,",
            "and their internal call order remains unchanged.",
            "",
            "Every worker's metered A100 seconds, including model loading and the synthetic",
            "equivalence smoke, are summed. Parallelism changes elapsed time, not the study meter.",
            "Scientific output remains blocked until the smoke report verifies identical engine,",
            "checkpoint, tokenizer, serving settings, and deterministic synthetic response.",
            "",
            "This amendment changes no cell, item, assignment, prompt, call ID, analysis, or claim",
            "rule in the sealed Freeze B protocol.",
            "",
        ]
    )


def ensure(
    out_dir: Path,
    admission_lock_sha256: str,
    *,
    workers: int,
    plan_sha256: str,
    amendment_id: str = AMENDMENT_ID,
    scope: str = "first sealed E2 cell Qwen/Qwen3-8B::multi_nli only",
) -> tuple[Path, dict[str, Any]]:
    found = sorted(out_dir.glob(f"*-{amendment_id}.json"))
    if found:
        payload = cast(dict[str, Any], json.loads(found[-1].read_text(encoding="utf-8")))
        expected = (admission_lock_sha256, plan_sha256, workers)
        actual = (
            payload.get("admission_lock_sha256"),
            payload.get("plan_sha256"),
            int(payload.get("max_concurrent_workers", 0)),
        )
        if actual != expected:
            raise RuntimeError(f"{found[-1]} belongs to a different E2 execution plan")
        smoke_path = found[-1].with_suffix(".smoke.json")
        if smoke_path.exists():
            payload = {
                **payload,
                "equivalence_smoke": json.loads(smoke_path.read_text(encoding="utf-8")),
                "equivalence_smoke_report": str(smoke_path),
            }
        return found[-1], payload
    payload = build(
        admission_lock_sha256,
        workers=workers,
        plan_sha256=plan_sha256,
        amendment_id=amendment_id,
        scope=scope,
    )
    base = f"{payload['stamp_utc']}-{payload['amendment_id']}"
    write_json(out_dir / f"{base}.json", payload)
    write_text(out_dir / f"{base}.md", render(payload))
    return out_dir / f"{base}.json", payload


def complete_smoke(
    amendment_path: Path, payload: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    if report.get("status") != "passed":
        raise RuntimeError("the E2 worker-equivalence smoke did not pass")
    smoke_path = amendment_path.with_suffix(".smoke.json")
    linked_report = {
        **report,
        "amendment_id": payload["amendment_id"],
        "amendment_content_hash": payload["content_hash"],
    }
    write_json(smoke_path, linked_report)
    return {
        **payload,
        "equivalence_smoke": linked_report,
        "equivalence_smoke_report": str(smoke_path),
    }
