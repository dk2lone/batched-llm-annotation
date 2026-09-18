"""Build every CPU-only B5 preparation artifact. No Pod, no model, no paid work."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb import b5, b5_power  # noqa: E402
from freezeb import extension as ext  # noqa: E402
from freezeb import extension_b4 as b4  # noqa: E402
from step0.hashing import read_jsonl, sha256_file, sha256_json, write_json  # noqa: E402

A4_TABLE = "decision-audit/20260918T054237Z-a1-a7/A4_TABLE.csv"
B3_ROOT = "extension-b3"
B4_ROOT = "extension-b4"
B4_CHUNK_CALLS = 32
DEFAULT_USD_PER_HOUR = 1.59
DEFAULT_BOOT_HOURS = 0.25
DEFAULT_RETRY_FRACTION = 0.02
CEILING_HOURS = 35.0
RESERVE_USD = 2.00


def b3_ledger(cell_key: str) -> Path:
    model_id, task_id = b5.split_cell(cell_key)
    return STUDY / B3_ROOT / b5.model_key(model_id) / "cells" / task_id / "events.jsonl"


def token_evidence(cell_key: str, doc_ids: list[str]) -> dict[str, Any]:
    """Derive the per-cell token model from the sealed B3 ledger of the same cell.

    Every count below was produced by that model's pinned tokenizer during the sealed
    B3 run. Writing `solo = S + t_j`, `joint_k = A + k x + sum t_j` and
    `exact_k = E + sum t_j`, the measured offsets identify `A`, `x`, `E`, `S` and the
    frame total `T = sum t_j` without any new tokenizer call.
    """
    path = b3_ledger(cell_key)
    solo: dict[str, int] = {}
    solo_output: list[int] = []
    joint_offsets: list[int] = []
    joint_output: list[int] = []
    exact_offsets: list[int] = []
    exact_output: list[int] = []
    pending: list[tuple[str, list[str], int, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if int(event["attempt"]) != 1:
                continue
            order = [str(value) for value in event["doc_order"]]
            prompt = int(event["prompt_tokens"] or 0)
            output = int(event["output_tokens"] or 0)
            condition = str(event["condition"])
            if condition == "solo":
                solo[order[0]] = prompt
                solo_output.append(output)
            else:
                pending.append((condition, order, prompt, output))
    if len(solo) != b5.N_ITEMS or set(solo) != set(doc_ids):
        raise ValueError(f"{cell_key} B3 solo arm does not cover the confirmation frame")
    for condition, order, prompt, output in pending:
        offset = prompt - sum(solo[doc_id] for doc_id in order)
        if condition == "joint":
            joint_offsets.append(offset)
            joint_output.append(output)
        else:
            exact_offsets.append(offset)
            exact_output.append(output)
    if not joint_offsets or not exact_offsets:
        raise ValueError(f"{cell_key} B3 ledger lacks a joint or exact-context arm")
    joint_offset = max(joint_offsets)
    joint_template = -joint_offset / (ext.BATCH_SIZE - 1)
    solo_output_mean = sum(solo_output) / len(solo_output)
    per_target = (sum(joint_output) / len(joint_output) - solo_output_mean) / (ext.BATCH_SIZE - 1)
    single_constant = joint_template + per_target
    exact_offset_mean = sum(exact_offsets) / len(exact_offsets)
    return {
        "cell_key": cell_key,
        "source_ledger": str(path.relative_to(STUDY.parent)),
        "source_ledger_sha256": sha256_file(path),
        "b3_batch_size": ext.BATCH_SIZE,
        "solo_prompt_tokens": solo,
        "solo_prompt_total": sum(solo.values()),
        "exact_offset_max": max(exact_offsets),
        "exact_offset_mean": exact_offset_mean,
        "joint_offset_distinct": len(set(joint_offsets)),
        "joint_template_tokens": joint_template,
        "target_tokens": per_target,
        "single_document_constant_tokens": single_constant,
        "exact_template_tokens": exact_offset_mean + ext.BATCH_SIZE * single_constant,
        "frame_document_tokens": sum(solo.values()) - b5.N_ITEMS * single_constant,
        "joint_response_frame_tokens": solo_output_mean - per_target,
        "exact_response_tokens": sum(exact_output) / len(exact_output),
        "solo_response_tokens": solo_output_mean,
    }


def runtime_evidence() -> dict[str, Any]:
    """Measured metered A100 PCIe hours and call shapes of the completed B4 lanes."""
    hours: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    prompt: dict[str, int] = defaultdict(int)
    output: dict[str, int] = defaultdict(int)
    runs: list[str] = []
    for directory in sorted((STUDY / B4_ROOT).glob("runpod*")):
        meter = directory / "meter.jsonl"
        receipts = directory / "receipts"
        if not meter.is_file() or not receipts.is_dir():
            continue
        settled = [row for row in read_jsonl(meter) if row.get("kind") == "settled"]
        starts = {str(row["phase_id"]): row for row in read_jsonl(meter) if row["kind"] == "start"}
        if not settled:
            continue
        resume = directory / "resume-record.json"
        imported = (
            set(cast(dict[str, str], json.loads(resume.read_text())["receipt_sha256"]))
            if resume.is_file()
            else set()
        )
        for row in settled:
            start = starts.get(str(row["phase_id"]))
            if start is not None:
                hours[str(start["cell_key"])] += float(row["charged_hours"])
        fresh = 0
        for path in sorted(receipts.glob("*.json")):
            if path.name in imported:
                continue
            payload = json.loads(path.read_text())
            fresh += 1
            for event in payload["events"]:
                cell_key = str(event["cell_key"])
                calls[cell_key] += 1
                prompt[cell_key] += int(event.get("prompt_tokens") or 0)
                output[cell_key] += int(event.get("output_tokens") or 0)
        runs.append(f"{directory.name}:{fresh}")
    anchors = []
    for cell_key in b5.CELLS:
        if not calls[cell_key] or hours[cell_key] <= 0:
            raise ValueError(f"{cell_key} has no completed B4 runtime evidence")
        anchors.append(
            {
                "cell_key": cell_key,
                "calls": calls[cell_key],
                "metered_hours": hours[cell_key],
                "mean_prompt_tokens": prompt[cell_key] / calls[cell_key],
                "mean_output_tokens": output[cell_key] / calls[cell_key],
                "chunk_calls": B4_CHUNK_CALLS,
                "seconds_per_call": hours[cell_key] * 3600 / calls[cell_key],
            }
        )
    return {
        "schema_version": "b5-runtime-evidence-v1",
        "source": "completed B4 Runpod A100 PCIe 80GB meters and durable receipts",
        "gpu_type_id": "NVIDIA A100 80GB PCIe",
        "contributing_runs": runs,
        "b4_anchors": anchors,
        "total_metered_hours": float(sum(row["metered_hours"] for row in anchors)),
        "total_calls": int(sum(int(row["calls"]) for row in anchors)),
    }


def planning_inputs() -> list[dict[str, Any]]:
    """Completed B3 marginal error and standardized dependence, planning use only."""
    rows = {
        str(row["cell"]): row for row in csv.DictReader((STUDY / A4_TABLE).open(encoding="utf-8"))
    }
    inputs = []
    for cell_key in b5.CELLS:
        row = rows[cell_key]
        inputs.append(
            {
                "cell_key": cell_key,
                "marginal_error": float(row["joint_marginal_error"]),
                "exact_standardized": float(row["exact_context_standardized"]),
                "observed_b3_standardized_contrast": float(row["standardized_joint_minus_exact"]),
                "observed_b3_raw_contrast": float(row["joint_minus_exact"]),
            }
        )
    return inputs


def build(output: Path, *, usd_per_hour: float, replicates: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frames: dict[str, list[dict[str, Any]]] = {}
    tokens: dict[str, dict[str, Any]] = {}
    digests: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    for cell_key in b5.CELLS:
        task_id = b5.split_cell(cell_key)[1]
        if task_id not in frames:
            frames[task_id] = b4.frame(STUDY, task_id)
        rows = frames[task_id]
        doc_ids = [str(row["doc_id"]) for row in rows]
        frame_records.append(
            {
                "cell_key": cell_key,
                "task_id": task_id,
                "items": len(rows),
                "ordered_doc_ids_sha256": sha256_json(doc_ids),
                "frame_file": f"ext-work/20260913-cpu-draft/{task_id}/confirm.jsonl",
                "frame_file_sha256": sha256_file(
                    STUDY / "ext-work/20260913-cpu-draft" / task_id / "confirm.jsonl"
                ),
                "sealed_identifier_sha256": sha256_file(
                    STUDY / b4.SEAL / "frame-identifiers" / task_id / "confirm.jsonl"
                ),
            }
        )
        tokens[cell_key] = token_evidence(cell_key, doc_ids)
        digests.append(b5.cell_digest(cell_key, rows))
        audits.append(b5.context_audit(cell_key, doc_ids, tokens[cell_key]))
    counts = {
        "joint": sum(int(row["counts"]["joint"]) for row in digests),
        "exact_context": sum(int(row["counts"]["exact_context"]) for row in digests),
        "total": sum(int(row["counts"]["total"]) for row in digests),
    }
    b5.verify_counts(counts)
    if not all(bool(row["fits"]) for row in audits):
        raise ValueError("a scheduled B5 prompt does not fit the sealed context limit")
    runtime = runtime_evidence()
    estimate = b5.forecast(
        tokens,
        runtime,
        usd_per_hour=usd_per_hour,
        boot_hours_per_model=DEFAULT_BOOT_HOURS,
        retry_fraction=DEFAULT_RETRY_FRACTION,
    )
    planning = planning_inputs()
    power = b5_power.finalize(b5_power.simulate(planning, n_replicates=replicates))
    schedule = {
        "schema_version": b5.SCHEMA_VERSION,
        "contract_version": b5.CONTRACT_VERSION,
        "built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "cells": list(b5.CELLS),
        "sizes": list(b5.SIZES),
        "assignments": b5.N_ASSIGNMENTS,
        "items_per_cell": b5.N_ITEMS,
        "expected_counts": b5.expected_counts(),
        "observed_counts": counts,
        "frames": frame_records,
        "nesting": [b5.nesting_report(cell_key) for cell_key in b5.CELLS],
        "cell_digests": digests,
        "chunk_calls": b5.CHUNK_CALLS,
        "prompt_rule_sha256": sha256_file(STUDY / "src/freezeb/prompts.py"),
        "parser_rule_sha256": sha256_file(STUDY / "src/freezeb/labels.py"),
        "schedule_rule_sha256": sha256_file(STUDY / "src/freezeb/b5.py"),
    }
    schedule["schedule_sha256"] = sha256_json(schedule)
    written = {
        "SCHEDULE_DIGEST.json": schedule,
        "TOKEN_EVIDENCE.json": {"cells": tokens},
        "RUNTIME_EVIDENCE.json": runtime,
        "CONTEXT_AUDIT.json": {"cells": audits},
        "FORECAST.json": {
            **estimate,
            "ceiling_metered_a100_hours": CEILING_HOURS,
            "reserve_usd": RESERVE_USD,
            "planning_point_balance_usd": b5.required_balance_usd(
                estimate, CEILING_HOURS, RESERVE_USD
            ),
            "full_launch_required_balance_usd": None,
            "full_launch_requires_smoke_calibration": True,
            "smoke_budget_metered_a100_hours": 2.0,
            "smoke_required_balance_usd": 2.0 * usd_per_hour + RESERVE_USD,
            "point_fits_ceiling": estimate["point_metered_a100_hours"] <= CEILING_HOURS,
            "upper_fits_ceiling": estimate["upper_metered_a100_hours"] <= CEILING_HOURS,
        },
        "POWER.json": power,
    }
    for name, payload in written.items():
        write_json(output / name, payload)
    manifest = {
        "schema_version": "b5-preparation-manifest-v1",
        "built_at_utc": schedule["built_at_utc"],
        "files_sha256": {name: sha256_file(output / name) for name in sorted(written)},
        "schedule_sha256": schedule["schedule_sha256"],
        "gpu_work_launched": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output / "MANIFEST.json", manifest)
    return {
        "output": str(output),
        "counts": counts,
        "point_metered_a100_hours": estimate["point_metered_a100_hours"],
        "upper_metered_a100_hours": estimate["upper_metered_a100_hours"],
        "full_launch_required_balance_usd": None,
        "smoke_required_balance_usd": written["FORECAST.json"]["smoke_required_balance_usd"],
        "context_fits": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "gpu_work_launched": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usd-per-hour", type=float, default=DEFAULT_USD_PER_HOUR)
    parser.add_argument("--replicates", type=int, default=b5_power.N_REPLICATES)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.output, usd_per_hour=args.usd_per_hour, replicates=args.replicates), indent=2
        )
    )


if __name__ == "__main__":
    main()
