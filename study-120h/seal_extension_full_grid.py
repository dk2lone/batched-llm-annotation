"""Seal the fixed X1R4 A8 4 x 4 grid without running a model."""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from freezeb import extension as ext  # noqa: E402
from freezeb import extension_screen as screen  # noqa: E402
from freezeb.selection import Cell, balanced_order  # noqa: E402
from step0.hashing import read_jsonl, sha256_file, sha256_json, write_json  # noqa: E402

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
SEAL = STUDY / "freeze/20260913T183353Z-X1R4"
RUNTIME_LOCK = STUDY / "runtime-lock/20260913T205646Z-X1R4"
A7 = STUDY / "admission-lock/20260914T012453Z-X1R4-A7"
M1 = STUDY / "ext-work/extension-m1/20260914T003735Z-5a8c63f2"
OUTPUT = STUDY / "full-grid-lock"

GRID_MODELS = ext.EXT_MODELS[:4]
GRID_TASKS = ext.EXT_TASKS
N_ITEMS = 8192
N_CELLS = 16


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_manifest(directory: Path, manifest_name: str = "MANIFEST.json") -> dict[str, Any]:
    manifest = load(directory / manifest_name)
    for relative, expected in manifest["files_sha256"].items():
        if sha256_file(directory / relative) != expected:
            raise ValueError(f"sealed file changed: {directory / relative}")
    if "manifest_sha256" in manifest:
        core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if sha256_json(core) != manifest["manifest_sha256"]:
            raise ValueError(f"manifest digest changed: {directory / manifest_name}")
    return manifest


def grid_cells() -> list[Cell]:
    cells = [
        Cell(model.repo, model.family, task.key, model.order, task_index)
        for model in GRID_MODELS
        for task_index, task in enumerate(GRID_TASKS, start=1)
    ]
    ordered = balanced_order(cells)
    if len(ordered) != N_CELLS:
        raise ValueError("the full grid must contain exactly sixteen cells")
    return ordered


def selected_plans() -> dict[str, Any]:
    source = load(SEAL / "B3_PLANS.json")
    wanted = {cell.key for cell in grid_cells()}
    chosen = [
        row
        for row in source["plans"]
        if row["cell_key"] in wanted and int(row["n_items"]) == N_ITEMS
    ]
    by_key = {row["cell_key"]: row for row in chosen}
    if len(chosen) != N_CELLS or set(by_key) != wanted:
        raise ValueError("the scientific seal lacks one or more full-grid schedules")
    expected = ext.b3_expected_counts(N_ITEMS)
    schedules: dict[str, Any] = {}
    for cell in grid_cells():
        row = by_key[cell.key]
        if row["counts"] != expected or row["assignments"] != 3 or row["batch_size"] != 16:
            raise ValueError(f"wrong frozen schedule shape for {cell.key}")
        schedules[cell.key] = {
            "n_items": N_ITEMS,
            "counts": row["counts"],
            "assignment_namespace": row["assignment_namespace"],
            "call_id_order_sha256": row["call_id_order_sha256"],
            "record_stream_sha256": row["record_stream_sha256"],
            "selected_plan_sha256": sha256_json(row),
        }
    return {
        "cells": schedules,
        "per_cell_counts": expected,
        "total_calls": sum(row["counts"]["total"] for row in schedules.values()),
        "source_plans_sha256": source["plans_sha256"],
    }


def frame_bindings() -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for task in GRID_TASKS:
        path = SEAL / "frame-identifiers" / task.key / "confirm.jsonl"
        rows = list(read_jsonl(path))
        ids = [str(row["doc_id"]) for row in rows]
        if len(ids) != N_ITEMS or len(set(ids)) != N_ITEMS:
            raise ValueError(f"{task.key} confirmation frame is not 8,192 unique items")
        bindings[task.key] = {
            "dataset": task.dataset,
            "dataset_revision": task.revision,
            "split": task.split,
            "n_items": N_ITEMS,
            "identifier_file_sha256": sha256_file(path),
            "ordered_doc_ids_sha256": sha256_json(ids),
            "reuse_rule": (
                "this exact ordered item frame is used by all four models and by joint, "
                "exact-context, solo, and fake-batch analyses"
            ),
        }
    return bindings


def build() -> dict[str, Any]:
    seal = screen.verify_seal(SEAL)
    runtime = screen.verify_runtime_lock(RUNTIME_LOCK, seal)
    a7_manifest = verify_manifest(A7)
    m1_inputs = load(M1 / "admission-inputs.json")
    m1_summary = load(M1 / "SUMMARY.json")
    if not m1_inputs["screen_complete"] or m1_inputs["extension_joint_outputs_seen"]:
        raise ValueError("the solo screen is incomplete or extension joint output exists")
    if not m1_summary["screen_complete"] or m1_summary["stopped"]:
        raise ValueError("the solo screen completion record is not clean")
    if m1_inputs["seal_manifest_sha256"] != seal["manifest_sha256"]:
        raise ValueError("the solo screen used a different scientific seal")
    if m1_inputs["runtime_lock_sha256"] != runtime["runtime_lock_sha256"]:
        raise ValueError("the solo screen used a different runtime lock")

    cells = grid_cells()
    plans = selected_plans()
    frames = frame_bindings()
    counts = ext.b3_expected_counts(N_ITEMS)
    prior_budget = load(A7 / "ADMISSION_LOCK.json")["budget_lock"]
    families = [model.family for model in GRID_MODELS]
    tasks = [task.key for task in GRID_TASKS]
    first_round = [cell.key for cell in cells[:4]]
    if len({cell.family for cell in cells[:4]}) != 4 or len({cell.task for cell in cells[:4]}) != 4:
        raise ValueError("the first checkpoint round must cover every model and task once")

    return {
        "schema_version": "ext-x1r4-a8-full-grid-lock-v1",
        "status": "scientific_schedule_sealed_budget_release_pending",
        "scientific_model_outputs_used": False,
        "supersedes_for_a8_execution": str(A7.relative_to(ROOT)),
        "historical_a7_preserved": True,
        "models": [
            {"model_id": model.repo, "family": model.family, "revision": model.revision}
            for model in GRID_MODELS
        ],
        "datasets": frames,
        "design": {
            "n_models": 4,
            "n_tasks": 4,
            "n_cells": N_CELLS,
            "items_per_cell": N_ITEMS,
            "batch_size": 16,
            "assignments": 3,
            "conditions": ["joint", "exact_context", "solo"],
            "fake_batch_control": (
                "regroup resolved exact-context and solo outputs using the frozen assignments; "
                "this is an analysis control and creates no additional model calls"
            ),
            "accuracy_and_parse_role": "reported moderators only; never eligibility gates",
            "phi_status": (
                "excluded for severe output-format failure before any extension joint output"
            ),
        },
        "ordered_cells": [cell.key for cell in cells],
        "execution_rounds": [
            {
                "round": index + 1,
                "cells": [cell.key for cell in cells[index * 4 : (index + 1) * 4]],
            }
            for index in range(4)
        ],
        "first_runtime_checkpoint": first_round,
        "selected_schedules": plans,
        "call_totals": {
            "per_cell": counts,
            "full_grid": {key: value * N_CELLS for key, value in counts.items()},
        },
        "analysis_lock": {
            "primary_estimand": (
                "within each cell, mean across three assignments of joint error covariance "
                "minus exact-context error covariance"
            ),
            "primary_malformed_treatment": "unresolved_as_correct",
            "malformed_treatments": [
                "unresolved_as_correct",
                "unresolved_as_error",
                "unresolved_dropped",
                "whole_call_discarded",
            ],
            "cell_tests": "4,999 one-sided design-respecting replay draws per cell",
            "multiplicity": "Holm correction across the fixed sixteen primary cell tests",
            "practical_floor": 0.002,
            "global": (
                "arithmetic mean of all sixteen cell effects; in this complete crossed grid this "
                "equals both equal-model and equal-task weighting"
            ),
            "margins": "four model means and four task means, each averaging four cells",
            "interaction": "cell effect - model mean - task mean + global mean",
            "leave_one_out": "report four leave-one-model-out and four leave-one-task-out means",
            "recurrence_gate": (
                "complete 16-cell grid; positive global effect with replay p <= 0.05; at least "
                "three model families and three tasks each contain a Holm-positive cell at or "
                "above 0.002; every leave-one-model-out and leave-one-task-out mean is positive; "
                "all four malformed-output effect treatments remain at or above 0.002"
            ),
            "incomplete_rule": (
                "an incomplete grid is reported as incomplete and cannot pass the recurrence gate"
            ),
            "moderators": ["reference-relative accuracy", "parse rate", "malformed-output rate"],
            "causal_scope": (
                "the exact-context contrast identifies the effect of bundled joint-output "
                "generation, not output-history alone"
            ),
        },
        "power_lock": {
            "items": N_ITEMS,
            "planning_error_rate": 0.50,
            "planning_alpha": 0.0025,
            "practical_effect": 0.004,
            "simulations": 6000,
            "estimated_power": 0.9726666666666667,
            "wilson_lower_bound": 0.9682,
            "choice_rule": (
                "fixed conservative maximum for every cell; solo accuracy and observed joint "
                "results do not alter the item count"
            ),
        },
        "budget_lock": {
            "meter_origin_a100_hours": prior_budget["meter_origin_a100_hours"],
            "hard_stop_a100_hours": prior_budget["hard_stop_a100_hours"],
            "protected_reserve_a100_hours": prior_budget["protected_reserve_a100_hours"],
            "complete_block_guard_multiplier": prior_budget["complete_block_guard_multiplier"],
            "conservative_prior_use_including_proxy_a100_hours": prior_budget[
                "conservative_prior_use_including_proxy_a100_hours"
            ],
            "start_formula": "prior + 1.6 * complete_block_base_forecast + 10 <= 80",
            "gpu_start_authorized_by_this_lock": False,
            "reason": "a measured full-grid block forecast has not been sealed",
            "release_requirement": (
                "before paid calls, seal a technical-only base forecast that clears the start "
                "formula; after each model's first 1,000 A8 attempts, replace it with a measured "
                "reforecast before releasing the rest of that model's cells"
            ),
            "outcome_blinding": (
                "cost and technical failure may stop a later complete block; predictions, "
                "accuracy, covariance, and p-values may not change order or item count"
            ),
        },
        "source_hashes": {
            "scientific_seal_manifest_sha256": seal["manifest_sha256"],
            "runtime_access_lock_manifest_sha256": runtime["manifest_sha256"],
            "historical_a7_manifest_sha256": a7_manifest["manifest_sha256"],
            "b3_plans_file_sha256": sha256_file(SEAL / "B3_PLANS.json"),
            "power_file_sha256": sha256_file(SEAL / "cpu-audits/power-x1.json"),
            "m1_admission_inputs_file_sha256": sha256_file(M1 / "admission-inputs.json"),
            "m1_summary_file_sha256": sha256_file(M1 / "SUMMARY.json"),
            "seal_script_file_sha256": sha256_file(Path(__file__)),
        },
        "claim_dimensions": {"model_families": families, "tasks": tasks},
    }


def report(payload: dict[str, Any]) -> str:
    totals = payload["call_totals"]["full_grid"]
    return "\n".join(
        [
            "# X1R4 A8 full-grid lock",
            "",
            "The CPU-only lock fixes all sixteen combinations of four models and four datasets at "
            "8,192 items per cell. It used no extension joint output and launched no GPU work.",
            "",
            f"The grid contains {totals['joint']:,} joint calls, "
            f"{totals['exact_context']:,} exact-context calls, and {totals['solo']:,} solo "
            f"calls, for {totals['total']:,} calls total. The same ordered 8,192 dataset items "
            "are used across all four models and all conditions.",
            "",
            "The primary analysis is joint-minus-exact error covariance over the complete grid, "
            "with 4,999 replay draws, sixteen-cell Holm correction, equal model/task weight, "
            "model and task margins, interaction terms, leave-one-out checks, and four frozen "
            "malformed-output treatments. Accuracy and parse rate cannot remove a cell.",
            "",
            "This lock does not authorize a GPU start. A separate technical-only forecast must "
            "clear prior + 1.6 × complete-block base forecast + 10 <= 80, followed by a measured "
            "reforecast after each model's first 1,000 A8 attempts. Scientific outcomes cannot "
            "alter the fixed order or 8,192-item count.",
            "",
        ]
    )


def seal() -> Path:
    payload = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = OUTPUT / f"{stamp}-X1R4-A8"
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".a8-stage-", dir=OUTPUT) as temporary:
        stage = Path(temporary)
        write_json(stage / "FULL_GRID_LOCK.json", payload)
        (stage / "REPORT.md").write_text(report(payload), encoding="utf-8")
        manifest = {
            "schema_version": "ext-x1r4-a8-full-grid-manifest-v1",
            "status": "sealed",
            "scientific_model_outputs": False,
            "gpu_work_launched": False,
            "gpu_start_authorized": False,
            "files_sha256": {
                "FULL_GRID_LOCK.json": sha256_file(stage / "FULL_GRID_LOCK.json"),
                "REPORT.md": sha256_file(stage / "REPORT.md"),
            },
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        write_json(stage / "MANIFEST.json", manifest)
        stage.rename(destination)
    return destination


if __name__ == "__main__":
    print(seal())
