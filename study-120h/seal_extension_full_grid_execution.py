"""Release the complete A8 grid without a compute-cost stopping rule."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from step0.hashing import sha256_file, sha256_json, write_json

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
FULL_GRID = STUDY / "full-grid-lock/20260914T032937Z-X1R4-A8"
BUDGET_RELEASE = STUDY / "budget-release/20260914T033645Z-X1R4-A8-budget"
PRIOR_EXECUTION = STUDY / "execution-release/20260914T034111Z-X1R4-A8-complete"
OUTPUT = STUDY / "execution-release"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a8 = _load_module(STUDY / "seal_extension_full_grid.py", "seal_extension_full_grid")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build() -> dict[str, Any]:
    grid_manifest = a8.verify_manifest(FULL_GRID)
    budget_manifest = a8.verify_manifest(BUDGET_RELEASE)
    prior_execution_manifest = a8.verify_manifest(PRIOR_EXECUTION)
    grid = load(FULL_GRID / "FULL_GRID_LOCK.json")
    budget = load(BUDGET_RELEASE / "BUDGET_RELEASE.json")
    if grid["design"]["n_cells"] != 16 or grid["call_totals"]["full_grid"]["total"] != 548864:
        raise ValueError("the active A8 lock is not the fixed sixteen-cell grid")
    if not budget["budget_gate"]["passed"]:
        raise ValueError("the historical budget release did not pass")
    if grid_manifest["gpu_work_launched"] or budget_manifest["gpu_work_launched"]:
        raise ValueError("a source lock unexpectedly records GPU work")
    task_order = list(grid["claim_dimensions"]["tasks"])
    ordered_cells = [
        f"{model['model_id']}::{task_id}" for model in grid["models"] for task_id in task_order
    ]
    if len(ordered_cells) != 16 or set(ordered_cells) != set(grid["ordered_cells"]):
        raise ValueError("the model-major order does not match the sealed scientific grid")
    calls_per_model = grid["call_totals"]["per_cell"]["total"] * len(task_order)
    return {
        "schema_version": "ext-x1r4-a8-model-major-execution-release-v2",
        "status": "complete_grid_released_model_by_model",
        "instruction_basis": (
            "Daniel directed that compute cost not constrain A8 and that the first-1,000-attempt "
            "pause be removed"
        ),
        "supersedes_budget_policy_only": [
            str(FULL_GRID.relative_to(ROOT)),
            str(BUDGET_RELEASE.relative_to(ROOT)),
        ],
        "supersedes_execution_order": str(PRIOR_EXECUTION.relative_to(ROOT)),
        "scientific_design_unchanged": True,
        "ordered_cells": ordered_cells,
        "model_runs": [
            {
                "model_id": model["model_id"],
                "family": model["family"],
                "cells": [f"{model['model_id']}::{task_id}" for task_id in task_order],
                "calls": calls_per_model,
            }
            for model in grid["models"]
        ],
        "items_per_cell": grid["design"]["items_per_cell"],
        "call_totals": grid["call_totals"],
        "execution_policy": {
            "released_calls": grid["call_totals"]["full_grid"]["total"],
            "released_cells": 16,
            "first_1000_attempt_pause_required": False,
            "runtime_reforecast_required": False,
            "a100_hour_stop": None,
            "protected_compute_reserve": None,
            "cost_based_block_admission": False,
            "metering_role": "descriptive reporting only",
            "run_order": (
                "run Qwen, Granite, Llama, then Mistral; within each model complete Civil "
                "Comments, MultiNLI, IMDb, then AG News"
            ),
            "scientific_outcome_rule": (
                "accuracy, parse rate, covariance, and p-values cannot remove, reorder, or resize "
                "a cell"
            ),
            "technical_failure_rule": (
                "retry under the frozen technical policy; report any irrecoverable incomplete cell"
            ),
        },
        "run_state": {
            "budget_policy_allows_complete_grid": True,
            "gpu_work_launched_by_this_release": False,
            "scientific_model_outputs_generated_by_this_release": False,
        },
        "source_hashes": {
            "full_grid_manifest_sha256": grid_manifest["manifest_sha256"],
            "full_grid_lock_sha256": sha256_file(FULL_GRID / "FULL_GRID_LOCK.json"),
            "budget_manifest_sha256": budget_manifest["manifest_sha256"],
            "budget_release_sha256": sha256_file(BUDGET_RELEASE / "BUDGET_RELEASE.json"),
            "prior_execution_manifest_sha256": prior_execution_manifest["manifest_sha256"],
            "execution_release_script_sha256": sha256_file(Path(__file__)),
        },
    }


def report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# X1R4 A8 complete execution release",
            "",
            "The active A8 policy releases the complete fixed sixteen-cell grid model by model, "
            "without a compute-cost stopping rule. Run Qwen, Granite, Llama, then Mistral. Within "
            "each model, complete Civil Comments, MultiNLI, IMDb, then AG News.",
            "",
            f"Each model run contains {payload['model_runs'][0]['calls']:,} calls across four "
            f"datasets. All {payload['call_totals']['full_grid']['total']:,} calls remain "
            "scheduled at 8,192 items per cell. Cost is recorded only for reporting, and "
            "scientific results cannot alter the grid.",
            "",
            "This CPU-only release launched no GPU work and generated no model output.",
            "",
        ]
    )


def seal() -> Path:
    payload = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = OUTPUT / f"{stamp}-X1R4-A8-model-major"
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".a8-execution-stage-", dir=OUTPUT) as temporary:
        stage = Path(temporary)
        write_json(stage / "EXECUTION_RELEASE.json", payload)
        (stage / "REPORT.md").write_text(report(payload), encoding="utf-8")
        manifest = {
            "schema_version": "ext-x1r4-a8-model-major-execution-manifest-v2",
            "status": "sealed",
            "complete_grid_released": True,
            "cost_gate_active": False,
            "gpu_work_launched": False,
            "scientific_model_outputs": False,
            "files_sha256": {
                "EXECUTION_RELEASE.json": sha256_file(stage / "EXECUTION_RELEASE.json"),
                "REPORT.md": sha256_file(stage / "REPORT.md"),
            },
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        write_json(stage / "MANIFEST.json", manifest)
        stage.rename(destination)
    return destination


if __name__ == "__main__":
    print(seal())
