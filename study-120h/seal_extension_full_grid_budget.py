"""Seal the technical-only A8 full-grid forecast without starting a GPU."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import statistics
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
RUNTIME_LOCK = STUDY / "runtime-lock/20260913T205646Z-X1R4"
ANCHOR = STUDY / "e2-out/qwen3-8b--multi_nli"
OUTPUT = STUDY / "budget-release"
ANCHOR_CELL = "Qwen/Qwen3-8B::multi_nli"
ANCHOR_ITEMS = 4096
TARGET_ITEMS = 8192
FIRST_REFORECAST_ATTEMPTS_PER_MODEL = 1000


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


def runtime_path(model_id: str) -> Path:
    return RUNTIME_LOCK / "evidence/runtime" / f"{model_id.replace('/', '--')}.json"


def median_measurement(evidence: dict[str, Any], case_id: str) -> float:
    values = [
        float(row["latency_seconds"])
        for row in evidence["events"]
        if row.get("phase") == "measure"
        and row.get("status") == "ok"
        and row.get("case_id") == case_id
    ]
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"runtime evidence needs two positive measurements for {case_id}")
    return float(statistics.median(values))


def work_units(evidence: dict[str, Any], task_id: str, counts: dict[str, int]) -> dict[str, Any]:
    latencies = {
        "joint": median_measurement(evidence, f"{task_id}::size16::joint"),
        "exact_context": median_measurement(evidence, f"{task_id}::size16::exact_context"),
        "solo": median_measurement(evidence, f"{task_id}::short::joint"),
    }
    weighted = {
        condition: latencies[condition] * int(counts[condition])
        for condition in ("joint", "exact_context", "solo")
    }
    return {
        "median_latency_seconds": latencies,
        "serial_latency_weight": weighted,
        "total_serial_latency_weight": sum(weighted.values()),
    }


def anchor_measurement() -> dict[str, Any]:
    plan = load(ANCHOR / "plan.json")
    summary = load(ANCHOR / "SUMMARY.json")
    final = load(ANCHOR / "reforecast-final.json")
    expected = {"joint": 768, "exact_context": 12288, "solo": 4096}
    if plan["cell_key"] != ANCHOR_CELL or plan["calls_by_condition"] != expected:
        raise ValueError("the timing anchor is not the expected 4,096-item B3 cell")
    complete = summary["completeness"]
    if (
        summary["failure"] is not None
        or not complete["complete"]
        or complete["n_first_attempts"] != plan["total_calls"]
        or complete["n_retry_attempts"] != 0
    ):
        raise ValueError("the timing anchor is not a clean complete cell")
    if not final["technical_only"] or final["real_attempts_used"] != plan["total_calls"]:
        raise ValueError("the final anchor reforecast is incomplete or not technical-only")
    model = ANCHOR_CELL.split("::", 1)[0]
    if summary["checkpoint_verification"]["model_id"] != model:
        raise ValueError("the anchor checkpoint identity changed")
    hours = float(summary["meter"]["a100_hours_total"])
    if hours <= 0:
        raise ValueError("the anchor must contain positive all-worker A100 time")
    return {
        "cell_key": ANCHOR_CELL,
        "items": ANCHOR_ITEMS,
        "calls": plan["total_calls"],
        "calls_by_condition": expected,
        "all_worker_a100_hours": hours,
        "workers": plan["worker_count"],
        "execution_blocks": plan["execution_blocks"],
        "retries": complete["n_retry_attempts"],
        "source_hashes": {
            "plan_sha256": sha256_file(ANCHOR / "plan.json"),
            "summary_sha256": sha256_file(ANCHOR / "SUMMARY.json"),
            "final_reforecast_sha256": sha256_file(ANCHOR / "reforecast-final.json"),
        },
    }


def budget_gate(
    prior_hours: float,
    base_forecast_hours: float,
    *,
    multiplier: float,
    reserve_hours: float,
    stop_hours: float,
) -> dict[str, Any]:
    values = (prior_hours, base_forecast_hours, multiplier, reserve_hours, stop_hours)
    if any(value < 0 for value in values) or multiplier < 1:
        raise ValueError("budget inputs must be nonnegative and the guard must be at least one")
    guarded = multiplier * base_forecast_hours
    projected = prior_hours + guarded + reserve_hours
    return {
        "passed": projected <= stop_hours,
        "prior_a100_hours": prior_hours,
        "base_forecast_a100_hours": base_forecast_hours,
        "guard_multiplier": multiplier,
        "guarded_forecast_a100_hours": guarded,
        "protected_reserve_a100_hours": reserve_hours,
        "projected_total_including_reserve_a100_hours": projected,
        "stop_a100_hours": stop_hours,
        "headroom_after_guard_and_reserve_a100_hours": stop_hours - projected,
    }


def build() -> dict[str, Any]:
    grid_manifest = a8.verify_manifest(FULL_GRID)
    runtime_manifest = a8.verify_manifest(RUNTIME_LOCK)
    grid = load(FULL_GRID / "FULL_GRID_LOCK.json")
    if grid["status"] != "scientific_schedule_sealed_budget_release_pending":
        raise ValueError("the active full-grid lock is not waiting for a budget release")
    if grid_manifest["scientific_model_outputs"] or grid_manifest["gpu_work_launched"]:
        raise ValueError("the full-grid lock is not CPU-only")
    if runtime_manifest["status"] != "sealed":
        raise ValueError("the synthetic runtime lock is not sealed")

    anchor = anchor_measurement()
    counts = grid["call_totals"]["per_cell"]
    runtime_report = load(RUNTIME_LOCK / "evidence/runtime/report.json")
    runtime_status = {row["model_id"]: row["status"] for row in runtime_report["models"]}
    evidence_by_model = {
        row["model_id"]: load(runtime_path(row["model_id"])) for row in grid["models"]
    }
    raw_units: dict[str, dict[str, Any]] = {}
    for cell_key in grid["ordered_cells"]:
        model_id, task_id = cell_key.split("::", 1)
        evidence = evidence_by_model[model_id]
        if evidence["model_id"] != model_id or runtime_status.get(model_id) != "passed":
            raise ValueError(f"invalid runtime evidence for {model_id}")
        raw_units[cell_key] = work_units(evidence, task_id, counts)
    anchor_units = raw_units[ANCHOR_CELL]["total_serial_latency_weight"]
    if anchor_units <= 0:
        raise ValueError("the anchor runtime weight must be positive")

    scaled_anchor_hours = anchor["all_worker_a100_hours"] * TARGET_ITEMS / ANCHOR_ITEMS
    cell_forecasts: dict[str, Any] = {}
    for cell_key in grid["ordered_cells"]:
        relative_weight = raw_units[cell_key]["total_serial_latency_weight"] / anchor_units
        cell_forecasts[cell_key] = {
            **raw_units[cell_key],
            "relative_to_qwen_multinli": relative_weight,
            "base_forecast_a100_hours": scaled_anchor_hours * relative_weight,
        }
    full_base = sum(row["base_forecast_a100_hours"] for row in cell_forecasts.values())

    budget = grid["budget_lock"]
    gate = budget_gate(
        float(budget["conservative_prior_use_including_proxy_a100_hours"]),
        full_base,
        multiplier=float(budget["complete_block_guard_multiplier"]),
        reserve_hours=float(budget["protected_reserve_a100_hours"]),
        stop_hours=float(budget["hard_stop_a100_hours"]),
    )
    if not gate["passed"]:
        decision = "blocked_full_grid_forecast_exceeds_stop"
        released_attempts = 0
    else:
        decision = "pass_release_first_1000_attempts_per_model_only"
        released_attempts = FIRST_REFORECAST_ATTEMPTS_PER_MODEL * len(grid["models"])

    return {
        "schema_version": "ext-x1r4-a8-budget-release-v1",
        "status": decision,
        "technical_only": True,
        "scientific_outputs_inspected": False,
        "gpu_work_launched": False,
        "forecast_method": {
            "absolute_anchor": (
                "completed 4,096-item Qwen/MultiNLI B3 all-worker meter, scaled linearly to "
                "8,192 items"
            ),
            "relative_weights": (
                "for each model-task cell, frozen median synthetic latencies for size-16 joint, "
                "size-16 exact-context, and short solo calls weighted by the fixed A8 call counts"
            ),
            "normalization_cell": ANCHOR_CELL,
            "conservatism": (
                "the anchor includes four worker model loads and is doubled in full; the separate "
                "1.6x budget guard remains applied"
            ),
            "limitations": [
                "relative synthetic latency may not reproduce full-frame production throughput",
                "linear scaling assumes the same average per-item workload from 4,096 to 8,192",
                "the mandatory production reforecast replaces this forecast after 1,000 attempts",
            ],
        },
        "anchor": anchor,
        "target": {
            "items_per_cell": TARGET_ITEMS,
            "n_cells": len(grid["ordered_cells"]),
            "total_calls": grid["call_totals"]["full_grid"]["total"],
            "scaled_anchor_a100_hours_at_8192": scaled_anchor_hours,
        },
        "cell_forecasts": cell_forecasts,
        "budget_gate": gate,
        "release": {
            "budget_gate_passed": gate["passed"],
            "paid_run_authorized_by_user": False,
            "released_attempts": released_attempts,
            "released_scope": (
                "the first 1,000 scheduled A8 attempts for each of the four models; no remaining "
                "attempt may start until that model's technical-only measured reforecast passes"
                if gate["passed"]
                else "none"
            ),
            "reforecast_rule": (
                "after each model reaches 1,000 A8 attempts, recompute its remaining complete "
                "blocks from all metered attempts and apply the same 1.6x guard and 10-hour reserve"
            ),
            "outcome_blinding": (
                "do not inspect predictions, accuracy, covariance, or p-values when deciding "
                "whether later blocks fit"
            ),
        },
        "source_hashes": {
            "full_grid_manifest_sha256": grid_manifest["manifest_sha256"],
            "full_grid_lock_sha256": sha256_file(FULL_GRID / "FULL_GRID_LOCK.json"),
            "runtime_lock_manifest_sha256": sha256_file(RUNTIME_LOCK / "MANIFEST.json"),
            "budget_script_sha256": sha256_file(Path(__file__)),
            **{
                f"runtime_{model_id.replace('/', '--')}_sha256": sha256_file(runtime_path(model_id))
                for model_id in evidence_by_model
            },
        },
    }


def report(payload: dict[str, Any]) -> str:
    gate = payload["budget_gate"]
    release = payload["release"]
    verdict = "PASS" if gate["passed"] else "BLOCKED"
    return "\n".join(
        [
            "# X1R4 A8 technical budget release",
            "",
            f"Budget verdict: **{verdict}**. This CPU-only forecast inspected timing and "
            "completion records, not model predictions or scientific results, and launched no GPU.",
            "",
            f"The sixteen-cell base forecast is {gate['base_forecast_a100_hours']:.3f} "
            f"A100-hours. After the 1.6x guard, {gate['prior_a100_hours']:.3f} prior hours, and "
            f"the {gate['protected_reserve_a100_hours']:.0f}-hour reserve, the projected total is "
            f"{gate['projected_total_including_reserve_a100_hours']:.3f} of "
            f"{gate['stop_a100_hours']:.0f} A100-hours. Guarded headroom is "
            f"{gate['headroom_after_guard_and_reserve_a100_hours']:.3f} hours.",
            "",
            f"Released scope: {release['released_scope']}. Paid execution still requires "
            "Daniel's separate authorization.",
            "",
            "The forecast uses the completed Qwen/MultiNLI B3 meter as its absolute anchor and "
            "sealed synthetic runtime measurements only as relative model-task weights. Production "
            "timing after 1,000 attempts replaces this forecast before any further release.",
            "",
        ]
    )


def seal() -> Path:
    payload = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = OUTPUT / f"{stamp}-X1R4-A8-budget"
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".a8-budget-stage-", dir=OUTPUT) as temporary:
        stage = Path(temporary)
        write_json(stage / "BUDGET_RELEASE.json", payload)
        (stage / "REPORT.md").write_text(report(payload), encoding="utf-8")
        manifest = {
            "schema_version": "ext-x1r4-a8-budget-manifest-v1",
            "status": "sealed",
            "budget_gate_passed": payload["budget_gate"]["passed"],
            "gpu_work_launched": False,
            "scientific_outputs_inspected": False,
            "files_sha256": {
                "BUDGET_RELEASE.json": sha256_file(stage / "BUDGET_RELEASE.json"),
                "REPORT.md": sha256_file(stage / "REPORT.md"),
            },
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        write_json(stage / "MANIFEST.json", manifest)
        stage.rename(destination)
    return destination


if __name__ == "__main__":
    print(seal())
