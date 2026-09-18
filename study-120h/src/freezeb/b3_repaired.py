"""Secondary B3 sensitivity analysis using exact linked output repairs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from step0 import stats
from step0.hashing import read_jsonl, sha256_file, sha256_json, sha256_text

from . import e2
from . import extension as ext
from . import extension_b3_analysis as b3

SCHEMA_VERSION = "ext-x1r4-a9-b3-repaired-sensitivity-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def recovery_evidence(
    study_root: Path, recovery_dir: Path, inventory_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Authenticate the complete recovery and return its B3 records/events."""
    inventory = _load(inventory_path)
    core = {key: value for key, value in inventory.items() if key != "manifest_sha256"}
    if inventory.get("manifest_sha256") != sha256_json(core):
        raise ValueError("recovery inventory manifest hash differs")
    report = _load(recovery_dir / "FULL_REPORT.json")
    if (
        report.get("inventory_sha256") != sha256_file(inventory_path)
        or report.get("complete") is not True
        or report.get("passed") is not True
        or report.get("completed_calls") != 4097
        or report.get("recovered_calls") != 4097
        or report.get("unresolved_calls") != 0
    ):
        raise ValueError("recovery completion report differs")

    events = list(read_jsonl(recovery_dir / "events.jsonl"))
    by_recovery_id = {str(event.get("recovery_id", "")): event for event in events}
    if len(events) != 4097 or len(by_recovery_id) != 4097:
        raise ValueError("recovery event ledger is incomplete or duplicated")
    receipt_events: dict[str, dict[str, Any]] = {}
    receipt_paths = sorted((recovery_dir / "receipts").glob("*.json"))
    for path in receipt_paths:
        payload = _load(path)
        if payload.get("chunk_id") != path.stem:
            raise ValueError("recovery receipt filename differs from its chunk")
        for event in payload.get("events", []):
            identity = str(event.get("recovery_id", ""))
            if identity in receipt_events:
                raise ValueError("duplicate recovery event across receipts")
            receipt_events[identity] = event
    if {key: sha256_json(value) for key, value in receipt_events.items()} != {
        key: sha256_json(value) for key, value in by_recovery_id.items()
    }:
        raise ValueError("recovery receipts do not reproduce the event ledger")

    b3_root = (study_root / "extension-b3").resolve()
    records: dict[str, dict[str, Any]] = {}
    repaired: dict[str, dict[str, Any]] = {}
    ledger_hashes: dict[Path, str] = {}
    for record in inventory.get("records", []):
        if record.get("study") != "b3":
            continue
        record_core = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != sha256_json(record_core):
            raise ValueError("B3 recovery inventory record hash differs")
        call_id = str(record["call_id"])
        if call_id in records:
            raise ValueError("duplicate B3 call in recovery inventory")
        ledger = Path(str(record["source_ledger"])).resolve()
        if not ledger.is_relative_to(b3_root):
            raise ValueError("B3 recovery source ledger is outside extension-b3")
        if ledger not in ledger_hashes:
            ledger_hashes[ledger] = sha256_file(ledger)
        ledger_digest = ledger_hashes[ledger]
        if ledger_digest != record["source_ledger_sha256"]:
            raise ValueError("B3 recovery source ledger hash differs")
        event = by_recovery_id.get(str(record["recovery_id"]))
        if event is None:
            raise ValueError("B3 recovery event is missing")
        raw = event.get("first_response")
        if not isinstance(raw, str) or event.get("output_sha256") != sha256_text(raw):
            raise ValueError("B3 recovery response hash differs")
        parsed = e2.parse_result(
            raw,
            [str(value) for value in record["target_ids"]],
            event.get("finish_reason"),
            task_id=str(record["task_id"]),
        )
        if any(event.get(key) != parsed[key] for key in ("status", "parse_status", "predictions")):
            raise ValueError("B3 recovery stored parse differs from response bytes")
        expected = {
            "recovery_id": record["recovery_id"],
            "source_attempt_id": record["source_attempt_id"],
            "source_event_sha256": record["source_event_sha256"],
            "inventory_record_sha256": record["record_sha256"],
            "json_schema_sha256": record["json_schema_sha256"],
            "model_id": record["model_id"],
            "task_id": record["task_id"],
            "target_ids": record["target_ids"],
            "prompt_sha256": record["prompt_sha256"],
        }
        if any(event.get(key) != value for key, value in expected.items()):
            raise ValueError("B3 recovery identity differs from its inventory record")
        if event.get("status") != "ok" or event.get("parse_status") != "ok":
            raise ValueError("B3 recovery contains an unresolved output")
        records[call_id], repaired[call_id] = record, event
    if len(records) != 3002:
        raise ValueError("B3 recovery inventory must contain exactly 3,002 calls")
    return (
        records,
        repaired,
        {
            "inventory_sha256": sha256_file(inventory_path),
            "recovery_events_sha256": sha256_file(recovery_dir / "events.jsonl"),
            "recovery_report_sha256": sha256_file(recovery_dir / "FULL_REPORT.json"),
            "recovery_receipts": len(receipt_paths),
        },
    )


def repaired_final_events(
    expected: dict[str, dict[str, Any]],
    original_events: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    repaired: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replace only terminal unresolved originals with their exact linked repairs."""
    final_by_call, _ = b3._final_events(original_events, expected)
    merged: list[dict[str, Any]] = []
    original_resolved = 0
    replaced = 0
    for call_id, original in final_by_call.items():
        expected_call = expected[call_id]
        target_ids = cast(list[str], original.get("target_ids"))
        predictions = original.get("predictions")
        resolved = (
            original.get("status") == "ok"
            and original.get("parse_status") == "ok"
            and isinstance(predictions, dict)
            and set(predictions) == set(target_ids)
        )
        if resolved:
            if call_id in records or call_id in repaired:
                raise ValueError("recovery attempted an originally resolved B3 call")
            merged.append(original)
            original_resolved += 1
            continue
        record, repair = records.get(call_id), repaired.get(call_id)
        if record is None or repair is None:
            raise ValueError("terminal unresolved B3 call lacks an exact repair")
        if (
            record["source_attempt_id"] != original.get("attempt_id")
            or record["source_event_sha256"] != sha256_json(original)
            or record["prompt_sha256"] != expected_call["prompt_sha256"]
            or record["target_ids"] != target_ids
            or sha256_json(record["target_ids"]) != expected_call["target_ids_sha256"]
            or record["doc_order_sha256"] != expected_call["doc_order_sha256"]
        ):
            raise ValueError("B3 repair is not linked to the final original attempt")
        b3._validate_event(repair, expected_call)
        merged.append(repair)
        replaced += 1
    expected_repairs = set(records) & set(expected)
    if replaced != len(expected_repairs):
        raise ValueError("B3 cell did not consume every applicable repair")
    return merged, {
        "original_resolved_calls": original_resolved,
        "repaired_calls": replaced,
        "revised_resolved_calls": original_resolved + replaced,
        "revised_unresolved_calls": 0,
    }


def _repaired_cell(
    original: b3.PreparedCell,
    study_root: Path,
    records: dict[str, dict[str, Any]],
    repaired: dict[str, dict[str, Any]],
) -> b3.PreparedCell:
    model_id, task_id = original.cell.model, original.cell.task
    rows = list(read_jsonl(study_root / "ext-work/20260913-cpu-draft" / task_id / "confirm.jsonl"))
    expected = {
        str(record["call_id"]): record
        for record in ext.b3_calls(model_id, task_id, rows, b3.N_ITEMS)
    }
    events_path = (
        study_root
        / "extension-b3"
        / model_id.replace("/", "--")
        / "cells"
        / task_id
        / "events.jsonl"
    )
    cell_ids = set(expected)
    cell_records = {key: value for key, value in records.items() if key in cell_ids}
    cell_repairs = {key: value for key, value in repaired.items() if key in cell_ids}
    final, repair_counts = repaired_final_events(
        expected, b3._read_events(events_path), cell_records, cell_repairs
    )
    joint = [
        b3._resolved_errors(final, rows, task_id, "joint", assignment)
        for assignment in range(1, ext.N_ASSIGNMENTS + 1)
    ]
    exact = [
        b3._resolved_errors(final, rows, task_id, "exact_context", assignment)
        for assignment in range(1, ext.N_ASSIGNMENTS + 1)
    ]
    solo = b3._resolved_errors(final, rows, task_id, "solo", None)
    if any(np.isnan(vector).any() for vector in [*joint, *exact, solo]):
        raise ValueError(f"repaired B3 cell remains unresolved: {original.cell.key}")
    sensitivity = b3._sensitivities(joint, exact, original.matrices)
    joint_covariances = [
        b3._covariance(vector, matrix)
        for vector, matrix in zip(joint, original.matrices, strict=True)
    ]
    exact_covariances = [
        b3._covariance(vector, matrix)
        for vector, matrix in zip(exact, original.matrices, strict=True)
    ]
    result = copy.deepcopy(original.result)
    result["source"]["recovery_substitution_verified"] = True
    result["source"]["repaired_calls"] = repair_counts["repaired_calls"]
    result["data_quality"] = {
        "by_condition": {
            condition: b3._call_quality(
                [event for event in final if event["condition"] == condition]
            )
            for condition in ext.B3_CONDITIONS
        },
        "joint_target_appearances": b3._accuracy(np.concatenate(joint)),
        "exact_target_appearances": b3._accuracy(np.concatenate(exact)),
        "solo_target_appearances": b3._accuracy(solo),
        "terminal_malformed_outputs": 0,
        "original_terminal_unresolved_calls": repair_counts["repaired_calls"],
        "repaired_calls": repair_counts["repaired_calls"],
        "n_calls_discarded_under_whole_call_rule": sensitivity["n_calls_discarded_total"],
    }
    result["accuracy"] = {
        "joint": b3._accuracy(np.concatenate(joint)),
        "exact_context": b3._accuracy(np.concatenate(exact)),
        "solo": b3._accuracy(solo),
    }
    primary = cast(dict[str, float], sensitivity["cell_mean"])[
        b3.TREATMENT_KEYS["unresolved_as_correct"]
    ]
    result["covariance_unresolved_as_correct"] = {
        "joint_per_assignment": joint_covariances,
        "exact_context_per_assignment": exact_covariances,
        "joint_mean": float(np.mean(joint_covariances)),
        "exact_context_mean": float(np.mean(exact_covariances)),
        "joint_minus_exact": primary,
    }
    result["fake_batch_controls_unresolved_as_correct"] = {
        "exact_context_per_assignment": exact_covariances,
        "exact_context_mean": float(np.mean(exact_covariances)),
        "solo_per_assignment": [b3._covariance(solo, matrix) for matrix in original.matrices],
        "solo_mean": float(np.mean([b3._covariance(solo, matrix) for matrix in original.matrices])),
    }
    result["malformed_sensitivity"] = sensitivity
    return b3.PreparedCell(
        original.cell, original.frame, joint, exact, solo, original.matrices, result
    )


def _summarize(
    cells: list[b3.PreparedCell], lock: dict[str, Any], n_draws: int, workers: int
) -> dict[str, Any]:
    replay = b3._replay_all(cells, n_draws, min(max(workers, 1), b3.N_CELLS))
    by_key = {cell.cell.key: cell for cell in cells}
    keys = [str(key) for key in lock["ordered_cells"]]
    if len(by_key) != b3.N_CELLS or set(by_key) != set(keys):
        raise ValueError("repaired cells differ from the sealed grid")
    observed = {key: replay[key][0] for key in keys}
    raw_p = {key: replay[key][2]["p_value_one_sided_upper"] for key in keys}
    holm = stats.holm(raw_p, b3.ALPHA)
    for key, cell in by_key.items():
        sensitivity_primary = cell.result["malformed_sensitivity"]["cell_mean"][
            b3.TREATMENT_KEYS["unresolved_as_correct"]
        ]
        if not np.isclose(observed[key], sensitivity_primary, atol=1e-15, rtol=0):
            raise ValueError(f"repaired primary implementations disagree for {key}")
        cell.result["primary_replay"] = replay[key][2]
        cell.result["cell_assessment"] = {
            "raw_p_value": raw_p[key],
            "holm_adjusted_p": holm["adjusted_p"][key],
            "holm_positive_at_practical_floor": bool(
                holm["adjusted_p"][key] <= b3.ALPHA and observed[key] >= b3.PRACTICAL_FLOOR
            ),
        }
    global_observed = float(np.mean([observed[key] for key in keys]))
    global_null = np.column_stack([replay[key][1] for key in keys]).mean(axis=1)
    global_p = float((1 + int((global_null >= global_observed).sum())) / (1 + n_draws))
    centred = global_null - global_null.mean()
    high_quantile, low_quantile = np.percentile(centred, [97.5, 2.5])
    family_of = {cell.cell.key: cell.cell.family for cell in cells}
    task_of = {cell.cell.key: cell.cell.task for cell in cells}
    families = [str(value) for value in lock["claim_dimensions"]["model_families"]]
    tasks = [str(value) for value in lock["claim_dimensions"]["tasks"]]
    model_margins = {
        family: float(np.mean([observed[key] for key in keys if family_of[key] == family]))
        for family in families
    }
    task_margins = {
        task: float(np.mean([observed[key] for key in keys if task_of[key] == task]))
        for task in tasks
    }
    interactions = {
        key: observed[key]
        - model_margins[family_of[key]]
        - task_margins[task_of[key]]
        + global_observed
        for key in keys
    }
    leave_one_model = {
        family: float(np.mean([observed[key] for key in keys if family_of[key] != family]))
        for family in families
    }
    leave_one_task = {
        task: float(np.mean([observed[key] for key in keys if task_of[key] != task]))
        for task in tasks
    }
    positive_cells = [
        key
        for key in keys
        if holm["adjusted_p"][key] <= b3.ALPHA and observed[key] >= b3.PRACTICAL_FLOOR
    ]
    positive_families = sorted({family_of[key] for key in positive_cells})
    positive_tasks = sorted({task_of[key] for key in positive_cells})
    treatment_cells = {
        treatment: {
            key: float(by_key[key].result["malformed_sensitivity"]["cell_mean"][stat_key])
            for key in keys
        }
        for treatment, stat_key in b3.TREATMENT_KEYS.items()
    }
    treatment_global = {
        treatment: float(np.mean(list(values.values())))
        for treatment, values in treatment_cells.items()
    }
    criteria = {
        "complete_16_cell_grid": len(cells) == b3.N_CELLS,
        "positive_global_effect": global_observed > 0,
        "global_replay_p_at_most_0.05": global_p <= b3.ALPHA,
        "at_least_three_positive_model_families": len(positive_families) >= 3,
        "at_least_three_positive_tasks": len(positive_tasks) >= 3,
        "all_leave_one_model_out_means_positive": min(leave_one_model.values()) > 0,
        "all_leave_one_task_out_means_positive": min(leave_one_task.values()) > 0,
        "all_four_global_malformed_treatments_at_least_0.002": all(
            np.isfinite(value) for value in treatment_global.values()
        )
        and min(treatment_global.values()) >= b3.PRACTICAL_FLOOR,
    }
    return {
        "design": {
            "n_models": 4,
            "n_tasks": 4,
            "n_cells": b3.N_CELLS,
            "items_per_cell": b3.N_ITEMS,
            "assignments": ext.N_ASSIGNMENTS,
            "batch_size": ext.BATCH_SIZE,
            "n_draws": n_draws,
            "multiplicity": "Holm across sixteen repaired cell tests",
            "practical_floor": b3.PRACTICAL_FLOOR,
            "alpha": b3.ALPHA,
        },
        "primary": {
            "global_equal_cell_delta": global_observed,
            "global_replay_p_one_sided": global_p,
            "global_effect_interval": {
                "point": global_observed,
                "low": global_observed - float(high_quantile),
                "high": global_observed - float(low_quantile),
            },
            "cell_delta": observed,
            "cell_raw_p": raw_p,
            "cell_holm_adjusted_p": holm["adjusted_p"],
            "holm_positive_cells_at_practical_floor": positive_cells,
            "positive_model_families": positive_families,
            "positive_tasks": positive_tasks,
            "model_margins": model_margins,
            "task_margins": task_margins,
            "interactions": interactions,
            "leave_one_model_out": leave_one_model,
            "leave_one_task_out": leave_one_task,
        },
        "malformed_output_sensitivity": {
            "cell_delta": treatment_cells,
            "global_equal_cell_delta": treatment_global,
        },
        "recurrence_gate": {
            "passed": all(criteria.values()),
            "criteria": criteria,
            "positive_model_family_count": len(positive_families),
            "positive_task_count": len(positive_tasks),
            "rule": lock["analysis_lock"]["recurrence_gate"],
        },
        "cells": [by_key[key].result for key in keys],
    }


def analyze(
    study_root: Path,
    recovery_dir: Path,
    inventory_path: Path,
    *,
    n_draws: int = b3.N_DRAWS,
    workers: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce A9 and return a separate exact-repair sensitivity analysis."""
    if n_draws < 319:
        raise ValueError("at least 319 draws are required")
    original = b3.analyze_grid(study_root, n_draws=n_draws, workers=workers)
    stored_original = _load(study_root / "extension-b3/analysis/analysis.json")
    if original != stored_original:
        raise ValueError("current code does not reproduce the stored A9 analysis")
    lock_root = study_root / "full-grid-lock/20260914T032937Z-X1R4-A8"
    lock = b3._load(lock_root / "FULL_GRID_LOCK.json")
    records, repaired, recovery_hashes = recovery_evidence(study_root, recovery_dir, inventory_path)
    prepared = [
        b3.prepare_cell(model, task_id, study_root, lock)
        for model in cast(list[dict[str, Any]], lock["models"])
        for task_id in cast(list[str], lock["claim_dimensions"]["tasks"])
    ]
    cells = [_repaired_cell(cell, study_root, records, repaired) for cell in prepared]
    used_repairs = sum(cell.result["data_quality"]["repaired_calls"] for cell in cells)
    if used_repairs != len(repaired):
        raise ValueError("some B3 recovery calls were not consumed")
    summary = _summarize(cells, lock, n_draws, workers)
    revised = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "secondary repaired-output sensitivity of X1R4 A9 B3 grid",
        "scope": original["scope"],
        "post_execution_technical_amendment": True,
        "original_result": "preserved_as_primary",
        "replacement_rule": "valid original output else exact linked repair-v1 output",
        "repair_calls_used": used_repairs,
        "source": {
            **original["source"],
            **recovery_hashes,
            "stored_original_analysis_sha256": sha256_file(
                study_root / "extension-b3/analysis/analysis.json"
            ),
            "original_analysis_reproduced_exactly": True,
        },
        **summary,
        "comparison_to_original": {
            "original_recurrence_gate_passed": original["recurrence_gate"]["passed"],
            "repaired_recurrence_gate_passed": summary["recurrence_gate"]["passed"],
            "original_global_delta": original["primary"]["global_equal_cell_delta"],
            "repaired_global_delta": summary["primary"]["global_equal_cell_delta"],
            "delta_change": summary["primary"]["global_equal_cell_delta"]
            - original["primary"]["global_equal_cell_delta"],
            "original_positive_cells": len(
                original["primary"]["holm_positive_cells_at_practical_floor"]
            ),
            "repaired_positive_cells": len(
                summary["primary"]["holm_positive_cells_at_practical_floor"]
            ),
        },
        "coverage_status": original["coverage_status"],
        "gpu_work_launched": False,
    }
    return original, revised


def render_report(result: dict[str, Any]) -> str:
    primary = result["primary"]
    comparison = result["comparison_to_original"]
    gate = result["recurrence_gate"]
    lines = [
        "# Repaired B3 recurrence sensitivity",
        "",
        "This is a post-execution sensitivity analysis. The sealed A9 analysis remains primary.",
        "",
        f"Exact linked repairs used: {result['repair_calls_used']:,}",
        f"Repaired recurrence gate: {'passes' if gate['passed'] else 'does not pass'}",
        (
            f"Global joint-minus-exact covariance: {primary['global_equal_cell_delta']:.6f} "
            f"(one-sided replay p = {primary['global_replay_p_one_sided']:.6f}; "
            f"95% shift interval [{primary['global_effect_interval']['low']:.6f}, "
            f"{primary['global_effect_interval']['high']:.6f}])."
        ),
        (
            f"Holm-positive cells at the 0.002 floor: "
            f"{len(primary['holm_positive_cells_at_practical_floor'])} of 16."
        ),
        "",
        "## Comparison to sealed A9",
        "",
        "| quantity | sealed A9 | repaired sensitivity |",
        "|---|---:|---:|",
        (
            f"| recurrence gate | {comparison['original_recurrence_gate_passed']} | "
            f"{comparison['repaired_recurrence_gate_passed']} |"
        ),
        (
            f"| global delta | {comparison['original_global_delta']:.6f} | "
            f"{comparison['repaired_global_delta']:.6f} |"
        ),
        (
            f"| Holm-positive cells | {comparison['original_positive_cells']} | "
            f"{comparison['repaired_positive_cells']} |"
        ),
        "",
        "## Repaired cells",
        "",
        "| family | task | repairs | delta | raw p | Holm p | positive |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    positive = set(primary["holm_positive_cells_at_practical_floor"])
    for cell in result["cells"]:
        key = cell["cell_key"]
        lines.append(
            f"| {cell['family']} | {cell['task_id']} | "
            f"{cell['data_quality']['repaired_calls']} | "
            f"{primary['cell_delta'][key]:.6f} | {primary['cell_raw_p'][key]:.6f} | "
            f"{primary['cell_holm_adjusted_p'][key]:.6f} | "
            f"{'yes' if key in positive else 'no'} |"
        )
    lines.extend(["", "## Gate checks", "", "| criterion | passed |", "|---|---:|"])
    for criterion, passed in gate["criteria"].items():
        lines.append(f"| {criterion} | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "All repaired vectors are complete. The four former malformed-output treatments "
            "therefore collapse to the same result.",
            "",
            "This analysis measures reference-relative disagreement dependence, not semantic "
            "accuracy or confidence-interval coverage.",
            "",
        ]
    )
    return "\n".join(lines)
