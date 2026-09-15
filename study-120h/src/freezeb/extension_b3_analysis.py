"""Prospectively sealed A9 analysis of the complete X1R4 B3 grid."""

from __future__ import annotations

import concurrent.futures
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from step0 import stats
from step0.hashing import read_jsonl, sha256_file, sha256_json

from . import assign
from . import extension as ext
from .labels import disagreement
from .selection import Cell

N_DRAWS = 4_999
N_ITEMS = 8_192
N_CELLS = 16
ALPHA = 0.05
PRACTICAL_FLOOR = 0.002
TREATMENT_KEYS = {
    "unresolved_as_correct": "delta_unresolved_as_correct",
    "unresolved_as_error": "delta_unresolved_as_error",
    "unresolved_dropped": "delta_unresolved_dropped",
    "whole_call_discarded": "delta_whole_call_discarded",
}
RETRYABLE = frozenset({"generation_failed", "truncated", "worker_failed"})


@dataclass
class PreparedCell:
    cell: Cell
    frame: assign.Frame
    joint: list[np.ndarray]
    exact: list[np.ndarray]
    solo: np.ndarray
    matrices: list[np.ndarray]
    result: dict[str, Any]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_events(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _verify_manifest(directory: Path) -> dict[str, Any]:
    manifest = _load(directory / "MANIFEST.json")
    for relative, digest in cast(dict[str, str], manifest["files_sha256"]).items():
        if sha256_file(directory / relative) != digest:
            raise ValueError(f"sealed file changed: {directory / relative}")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if "manifest_sha256" in manifest and sha256_json(core) != manifest["manifest_sha256"]:
        raise ValueError(f"manifest digest changed: {directory / 'MANIFEST.json'}")
    return manifest


def _model_dir(model_id: str) -> str:
    return model_id.replace("/", "--")


def _groups(matrix: np.ndarray, n_items: int) -> np.ndarray:
    groups = np.empty(n_items, dtype=np.int64)
    for call_index in range(matrix.shape[1]):
        groups[matrix[:, call_index]] = call_index
    return groups


def _final_events(
    events: list[dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    by_attempt: dict[str, dict[str, Any]] = {}
    by_call: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        call_id = str(event.get("call_id", ""))
        if attempt_id in by_attempt:
            if event != by_attempt[attempt_id]:
                raise ValueError(f"conflicting duplicate attempt {attempt_id}")
            continue
        if call_id not in expected:
            raise ValueError(f"event does not belong to the sealed cell: {call_id}")
        _validate_event(event, expected[call_id])
        by_attempt[attempt_id] = event
        by_call.setdefault(call_id, []).append(event)
    if set(by_call) != set(expected):
        raise ValueError(f"ledger covers {len(by_call)} calls, expected {len(expected)}")

    final: dict[str, dict[str, Any]] = {}
    for call_id, attempts in by_call.items():
        attempts.sort(key=lambda row: int(row.get("attempt", 0)))
        if [int(row.get("attempt", 0)) for row in attempts] not in ([1], [1, 2]):
            raise ValueError(f"invalid attempt sequence for {call_id}")
        first = attempts[0]
        if first.get("attempt_id") != f"{call_id}::attempt::1":
            raise ValueError(f"invalid first-attempt identity for {call_id}")
        if first.get("status") in RETRYABLE and len(attempts) != 2:
            raise ValueError(f"missing required retry for {call_id}")
        if len(attempts) == 2:
            retry = attempts[1]
            if first.get("status") not in RETRYABLE:
                raise ValueError(f"unpermitted retry for {call_id}")
            if retry.get("attempt_id") != f"{call_id}::attempt::2" or retry.get(
                "parent_attempt_id"
            ) != first.get("attempt_id"):
                raise ValueError(f"invalid retry linkage for {call_id}")
        final[call_id] = attempts[-1]
    return final, {
        "stored_attempts": len(by_attempt),
        "first_attempts": len(by_call),
        "retry_attempts": len(by_attempt) - len(by_call),
    }


def _validate_event(event: dict[str, Any], expected: dict[str, Any]) -> None:
    expected_model, expected_task = str(expected["cell_key"]).rsplit("::", 1)
    if event.get("model_id") != expected_model or event.get("task_id") != expected_task:
        raise ValueError(f"event model or task differs from the seal: {expected['call_id']}")
    fields = (
        "cell_key",
        "condition",
        "assignment",
        "call_index",
        "execution_block",
        "batch_size",
        "prompt_sha256",
    )
    if any(event.get(field) != expected.get(field) for field in fields):
        raise ValueError(f"event metadata differs from the seal: {expected['call_id']}")
    if sha256_json(event.get("doc_order")) != expected["doc_order_sha256"]:
        raise ValueError(f"document order differs from the seal: {expected['call_id']}")
    if sha256_json(event.get("target_ids")) != expected["target_ids_sha256"]:
        raise ValueError(f"target IDs differ from the seal: {expected['call_id']}")


def _resolved_errors(
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    task_id: str,
    condition: str,
    assignment_number: int | None,
) -> np.ndarray:
    index = {str(row["doc_id"]): position for position, row in enumerate(rows)}
    reference = {str(row["doc_id"]): row["reference_normalized"] for row in rows}
    vector = np.full(len(rows), np.nan)
    seen: set[str] = set()
    selected = [
        event
        for event in events
        if event["condition"] == condition and event.get("assignment") == assignment_number
    ]
    for event in selected:
        target_ids = [str(value) for value in cast(list[str], event["target_ids"])]
        if seen.intersection(target_ids):
            raise ValueError(f"duplicate {condition} targets in {task_id}")
        seen.update(target_ids)
        predictions = event.get("predictions")
        resolved = (
            event.get("status") == "ok"
            and event.get("parse_status") == "ok"
            and isinstance(predictions, dict)
            and set(predictions) == set(target_ids)
        )
        if resolved:
            parsed = cast(dict[str, Any], predictions)
            for doc_id in target_ids:
                vector[index[doc_id]] = disagreement(parsed[doc_id], reference[doc_id], task_id)
    if seen != set(index):
        raise ValueError(
            f"{condition} assignment {assignment_number} covers {len(seen)} targets, "
            f"expected {len(index)}"
        )
    return vector


def _accuracy(vector: np.ndarray) -> dict[str, Any]:
    resolved = vector[~np.isnan(vector)]
    return {
        "target_appearances": int(vector.size),
        "resolved": int(resolved.size),
        "unresolved": int(np.isnan(vector).sum()),
        "parse_rate": float(resolved.size / vector.size),
        "error_rate_resolved": float(resolved.mean()) if resolved.size else None,
        "accuracy_resolved": float(1.0 - resolved.mean()) if resolved.size else None,
    }


def _covariance(vector: np.ndarray, matrix: np.ndarray, fill: float = 0.0) -> float:
    return stats.covariance_from_stratified(np.where(np.isnan(vector), fill, vector)[matrix].T)


def _sensitivities(
    joint: list[np.ndarray], exact: list[np.ndarray], matrices: list[np.ndarray]
) -> dict[str, Any]:
    per_assignment: list[dict[str, float]] = []
    for joint_values, exact_values, matrix in zip(joint, exact, matrices, strict=True):
        groups = _groups(matrix, joint_values.size)
        row = stats.malformed_sensitivity(
            joint_values,
            exact_values,
            groups,
            np.isnan(joint_values),
            np.isnan(exact_values),
        )
        row["failure_covariance_joint"] = stats.failure_clustering(np.isnan(joint_values), groups)
        row["failure_covariance_exact"] = stats.failure_clustering(np.isnan(exact_values), groups)
        per_assignment.append(row)
    keys = (*TREATMENT_KEYS.values(), "failure_covariance_joint", "failure_covariance_exact")
    return {
        "per_assignment": per_assignment,
        "cell_mean": {key: float(np.nanmean([row[key] for row in per_assignment])) for key in keys},
        "n_calls_discarded_total": int(sum(row["n_calls_discarded"] for row in per_assignment)),
    }


def _call_quality(events: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(str(event.get("status")) for event in events)
    parse = Counter(str(event.get("parse_status")) for event in events)
    unresolved_calls = sum(
        event.get("status") != "ok"
        or event.get("parse_status") != "ok"
        or not isinstance(event.get("predictions"), dict)
        for event in events
    )
    return {
        "calls": len(events),
        "resolved_calls": len(events) - unresolved_calls,
        "unresolved_calls": unresolved_calls,
        "call_parse_rate": float((len(events) - unresolved_calls) / len(events)),
        "status_counts": dict(sorted(status.items())),
        "parse_status_counts": dict(sorted(parse.items())),
    }


def prepare_cell(
    model: dict[str, Any],
    task_id: str,
    study_root: Path,
    lock: dict[str, Any],
) -> PreparedCell:
    model_id = str(model["model_id"])
    family = str(model["family"])
    cell_key = f"{model_id}::{task_id}"
    frame_path = study_root / "ext-work/20260913-cpu-draft" / task_id / "confirm.jsonl"
    rows = list(read_jsonl(frame_path))
    if len(rows) != N_ITEMS:
        raise ValueError(f"{task_id} frame has {len(rows)} rows, expected {N_ITEMS}")
    ids = [str(row["doc_id"]) for row in rows]
    locked_frame = lock["datasets"][task_id]
    identifier_path = (
        study_root / "freeze/20260913T183353Z-X1R4/frame-identifiers" / task_id / "confirm.jsonl"
    )
    if (
        len(set(ids)) != N_ITEMS
        or sha256_json(ids) != locked_frame["ordered_doc_ids_sha256"]
        or sha256_file(identifier_path) != locked_frame["identifier_file_sha256"]
    ):
        raise ValueError(f"{task_id} frame differs from the A8 lock")

    run_root = study_root / "extension-b3" / _model_dir(model_id)
    plan_path = run_root / "plan.json"
    summary_path = run_root / "SUMMARY.json"
    completeness_path = run_root / "cells" / task_id / "completeness.json"
    events_path = run_root / "cells" / task_id / "events.jsonl"
    plan = _load(plan_path)
    summary = _load(summary_path)
    completeness = _load(completeness_path)
    core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("model_id") != model_id
        or plan.get("model_revision") != model["revision"]
        or sha256_json(core) != plan.get("plan_sha256")
        or summary.get("plan_sha256") != plan.get("plan_sha256")
    ):
        raise ValueError(f"{model_id} plan or summary identity changed")
    plan_cell = next(
        (row for row in cast(list[dict[str, Any]], plan["cells"]) if row["cell_key"] == cell_key),
        None,
    )
    lock_path = study_root / "full-grid-lock/20260914T032937Z-X1R4-A8/FULL_GRID_LOCK.json"
    if (
        plan.get("dataset_order") != list(ext.EXT_TASK_KEYS)
        or plan.get("total_calls") != 137_216
        or plan.get("full_grid_lock_sha256") != sha256_file(lock_path)
        or plan_cell is None
    ):
        raise ValueError(f"{model_id} execution plan differs from the A8 release")
    if completeness.get("complete") is not True or completeness.get("pending_attempts") != 0:
        raise ValueError(f"{cell_key} is not execution-complete")

    digest = ext.b3_plan_digest(model_id, task_id, rows, N_ITEMS)
    locked_schedule = lock["selected_schedules"]["cells"][cell_key]
    for key in ("call_id_order_sha256", "record_stream_sha256"):
        if digest[key] != locked_schedule[key]:
            raise ValueError(f"{cell_key} {key} differs from the A8 lock")
    if sha256_json(digest) != locked_schedule["selected_plan_sha256"]:
        raise ValueError(f"{cell_key} selected plan digest differs from the A8 lock")
    if (
        plan_cell["call_id_order_sha256"] != digest["call_id_order_sha256"]
        or plan_cell["item_id_order_sha256"] != locked_frame["ordered_doc_ids_sha256"]
        or plan_cell["calls"] != ext.b3_expected_counts(N_ITEMS)["total"]
    ):
        raise ValueError(f"{cell_key} local execution plan differs from the sealed schedule")

    expected = {
        str(record["call_id"]): record for record in ext.b3_calls(model_id, task_id, rows, N_ITEMS)
    }
    events = _read_events(events_path)
    final_by_call, attempt_counts = _final_events(events, expected)
    if (
        attempt_counts["first_attempts"] != completeness["first_attempts"]
        or attempt_counts["retry_attempts"] != completeness["retry_attempts"]
    ):
        raise ValueError(f"{cell_key} completeness record differs from its ledger")

    final = list(final_by_call.values())
    irrecoverable = [
        event
        for event in final
        if event.get("status") in {"generation_failed", "worker_failed", "context_excluded"}
    ]
    if irrecoverable:
        raise ValueError(f"{cell_key} has {len(irrecoverable)} irrecoverable terminal failures")
    terminal_malformed_outputs = sum(event.get("status") == "truncated" for event in final)
    joint = [
        _resolved_errors(final, rows, task_id, "joint", assignment_number)
        for assignment_number in range(1, ext.N_ASSIGNMENTS + 1)
    ]
    exact = [
        _resolved_errors(final, rows, task_id, "exact_context", assignment_number)
        for assignment_number in range(1, ext.N_ASSIGNMENTS + 1)
    ]
    solo = _resolved_errors(final, rows, task_id, "solo", None)
    frame = assign.Frame(tuple(ids), tuple(int(row["n_chars"]) for row in rows))
    matrices = assign.observed_stat_matrices(
        frame, cell_key, ext.BATCH_SIZE, namespace=ext.ASSIGN_NAMESPACE
    )
    sensitivity = _sensitivities(joint, exact, matrices)
    primary = cast(dict[str, float], sensitivity["cell_mean"])[
        TREATMENT_KEYS["unresolved_as_correct"]
    ]
    joint_covariances = [
        _covariance(vector, matrix) for vector, matrix in zip(joint, matrices, strict=True)
    ]
    exact_covariances = [
        _covariance(vector, matrix) for vector, matrix in zip(exact, matrices, strict=True)
    ]
    solo_covariances = [_covariance(solo, matrix) for matrix in matrices]
    conditions = {
        condition: _call_quality([event for event in final if event["condition"] == condition])
        for condition in ext.B3_CONDITIONS
    }
    result = {
        "cell_key": cell_key,
        "family": family,
        "model_id": model_id,
        "task_id": task_id,
        "source": {
            "events_path": str(events_path.relative_to(study_root.parent)),
            "events_sha256": sha256_file(events_path),
            "plan_sha256_file": sha256_file(plan_path),
            "summary_sha256": sha256_file(summary_path),
            "completeness_sha256": sha256_file(completeness_path),
            "frame_sha256": sha256_file(frame_path),
            **attempt_counts,
            "sealed_call_metadata_verified": True,
        },
        "data_quality": {
            "by_condition": conditions,
            "joint_target_appearances": _accuracy(np.concatenate(joint)),
            "exact_target_appearances": _accuracy(np.concatenate(exact)),
            "solo_target_appearances": _accuracy(solo),
            "terminal_malformed_outputs": terminal_malformed_outputs,
            "n_calls_discarded_under_whole_call_rule": sensitivity["n_calls_discarded_total"],
        },
        "accuracy": {
            "joint": _accuracy(np.concatenate(joint)),
            "exact_context": _accuracy(np.concatenate(exact)),
            "solo": _accuracy(solo),
        },
        "covariance_unresolved_as_correct": {
            "joint_per_assignment": joint_covariances,
            "exact_context_per_assignment": exact_covariances,
            "joint_mean": float(np.mean(joint_covariances)),
            "exact_context_mean": float(np.mean(exact_covariances)),
            "joint_minus_exact": primary,
        },
        "fake_batch_controls_unresolved_as_correct": {
            "exact_context_per_assignment": exact_covariances,
            "exact_context_mean": float(np.mean(exact_covariances)),
            "solo_per_assignment": solo_covariances,
            "solo_mean": float(np.mean(solo_covariances)),
        },
        "malformed_sensitivity": sensitivity,
    }
    return PreparedCell(
        Cell(
            model_id,
            family,
            task_id,
            int(next(m.order for m in ext.EXT_MODELS if m.repo == model_id)),
            ext.EXT_TASK_KEYS.index(task_id) + 1,
        ),
        frame,
        joint,
        exact,
        solo,
        matrices,
        result,
    )


def _mean_delta(
    joint: list[np.ndarray], exact: list[np.ndarray], matrices: list[np.ndarray]
) -> float:
    return float(
        np.mean(
            [
                _covariance(joint_values, matrix) - _covariance(exact_values, matrix)
                for joint_values, exact_values, matrix in zip(joint, exact, matrices, strict=True)
            ]
        )
    )


def _replay_cell(cell: PreparedCell, n_draws: int) -> tuple[str, float, np.ndarray, dict[str, Any]]:
    observed = _mean_delta(cell.joint, cell.exact, cell.matrices)
    null = np.empty(n_draws, dtype=float)
    for draw in range(n_draws):
        matrices = assign.replay_stat_matrices(
            cell.frame,
            cell.cell.key,
            ext.BATCH_SIZE,
            draw,
            namespace=ext.ASSIGN_NAMESPACE,
        )
        null[draw] = _mean_delta(cell.joint, cell.exact, matrices)
    p_value = float((1 + int((null >= observed).sum())) / (1 + n_draws))
    centred = null - null.mean()
    high_quantile, low_quantile = np.percentile(centred, [97.5, 2.5])
    replay = {
        "observed": observed,
        "p_value_one_sided_upper": p_value,
        "n_draws": n_draws,
        "min_reachable_p": 1.0 / (1 + n_draws),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "effect_interval": {
            "point": observed,
            "low": observed - float(high_quantile),
            "high": observed - float(low_quantile),
        },
        "effect_interval_method": (
            "design-respecting shift interval from mean-centred replay quantiles"
        ),
    }
    return cell.cell.key, observed, null, replay


def _replay_all(
    cells: list[PreparedCell], n_draws: int, workers: int
) -> dict[str, tuple[float, np.ndarray, dict[str, Any]]]:
    if workers <= 1:
        rows = [_replay_cell(cell, n_draws) for cell in cells]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_replay_payload, [(cell, n_draws) for cell in cells]))
    return {key: (observed, null, replay) for key, observed, null, replay in rows}


def _replay_payload(
    payload: tuple[PreparedCell, int],
) -> tuple[str, float, np.ndarray, dict[str, Any]]:
    return _replay_cell(*payload)


def _safe_number(value: float) -> float | None:
    return None if not math.isfinite(value) else float(value)


def analyze_grid(
    study_root: Path,
    *,
    n_draws: int = N_DRAWS,
    workers: int = 1,
) -> dict[str, Any]:
    if n_draws < 319:
        raise ValueError("at least 319 draws are required to resolve a sixteen-test Holm family")
    lock_root = study_root / "full-grid-lock/20260914T032937Z-X1R4-A8"
    manifest = _verify_manifest(lock_root)
    lock_path = lock_root / "FULL_GRID_LOCK.json"
    lock = _load(lock_path)
    if lock["design"]["n_cells"] != N_CELLS or lock["design"]["items_per_cell"] != N_ITEMS:
        raise ValueError("A9 requires the complete sealed 16-cell, 8,192-item grid")
    if lock["analysis_lock"]["malformed_treatments"] != list(TREATMENT_KEYS):
        raise ValueError("A8 malformed-output treatments changed")
    if lock["analysis_lock"]["primary_malformed_treatment"] != "unresolved_as_correct":
        raise ValueError("A8 primary malformed-output treatment changed")

    cells = [
        prepare_cell(model, task_id, study_root, lock)
        for model in cast(list[dict[str, Any]], lock["models"])
        for task_id in cast(list[str], lock["claim_dimensions"]["tasks"])
    ]
    by_key = {cell.cell.key: cell for cell in cells}
    if len(by_key) != N_CELLS or set(by_key) != set(lock["ordered_cells"]):
        raise ValueError("prepared cells differ from the sealed grid")

    replay = _replay_all(cells, n_draws, min(max(workers, 1), N_CELLS))
    raw_p = {key: values[2]["p_value_one_sided_upper"] for key, values in replay.items()}
    holm = stats.holm(raw_p, ALPHA)
    observed = {key: values[0] for key, values in replay.items()}
    for key, cell in by_key.items():
        sensitivity_primary = cell.result["malformed_sensitivity"]["cell_mean"][
            TREATMENT_KEYS["unresolved_as_correct"]
        ]
        if not math.isclose(observed[key], sensitivity_primary, abs_tol=1e-15):
            raise ValueError(f"primary implementations disagree for {key}")
        cell.result["primary_replay"] = replay[key][2]
        cell.result["cell_assessment"] = {
            "raw_p_value": raw_p[key],
            "holm_adjusted_p": holm["adjusted_p"][key],
            "holm_positive_at_practical_floor": bool(
                holm["adjusted_p"][key] <= ALPHA and observed[key] >= PRACTICAL_FLOOR
            ),
        }

    keys = [str(key) for key in lock["ordered_cells"]]
    global_observed = float(np.mean([observed[key] for key in keys]))
    null_matrix = np.column_stack([replay[key][1] for key in keys])
    global_null = null_matrix.mean(axis=1)
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
        key for key in keys if holm["adjusted_p"][key] <= ALPHA and observed[key] >= PRACTICAL_FLOOR
    ]
    positive_families = sorted({family_of[key] for key in positive_cells})
    positive_tasks = sorted({task_of[key] for key in positive_cells})
    treatment_cells = {
        treatment: {
            key: float(by_key[key].result["malformed_sensitivity"]["cell_mean"][stat_key])
            for key in keys
        }
        for treatment, stat_key in TREATMENT_KEYS.items()
    }
    treatment_global = {
        treatment: float(np.mean(list(values.values())))
        for treatment, values in treatment_cells.items()
    }
    treatments_finite = all(math.isfinite(value) for value in treatment_global.values())
    criteria = {
        "complete_16_cell_grid": len(cells) == N_CELLS,
        "positive_global_effect": global_observed > 0,
        "global_replay_p_at_most_0.05": global_p <= ALPHA,
        "at_least_three_positive_model_families": len(positive_families) >= 3,
        "at_least_three_positive_tasks": len(positive_tasks) >= 3,
        "all_leave_one_model_out_means_positive": min(leave_one_model.values()) > 0,
        "all_leave_one_task_out_means_positive": min(leave_one_task.values()) > 0,
        "all_four_global_malformed_treatments_at_least_0.002": treatments_finite
        and min(treatment_global.values()) >= PRACTICAL_FLOOR,
    }
    passed = all(criteria.values())
    return {
        "schema_version": "ext-x1r4-a9-full-grid-analysis-v1",
        "analysis": "X1R4 A9 complete 4 x 4 B3 grid",
        "scope": (
            "sixteen frozen model-task cells; reference-relative disagreement only; "
            "no measured confidence-interval coverage claim"
        ),
        "source": {
            "full_grid_lock_path": str(lock_path.relative_to(study_root.parent)),
            "full_grid_lock_sha256": sha256_file(lock_path),
            "full_grid_manifest_sha256": manifest.get("manifest_sha256"),
            "all_cell_ledgers_and_schedules_verified": True,
        },
        "design": {
            "n_models": 4,
            "n_tasks": 4,
            "n_cells": N_CELLS,
            "items_per_cell": N_ITEMS,
            "assignments": ext.N_ASSIGNMENTS,
            "batch_size": ext.BATCH_SIZE,
            "n_draws": n_draws,
            "multiplicity": "Holm across sixteen primary cell tests",
            "primary_malformed_treatment": "unresolved_as_correct",
            "practical_floor": PRACTICAL_FLOOR,
            "alpha": ALPHA,
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
            "passed": passed,
            "criteria": criteria,
            "positive_model_family_count": len(positive_families),
            "positive_task_count": len(positive_tasks),
            "rule": lock["analysis_lock"]["recurrence_gate"],
        },
        "cells": [by_key[key].result for key in keys],
        "coverage_status": {
            "measured": False,
            "reason": "A9 contains no independent repeated-sampling stream; B4 is required",
        },
    }


def render_report(result: dict[str, Any]) -> str:
    primary = result["primary"]
    gate = result["recurrence_gate"]
    verdict = "passes" if gate["passed"] else "does not pass"
    lines = [
        "# X1R4 A9 full-grid analysis",
        "",
        "## Result",
        "",
        f"The frozen cross-family recurrence gate **{verdict}**.",
        "",
        (
            "The equal-cell joint-minus-exact covariance effect is "
            f"{primary['global_equal_cell_delta']:.6f} "
            f"(one-sided replay p = {primary['global_replay_p_one_sided']:.6f}; "
            f"95% shift interval [{primary['global_effect_interval']['low']:.6f}, "
            f"{primary['global_effect_interval']['high']:.6f}])."
        ),
        "",
        (
            f"{len(primary['holm_positive_cells_at_practical_floor'])} of 16 cells are "
            "Holm-positive at the 0.002 practical floor, covering "
            f"{len(primary['positive_model_families'])} model families and "
            f"{len(primary['positive_tasks'])} tasks."
        ),
        "",
        "This analysis measures reference-relative disagreement dependence. It does not measure "
        "confidence-interval coverage; that requires the separate B4 repeated-sampling stage.",
        "",
        "## Cells",
        "",
        "| family | task | joint cov. | exact cov. | delta | raw p | Holm p | interaction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in result["cells"]:
        cov = cell["covariance_unresolved_as_correct"]
        assessment = cell["cell_assessment"]
        lines.append(
            f"| {cell['family']} | {cell['task_id']} | {cov['joint_mean']:.6f} | "
            f"{cov['exact_context_mean']:.6f} | {cov['joint_minus_exact']:.6f} | "
            f"{assessment['raw_p_value']:.6f} | {assessment['holm_adjusted_p']:.6f} | "
            f"{primary['interactions'][cell['cell_key']]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Malformed-output sensitivity",
            "",
            "| treatment | global delta | at least 0.002 |",
            "|---|---:|---:|",
        ]
    )
    for treatment, value in result["malformed_output_sensitivity"][
        "global_equal_cell_delta"
    ].items():
        lines.append(
            f"| {treatment} | {value:.6f} | {'yes' if value >= PRACTICAL_FLOOR else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Model and task margins",
            "",
            "| margin | delta |",
            "|---|---:|",
        ]
    )
    for family, value in primary["model_margins"].items():
        lines.append(f"| model: {family} | {value:.6f} |")
    for task, value in primary["task_margins"].items():
        lines.append(f"| task: {task} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Leave-one-out checks",
            "",
            "| omitted margin | remaining-grid delta |",
            "|---|---:|",
        ]
    )
    for family, value in primary["leave_one_model_out"].items():
        lines.append(f"| model: {family} | {value:.6f} |")
    for task, value in primary["leave_one_task_out"].items():
        lines.append(f"| task: {task} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Accuracy and parse moderators",
            "",
            "| family | task | joint acc. | exact acc. | solo acc. | joint parse | "
            "exact parse | solo parse | terminal malformed |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in result["cells"]:
        accuracy = cell["accuracy"]
        quality = cell["data_quality"]["by_condition"]
        lines.append(
            f"| {cell['family']} | {cell['task_id']} | "
            f"{accuracy['joint']['accuracy_resolved']:.6f} | "
            f"{accuracy['exact_context']['accuracy_resolved']:.6f} | "
            f"{accuracy['solo']['accuracy_resolved']:.6f} | "
            f"{quality['joint']['call_parse_rate']:.6f} | "
            f"{quality['exact_context']['call_parse_rate']:.6f} | "
            f"{quality['solo']['call_parse_rate']:.6f} | "
            f"{cell['data_quality']['terminal_malformed_outputs']} |"
        )
    lines.extend(
        [
            "",
            "## Fake-batch controls",
            "",
            "| family | task | exact-context fake covariance | solo fake covariance |",
            "|---|---|---:|---:|",
        ]
    )
    for cell in result["cells"]:
        fake = cell["fake_batch_controls_unresolved_as_correct"]
        lines.append(
            f"| {cell['family']} | {cell['task_id']} | "
            f"{fake['exact_context_mean']:.6f} | {fake['solo_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate checks",
            "",
            "| criterion | passed |",
            "|---|---:|",
        ]
    )
    for criterion, passed in gate["criteria"].items():
        lines.append(f"| {criterion} | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "Every clean event ledger, retry link, call identity, prompt hash, assignment, frame "
            "binding, execution plan, and sealed schedule was verified before analysis. The "
            "excluded interrupted Llama run was not read as scientific evidence.",
            "",
        ]
    )
    return "\n".join(lines)
