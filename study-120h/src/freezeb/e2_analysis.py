"""Frozen CPU-only analysis for completed Freeze B E2 cells."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

from step0 import stats
from step0.hashing import sha256_file

from . import assign, e2
from .labels import disagreement

N_DRAWS = 4_999
ALPHA = 0.05
PRACTICAL_FLOOR = 0.002
PRIMARY_FAMILY_SIZE = 16


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _groups(matrix: np.ndarray, n_items: int) -> np.ndarray:
    groups = np.empty(n_items, dtype=np.int64)
    for call_index in range(matrix.shape[1]):
        groups[matrix[:, call_index]] = call_index
    return groups


def _summary(
    vectors: list[np.ndarray], matrices: list[np.ndarray], fill: float = 0.0
) -> dict[str, Any]:
    per_assignment: list[dict[str, float]] = []
    for vector, matrix in zip(vectors, matrices, strict=True):
        resolved = np.where(np.isnan(vector), fill, vector)
        covariance = stats.covariance_from_stratified(resolved[matrix].T)
        error_rate = float(resolved.mean())
        denominator = error_rate * (1.0 - error_rate)
        per_assignment.append(
            {
                "error_rate": error_rate,
                "accuracy": 1.0 - error_rate,
                "covariance": covariance,
                "icc": covariance / denominator if denominator > 0 else math.nan,
            }
        )
    return {
        "per_assignment": per_assignment,
        "mean_error_rate": float(np.mean([row["error_rate"] for row in per_assignment])),
        "mean_accuracy": float(np.mean([row["accuracy"] for row in per_assignment])),
        "mean_covariance": float(np.mean([row["covariance"] for row in per_assignment])),
        "mean_icc": float(np.mean([row["icc"] for row in per_assignment])),
        "unresolved_rule": "errors as correct" if fill == 0.0 else "errors as errors",
    }


def _vectors(
    events: list[dict[str, Any]], rows: list[dict[str, Any]], condition: str
) -> list[np.ndarray]:
    index_of = {str(row["doc_id"]): index for index, row in enumerate(rows)}
    reference = {str(row["doc_id"]): row["reference_normalized"] for row in rows}
    vectors: list[np.ndarray] = []
    for assignment in range(1, assign.N_ASSIGNMENTS + 1):
        vector = np.full(len(rows), np.nan)
        seen: set[str] = set()
        selected = [
            event
            for event in events
            if event["condition"] == condition and int(event["assignment"]) == assignment
        ]
        for event in selected:
            predictions = cast(dict[str, Any], event.get("predictions") or {})
            for doc_id in cast(list[str], event["target_ids"]):
                if doc_id in seen:
                    raise ValueError(
                        f"duplicate {condition} assignment {assignment} target {doc_id}"
                    )
                seen.add(doc_id)
                if event.get("status") == "ok" and doc_id in predictions:
                    vector[index_of[doc_id]] = disagreement(
                        predictions[doc_id], reference[doc_id], e2.TASK_ID
                    )
        if seen != set(index_of):
            raise ValueError(
                f"{condition} assignment {assignment} covers {len(seen)} targets, "
                f"expected {len(rows)}"
            )
        vectors.append(vector)
    return vectors


def _solo(events: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = {str(row["doc_id"]): row["reference_normalized"] for row in rows}
    solo = [event for event in events if event["condition"] == "solo"]
    if len(solo) != len(rows) or len({event["target_ids"][0] for event in solo}) != len(rows):
        raise ValueError("solo events do not cover the selected frame exactly once")
    values = []
    for event in solo:
        doc_id = str(event["target_ids"][0])
        predictions = cast(dict[str, Any], event.get("predictions") or {})
        values.append(
            math.nan
            if event.get("status") != "ok" or doc_id not in predictions
            else float(disagreement(predictions[doc_id], reference[doc_id], e2.TASK_ID))
        )
    array = np.asarray(values)
    resolved = array[~np.isnan(array)]
    return {
        "n_items": len(array),
        "n_unresolved": int(np.isnan(array).sum()),
        "error_rate_resolved": float(resolved.mean()) if resolved.size else None,
        "accuracy_resolved": float(1.0 - resolved.mean()) if resolved.size else None,
    }


def _validate_events(
    rows: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    jobs = e2.planned_jobs(rows)
    complete = e2.completeness(jobs, events)
    if not complete["complete"]:
        raise ValueError(f"E2 ledger is incomplete: {complete}")
    first = [event for event in events if int(event["attempt"]) == 1]
    by_call = {str(event["call_id"]): event for event in first}
    if len(first) != len(by_call) or set(by_call) != {str(job["call_id"]) for job in jobs}:
        raise ValueError("E2 first-response call IDs differ from the sealed plan")
    fields = (
        "attempt_id",
        "cell_key",
        "condition",
        "assignment",
        "call_index",
        "execution_block",
        "batch_size",
        "doc_order",
        "target_ids",
    )
    for job in jobs:
        event = by_call[str(job["call_id"])]
        if any(event.get(field) != job.get(field) for field in fields):
            raise ValueError(f"E2 event metadata differs for {job['call_id']}")
    return first


def _sensitivities(
    joint: list[np.ndarray], exact: list[np.ndarray], matrices: list[np.ndarray]
) -> dict[str, Any]:
    per_assignment = []
    for j_values, e_values, matrix in zip(joint, exact, matrices, strict=True):
        groups = _groups(matrix, j_values.size)
        row = stats.malformed_sensitivity(
            j_values,
            e_values,
            groups,
            np.isnan(j_values),
            np.isnan(e_values),
        )
        row["failure_covariance_joint"] = stats.failure_clustering(np.isnan(j_values), groups)
        row["failure_covariance_exact"] = stats.failure_clustering(np.isnan(e_values), groups)
        per_assignment.append(row)
    keys = (
        "delta_unresolved_as_correct",
        "delta_unresolved_as_error",
        "delta_unresolved_dropped",
        "delta_whole_call_discarded",
        "failure_covariance_joint",
        "failure_covariance_exact",
    )
    return {
        "per_assignment": per_assignment,
        "cell_mean": {
            key: float(np.mean([assignment[key] for assignment in per_assignment])) for key in keys
        },
        "n_calls_discarded_total": int(sum(row["n_calls_discarded"] for row in per_assignment)),
    }


def _randomization(
    joint: list[np.ndarray],
    exact: list[np.ndarray],
    frame: assign.Frame,
    matrices: list[np.ndarray],
    n_draws: int,
) -> dict[str, Any]:
    result = {}
    for name, fill in (("errors_as_correct", 0.0), ("errors_as_errors", 1.0)):
        result[name] = stats.randomization_test(
            [np.where(np.isnan(values), fill, values) for values in joint],
            [np.where(np.isnan(values), fill, values) for values in exact],
            matrices,
            n_draws,
            lambda draw: assign.replay_stat_matrices(frame, e2.FIRST_CELL, e2.BATCH_SIZE, draw),
            ALPHA,
        )
    return result


def _fake_batch(
    joint: list[np.ndarray],
    exact: list[np.ndarray],
    frame: assign.Frame,
    matrices: list[np.ndarray],
    n_draws: int,
) -> dict[str, Any]:
    j_resolved = [np.where(np.isnan(values), 0.0, values) for values in joint]
    e_resolved = [np.where(np.isnan(values), 0.0, values) for values in exact]
    joint_covariance = float(
        np.mean(
            [
                stats.covariance_from_stratified(values[matrix].T)
                for values, matrix in zip(j_resolved, matrices, strict=True)
            ]
        )
    )
    observed_fake = float(
        np.mean(
            [
                stats.covariance_from_stratified(values[matrix].T)
                for values, matrix in zip(e_resolved, matrices, strict=True)
            ]
        )
    )
    regrouped = np.empty(n_draws)
    for draw in range(n_draws):
        replay = assign.replay_stat_matrices(frame, e2.FIRST_CELL, e2.BATCH_SIZE, draw)
        regrouped[draw] = float(
            np.mean(
                [
                    stats.covariance_from_stratified(values[matrix].T)
                    for values, matrix in zip(e_resolved, replay, strict=True)
                ]
            )
        )
    return {
        "joint_covariance": joint_covariance,
        "observed_exact_control_covariance": observed_fake,
        "joint_minus_observed_exact": joint_covariance - observed_fake,
        "p_value_joint_vs_regrouped_exact_one_sided": float(
            (1 + int((regrouped >= joint_covariance).sum())) / (1 + n_draws)
        ),
        "n_regroupings": n_draws,
        "regrouped_exact_covariance_mean": float(regrouped.mean()),
        "regrouped_exact_covariance_sd": float(regrouped.std(ddof=1)),
    }


def analyze(
    events_path: Path,
    source_root: Path,
    seal_root: Path,
    admission_lock_path: Path,
    plan_path: Path,
    summary_path: Path,
    n_draws: int = N_DRAWS,
) -> dict[str, Any]:
    """Analyze one complete fresh cell under the sealed Freeze B E2 rules."""
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    admission = cast(dict[str, Any], json.loads(admission_lock_path.read_text(encoding="utf-8")))
    rows = e2.verify_confirm_frame(source_root, seal_root, admission)
    events = _read_jsonl(events_path)
    first = _validate_events(rows, events)
    summary = cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))
    if (
        summary.get("failure") is not None
        or summary.get("completeness", {}).get("complete") is not True
    ):
        raise ValueError("technical summary does not certify a complete E2 cell")
    plan = cast(dict[str, Any], json.loads(plan_path.read_text(encoding="utf-8")))
    if plan.get("plan_sha256") != summary.get("plan_sha256"):
        raise ValueError("executed plan and technical summary hashes differ")

    frame = assign.Frame(
        tuple(str(row["doc_id"]) for row in rows), tuple(int(row["n_chars"]) for row in rows)
    )
    matrices = assign.observed_stat_matrices(frame, e2.FIRST_CELL, e2.BATCH_SIZE)
    joint = _vectors(first, rows, "joint")
    exact = _vectors(first, rows, "exact_context")
    sensitivities = _sensitivities(joint, exact, matrices)
    randomization = _randomization(joint, exact, frame, matrices, n_draws)
    fake_batch = _fake_batch(joint, exact, frame, matrices, n_draws)
    covariance = {
        "joint": _summary(joint, matrices),
        "exact_context": _summary(exact, matrices),
    }
    p_value = float(randomization["errors_as_correct"]["p_value_one_sided_upper"])
    holm_family = {e2.FIRST_CELL: p_value} | {
        f"unrun_primary_slot_{slot:02d}": 1.0 for slot in range(2, PRIMARY_FAMILY_SIZE + 1)
    }
    adjusted = float(stats.holm(holm_family, ALPHA)["adjusted_p"][e2.FIRST_CELL])
    primary_cells = cast(list[str], admission["ordered_cells"]["primary"])
    bounds = cast(dict[str, float], sensitivities["cell_mean"])
    required = [
        bounds["delta_unresolved_as_correct"],
        bounds["delta_unresolved_as_error"],
        bounds["delta_unresolved_dropped"],
        bounds["delta_whole_call_discarded"],
    ]
    status_counts = Counter(str(event["status"]) for event in first)
    parse_counts = Counter(f"{event['condition']}::{event['parse_status']}" for event in first)
    unresolved = {
        condition: int(sum(np.isnan(vector).sum() for vector in vectors))
        for condition, vectors in (("joint", joint), ("exact_context", exact))
    }
    return {
        "analysis": "freezeb_e2_first_cell",
        "cell_key": e2.FIRST_CELL,
        "scope": (
            "one fresh E2 cell; not the historical L1 replication, not a panel recurrence result, "
            "and not a measured confidence-interval coverage experiment"
        ),
        "source": {
            "events_path": str(events_path),
            "events_sha256": sha256_file(events_path),
            "plan_path": str(plan_path),
            "plan_sha256_file": sha256_file(plan_path),
            "executed_plan_sha256": plan["plan_sha256"],
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "admission_lock_sha256": sha256_file(admission_lock_path),
            "n_first_calls": len(first),
            "n_retry_calls": len(events) - len(first),
            "assignment_and_call_metadata_verified": True,
            "technical_summary_verified": True,
        },
        "data_quality": {
            "status_counts": dict(sorted(status_counts.items())),
            "parse_counts_by_condition": dict(sorted(parse_counts.items())),
            "unresolved_target_rows": unresolved,
            "n_calls_discarded_under_whole_call_rule": sensitivities["n_calls_discarded_total"],
        },
        "accuracy": {
            "joint": {
                "per_assignment": [
                    row["accuracy"] for row in covariance["joint"]["per_assignment"]
                ],
                "mean": covariance["joint"]["mean_accuracy"],
            },
            "exact_context": {
                "per_assignment": [
                    row["accuracy"] for row in covariance["exact_context"]["per_assignment"]
                ],
                "mean": covariance["exact_context"]["mean_accuracy"],
            },
            "joint_minus_exact_mean": (
                covariance["joint"]["mean_accuracy"] - covariance["exact_context"]["mean_accuracy"]
            ),
            "solo": _solo(first, rows),
        },
        "covariance": covariance,
        "primary_sensitivity": sensitivities,
        "primary_randomization": randomization,
        "fake_batch_falsification": fake_batch,
        "cell_assessment": {
            "raw_p_value": p_value,
            "holm_adjusted_p_with_16_slots": adjusted,
            "all_four_effect_bounds_at_least_0.002": min(required) >= PRACTICAL_FLOOR,
            "holm_positive": adjusted <= ALPHA,
            "descriptively_positive_and_practical": bool(
                min(required) >= PRACTICAL_FLOOR and adjusted <= ALPHA
            ),
            "practical_floor": PRACTICAL_FLOOR,
            "alpha": ALPHA,
        },
        "panel_status": {
            "claim_available": False,
            "reason": "incomplete_primary_panel_and_locked_family_coverage_insufficient",
            "completed_primary_cells": 1,
            "admitted_primary_cells": len(primary_cells),
            "protocol_target_cells": PRIMARY_FAMILY_SIZE,
            "locked_recurrence_scope": admission["claim_scope"]["recurrence_scope"],
            "distinct_locked_model_families": admission["claim_scope"][
                "distinct_primary_model_families"
            ],
        },
        "coverage_status": {
            "measured": False,
            "reason": "E2 has no independent repeated-sampling evaluation stream; E5 is required",
        },
        "n_draws": n_draws,
    }


def render_report(result: dict[str, Any]) -> str:
    covariance = result["covariance"]
    sensitivity = result["primary_sensitivity"]["cell_mean"]
    replay = result["primary_randomization"]["errors_as_correct"]
    accuracy = result["accuracy"]
    assessment = result["cell_assessment"]
    rows = []
    for assignment in range(assign.N_ASSIGNMENTS):
        joint = covariance["joint"]["per_assignment"][assignment]
        exact = covariance["exact_context"]["per_assignment"][assignment]
        rows.append(
            f"| {assignment + 1} | {joint['accuracy']:.6f} | {exact['accuracy']:.6f} | "
            f"{joint['covariance']:.6f} | {exact['covariance']:.6f} | "
            f"{joint['covariance'] - exact['covariance']:.6f} |"
        )
    finding = (
        "This cell is positive and practically meaningful under its interim 16-slot Holm test."
        if assessment["descriptively_positive_and_practical"]
        else "This cell does not satisfy both the frozen practical and interim multiplicity rules."
    )
    lines = [
        "# Freeze B E2: Qwen3-8B / MultiNLI",
        "",
        "## Cell assessment",
        "",
        finding,
        "",
        (
            "This is one fresh E2 cell. It cannot establish panel recurrence or measured "
            "interval coverage."
        ),
        "",
        "## Raw assignment results",
        "",
        (
            "| assignment | joint accuracy | exact accuracy | joint covariance | "
            "exact covariance | delta |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Primary result",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| joint covariance | {covariance['joint']['mean_covariance']:.6f} |",
        f"| exact-context covariance | {covariance['exact_context']['mean_covariance']:.6f} |",
        (
            "| joint minus exact, errors as correct | "
            f"{sensitivity['delta_unresolved_as_correct']:.6f} |"
        ),
        (
            "| joint minus exact, errors as errors | "
            f"{sensitivity['delta_unresolved_as_error']:.6f} |"
        ),
        (
            "| joint minus exact, unresolved dropped | "
            f"{sensitivity['delta_unresolved_dropped']:.6f} |"
        ),
        (
            "| joint minus exact, whole calls discarded | "
            f"{sensitivity['delta_whole_call_discarded']:.6f} |"
        ),
        f"| replay p, one-sided | {assessment['raw_p_value']:.6f} |",
        f"| Holm p, 16 primary slots | {assessment['holm_adjusted_p_with_16_slots']:.6f} |",
        "",
        f"The replay-centred 95% shift interval is [{replay['effect_interval']['low']:.6f}, "
        f"{replay['effect_interval']['high']:.6f}].",
        "",
        "## Accuracy and falsification",
        "",
        f"Mean joint accuracy is {accuracy['joint']['mean']:.6f}; mean exact-context accuracy is "
        f"{accuracy['exact_context']['mean']:.6f}; their signed difference is "
        f"{accuracy['joint_minus_exact_mean']:.6f}. Solo accuracy is "
        f"{accuracy['solo']['accuracy_resolved']:.6f}.",
        "",
        f"The observed exact-control covariance is "
        f"{result['fake_batch_falsification']['observed_exact_control_covariance']:.6f}. The "
        "exact-control regrouping is reported as a falsification check, separate from the primary "
        "paired replay test.",
        "",
        "## Boundaries",
        "",
        (
            f"All {result['source']['n_first_calls']:,} sealed first calls and their assignment "
            "metadata were verified before calculation. "
            f"{result['data_quality']['unresolved_target_rows']['exact_context']} exact-context "
            "target rows and "
            f"{result['data_quality']['unresolved_target_rows']['joint']} joint target rows were "
            f"unresolved; {accuracy['solo']['n_unresolved']} solo calls were unresolved. "
            f"{result['source']['n_retry_calls']} retries were used."
        ),
        "",
        "Only one primary cell is complete. The admission lock contains ten primary cells from two "
        "model families and already marks the recurrence scope unsupported, so this result cannot "
        "be promoted to an across-model or across-task claim.",
        "",
        (
            "No confidence-interval coverage was measured. That requires the independently "
            "generated E5 repeated-sampling stream."
        ),
        "",
    ]
    return "\n".join(lines)
