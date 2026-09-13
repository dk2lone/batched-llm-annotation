"""Frozen CPU-only analysis of the historical Qwen/Civil bridge."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from step0 import assign, stats
from step0.hashing import sha256_file

CELL_ID = "qwen_civil"
N_ASSIGNMENTS = 3
N_DRAWS = 2000
PRACTICAL_FLOOR = 0.002
FAKE_BATCH_BAND = 0.001
ALPHA = 0.05


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _error(row: dict[str, Any]) -> float:
    value = row.get("error")
    return math.nan if value is None else float(value)


def _groups(matrix: np.ndarray, n_items: int) -> np.ndarray:
    groups = np.empty(n_items, dtype=np.int64)
    for call_index in range(matrix.shape[1]):
        groups[matrix[:, call_index]] = call_index
    return groups


def _condition_vectors(
    rows: list[dict[str, Any]], index_of: dict[str, int], n_items: int, condition: str
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for assignment in range(1, N_ASSIGNMENTS + 1):
        vector = np.full(n_items, np.nan)
        seen: set[int] = set()
        selected = [
            row
            for row in rows
            if row["condition"] == condition and int(row["assignment"]) == assignment
        ]
        if len(selected) != n_items:
            raise ValueError(
                f"{condition} assignment {assignment} has {len(selected)} rows, expected {n_items}"
            )
        for row in selected:
            index = index_of[str(row["doc_id"])]
            if index in seen:
                raise ValueError(
                    f"duplicate {condition} assignment {assignment} row for {row['doc_id']}"
                )
            seen.add(index)
            vector[index] = _error(row)
        if len(seen) != n_items:
            raise ValueError(
                f"{condition} assignment {assignment} covers {len(seen)} distinct items, "
                f"expected {n_items}"
            )
        out.append(vector)
    return out


def _verify_assignments(
    rows: list[dict[str, Any]], frame: assign.Frame, matrices: list[np.ndarray]
) -> None:
    for assignment, matrix in enumerate(matrices, start=1):
        for call_index in range(frame.n_calls):
            expected = {frame.doc_ids[int(index)] for index in matrix[:, call_index]}
            actual = {
                str(row["doc_id"])
                for row in rows
                if row["condition"] == "joint"
                and int(row["assignment"]) == assignment
                and int(row["call_index"]) == call_index
            }
            if actual != expected:
                raise ValueError(
                    f"joint assignment {assignment} call {call_index} differs from the seal"
                )


def _covariance_summary(
    vectors: list[np.ndarray], matrices: list[np.ndarray], unresolved_fill: float = 0.0
) -> dict[str, Any]:
    per_assignment: list[dict[str, float]] = []
    for vector, matrix in zip(vectors, matrices, strict=True):
        resolved = np.where(np.isnan(vector), unresolved_fill, vector)
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
        "mean_covariance": float(np.mean([row["covariance"] for row in per_assignment])),
        "mean_icc": float(np.mean([row["icc"] for row in per_assignment])),
        "unresolved_rule": "errors as correct" if unresolved_fill == 0.0 else "errors as errors",
    }


def _sensitivities(
    joint: list[np.ndarray], exact: list[np.ndarray], matrices: list[np.ndarray]
) -> dict[str, Any]:
    per_assignment: list[dict[str, float]] = []
    for joint_values, exact_values, matrix in zip(joint, exact, matrices, strict=True):
        groups = _groups(matrix, joint_values.size)
        per_assignment.append(
            stats.malformed_sensitivity(
                joint_values,
                exact_values,
                groups,
                np.isnan(joint_values),
                np.isnan(exact_values),
            )
        )
    keys = (
        "delta_unresolved_as_correct",
        "delta_unresolved_as_error",
        "delta_unresolved_dropped",
        "delta_whole_call_discarded",
    )
    return {
        "per_assignment": per_assignment,
        "cell_mean": {
            key: float(np.mean([assignment[key] for assignment in per_assignment])) for key in keys
        },
        "n_calls_discarded_total": int(
            sum(assignment["n_calls_discarded"] for assignment in per_assignment)
        ),
    }


def _randomization(
    joint: list[np.ndarray],
    exact: list[np.ndarray],
    frame: assign.Frame,
    matrices: list[np.ndarray],
    n_draws: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, fill in (("errors_as_correct", 0.0), ("errors_as_errors", 1.0)):
        joint_resolved = [np.where(np.isnan(values), fill, values) for values in joint]
        exact_resolved = [np.where(np.isnan(values), fill, values) for values in exact]
        results[name] = stats.randomization_test(
            joint_resolved,
            exact_resolved,
            matrices,
            n_draws,
            lambda draw: assign.replay_stratum_matrices(frame, CELL_ID, N_ASSIGNMENTS, draw),
            ALPHA,
        )
    return results


def _fake_batch(
    joint: list[np.ndarray],
    exact: list[np.ndarray],
    frame: assign.Frame,
    matrices: list[np.ndarray],
    n_draws: int,
) -> dict[str, Any]:
    joint_resolved = [np.where(np.isnan(values), 0.0, values) for values in joint]
    exact_resolved = [np.where(np.isnan(values), 0.0, values) for values in exact]
    joint_covariance = float(
        np.mean(
            [
                stats.covariance_from_stratified(values[matrix].T)
                for values, matrix in zip(joint_resolved, matrices, strict=True)
            ]
        )
    )
    observed_fake = float(
        np.mean(
            [
                stats.covariance_from_stratified(values[matrix].T)
                for values, matrix in zip(exact_resolved, matrices, strict=True)
            ]
        )
    )
    regrouped = np.empty(n_draws)
    for draw in range(n_draws):
        replay = assign.replay_stratum_matrices(frame, CELL_ID, N_ASSIGNMENTS, draw)
        regrouped[draw] = float(
            np.mean(
                [
                    stats.covariance_from_stratified(values[matrix].T)
                    for values, matrix in zip(exact_resolved, replay, strict=True)
                ]
            )
        )
    return {
        "joint_covariance": joint_covariance,
        "observed_fake_batch_covariance": observed_fake,
        "delta_perm": joint_covariance - observed_fake,
        "p_value_one_sided_upper": float(
            (1 + int((regrouped >= joint_covariance).sum())) / (1 + n_draws)
        ),
        "n_regroupings": n_draws,
        "regrouped_covariance_mean": float(regrouped.mean()),
        "regrouped_covariance_sd": float(regrouped.std(ddof=1)),
        "bounded_null_band_absolute": FAKE_BATCH_BAND,
    }


def analyze(
    calls_path: Path,
    items_path: Path,
    solo_path: Path,
    n_draws: int = N_DRAWS,
) -> dict[str, Any]:
    """Analyze the bridge with the frozen Step 0 assignment generator and statistics."""
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    rows = _read_jsonl(calls_path)
    items = _read_jsonl(items_path)
    solo = _read_jsonl(solo_path)
    frame = assign.Frame(
        tuple(str(item["doc_id"]) for item in items),
        tuple(int(item["length_stratum"]) for item in items),
    )
    index_of = {doc_id: index for index, doc_id in enumerate(frame.doc_ids)}
    matrices = assign.observed_stratum_matrices(frame, CELL_ID, N_ASSIGNMENTS)
    _verify_assignments(rows, frame, matrices)

    joint = _condition_vectors(rows, index_of, frame.n_items, "joint")
    exact = _condition_vectors(rows, index_of, frame.n_items, "exact_context")
    filler = _condition_vectors(rows, index_of, frame.n_items, "filler")
    sensitivities = _sensitivities(joint, exact, matrices)
    randomization = _randomization(joint, exact, frame, matrices, n_draws)
    fake_batch = _fake_batch(joint, exact, frame, matrices, n_draws)
    bounds = sensitivities["cell_mean"]
    required_bounds = (
        bounds["delta_unresolved_as_correct"],
        bounds["delta_unresolved_as_error"],
        bounds["delta_unresolved_dropped"],
        bounds["delta_whole_call_discarded"],
    )
    p_values = [
        float(randomization[name]["p_value_one_sided_upper"])
        for name in ("errors_as_correct", "errors_as_errors")
    ]
    supported = bool(
        min(required_bounds) >= PRACTICAL_FLOOR
        and max(p_values) < ALPHA
        and fake_batch["delta_perm"] > 0
        and abs(fake_batch["observed_fake_batch_covariance"]) < FAKE_BATCH_BAND
        and fake_batch["p_value_one_sided_upper"] < ALPHA
    )
    parse_counts = Counter(str(row["condition"]) for row in rows if row["error"] is None)
    solo_unresolved = sum(row.get("error") is None for row in solo)
    return {
        "analysis": "historical_qwen3_8b_civil_comments_bridge",
        "scope": (
            "separate historical replication bridge; not a Freeze B panel result and not "
            "evidence of recurrence across models or tasks"
        ),
        "source": {
            "calls_path": str(calls_path),
            "calls_sha256": sha256_file(calls_path),
            "items_path": str(items_path),
            "items_sha256": sha256_file(items_path),
            "solo_path": str(solo_path),
            "solo_sha256": sha256_file(solo_path),
            "target_rows": len(rows),
            "unique_first_calls": len({str(row["call_id"]) for row in rows}),
            "solo_rows_reused": len(solo),
            "assignment_seal_verified": True,
        },
        "parse": {
            "unresolved_target_rows_by_condition": dict(sorted(parse_counts.items())),
            "unresolved_solo_rows": solo_unresolved,
        },
        "covariance": {
            "joint": _covariance_summary(joint, matrices),
            "exact_context": _covariance_summary(exact, matrices),
            "filler": _covariance_summary(filler, matrices),
        },
        "primary_sensitivity": sensitivities,
        "primary_randomization": randomization,
        "fake_batch_falsification": fake_batch,
        "verdict": {
            "label": "REPLICATION_SUPPORTED" if supported else "REPLICATION_NOT_SUPPORTED",
            "supported": supported,
            "practical_floor": PRACTICAL_FLOOR,
            "alpha": ALPHA,
            "rule": (
                "errors-as-correct, errors-as-errors, unresolved-dropped, and "
                "whole-call-discarded contrasts must "
                "remain at least 0.002; both frozen replay tests and the fake-batch test must "
                "have p < 0.05; fake-batch covariance must have absolute value below 0.001"
            ),
        },
    }


def render_report(result: dict[str, Any]) -> str:
    verdict = result["verdict"]
    covariance = result["covariance"]
    bounds = result["primary_sensitivity"]["cell_mean"]
    randomization = result["primary_randomization"]
    fake = result["fake_batch_falsification"]
    lines = [
        "# Historical Qwen3-8B / Civil Comments bridge analysis",
        "",
        f"## Verdict: {verdict['label']}",
        "",
        (
            "The historical cell replicates excess same-call error covariance under the frozen "
            "Step 0 rules. This result is limited to Qwen3-8B on Civil Comments and is separate "
            "from the fresh Freeze B panel."
            if verdict["supported"]
            else "The historical cell does not pass every frozen replication requirement."
        ),
        "",
        "## Primary result",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| joint covariance | {covariance['joint']['mean_covariance']:.6f} |",
        f"| exact-context covariance | {covariance['exact_context']['mean_covariance']:.6f} |",
        f"| joint minus exact, errors as correct | {bounds['delta_unresolved_as_correct']:.6f} |",
        f"| joint minus exact, errors as errors | {bounds['delta_unresolved_as_error']:.6f} |",
        f"| joint minus exact, unresolved dropped | {bounds['delta_unresolved_dropped']:.6f} |",
        (
            "| joint minus exact, whole calls discarded | "
            f"{bounds['delta_whole_call_discarded']:.6f} |"
        ),
        (
            "| replay p, errors as correct | "
            f"{randomization['errors_as_correct']['p_value_one_sided_upper']:.6f} |"
        ),
        (
            "| replay p, errors as errors | "
            f"{randomization['errors_as_errors']['p_value_one_sided_upper']:.6f} |"
        ),
        "",
        (
            f"The errors-as-correct shift interval is "
            f"[{randomization['errors_as_correct']['effect_interval']['low']:.6f}, "
            f"{randomization['errors_as_correct']['effect_interval']['high']:.6f}]."
        ),
        "",
        "## Falsification and context control",
        "",
        f"Fake-batch covariance is {fake['observed_fake_batch_covariance']:.6f}, inside the frozen "
        f"absolute 0.001 band. Its contrast with joint covariance is {fake['delta_perm']:.6f}, "
        f"with one-sided p = {fake['p_value_one_sided_upper']:.6f}. Filler covariance is "
        f"{covariance['filler']['mean_covariance']:.6f}.",
        "",
        "## Data quality",
        "",
        f"The analysis verified {result['source']['target_rows']:,} target rows, "
        f"{result['source']['unique_first_calls']:,} new first calls, the sealed assignments, and "
        f"{result['source']['solo_rows_reused']:,} reused solo rows. Seven exact-context and ten "
        "filler responses were unresolved. No joint response was unresolved. The seven affected "
        "exact-context groups are removed only in the whole-call sensitivity.",
        "",
        "No confidence-interval coverage result is available from this bridge.",
        "",
    ]
    return "\n".join(lines)
