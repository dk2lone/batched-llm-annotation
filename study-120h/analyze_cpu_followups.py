"""CPU-only follow-up analyses for the B3/B4/B5 manuscript results.

This script reads authenticated existing outputs and writes a new analysis bundle. It never
modifies a sealed protocol, execution ledger, recovery record, or prior analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

from freezeb import b4_revised  # noqa: E402
from freezeb import extension as ext  # noqa: E402
from freezeb import extension_b3_analysis as b3  # noqa: E402
from freezeb import extension_b4 as b4  # noqa: E402
from step0 import stats  # noqa: E402
from step0.hashing import sha256_file  # noqa: E402

B4_REVISED = HERE / "extension-b4/revised-analysis-20260918T034636Z"
B3_ANALYSIS = HERE / "extension-b3/analysis/analysis.json"
B5_ANALYSIS = HERE / "extension-b5/run-01/analysis-final-a/ANALYSIS.json"
FLOOR = 0.002
ALPHA = 0.05
BOOT_DRAWS = 2_000
PAIR_BOOT_DRAWS = 20_000


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def clean_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def wilson_interval(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float]:
    flat = np.asarray(values, dtype=float).ravel()
    if flat.size == 0 or not np.isfinite(flat).all():
        return math.nan, math.nan, math.nan
    n = flat.size
    point = float(flat.mean())
    z = float(sps.norm.ppf(1 - alpha / 2))
    denominator = 1 + z**2 / n
    centre = (point + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(point * (1 - point) / n + z**2 / (4 * n**2)) / denominator
    return point, max(0.0, centre - half), min(1.0, centre + half)


def scaled_interval(point: float, low: float, high: float, factor: float) -> tuple[float, float]:
    return max(0.0, point - factor * (point - low)), min(1.0, point + factor * (high - point))


def coverage(low: np.ndarray, high: np.ndarray, target: float) -> dict[str, Any]:
    return stats.coverage(np.asarray(low), np.asarray(high), target)


def paired_comparison(
    base_hits: np.ndarray, method_hits: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    base = np.asarray(base_hits, dtype=bool)
    method = np.asarray(method_hits, dtype=bool)
    if base.shape != method.shape:
        raise ValueError("paired coverage indicators must have the same shape")
    gain = (~base) & method
    loss = base & (~method)
    discordant = int(gain.sum() + loss.sum())
    p_value = float(sps.binomtest(int(gain.sum()), discordant, 0.5).pvalue) if discordant else 1.0
    differences = method.astype(float) - base.astype(float)
    indices = rng.integers(0, differences.size, size=(PAIR_BOOT_DRAWS, differences.size))
    draws = differences[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "coverage_difference": float(differences.mean()),
        "paired_bootstrap_95": [float(low), float(high)],
        "gained_replicates": int(gain.sum()),
        "lost_replicates": int(loss.sum()),
        "exact_mcnemar_p_two_sided": p_value,
    }


def reconstruct_b4() -> dict[str, dict[str, Any]]:
    manifest = load(B4_REVISED / "MANIFEST.json")
    base_run = Path(str(manifest["base_run"]))
    recovery_dir = Path(str(manifest["recovery_dir"]))
    inventory_path = Path(str(manifest["inventory"]))
    records, repaired, hashes = b4_revised.recovery_evidence(base_run, recovery_dir, inventory_path)
    output: dict[str, dict[str, Any]] = {}
    for cell in b4.CELLS:
        rows = b4.frame(HERE, cell.rsplit("::", 1)[1])
        arrays: dict[str, np.ndarray] = {}
        originals: dict[str, np.ndarray] = {}
        reports: dict[str, Any] = {}
        for stream, count in (("reference", 250), ("evaluation", 400)):
            path = b4.stream_path(base_run, cell, stream)
            planned = list(b4.jobs(cell, rows, stream, count, render=False))
            stream_ids = {str(job["call_id"]) for job in planned}
            stream_records = {key: value for key, value in records.items() if key in stream_ids}
            stream_repairs = {key: value for key, value in repaired.items() if key in stream_ids}
            stored = b4.events(path)
            arrays[stream], reports[stream] = b4_revised.revised_stream_errors(
                planned, stored, rows, stream_records, stream_repairs
            )
            originals[stream], _ = b4.stream_errors(planned, stored, rows)
        output[cell] = {
            "arrays": arrays,
            "originals": originals,
            "reports": reports,
            "recovery_hashes": hashes,
        }
    return output


def analyze_e2(b4_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(2026091902)
    cells: list[dict[str, Any]] = []
    for cell in b4.CELLS:
        reference = cast(dict[str, np.ndarray], b4_data[cell]["arrays"])["reference"]
        evaluation = cast(dict[str, np.ndarray], b4_data[cell]["arrays"])["evaluation"]
        target = float(reference.mean(axis=(1, 2)).mean())
        fpc = stats.fpc(512, 8192)
        method_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        widths: dict[str, float] = {}
        for method in ("iid_wald", "iid_wilson", "call_t", "iid_wilson_fpc", "call_t_fpc"):
            lows: list[float] = []
            highs: list[float] = []
            for block in evaluation:
                flat = block.ravel()
                if method.startswith("iid_wald"):
                    interval = stats.wald_interval(flat)
                    point, low, high = interval.point, interval.low, interval.high
                elif method.startswith("iid_wilson"):
                    point, low, high = wilson_interval(flat)
                else:
                    interval = stats.call_t_interval(block.mean(axis=1))
                    point, low, high = interval.point, interval.low, interval.high
                if method.endswith("_fpc"):
                    low, high = scaled_interval(float(point), float(low), float(high), fpc)
                lows.append(float(low))
                highs.append(float(high))
            low_array, high_array = np.asarray(lows), np.asarray(highs)
            method_bounds[method] = (low_array, high_array)
            widths[method] = float(np.mean(high_array - low_array))

        boot_low: list[float] = []
        boot_high: list[float] = []
        boot_rng = np.random.default_rng(
            int.from_bytes(cell.encode("utf-8"), "little", signed=False) % (2**63 - 1)
        )
        for block in evaluation:
            interval = stats.whole_call_bootstrap(block.mean(axis=1), 2_000, boot_rng)
            boot_low.append(interval.low)
            boot_high.append(interval.high)
        method_bounds["whole_call_bootstrap"] = (np.asarray(boot_low), np.asarray(boot_high))
        widths["whole_call_bootstrap"] = float(
            np.mean(np.asarray(boot_high) - np.asarray(boot_low))
        )

        rows: dict[str, Any] = {}
        hits: dict[str, np.ndarray] = {}
        for method, (low, high) in method_bounds.items():
            rows[method] = {**coverage(low, high, target), "mean_width": widths[method]}
            hits[method] = (low <= target) & (target <= high)
        comparisons = {
            method: paired_comparison(hits["iid_wald"], indicators, rng)
            for method, indicators in hits.items()
            if method != "iid_wald"
        }

        original_reference = cast(dict[str, np.ndarray], b4_data[cell]["originals"])["reference"]
        original_evaluation = cast(dict[str, np.ndarray], b4_data[cell]["originals"])["evaluation"]
        reference_complete = np.isfinite(original_reference).all(axis=(1, 2))
        evaluation_complete = np.isfinite(original_evaluation).all(axis=(1, 2))
        original_target = float(original_reference[reference_complete].mean())
        repair_free_intervals = [
            stats.wald_interval(block.ravel()) for block in original_evaluation[evaluation_complete]
        ]
        repair_free = coverage(
            np.asarray([interval.low for interval in repair_free_intervals]),
            np.asarray([interval.high for interval in repair_free_intervals]),
            original_target,
        )
        cells.append(
            {
                "cell_key": cell,
                "target": target,
                "finite_population_correction": fpc,
                "methods": rows,
                "paired_vs_iid_wald": comparisons,
                "repair_free_subset": {
                    "reference_replicates": int(reference_complete.sum()),
                    "evaluation_replicates": int(evaluation_complete.sum()),
                    "target": original_target,
                    "iid_wald": repair_free,
                    "role": "descriptive subset only; replicate completeness is not randomized",
                },
                "repaired_calls": sum(
                    report["repaired_calls"]
                    for report in cast(dict[str, dict[str, Any]], b4_data[cell]["reports"]).values()
                ),
            }
        )
    return {
        "analysis": "E2 interval-method and recovery robustness",
        "cells": cells,
        "paired_bootstrap_draws": PAIR_BOOT_DRAWS,
        "whole_call_bootstrap_draws": 2_000,
        "notes": [
            "Coverage intervals are exact Clopper-Pearson intervals over 400 indicators.",
            "Paired comparisons operate on the same evaluation replicates.",
            "FPC variants multiply each method's half-width by sqrt(1-512/8192).",
            "Whole-call bootstrap rows are fresh 2,000-draw recomputations with "
            "deterministic cell-derived seeds; the manuscript retains the sealed B4 "
            "bootstrap summary.",
        ],
    }


def prepare_b3_cells() -> list[b3.PreparedCell]:
    lock_root = HERE / "full-grid-lock/20260914T032937Z-X1R4-A8"
    b3._verify_manifest(lock_root)
    lock = b3._load(lock_root / "FULL_GRID_LOCK.json")
    cells = [
        b3.prepare_cell(model, task, HERE, lock)
        for model in cast(list[dict[str, Any]], lock["models"])
        for task in cast(list[str], lock["claim_dimensions"]["tasks"])
    ]
    by_key = {cell.cell.key: cell for cell in cells}
    return [by_key[key] for key in cast(list[str], lock["ordered_cells"])]


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    return cast(dict[str, float], stats.holm(p_values, ALPHA)["adjusted_p"])


def analyze_e3(cells: list[b3.PreparedCell]) -> dict[str, Any]:
    replay = b3._replay_all(cells, b3.N_DRAWS, 1)
    raw: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        observed, null, original = replay[cell.cell.key]
        centred = null - null.mean()
        p_floor = float((1 + int((centred >= observed - FLOOR).sum())) / (1 + len(centred)))
        raw[cell.cell.key] = p_floor
        rows.append(
            {
                "cell_key": cell.cell.key,
                "delta": observed,
                "p_zero": original["p_value_one_sided_upper"],
                "p_floor_shift_sensitivity": p_floor,
                "effect_interval": original["effect_interval"],
            }
        )
    adjusted = holm_adjust(raw)
    for row in rows:
        row["holm_p_floor_shift_sensitivity"] = adjusted[row["cell_key"]]
        row["lower_bound_above_floor"] = row["effect_interval"]["low"] > FLOOR
        row["passes_shifted_floor_sensitivity"] = adjusted[row["cell_key"]] <= ALPHA
    return {
        "analysis": "E3 practical-floor shifted-replay sensitivity",
        "floor": FLOOR,
        "cells": rows,
        "passing_cells": [
            row["cell_key"] for row in rows if row["passes_shifted_floor_sensitivity"]
        ],
        "interpretation_boundary": (
            "This inverts the existing mean-centred replay distribution as a location-shift "
            "sensitivity. It is not an exact randomization test of the composite causal null "
            "delta <= 0.002."
        ),
    }


def covariance(matrix: np.ndarray) -> float:
    return stats.covariance_from_stratified(np.asarray(matrix, dtype=float))


def condition_standardized(cov: float, values: np.ndarray) -> float:
    p = float(np.asarray(values).mean())
    variance = p * (1 - p)
    return cov / variance if variance > 0 else math.nan


def call_bootstrap(cells: list[b3.PreparedCell]) -> dict[str, Any]:
    rng = np.random.default_rng(2026091904)
    draws = np.empty(BOOT_DRAWS, dtype=float)
    prepared: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for cell in cells:
        assignments: list[tuple[np.ndarray, np.ndarray]] = []
        for joint, exact, matrix in zip(cell.joint, cell.exact, cell.matrices, strict=True):
            assignments.append(
                (
                    np.where(np.isnan(joint), 0.0, joint)[matrix].T,
                    np.where(np.isnan(exact), 0.0, exact)[matrix].T,
                )
            )
        prepared.append(assignments)
    for draw in range(BOOT_DRAWS):
        cell_values = []
        for assignments in prepared:
            assignment_values = []
            for joint_calls, exact_calls in assignments:
                indices = rng.integers(0, joint_calls.shape[0], size=joint_calls.shape[0])
                assignment_values.append(
                    covariance(joint_calls[indices]) - covariance(exact_calls[indices])
                )
            cell_values.append(float(np.mean(assignment_values)))
        draws[draw] = float(np.mean(cell_values))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "draws": BOOT_DRAWS,
        "mean": float(draws.mean()),
        "interval_95": [float(low), float(high)],
        "boundary": (
            "Paired calls are resampled within each realized assignment. This is conditional "
            "on realized item outputs and does not model item reuse across assignments."
        ),
    }


def analyze_e4(cells: list[b3.PreparedCell]) -> dict[str, Any]:
    assignment_global = []
    per_cell = []
    for assignment in range(ext.N_ASSIGNMENTS):
        assignment_global.append(
            float(
                np.mean(
                    [
                        covariance(
                            np.where(
                                np.isnan(cell.joint[assignment]),
                                0.0,
                                cell.joint[assignment],
                            )[cell.matrices[assignment]].T
                        )
                        - covariance(
                            np.where(
                                np.isnan(cell.exact[assignment]),
                                0.0,
                                cell.exact[assignment],
                            )[cell.matrices[assignment]].T
                        )
                        for cell in cells
                    ]
                )
            )
        )
    for cell in cells:
        raw_deltas = []
        standardized_deltas = []
        for joint, exact, matrix in zip(cell.joint, cell.exact, cell.matrices, strict=True):
            joint_calls = np.where(np.isnan(joint), 0.0, joint)[matrix].T
            exact_calls = np.where(np.isnan(exact), 0.0, exact)[matrix].T
            joint_cov, exact_cov = covariance(joint_calls), covariance(exact_calls)
            raw_deltas.append(joint_cov - exact_cov)
            standardized_deltas.append(
                condition_standardized(joint_cov, joint_calls)
                - condition_standardized(exact_cov, exact_calls)
            )
        standardized = float(np.mean(standardized_deltas))
        design_effect = 1 + (ext.BATCH_SIZE - 1) * standardized
        per_cell.append(
            {
                "cell_key": cell.cell.key,
                "raw_delta_per_assignment": [float(value) for value in raw_deltas],
                "raw_delta": float(np.mean(raw_deltas)),
                "standardized_excess_dependence": standardized,
                "excess_design_effect": design_effect,
                "effective_n_from_8192": 8192 / design_effect if design_effect > 0 else None,
            }
        )
    leave_one_assignment_out = [
        float(np.mean([value for index, value in enumerate(assignment_global) if index != omitted]))
        for omitted in range(ext.N_ASSIGNMENTS)
    ]
    return {
        "analysis": "E4 assignment and scale robustness",
        "global_delta_per_assignment": assignment_global,
        "global_delta_leave_one_assignment_out": leave_one_assignment_out,
        "cell_results": per_cell,
        "global_standardized_excess_dependence": float(
            np.mean([row["standardized_excess_dependence"] for row in per_cell])
        ),
        "call_bootstrap": call_bootstrap(cells),
        "design_effect_note": (
            "1 + 15 times standardized joint-minus-exact dependence; this isolates excess "
            "dependence relative to exact context and is not the total joint-arm design effect."
        ),
    }


def analyze_e5(cells: list[b3.PreparedCell]) -> dict[str, Any]:
    position_rows = []
    distance_rows = []
    all_joint_positions: list[np.ndarray] = []
    all_exact_positions: list[np.ndarray] = []
    all_joint_distance: dict[int, list[float]] = {distance: [] for distance in range(1, 16)}
    all_exact_distance: dict[int, list[float]] = {distance: [] for distance in range(1, 16)}
    for cell in cells:
        for assignment, (joint, exact, matrix) in enumerate(
            zip(cell.joint, cell.exact, cell.matrices, strict=True), start=1
        ):
            joint_calls = np.where(np.isnan(joint), 0.0, joint)[matrix].T
            exact_calls = np.where(np.isnan(exact), 0.0, exact)[matrix].T
            all_joint_positions.append(joint_calls.mean(axis=0))
            all_exact_positions.append(exact_calls.mean(axis=0))
            joint_centred = joint_calls - joint_calls.mean()
            exact_centred = exact_calls - exact_calls.mean()
            for distance in range(1, 16):
                joint_product = joint_centred[:, :-distance] * joint_centred[:, distance:]
                exact_product = exact_centred[:, :-distance] * exact_centred[:, distance:]
                all_joint_distance[distance].append(float(joint_product.mean()))
                all_exact_distance[distance].append(float(exact_product.mean()))
            position_rows.append(
                {
                    "cell_key": cell.cell.key,
                    "assignment": assignment,
                    "joint_error_by_position": [float(v) for v in joint_calls.mean(axis=0)],
                    "exact_error_by_position": [float(v) for v in exact_calls.mean(axis=0)],
                }
            )
    joint_position = np.mean(np.vstack(all_joint_positions), axis=0)
    exact_position = np.mean(np.vstack(all_exact_positions), axis=0)
    position_delta = joint_position - exact_position
    position_slope = sps.linregress(np.arange(1, 17), position_delta)
    for distance in range(1, 16):
        joint = float(np.mean(all_joint_distance[distance]))
        exact = float(np.mean(all_exact_distance[distance]))
        distance_rows.append(
            {
                "distance": distance,
                "joint_covariance": joint,
                "exact_covariance": exact,
                "delta": joint - exact,
            }
        )
    distance_slope = sps.linregress(
        [row["distance"] for row in distance_rows], [row["delta"] for row in distance_rows]
    )
    return {
        "analysis": "E5 exploratory output-position and pair-distance patterns",
        "position": {
            "joint_error": [float(v) for v in joint_position],
            "exact_error": [float(v) for v in exact_position],
            "joint_minus_exact_error": [float(v) for v in position_delta],
            "linear_slope_per_position": float(position_slope.slope),
            "slope_p_two_sided_descriptive": float(position_slope.pvalue),
        },
        "pair_distance": {
            "rows": distance_rows,
            "linear_slope_per_position_gap": float(distance_slope.slope),
            "slope_p_two_sided_descriptive": float(distance_slope.pvalue),
        },
        "cell_assignment_position_rows": position_rows,
        "interpretation_boundary": (
            "Post hoc descriptive analysis of fixed outputs. Position and distance were not "
            "randomized as separate interventions, so slopes do not identify a mechanism."
        ),
    }


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# CPU follow-up analyses E2-E5",
        "",
        "All analyses use existing B3/B4/B5 outputs. No model inference or GPU work was launched.",
        "",
        "## E2: interval and recovery robustness",
        "",
        "Whole-call bootstrap rows are a fresh seeded 2,000-draw recomputation for each "
        "replicate. Small differences from the sealed B4 bootstrap summary are Monte Carlo "
        "variation; the manuscript retains the prespecified B4 values.",
        "",
    ]
    e2 = result["e2"]
    lines.extend(
        markdown_table(
            ["cell", "method", "coverage", "95% MC interval", "width", "vs Wald"],
            [
                [
                    cell["cell_key"],
                    method,
                    f"{values['point']:.4f}",
                    f"[{values['low']:.4f}, {values['high']:.4f}]",
                    f"{values['mean_width']:.4f}",
                    (
                        "reference"
                        if method == "iid_wald"
                        else f"{cell['paired_vs_iid_wald'][method]['coverage_difference']:+.4f}"
                    ),
                ]
                for cell in e2["cells"]
                for method, values in cell["methods"].items()
            ],
        )
    )
    lines.extend(["", "Repair-free complete-replicate subsets are descriptive only:", ""])
    lines.extend(
        markdown_table(
            ["cell", "reference reps", "evaluation reps", "target", "IID coverage"],
            [
                [
                    cell["cell_key"],
                    str(cell["repair_free_subset"]["reference_replicates"]),
                    str(cell["repair_free_subset"]["evaluation_replicates"]),
                    f"{cell['repair_free_subset']['target']:.6f}",
                    f"{cell['repair_free_subset']['iid_wald']['point']:.4f}",
                ]
                for cell in e2["cells"]
            ],
        )
    )
    e3 = result["e3"]
    lines.extend(["", "## E3: direct practical-floor sensitivity", ""])
    lines.extend(
        markdown_table(
            ["cell", "delta", "Holm p vs 0.002", "shift CI low", "passes"],
            [
                [
                    row["cell_key"],
                    f"{row['delta']:.6f}",
                    f"{row['holm_p_floor_shift_sensitivity']:.4f}",
                    f"{row['effect_interval']['low']:.6f}",
                    "yes" if row["passes_shifted_floor_sensitivity"] else "no",
                ]
                for row in e3["cells"]
            ],
        )
    )
    lines.extend(["", e3["interpretation_boundary"], ""])
    e4 = result["e4"]
    lines.extend(
        [
            "## E4: assignment and scale robustness",
            "",
            "Global delta by assignment: "
            + ", ".join(f"{value:.6f}" for value in e4["global_delta_per_assignment"])
            + ".",
            "",
            "Leave-one-assignment-out global delta: "
            + ", ".join(f"{value:.6f}" for value in e4["global_delta_leave_one_assignment_out"])
            + ".",
            "",
            f"Conditional paired-call bootstrap 95% interval: "
            f"[{e4['call_bootstrap']['interval_95'][0]:.6f}, "
            f"{e4['call_bootstrap']['interval_95'][1]:.6f}].",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["cell", "raw delta", "standardized delta", "excess DE", "effective N"],
            [
                [
                    row["cell_key"],
                    f"{row['raw_delta']:.6f}",
                    f"{row['standardized_excess_dependence']:.4f}",
                    f"{row['excess_design_effect']:.3f}",
                    f"{row['effective_n_from_8192']:.0f}",
                ]
                for row in e4["cell_results"]
            ],
        )
    )
    e5 = result["e5"]
    lines.extend(
        [
            "",
            "## E5: exploratory output-position patterns",
            "",
            f"Joint-minus-exact error slope per output position: "
            f"{e5['position']['linear_slope_per_position']:.6f} "
            f"(descriptive p={e5['position']['slope_p_two_sided_descriptive']:.4f}).",
            "",
            f"Excess covariance slope per pair-distance step: "
            f"{e5['pair_distance']['linear_slope_per_position_gap']:.6f} "
            f"(descriptive p={e5['pair_distance']['slope_p_two_sided_descriptive']:.4f}).",
            "",
            e5["interpretation_boundary"],
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["distance", "joint cov.", "exact cov.", "delta"],
            [
                [
                    str(row["distance"]),
                    f"{row['joint_covariance']:.6f}",
                    f"{row['exact_covariance']:.6f}",
                    f"{row['delta']:.6f}",
                ]
                for row in e5["pair_distance"]["rows"]
            ],
        )
    )
    lines.extend(["", "## Input integrity", ""])
    for path, digest in result["input_sha256"].items():
        lines.append(f"- `{path}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def write_csvs(out: Path, result: dict[str, Any]) -> None:
    with (out / "e2_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "method", "coverage", "low", "high", "mean_width"])
        for cell in result["e2"]["cells"]:
            for method, values in cell["methods"].items():
                writer.writerow(
                    [
                        cell["cell_key"],
                        method,
                        values["point"],
                        values["low"],
                        values["high"],
                        values["mean_width"],
                    ]
                )
    with (out / "e3_floor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["e3"]["cells"][0]))
        writer.writeheader()
        writer.writerows(result["e3"]["cells"])
    with (out / "e4_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "cell_key",
            "raw_delta",
            "standardized_excess_dependence",
            "excess_design_effect",
            "effective_n_from_8192",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["e4"]["cell_results"])
    with (out / "e5_pair_distance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["e5"]["pair_distance"]["rows"][0]))
        writer.writeheader()
        writer.writerows(result["e5"]["pair_distance"]["rows"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "cpu-followups/20260919-e2-e5",
        help="new output directory; an existing directory is refused",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {out}")

    b4_data = reconstruct_b4()
    b3_cells = prepare_b3_cells()
    result = {
        "schema_version": "cpu-followups-e2-e5-v1",
        "gpu_work_launched": False,
        "e2": analyze_e2(b4_data),
        "e3": analyze_e3(b3_cells),
        "e4": analyze_e4(b3_cells),
        "e5": analyze_e5(b3_cells),
        "input_sha256": {
            str(B4_REVISED / "REVISED_ANALYSIS.json"): sha256_file(
                B4_REVISED / "REVISED_ANALYSIS.json"
            ),
            str(B3_ANALYSIS): sha256_file(B3_ANALYSIS),
            str(B5_ANALYSIS): sha256_file(B5_ANALYSIS),
        },
    }
    out.mkdir(parents=True)
    (out / "ANALYSIS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    write_csvs(out, result)
    print(out)


if __name__ == "__main__":
    main()
