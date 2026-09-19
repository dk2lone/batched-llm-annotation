"""Frozen B5 estimands: standardized joint-minus-exact dependence against batch size.

The primary estimand of one cell and one size is the standardized joint dependence
minus the standardized exact-context dependence, where standardized dependence is the
pair-weighted within-call error covariance divided by `p (1 - p)` and `p` is the
marginal error probability of that condition. The primary cross-size statistic is the
ordinary least squares slope of that contrast on `log2(batch size)`, fitted inside each
cell, then averaged with equal weight over the analyzed cells.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from step0 import stats
from step0.hashing import derived_seed, sha256_json

from . import b5
from .labels import disagreement

N_BOOTSTRAP = 2_000
N_REPLAY = 999
ALPHA = 0.05
SUPER_SIZE = max(b5.SIZES)

# Practical thresholds on the standardized scale. The sealed B3 raw floor of 0.002 is a
# covariance, so it is not transferred. The standardized contrast is the excess within-call
# error correlation, and the approximate variance inflation of a size-k call is
# 1 + (k - 1) * correlation. The existing B4 practical inflation floor is 1.25, which at the
# sealed batch size 16 needs a correlation of 0.25 / 15.
PRACTICAL_INFLATION = 1.25
PRACTICAL_CONTRAST = (PRACTICAL_INFLATION - 1.0) / (16 - 1)
# One doubling step that moves a flat zero contrast to the practical contrast across the
# four doublings from size 2 to size 32.
PRACTICAL_SLOPE = PRACTICAL_CONTRAST / (len(b5.SIZES) - 1)

OUTCOME_INTERPRETATIONS = {
    "positive_monotonic": (
        "the cross-family slope interval lies wholly above zero and every cell slope is "
        "positive: pairwise error dependence strengthens with batch size"
    ),
    "flat_positive": (
        "per-size contrasts are above the practical contrast at every size and the slope "
        "interval contains zero and excludes the practical slope: batching introduces "
        "dependence, but dependence does not materially grow with size"
    ),
    "threshold": (
        "contrasts at the two smallest sizes lie below the practical contrast and at least "
        "the largest size lies above it: evidence for a practical batch-size boundary"
    ),
    "non_monotonic": (
        "the per-size contrasts are not ordered in size and the slope interval contains "
        "zero while some per-size intervals exclude it: dependence varies with size "
        "without a simple dose response"
    ),
    "null": (
        "every per-size contrast interval contains zero and lies below the practical "
        "contrast: the earlier size-16 effect does not replicate under the uniform contract"
    ),
    "inconclusive": (
        "the slope interval is wider than twice the practical slope or a cell slope "
        "interval is wider than four times it: the experiment is inconclusive, and the "
        "absence of significance is not evidence of equivalence"
    ),
}


def _standardized(covariance: float, marginal: float) -> float:
    spread = marginal * (1.0 - marginal)
    return float(covariance / spread) if spread > 0 else math.nan


def covariance(values: np.ndarray, n_groups: int, k: int) -> float:
    """Pair-weighted within-call covariance of a permutation-ordered error vector.

    `values` is arranged in canonical permutation order, so reshaping to `(n_groups, k)`
    rebuilds the scheduled groups. Unresolved labels are NaN and drop out, which shrinks
    the affected call. This matches `stats.pair_weighted_covariance` exactly.
    """
    matrix = values.reshape(n_groups, k)
    mask = ~np.isnan(matrix)
    resolved = int(mask.sum())
    if resolved == 0:
        return math.nan
    mean = float(matrix[mask].mean())
    centred = np.where(mask, matrix - mean, 0.0)
    sums = centred.sum(axis=1)
    squares = (centred**2).sum(axis=1)
    counts = mask.sum(axis=1)
    pairs = float((counts * (counts - 1)).sum())
    if pairs <= 0:
        return math.nan
    return float(((sums**2 - squares).sum()) / pairs)


def _marginal(values: np.ndarray) -> float:
    resolved = values[~np.isnan(values)]
    return float(resolved.mean()) if resolved.size else math.nan


def condition_summary(values: np.ndarray, n_groups: int, k: int) -> dict[str, float]:
    raw = covariance(values, n_groups, k)
    marginal = _marginal(values)
    return {
        "covariance": raw,
        "marginal_error": marginal,
        "accuracy": 1.0 - marginal,
        "standardized": _standardized(raw, marginal),
        "variance_inflation": 1.0 + (k - 1) * _standardized(raw, marginal),
    }


def _contrast(joint: np.ndarray, exact: np.ndarray, n_groups: int, k: int) -> tuple[float, float]:
    joint_row = condition_summary(joint, n_groups, k)
    exact_row = condition_summary(exact, n_groups, k)
    return (
        joint_row["standardized"] - exact_row["standardized"],
        joint_row["covariance"] - exact_row["covariance"],
    )


def _slope(sizes: list[int], values: list[float]) -> float:
    x = np.log2(np.asarray(sizes, dtype=float))
    y = np.asarray(values, dtype=float)
    if np.isnan(y).any():
        return math.nan
    centred = x - x.mean()
    return float((centred * y).sum() / (centred**2).sum())


def _homogeneity(values: np.ndarray, n_groups: int, k: int) -> float:
    """Mean within-call agreement of the error indicator over unordered target pairs."""
    matrix = values.reshape(n_groups, k)
    mask = ~np.isnan(matrix)
    counts = mask.sum(axis=1)
    ones = np.where(mask, np.nan_to_num(matrix), 0.0).sum(axis=1)
    zeros = counts - ones
    agree = ones * (ones - 1) + zeros * (zeros - 1)
    pairs = float((counts * (counts - 1)).sum())
    return float(agree.sum() / pairs) if pairs > 0 else math.nan


class CellData:
    """Permutation-ordered error vectors of one cell, ready for resampling."""

    def __init__(self, cell_key: str, doc_ids: list[str], references: dict[str, Any]) -> None:
        self.cell_key = cell_key
        self.model_id, self.task_id = b5.split_cell(cell_key)
        self.doc_ids = doc_ids
        self.references = references
        self.position = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        self.order = {
            assignment: b5.canonical_permutation(cell_key, assignment, len(doc_ids))
            for assignment in range(1, b5.N_ASSIGNMENTS + 1)
        }
        self.values: dict[tuple[int, int, str], np.ndarray] = {}
        self.quality: dict[tuple[int, int, str], dict[str, int]] = {}

    def load(self, final_events: dict[str, dict[str, Any]]) -> None:
        for k in b5.SIZES:
            for assignment in range(1, b5.N_ASSIGNMENTS + 1):
                for condition in b5.CONDITIONS:
                    vector, counts = self._vector(final_events, k, assignment, condition)
                    self.values[(k, assignment, condition)] = vector
                    self.quality[(k, assignment, condition)] = counts

    def _vector(
        self,
        final_events: dict[str, dict[str, Any]],
        k: int,
        assignment: int,
        condition: str,
    ) -> tuple[np.ndarray, dict[str, int]]:
        raw = np.full(len(self.doc_ids), np.nan)
        seen: set[str] = set()
        counts = {"calls": 0, "resolved_calls": 0, "terminal_failures": 0, "parse_failures": 0}
        for event in final_events.values():
            if (
                int(event["batch_size"]) != k
                or int(event["assignment"]) != assignment
                or str(event["condition"]) != condition
            ):
                continue
            counts["calls"] += 1
            targets = [str(value) for value in cast(list[str], event["target_ids"])]
            if seen.intersection(targets):
                raise ValueError(f"duplicate {condition} targets in {self.cell_key}")
            seen.update(targets)
            predictions = event.get("predictions")
            resolved = (
                event.get("status") == "ok"
                and event.get("parse_status") == "ok"
                and isinstance(predictions, dict)
                and set(predictions) == set(targets)
            )
            if not resolved:
                counts["terminal_failures"] += 1
                counts["parse_failures"] += int(event.get("parse_status") != "ok")
                continue
            counts["resolved_calls"] += 1
            parsed = cast(dict[str, Any], predictions)
            for doc_id in targets:
                raw[self.position[doc_id]] = disagreement(
                    parsed[doc_id], self.references[doc_id], self.task_id
                )
        if seen != set(self.position):
            raise ValueError(
                f"{self.cell_key} k={k} a={assignment} {condition} covers {len(seen)} targets"
            )
        return raw[self.order[assignment]], counts

    def blocks(self, assignment: int, condition: str, k: int) -> np.ndarray:
        """Size-32 super-group blocks, the frozen resampling unit."""
        values = self.values[(k, assignment, condition)]
        return values.reshape(values.size // SUPER_SIZE, SUPER_SIZE)


def cell_table(cell: CellData) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for k in b5.SIZES:
        per_assignment: list[dict[str, Any]] = []
        for assignment in range(1, b5.N_ASSIGNMENTS + 1):
            joint = cell.values[(k, assignment, "joint")]
            exact = cell.values[(k, assignment, "exact_context")]
            n_groups = joint.size // k
            standardized, raw = _contrast(joint, exact, n_groups, k)
            per_assignment.append(
                {
                    "assignment": assignment,
                    "joint": condition_summary(joint, n_groups, k),
                    "exact_context": condition_summary(exact, n_groups, k),
                    "standardized_contrast": standardized,
                    "raw_contrast": raw,
                    "joint_label_homogeneity": _homogeneity(joint, n_groups, k),
                    "exact_label_homogeneity": _homogeneity(exact, n_groups, k),
                    "quality": {
                        "joint": cell.quality[(k, assignment, "joint")],
                        "exact_context": cell.quality[(k, assignment, "exact_context")],
                    },
                }
            )
        rows.append(
            {
                "batch_size": k,
                "log2_batch_size": math.log2(k),
                "per_assignment": per_assignment,
                "standardized_contrast": float(
                    np.mean([row["standardized_contrast"] for row in per_assignment])
                ),
                "raw_contrast": float(np.mean([row["raw_contrast"] for row in per_assignment])),
                "joint_covariance": float(
                    np.mean([row["joint"]["covariance"] for row in per_assignment])
                ),
                "exact_covariance": float(
                    np.mean([row["exact_context"]["covariance"] for row in per_assignment])
                ),
                "joint_standardized": float(
                    np.mean([row["joint"]["standardized"] for row in per_assignment])
                ),
                "exact_standardized": float(
                    np.mean([row["exact_context"]["standardized"] for row in per_assignment])
                ),
                "marginal_error_joint": float(
                    np.mean([row["joint"]["marginal_error"] for row in per_assignment])
                ),
                "marginal_error_exact": float(
                    np.mean([row["exact_context"]["marginal_error"] for row in per_assignment])
                ),
                "approximate_variance_inflation": float(
                    np.mean([row["joint"]["variance_inflation"] for row in per_assignment])
                ),
                "assignment_range": float(
                    np.max([row["standardized_contrast"] for row in per_assignment])
                    - np.min([row["standardized_contrast"] for row in per_assignment])
                ),
            }
        )
    sizes = [int(row["batch_size"]) for row in rows]
    return {
        "cell_key": cell.cell_key,
        "model_id": cell.model_id,
        "task_id": cell.task_id,
        "by_size": rows,
        "slope_standardized": _slope(sizes, [float(row["standardized_contrast"]) for row in rows]),
        "slope_raw": _slope(sizes, [float(row["raw_contrast"]) for row in rows]),
    }


def _draw_contrasts(cell: CellData, picks: dict[int, np.ndarray]) -> list[float]:
    """Recompute one cell's per-size contrasts on a resampled set of super-groups."""
    values = []
    for k in b5.SIZES:
        per_assignment = []
        for assignment in range(1, b5.N_ASSIGNMENTS + 1):
            index = picks[assignment]
            joint = cell.blocks(assignment, "joint", k)[index].reshape(-1)
            exact = cell.blocks(assignment, "exact_context", k)[index].reshape(-1)
            per_assignment.append(_contrast(joint, exact, joint.size // k, k)[0])
        values.append(float(np.mean(per_assignment)))
    return values


def bootstrap(
    cells: list[CellData], n_draws: int = N_BOOTSTRAP
) -> dict[str, Any]:
    """Cluster bootstrap over size-32 super-groups, independently within each assignment.

    A super-group is the top of the nesting hierarchy, so resampling it carries its
    subgroups at every size together. This keeps the assignment structure and never
    treats target rows as independent.
    """
    if n_draws < 200:
        raise ValueError("the frozen interval needs at least 200 bootstrap draws")
    n_blocks = b5.N_ITEMS // SUPER_SIZE
    per_cell_slopes = np.empty((n_draws, len(cells)))
    per_cell_sizes = np.empty((n_draws, len(cells), len(b5.SIZES)))
    for draw in range(n_draws):
        for cell_index, cell in enumerate(cells):
            picks = {
                assignment: np.random.default_rng(
                    derived_seed(
                        b5.BOOTSTRAP_NAMESPACE, cell.cell_key, assignment, "draw", draw
                    )
                ).integers(0, n_blocks, n_blocks)
                for assignment in range(1, b5.N_ASSIGNMENTS + 1)
            }
            contrasts = _draw_contrasts(cell, picks)
            per_cell_sizes[draw, cell_index] = contrasts
            per_cell_slopes[draw, cell_index] = _slope(list(b5.SIZES), contrasts)
    cross = per_cell_slopes.mean(axis=1)
    return {
        "n_draws": n_draws,
        "unit": "size-32 super-group within cell and assignment",
        "cross_family_slope": cross,
        "cell_slope": per_cell_slopes,
        "cell_size_contrast": per_cell_sizes,
    }


def _interval(samples: np.ndarray) -> dict[str, float | None]:
    finite = samples[np.isfinite(samples)]
    if finite.size < 2:
        return {"low": None, "high": None, "width": None}
    low, high = (float(value) for value in np.percentile(finite, [2.5, 97.5]))
    return {"low": low, "high": high, "width": high - low}


def _two_sided_p(samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return math.nan
    below = float((finite <= 0).mean())
    above = float((finite >= 0).mean())
    return float(min(1.0, 2.0 * min(below, above)))


def replay_test(cell: CellData, k: int, n_draws: int = N_REPLAY) -> dict[str, Any]:
    """Within-cell regrouping permutation test of the size-k contrast against zero.

    Under no within-call dependence the contrast does not depend on which items share a
    call, so a fresh grouping of the observed labels is a valid null.
    """
    observed = float(
        np.mean(
            [
                _contrast(
                    cell.values[(k, assignment, "joint")],
                    cell.values[(k, assignment, "exact_context")],
                    b5.N_ITEMS // k,
                    k,
                )[0]
                for assignment in range(1, b5.N_ASSIGNMENTS + 1)
            ]
        )
    )
    null = np.empty(n_draws)
    for draw in range(n_draws):
        values = []
        for assignment in range(1, b5.N_ASSIGNMENTS + 1):
            rng = np.random.default_rng(
                derived_seed(b5.BOOTSTRAP_NAMESPACE, cell.cell_key, k, assignment, "replay", draw)
            )
            shuffle = rng.permutation(b5.N_ITEMS)
            joint = cell.values[(k, assignment, "joint")][shuffle]
            exact = cell.values[(k, assignment, "exact_context")][shuffle]
            values.append(_contrast(joint, exact, b5.N_ITEMS // k, k)[0])
        null[draw] = float(np.mean(values))
    p_value = float((1 + int((null >= observed).sum())) / (1 + n_draws))
    return {
        "batch_size": k,
        "observed": observed,
        "p_value_one_sided_upper": p_value,
        "n_draws": n_draws,
        "min_reachable_p": 1.0 / (1 + n_draws),
    }


def classify(result: dict[str, Any]) -> dict[str, Any]:
    """Apply the prespecified outcome interpretations in their frozen order."""
    primary = result["primary"]
    interval = primary["cross_family_slope_interval"]
    per_size = result["cross_family"]["standardized_contrast_by_size"]
    size_intervals = result["cross_family"]["standardized_contrast_interval_by_size"]
    cell_slopes = [float(row["slope_standardized"]) for row in result["cells"]]
    cell_widths = [
        float(row["slope_interval"]["width"])
        for row in result["cells"]
        if row["slope_interval"]["width"] is not None
    ]
    width = interval["width"]
    unstable = width is None or float(width) > 2.0 * PRACTICAL_SLOPE or any(
        value > 4.0 * PRACTICAL_SLOPE for value in cell_widths
    )
    sizes = list(b5.SIZES)
    values = [float(per_size[str(k)]) for k in sizes]
    bounds = [size_intervals[str(k)] for k in sizes]
    contains_zero = [
        bound["low"] is None or (float(bound["low"]) <= 0.0 <= float(bound["high"]))
        for bound in bounds
    ]
    above_practical = [value >= PRACTICAL_CONTRAST for value in values]
    ordered = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    labels = []
    if unstable:
        labels.append("inconclusive")
    if (
        width is not None
        and interval["low"] is not None
        and float(interval["low"]) > 0.0
        and all(value > 0 for value in cell_slopes)
        and ordered
    ):
        labels.append("positive_monotonic")
    if all(contains_zero) and all(value < PRACTICAL_CONTRAST for value in values):
        labels.append("null")
    if (
        all(above_practical)
        and interval["low"] is not None
        and float(interval["low"]) <= 0.0 <= float(interval["high"])
        and float(interval["high"]) < PRACTICAL_SLOPE
    ):
        labels.append("flat_positive")
    if not above_practical[0] and not above_practical[1] and above_practical[-1]:
        labels.append("threshold")
    if not ordered and any(not flag for flag in contains_zero):
        labels.append("non_monotonic")
    return {
        "matched": labels,
        "practical_contrast": PRACTICAL_CONTRAST,
        "practical_slope": PRACTICAL_SLOPE,
        "practical_inflation": PRACTICAL_INFLATION,
        "interpretations": OUTCOME_INTERPRETATIONS,
        "note": (
            "these labels are descriptive readings of the frozen thresholds; success is "
            "not defined as a positive slope, and an inconclusive label never becomes an "
            "equivalence claim"
        ),
    }


def _categorical_row(tables: list[dict[str, Any]], position: int) -> dict[str, Any]:
    """One size read as a category, with Holm across the four cell permutation tests."""
    replay_p = {
        str(table["cell_key"]): float(table["replay"][position]["p_value_one_sided_upper"])
        for table in tables
    }
    return {
        "cells": {
            str(table["cell_key"]): float(table["by_size"][position]["standardized_contrast"])
            for table in tables
        },
        "replay_p": replay_p,
        "holm_adjusted_p": stats.holm(replay_p, ALPHA)["adjusted_p"],
    }


def analyze(
    cells: list[CellData],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    n_replay: int = N_REPLAY,
    historical: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not cells:
        raise ValueError("analysis needs at least one complete cell")
    tables = [cell_table(cell) for cell in cells]
    draws = bootstrap(cells, n_bootstrap)
    slopes = cast(np.ndarray, draws["cell_slope"])
    sizes_draws = cast(np.ndarray, draws["cell_size_contrast"])
    cross = cast(np.ndarray, draws["cross_family_slope"])
    observed_cross = float(np.mean([float(row["slope_standardized"]) for row in tables]))
    for index, table in enumerate(tables):
        table["slope_interval"] = _interval(slopes[:, index])
        table["slope_two_sided_p"] = _two_sided_p(slopes[:, index])
        table["size_contrast_interval"] = {
            str(k): _interval(sizes_draws[:, index, position])
            for position, k in enumerate(b5.SIZES)
        }
        table["replay"] = [replay_test(cells[index], k, n_replay) for k in b5.SIZES]
    cross_by_size = sizes_draws.mean(axis=1)
    leave_one_out = {
        cells[index].model_id: float(
            np.mean(
                [
                    float(row["slope_standardized"])
                    for position, row in enumerate(tables)
                    if position != index
                ]
            )
        )
        for index in range(len(cells))
    }
    result: dict[str, Any] = {
        "schema_version": "b5-analysis-v1",
        "design": {
            "cells": [cell.cell_key for cell in cells],
            "sizes": list(b5.SIZES),
            "assignments": b5.N_ASSIGNMENTS,
            "items_per_cell": b5.N_ITEMS,
            "primary_estimand": (
                "standardized joint dependence minus standardized exact-context dependence, "
                "where standardized dependence is covariance divided by p (1 - p)"
            ),
            "primary_statistic": (
                "ordinary least squares slope on log2(batch size) inside each cell, then the "
                f"equal-weight mean over the {len(cells)} analyzed cells"
            ),
            "uncertainty": (
                "cluster bootstrap over size-32 super-groups within cell and assignment"
            ),
            "n_bootstrap": n_bootstrap,
            "n_replay": n_replay,
            "alpha": ALPHA,
        },
        "primary": {
            "cross_family_slope": observed_cross,
            "cross_family_slope_interval": _interval(cross),
            "cross_family_slope_two_sided_p": _two_sided_p(cross),
            "rejects_zero_at_alpha": bool(
                _interval(cross)["low"] is not None
                and not (
                    float(cast(float, _interval(cross)["low"]))
                    <= 0.0
                    <= float(cast(float, _interval(cross)["high"]))
                )
            ),
        },
        "cross_family": {
            "standardized_contrast_by_size": {
                str(k): float(
                    np.mean([float(table["by_size"][position]["standardized_contrast"])
                             for table in tables])
                )
                for position, k in enumerate(b5.SIZES)
            },
            "raw_contrast_by_size": {
                str(k): float(
                    np.mean([float(table["by_size"][position]["raw_contrast"])
                             for table in tables])
                )
                for position, k in enumerate(b5.SIZES)
            },
            "standardized_contrast_interval_by_size": {
                str(k): _interval(cross_by_size[:, position])
                for position, k in enumerate(b5.SIZES)
            },
            "raw_slope_cross_family": float(
                np.mean([float(table["slope_raw"]) for table in tables])
            ),
            "leave_one_family_out_slope": leave_one_out,
            "model_slopes": {
                table["model_id"]: float(table["slope_standardized"]) for table in tables
            },
            "categorical_by_size": {
                str(k): _categorical_row(tables, position) for position, k in enumerate(b5.SIZES)
            },
        },
        "cells": tables,
        "historical_size_16_b3": {
            "values": historical or {},
            "scale": "raw joint-minus-exact covariance from the sealed A9 analysis",
            "use": (
                "descriptive comparison only; the sealed A9 result used the earlier output "
                "contract and is never pooled into this experiment"
            ),
        },
    }
    result["outcome"] = classify(result)
    result["result_sha256"] = sha256_json(result)
    return result
