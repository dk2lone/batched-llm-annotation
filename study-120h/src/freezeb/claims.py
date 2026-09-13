"""Predeclared panel and repeated-sampling claim gates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest

from step0.stats import covariance_from_stratified, holm

from .selection import Cell


@dataclass(frozen=True)
class CellData:
    cell: Cell
    joint: list[np.ndarray]
    exact: list[np.ndarray]
    observed: list[np.ndarray]
    replay: Callable[[int], list[np.ndarray]]


def cell_delta(data: CellData, matrices: list[np.ndarray]) -> float:
    if not (len(data.joint) == len(data.exact) == len(matrices) == 3):
        raise ValueError("three aligned assignments are required")
    effects = []
    for joint, exact, matrix in zip(data.joint, data.exact, matrices, strict=True):
        if matrix.ndim != 2 or matrix.shape[0] != 16:
            raise ValueError("E2 matrices must be (16 strata, calls)")
        if joint.shape != exact.shape or joint.size != matrix.size:
            raise ValueError("prediction arrays and assignment matrix must align")
        if np.isnan(joint).any() or np.isnan(exact).any():
            raise ValueError("resolve failures under a frozen bound first")
        effects.append(
            covariance_from_stratified(joint[matrix].T)
            - covariance_from_stratified(exact[matrix].T)
        )
    return float(np.mean(effects))


def e2_panel_gate(
    data: list[CellData], primary: list[Cell], *, n_draws: int = 4999
) -> dict[str, Any]:
    """Equal-cell global replay and 16-test Holm gate; missing cells cannot pass."""
    if n_draws < 319:
        raise ValueError("too few draws to resolve a 16-test Holm family")
    if len(primary) != 16 or len({c.key for c in primary}) != 16:
        raise ValueError("the full recurrence claim requires 16 distinct primary cells")
    by_key = {d.cell.key: d for d in data}
    if len(by_key) != len(data) or not set(by_key) <= {c.key for c in primary}:
        raise ValueError("results do not match the locked primary panel")
    if len(by_key) != 16:
        return {"passed": False, "reason": "incomplete_primary_panel",
                "completed": len(by_key), "required": 16}
    ordered = [by_key[c.key] for c in primary]
    observed = np.asarray([cell_delta(d, d.observed) for d in ordered])
    null = np.asarray([[cell_delta(d, d.replay(draw)) for d in ordered]
                       for draw in range(n_draws)])
    p_cell = {d.cell.key: float((1 + np.sum(null[:, i] >= observed[i])) / (1 + n_draws))
              for i, d in enumerate(ordered)}
    adjusted = holm(p_cell)["adjusted_p"]
    global_observed = float(observed.mean())
    global_p = float((1 + np.sum(null.mean(axis=1) >= global_observed)) / (1 + n_draws))
    positive = [d.cell for i, d in enumerate(ordered)
                if adjusted[d.cell.key] <= 0.05 and observed[i] >= 0.002]
    families = {c.family for c in positive}
    tasks = {c.task for c in positive}
    family_loo = {
        family: float(np.mean([observed[i] for i, d in enumerate(ordered)
                               if d.cell.family != family]))
        for family in {d.cell.family for d in ordered}
    }
    task_loo = {
        task: float(np.mean([observed[i] for i, d in enumerate(ordered)
                             if d.cell.task != task]))
        for task in {d.cell.task for d in ordered}
    }
    passed = (
        global_observed > 0 and global_p <= 0.05 and len(families) >= 3
        and len(tasks) >= 3 and min(family_loo.values()) > 0
        and min(task_loo.values()) > 0
    )
    return {
        "passed": bool(passed), "global_equal_cell_delta": global_observed,
        "global_p": global_p, "cell_delta": {d.cell.key: float(observed[i])
                                               for i, d in enumerate(ordered)},
        "cell_p": p_cell, "holm_adjusted_p": adjusted,
        "positive_families": sorted(families), "positive_tasks": sorted(tasks),
        "leave_one_family_out": family_loo, "leave_one_task_out": task_loo,
        "n_draws": n_draws,
    }


def e5_coverage_gate(
    coverage_rows: list[dict[str, Any]], primary_keys: list[str]
) -> dict[str, Any]:
    """Undercoverage needs two prespecified cells after six-test Holm correction."""
    if len(primary_keys) != 6 or len(set(primary_keys)) != 6:
        raise ValueError("six distinct primary E5 cells are required")
    by_key = {r["cell_key"]: r for r in coverage_rows}
    if len(by_key) != len(coverage_rows) or not set(by_key) <= set(primary_keys):
        raise ValueError("coverage rows do not match locked E5 cells")
    p_values = {}
    for key in primary_keys:
        row = by_key.get(key)
        if row is None or row["n_evaluation"] < 400 or not row["reference_precise"]:
            p_values[key] = 1.0
        else:
            p_values[key] = float(binomtest(row["n_covered"], row["n_evaluation"],
                                           0.95, alternative="less").pvalue)
    adjusted = holm(p_values)["adjusted_p"]
    supported = [key for key in primary_keys if key in by_key and
                 by_key[key]["n_evaluation"] >= 400 and by_key[key]["reference_precise"]
                 and by_key[key]["coverage"] <= 0.90
                 and by_key[key]["worst_reference_upper_coverage"] < 0.95
                 and adjusted[key] <= 0.05]
    return {"passed": len(supported) >= 2, "supported_cells": supported,
            "holm_adjusted_p": adjusted, "family": primary_keys}
