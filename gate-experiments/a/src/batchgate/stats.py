"""Estimators for the gate: one-way ICC, cluster bootstrap, permutation nulls.

Every function here works on binary error indicators and returns NaN rather than a
fabricated number when the estimand is undefined. Negative ICC estimates are real
and are returned unchanged; clipping happens only where a design effect needs a
non-negative input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n_valid: int

    def as_dict(self) -> dict[str, float | int | None]:
        def clean(x: float) -> float | None:
            return None if math.isnan(x) else float(x)

        return {
            "point": clean(self.point),
            "ci_low": clean(self.low),
            "ci_high": clean(self.high),
            "n_bootstrap_valid": self.n_valid,
        }


def group_stats(
    values: FloatArray, groups: np.ndarray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-group count, sum and sum of squares. Sufficient for the one-way ANOVA."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    size = counts.size
    sums = np.bincount(inverse, weights=values, minlength=size)
    sumsq = np.bincount(inverse, weights=values * values, minlength=size)
    return counts.astype(float), sums, sumsq


def icc_from_group_stats(counts: FloatArray, sums: FloatArray, sumsq: FloatArray) -> float:
    m = counts.size
    n = float(counts.sum())
    if m < 2 or n - m < 1:
        return math.nan
    grand = float(sums.sum()) / n
    means = sums / counts
    msb = float((counts * (means - grand) ** 2).sum()) / (m - 1)
    msw = float((sumsq - sums**2 / counts).sum()) / (n - m)
    k0 = (n - float((counts**2).sum()) / n) / (m - 1)
    denom = msb + (k0 - 1) * msw
    if denom <= 0:
        return math.nan
    return (msb - msw) / denom


def icc_oneway(values: FloatArray, groups: np.ndarray) -> float:
    """ICC(1) from a one-way random effects ANOVA.

    Unequal batch sizes use the standard k0 correction, so equal and near-equal
    sizes both work. All-constant input has no variance to partition and returns
    NaN instead of a division guarded by an epsilon.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan
    return icc_from_group_stats(*group_stats(values, np.asarray(groups)))


def design_effect(rho: float, k: float) -> float:
    if math.isnan(rho):
        return math.nan
    return 1.0 + (k - 1.0) * max(0.0, rho)


def effective_n(n: int, rho: float, k: float) -> float:
    d = design_effect(rho, k)
    return math.nan if math.isnan(d) or d <= 0 else n / d


def percentile_interval(samples: FloatArray, point: float, alpha: float) -> Interval:
    valid = samples[~np.isnan(samples)]
    if valid.size < 2:
        return Interval(point, math.nan, math.nan, int(valid.size))
    low, high = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(low), float(high), int(valid.size))


def cluster_bootstrap_icc(
    values: FloatArray, groups: np.ndarray, n_boot: int, rng: np.random.Generator
) -> FloatArray:
    """Resample whole batches with replacement. A batch drawn twice counts twice."""
    values = np.asarray(values, dtype=float)
    counts, sums, sumsq = group_stats(values, np.asarray(groups))
    m = counts.size
    if m < 2:
        return np.full(n_boot, math.nan)
    draws = rng.integers(0, m, size=(n_boot, m))
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = draws[i]
        out[i] = icc_from_group_stats(counts[idx], sums[idx], sumsq[idx])
    return out


def permutation_null_icc(
    values: FloatArray, groups: np.ndarray, n_perm: int, rng: np.random.Generator
) -> FloatArray:
    """Fake batches: permute membership while every batch keeps its size."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        out[i] = icc_oneway(rng.permutation(values), groups)
    return out


def upper_p_value(observed: float, null_samples: FloatArray) -> float:
    if math.isnan(observed):
        return math.nan
    valid = null_samples[~np.isnan(null_samples)]
    if valid.size == 0:
        return math.nan
    return float((1 + int((valid >= observed).sum())) / (1 + valid.size))


def bootstrap_mean(
    values: FloatArray, n_boot: int, rng: np.random.Generator, alpha: float
) -> Interval:
    """Resample the unit of independence, which for paired deltas is the document."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return Interval(math.nan, math.nan, math.nan, 0)
    point = float(values.mean())
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[idx].mean(axis=1)
    return percentile_interval(samples, point, alpha)


def between_group_ss(values: FloatArray, groups: np.ndarray) -> float:
    counts, sums, _ = group_stats(np.asarray(values, dtype=float), np.asarray(groups))
    grand = float(sums.sum()) / float(counts.sum())
    means = sums / counts
    return float((counts * (means - grand) ** 2).sum())


def correlation(x: FloatArray, y: FloatArray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def slot_effect(
    values: FloatArray, slots: np.ndarray, n_perm: int, rng: np.random.Generator
) -> dict[str, object]:
    """Two position tests: any difference between slots, and a monotone trend."""
    values = np.asarray(values, dtype=float)
    slots = np.asarray(slots)
    counts, sums, _ = group_stats(values, slots)
    unique_slots = np.unique(slots)
    observed_ss = between_group_ss(values, slots)
    observed_r = correlation(slots.astype(float), values)
    null_ss = np.empty(n_perm)
    null_r = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(values)
        null_ss[i] = between_group_ss(shuffled, slots)
        null_r[i] = correlation(slots.astype(float), shuffled)
    p_any = upper_p_value(observed_ss, null_ss)
    p_trend = (
        math.nan
        if math.isnan(observed_r)
        else float((1 + int((np.abs(null_r) >= abs(observed_r)).sum())) / (1 + n_perm))
    )
    return {
        "error_rate_by_slot": {
            int(s): float(sums[i] / counts[i]) for i, s in enumerate(unique_slots)
        },
        "n_by_slot": {int(s): int(counts[i]) for i, s in enumerate(unique_slots)},
        "between_slot_ss": observed_ss,
        "p_any_slot_difference": p_any,
        "slot_error_correlation": None if math.isnan(observed_r) else observed_r,
        "p_monotone_trend": None if math.isnan(p_trend) else p_trend,
        "n_permutations": n_perm,
    }
