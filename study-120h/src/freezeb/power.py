"""CPU-only size diagnostics using null and exchangeable-call error generators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from step0.hashing import derived_seed

SIZES = (2048, 4096, 8192)
ERROR_GRID = (0.05, 0.10, 0.20, 0.35, 0.50)
K = 16
ASSIGNMENTS = 3
DESIGN_EFFECT = 0.004
PRACTICAL_FLOOR = 0.002
FAMILY_ALPHA = 0.05 / 16
POWER_TARGET = 0.80
NAMESPACE = "freezeb-power-v1"


def _covariance(calls: np.ndarray) -> np.ndarray:
    """Pair-weighted covariance across all equal-size calls in each simulation."""
    centred = calls - calls.mean(axis=(2, 3), keepdims=True)
    sums = centred.sum(axis=3)
    pairs = calls.shape[2] * K * (K - 1)
    values: np.ndarray = (
        (sums**2).sum(axis=2) - (centred**2).sum(axis=(2, 3))
    ) / pairs
    mean: np.ndarray = values.mean(axis=1)
    return mean


def _draw(rng: np.random.Generator, sims: int, n: int, p: float, delta: float) -> np.ndarray:
    shape = (sims, ASSIGNMENTS, n // K)
    if delta == 0:
        rates: np.ndarray | float = p
    else:
        rho = delta / (p * (1 - p))
        if not 0 < rho < 1:
            raise ValueError("effect is not reachable at this error rate")
        concentration = 1 / rho - 1
        rates = rng.beta(p * concentration, (1 - p) * concentration, size=shape)[..., None]
    return (rng.random((*shape, K)) < rates).astype(np.float32)


def simulate(
    p: float,
    n: int,
    *,
    sims: int = 2000,
    alpha: float = FAMILY_ALPHA,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Conservative Bonferroni proxy for a Holm family of `0.05 / alpha` cells.

    This approximates test power for planning, not the final design-respecting
    randomization p-value. The seed excludes alpha on purpose, so two family sizes
    are compared on one null sample and only the threshold moves. Every synthetic
    call has the planned batch size and three assignments. Independent
    item-difficulty and failure scenarios need sensitivity diagnostics before the
    final protocol seal.
    """
    if n not in SIZES or not 0 < p < 1 or sims < 100:
        raise ValueError("invalid planning scenario")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    rng = np.random.default_rng(derived_seed(namespace, p, n, sims))
    null = _covariance(_draw(rng, sims, n, p, 0)) - _covariance(_draw(rng, sims, n, p, 0))
    critical = float(np.quantile(null, 1 - alpha))
    alternative = _covariance(_draw(rng, sims, n, p, DESIGN_EFFECT)) - _covariance(
        _draw(rng, sims, n, p, 0)
    )
    return {
        "namespace": namespace,
        "p": p,
        "n": float(n),
        "delta": DESIGN_EFFECT,
        "alpha": alpha,
        "sims": float(sims),
        "critical": critical,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "alternative_mean": float(alternative.mean()),
        "effect_bias": float(alternative.mean() - DESIGN_EFFECT),
        "power": float((alternative >= critical).mean()),
        "power_mc_se": math.sqrt(float((alternative >= critical).mean()) *
                                 (1 - float((alternative >= critical).mean())) / sims),
    }


def choose_n(p: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the next higher-error grid rate to avoid optimistic interpolation."""
    if not 0 < p < 1:
        raise ValueError("error rate must be between zero and one")
    effective_p = min(p, 1 - p)
    eligible = [rate for rate in ERROR_GRID if rate >= effective_p]
    grid_p = min(eligible) if eligible else ERROR_GRID[-1]
    scenarios = sorted((r for r in rows if r["p"] == grid_p), key=lambda r: r["n"])
    if len(scenarios) != len(SIZES):
        raise ValueError("the size table is incomplete")
    def lower(r: dict[str, Any]) -> float:
        z = 1.959963984540054
        n, rate = float(r["sims"]), float(r["power"])
        return ((rate + z * z / (2 * n)) - z * math.sqrt(
            rate * (1 - rate) / n + z * z / (4 * n * n)
        )) / (1 + z * z / n)

    selected = next((r for r in scenarios if lower(r) >= POWER_TARGET), scenarios[-1])
    return {
        "solo_error_rate": p,
        "grid_rate": grid_p,
        "n_items": int(selected["n"]),
        "proxy_power": selected["power"],
        "power_wilson_lower_95": lower(selected),
        "power_target_met": lower(selected) >= POWER_TARGET,
        "information_poor": effective_p < ERROR_GRID[0],
    }
