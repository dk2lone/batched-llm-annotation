"""CPU precision and power simulation for the B5 dose-response design.

Planning inputs come from the completed B3 grid only. The simulation never tunes the
design toward the observed B3 direction: the six simulated shapes include a null, a
decreasing arm and an inconclusive arm, and the reported decision rule is the frozen
one.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from step0.hashing import derived_seed, sha256_json

from . import b5
from .b5_analysis import PRACTICAL_CONTRAST, PRACTICAL_SLOPE, _contrast, _slope

N_REPLICATES = 200
ITEM_LOGIT_SD = 1.0


def scenario_contrasts() -> dict[str, dict[int, float]]:
    """The six frozen result shapes, expressed as the standardized contrast per size."""
    step = PRACTICAL_SLOPE
    return {
        "positive_monotonic": {k: step * math.log2(k) for k in b5.SIZES},
        "flat_positive": dict.fromkeys(b5.SIZES, PRACTICAL_CONTRAST),
        "threshold": {
            k: 0.0 if k <= 4 else PRACTICAL_CONTRAST * (math.log2(k) - 2) / 3 for k in b5.SIZES
        },
        "non_monotonic": {
            k: PRACTICAL_CONTRAST * (1.0 - abs(math.log2(k) - 3.0) / 2.0) for k in b5.SIZES
        },
        "decreasing": {k: PRACTICAL_CONTRAST * (5.0 - math.log2(k)) / 4.0 for k in b5.SIZES},
        "null": dict.fromkeys(b5.SIZES, 0.0),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-values)))


def item_rates(marginal: float, n_items: int, seed: int) -> np.ndarray:
    """Fixed per-item error rates, so item reuse across sizes carries real structure."""
    rng = np.random.default_rng(seed)
    logit = math.log(marginal / (1.0 - marginal))
    rates = _sigmoid(logit + ITEM_LOGIT_SD * rng.standard_normal(n_items))
    return np.asarray(rates * (marginal / float(rates.mean())), dtype=float).clip(1e-4, 1 - 1e-4)


def attenuation(rates: np.ndarray, seed: int, n_pairs: int = 200_000) -> float:
    """Realized standardized dependence per unit of requested sharing.

    Sharing one uniform gives `Cov = rho (min(p_i, p_j) - p_i p_j)`. Heterogeneous item
    rates make that smaller than `rho p (1 - p)`, so the requested sharing is divided by
    this factor and the simulated contrast matches its nominal value at every size.
    """
    rng = np.random.default_rng(seed)
    left = rates[rng.integers(0, rates.size, n_pairs)]
    right = rates[rng.integers(0, rates.size, n_pairs)]
    mean = float(rates.mean())
    return float(np.mean(np.minimum(left, right) - left * right) / (mean * (1.0 - mean)))


def draw_errors(
    rates: np.ndarray, n_groups: int, k: int, correlation: float, rng: np.random.Generator
) -> np.ndarray:
    """Exchangeable within-call binary errors with the requested pair correlation.

    Each call draws one shared uniform. A target copies it with probability
    `sqrt(correlation)` and otherwise draws its own, then compares against its own error
    rate. Sharing the uniform rather than a shared outcome keeps every item's marginal
    exactly `p_i`, so item heterogeneity never leaks into the covariance and the two
    conditions differ only in the requested correlation.
    """
    if not 0.0 <= correlation < 1.0:
        raise ValueError("the simulated correlation must lie in [0, 1)")
    share = math.sqrt(correlation)
    matrix = rates.reshape(n_groups, k)
    shared = np.broadcast_to(rng.random(n_groups)[:, None], matrix.shape)
    own = rng.random(matrix.shape)
    copy = rng.random(matrix.shape) < share
    return (np.where(copy, shared, own) < matrix).astype(float).reshape(-1)


def replicate(
    planning: list[dict[str, Any]], contrasts: dict[int, float], seed: int
) -> tuple[float, dict[int, float]]:
    """One simulated experiment: four cells, five sizes, three assignments."""
    cell_slopes = []
    by_size: dict[int, list[float]] = {k: [] for k in b5.SIZES}
    for cell in planning:
        rates = item_rates(
            float(cell["marginal_error"]),
            b5.N_ITEMS,
            derived_seed(b5.POWER_NAMESPACE, cell["cell_key"], "items"),
        )
        ordered = {
            assignment: rates[b5.canonical_permutation(str(cell["cell_key"]), assignment)]
            for assignment in range(1, b5.N_ASSIGNMENTS + 1)
        }
        scale = attenuation(
            rates, derived_seed(b5.POWER_NAMESPACE, cell["cell_key"], "attenuation")
        )
        values = []
        for k in b5.SIZES:
            n_groups = b5.N_ITEMS // k
            per_assignment = []
            for assignment in range(1, b5.N_ASSIGNMENTS + 1):
                rng = np.random.default_rng(
                    derived_seed(b5.POWER_NAMESPACE, cell["cell_key"], seed, k, assignment)
                )
                exact_rho = max(float(cell["exact_standardized"]), 0.0) / scale
                joint = draw_errors(
                    ordered[assignment],
                    n_groups,
                    k,
                    max(exact_rho + contrasts[k] / scale, 0.0),
                    rng,
                )
                exact = draw_errors(ordered[assignment], n_groups, k, exact_rho, rng)
                per_assignment.append(_contrast(joint, exact, n_groups, k)[0])
            mean = float(np.mean(per_assignment))
            values.append(mean)
            by_size[k].append(mean)
        cell_slopes.append(_slope(list(b5.SIZES), values))
    return float(np.mean(cell_slopes)), {k: float(np.mean(values)) for k, values in by_size.items()}


def simulate(
    planning: list[dict[str, Any]], *, n_replicates: int = N_REPLICATES
) -> dict[str, Any]:
    if n_replicates < 50:
        raise ValueError("the precision simulation needs at least 50 replicates")
    shapes = scenario_contrasts()
    results: dict[str, dict[str, Any]] = {}
    for name, contrasts in shapes.items():
        slopes = np.empty(n_replicates)
        sizes: dict[int, list[float]] = {k: [] for k in b5.SIZES}
        for index in range(n_replicates):
            slope, per_size = replicate(planning, contrasts, index)
            slopes[index] = slope
            for k, value in per_size.items():
                sizes[k].append(value)
        standard_error = float(np.std(slopes, ddof=1))
        rejects = float(np.mean(np.abs(slopes) > 1.959963984540054 * standard_error))
        results[name] = {
            "true_contrast_by_size": {str(k): contrasts[k] for k in b5.SIZES},
            "true_slope": _slope(list(b5.SIZES), [contrasts[k] for k in b5.SIZES]),
            "mean_estimated_slope": float(np.mean(slopes)),
            "slope_standard_error": standard_error,
            "slope_95_half_width": 1.959963984540054 * standard_error,
            "rejection_rate_against_zero": rejects,
            "per_size_standard_error": {
                str(k): float(np.std(values, ddof=1)) for k, values in sizes.items()
            },
            "per_size_mean_estimate": {
                str(k): float(np.mean(values)) for k, values in sizes.items()
            },
        }
    resolvable = {
        name: bool(row["slope_95_half_width"] < PRACTICAL_SLOPE)
        for name, row in results.items()
    }
    return {
        "schema_version": "b5-power-v1",
        "namespace": b5.POWER_NAMESPACE,
        "n_replicates": n_replicates,
        "planning_inputs": planning,
        "planning_source": (
            "completed B3 marginal error rates and standardized dependence of the same four "
            "cells; no B5 output exists"
        ),
        "item_logit_sd": ITEM_LOGIT_SD,
        "practical_contrast": PRACTICAL_CONTRAST,
        "practical_slope": PRACTICAL_SLOPE,
        "scenarios": results,
        "half_width_below_practical_slope": resolvable,
        "what_8192_items_resolve": (
            "8,192 items, three assignments and five sizes resolve a cross-family slope "
            "whose half width is reported above; a per-size contrast smaller than its own "
            "reported standard error is not resolvable by this design, and no reduction of "
            "the four-family scope is used to buy precision"
        ),
        "record_sha256": "",
    }


def finalize(record: dict[str, Any]) -> dict[str, Any]:
    core = {key: value for key, value in record.items() if key != "record_sha256"}
    return {**core, "record_sha256": sha256_json(core)}
