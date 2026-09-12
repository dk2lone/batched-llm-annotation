"""Distance-concentration metrics and the one naming convention the gate depends on.

`distance_concentrated`   = distances to the demonstration rows are more alike,
                            so relative contrast is LOWER.
`distance_deconcentrated` = distances are less alike, so relative contrast is HIGHER.

`tests/test_metrics.py` pins this direction so the two labels cannot be swapped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

EPS = 1e-12

CONCENTRATED = "distance_concentrated"
DECONCENTRATED = "distance_deconcentrated"


def relative_contrast(distances: np.ndarray) -> float:
    """(max - min) / min. Lower means more concentrated."""
    d = np.asarray(distances, dtype=np.float64)
    return float((d.max() - d.min()) / (d.min() + EPS))


def nearest_farthest_ratio(distances: np.ndarray) -> float:
    """min / max. Higher means more concentrated."""
    d = np.asarray(distances, dtype=np.float64)
    return float(d.min() / (d.max() + EPS))


def coefficient_of_variation(distances: np.ndarray) -> float:
    """std / mean. Lower means more concentrated."""
    d = np.asarray(distances, dtype=np.float64)
    return float(d.std() / (d.mean() + EPS))


def concentration_label(relative_contrast_value: float, reference_value: float) -> str:
    """Label a distance set against a reference set.

    Lower relative contrast than the reference means CONCENTRATED.
    """
    return CONCENTRATED if relative_contrast_value < reference_value else DECONCENTRATED


@dataclass(frozen=True)
class DistanceStats:
    relative_contrast: float
    nearest_farthest_ratio: float
    coefficient_of_variation: float
    d_min: float
    d_max: float
    d_mean: float
    d_std: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def distance_stats(distances: np.ndarray) -> DistanceStats:
    d = np.asarray(distances, dtype=np.float64).ravel()
    if d.size < 2:
        raise ValueError("need at least two distances to measure concentration")
    return DistanceStats(
        relative_contrast=relative_contrast(d),
        nearest_farthest_ratio=nearest_farthest_ratio(d),
        coefficient_of_variation=coefficient_of_variation(d),
        d_min=float(d.min()),
        d_max=float(d.max()),
        d_mean=float(d.mean()),
        d_std=float(d.std()),
        n=int(d.size),
    )


def euclidean_distances(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Distances from one query vector to every row of ``rows``."""
    q = np.asarray(query, dtype=np.float64).ravel()
    r = np.asarray(rows, dtype=np.float64)
    return np.linalg.norm(r - q[None, :], axis=1)


def cosine_distances(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64).ravel()
    r = np.asarray(rows, dtype=np.float64)
    qn = q / (np.linalg.norm(q) + EPS)
    rn = r / (np.linalg.norm(r, axis=1, keepdims=True) + EPS)
    return 1.0 - rn @ qn


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank agreement between two distance vectors. None when it is undefined."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    rho = float(spearmanr(x, y).statistic)
    if np.isnan(rho):
        return None
    if np.isclose(abs(rho), 1.0):
        return float(np.sign(rho))
    return rho


def mean_stats(stats: list[DistanceStats]) -> dict[str, float]:
    """Average a list of per-query stats into one summary."""
    if not stats:
        return {}
    keys = ["relative_contrast", "nearest_farthest_ratio", "coefficient_of_variation"]
    return {k: float(np.mean([getattr(s, k) for s in stats])) for k in keys}
