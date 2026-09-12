"""The three matched synthetic arms, built from one shared latent 2D dataset.

Every arm at a given (dim, seed) uses the same latent points, the same labels, the
same row order and the same feature names. Only the mapping from the latent points
to the observed columns changes:

  sparse_zero_pad            [z0, z1, 0, 0, ...]
  dense_redundant_projection z @ R, R is 2 x dim with i.i.d. uniform entries
  independent_nuisance       [z0, z1, e2, ..., e_{dim-1}], e_j i.i.d. normal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ARMS = ("sparse_zero_pad", "dense_redundant_projection", "independent_nuisance")
LABEL_LETTERS = ("A", "B")


@dataclass(frozen=True)
class LatentDataset:
    """The shared latent 2D rows. Identical for all three arms."""

    seed: int
    demo_z: np.ndarray  # (n_demo, 2)
    demo_y: np.ndarray  # (n_demo,) int 0/1
    query_z: np.ndarray  # (n_query, 2)
    query_y: np.ndarray  # (n_query,)


@dataclass(frozen=True)
class ArmDataset:
    arm: str
    dim: int
    seed: int
    demo_x: np.ndarray  # (n_demo, dim)
    query_x: np.ndarray  # (n_query, dim)
    latent: LatentDataset
    feature_names: tuple[str, ...]

    @property
    def demo_y(self) -> np.ndarray:
        return self.latent.demo_y

    @property
    def query_y(self) -> np.ndarray:
        return self.latent.query_y


def _sample_quadrant_points(
    rng: np.random.Generator, n: int, sx: int, sy: int, offset: float, sigma: float, margin: float
) -> np.ndarray:
    """Draw ``n`` points in one XOR quadrant, rejecting anything inside the margin band."""
    out = np.empty((n, 2), dtype=np.float64)
    filled = 0
    center = np.array([sx * offset, sy * offset], dtype=np.float64)
    while filled < n:
        cand = rng.normal(loc=center, scale=sigma, size=(max(n - filled, 1) * 4, 2))
        keep = cand[(np.abs(cand[:, 0]) >= margin) & (np.abs(cand[:, 1]) >= margin)]
        keep = keep[(np.sign(keep[:, 0]) == sx) & (np.sign(keep[:, 1]) == sy)]
        take = min(len(keep), n - filled)
        out[filled : filled + take] = keep[:take]
        filled += take
    return out


def make_latent(
    seed: int,
    n_demo: int,
    n_query: int,
    cluster_offset: float,
    cluster_sigma: float,
    margin: float,
) -> LatentDataset:
    """Build the shared latent 2D dataset with an exact class balance.

    The rule is XOR over the two latent coordinates: label = 1 when z0 * z1 > 0.
    Both coordinates are needed, so no single column can solve the task.
    """
    if n_demo % 4 or n_query % 4:
        raise ValueError("n_demo and n_query must be multiples of 4 to balance four quadrants")
    rng = np.random.default_rng(1000 + seed)
    quadrants = [(1, 1, 1), (-1, -1, 1), (1, -1, 0), (-1, 1, 0)]

    def draw(n_total: int) -> tuple[np.ndarray, np.ndarray]:
        per = n_total // 4
        zs, ys = [], []
        for sx, sy, label in quadrants:
            zs.append(
                _sample_quadrant_points(rng, per, sx, sy, cluster_offset, cluster_sigma, margin)
            )
            ys.append(np.full(per, label, dtype=np.int64))
        z = np.concatenate(zs)
        y = np.concatenate(ys)
        order = rng.permutation(len(z))
        return z[order], y[order]

    demo_z, demo_y = draw(n_demo)
    query_z, query_y = draw(n_query)
    return LatentDataset(seed=seed, demo_z=demo_z, demo_y=demo_y, query_z=query_z, query_y=query_y)


def _dense_projection(
    rng: np.random.Generator, dim: int, z_all: np.ndarray, scale: str
) -> tuple[np.ndarray, np.ndarray]:
    """Source-paper-style dense expansion: i.i.d. uniform entries, fresh per dimension."""
    proj = rng.uniform(-1.0, 1.0, size=(2, dim))
    x = z_all @ proj
    if scale == "unit_std":
        std = x.std(axis=0)
        std[std < 1e-9] = 1.0
        x = x / std[None, :]
    elif scale != "raw":
        raise ValueError(f"unknown dense_column_scale: {scale}")
    return x, proj


def build_arm(
    arm: str,
    dim: int,
    latent: LatentDataset,
    nuisance_scale: float,
    dense_column_scale: str,
) -> ArmDataset:
    """Map the shared latent rows into ``dim`` observed columns for one arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if dim < 2:
        raise ValueError("dim must be at least 2")
    z_all = np.concatenate([latent.demo_z, latent.query_z])
    n_demo = len(latent.demo_z)

    if arm == "sparse_zero_pad":
        x_all = np.zeros((len(z_all), dim), dtype=np.float64)
        x_all[:, :2] = z_all
    elif arm == "dense_redundant_projection":
        rng = np.random.default_rng(2000 + latent.seed * 100 + dim)
        x_all, _ = _dense_projection(rng, dim, z_all, dense_column_scale)
    else:
        rng = np.random.default_rng(3000 + latent.seed * 100 + dim)
        x_all = np.zeros((len(z_all), dim), dtype=np.float64)
        x_all[:, :2] = z_all
        if dim > 2:
            latent_std = float(z_all.std())
            noise = rng.normal(0.0, nuisance_scale * latent_std, size=(len(z_all), dim - 2))
            x_all[:, 2:] = noise

    return ArmDataset(
        arm=arm,
        dim=dim,
        seed=latent.seed,
        demo_x=x_all[:n_demo],
        query_x=x_all[n_demo:],
        latent=latent,
        feature_names=tuple(f"f{i}" for i in range(dim)),
    )


def label_letters(y: np.ndarray) -> list[str]:
    return [LABEL_LETTERS[int(v)] for v in y]


def arm_summary(arm: ArmDataset) -> dict[str, Any]:
    return {
        "arm": arm.arm,
        "dim": arm.dim,
        "seed": arm.seed,
        "n_demo": int(len(arm.demo_x)),
        "n_query": int(len(arm.query_x)),
        "demo_class_balance": [int((arm.demo_y == c).sum()) for c in (0, 1)],
        "query_class_balance": [int((arm.query_y == c).sum()) for c in (0, 1)],
        "feature_names": list(arm.feature_names),
        "observed_abs_max": float(np.abs(arm.demo_x).max()),
    }
