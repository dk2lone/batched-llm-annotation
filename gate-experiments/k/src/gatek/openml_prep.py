"""Stage 2: an optional random-projection slice over five OpenML datasets.

This module downloads data. It runs only when a user invokes `gatek prepare-openml`.
The tests never import it and the smoke command never calls it.

This is an expansion after the synthetic gate, not part of its pass/fail evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .metrics import distance_stats, euclidean_distances


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def random_projection(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
    """Entries drawn i.i.d. uniform, a fresh matrix per target dimensionality."""
    return rng.uniform(-1.0, 1.0, size=(n_in, n_out))


def prepare_openml(cfg: Config, out_dir: Path) -> dict[str, Any]:
    """Download, preprocess, project and freeze the reproduction slice."""
    from sklearn.datasets import fetch_openml  # noqa: PLC0415  (network, invoked on purpose)

    target_dir = out_dir / "openml"
    target_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    sources: list[dict[str, Any]] = []

    for data_id in cfg.openml.dataset_ids:
        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        frame = bunch.frame.dropna()
        target_name = bunch.target_names[0]
        y_raw = frame[target_name]
        classes = list(y_raw.value_counts().index[:2])
        keep = y_raw.isin(classes)
        frame = frame[keep]
        y = (frame[target_name] == classes[1]).to_numpy().astype(np.int64)
        x = frame.drop(columns=[target_name]).select_dtypes(include="number").to_numpy(dtype=float)
        if x.shape[1] < 2:
            raise ValueError(f"OpenML {data_id} has fewer than two numeric features")

        rng_rows = np.random.default_rng(10_000 + data_id)
        n_take = cfg.openml.n_demo + cfg.openml.n_query
        idx = _stratified_indices(rng_rows, y, n_take)
        x, y = x[idx], y[idx]
        if cfg.openml.standardize_before_projection:
            mean, std = x.mean(axis=0), x.std(axis=0)
            std[std < 1e-9] = 1.0
            x = (x - mean) / std

        per_dim: dict[str, Any] = {}
        for dim in cfg.openml.dims:
            for pseed in cfg.openml.projection_seeds:
                rng = np.random.default_rng(20_000 + data_id * 1000 + dim * 10 + pseed)
                proj = random_projection(rng, x.shape[1], dim)
                xp = x @ proj
                key = f"openml{data_id}__d{dim}__p{pseed}"
                arrays[f"{key}__x"] = xp
                arrays[f"{key}__y"] = y
                demo, query = xp[: cfg.openml.n_demo], xp[cfg.openml.n_demo :]
                stats = [distance_stats(euclidean_distances(q, demo)) for q in query]
                per_dim[key] = {
                    "projection_seed": pseed,
                    "projection_sha256": _digest(proj),
                    "x_sha256": _digest(xp),
                    "mean_relative_contrast": float(np.mean([s.relative_contrast for s in stats])),
                }
        sources.append(
            {
                "data_id": data_id,
                "name": bunch.details.get("name"),
                "version": bunch.details.get("version"),
                "openml_url": bunch.details.get("url"),
                "n_rows_used": int(len(y)),
                "n_native_features": int(x.shape[1]),
                "class_balance": [int((y == c).sum()) for c in (0, 1)],
                "kept_classes": [str(c) for c in classes],
                "projections": per_dim,
            }
        )

    npz_path = target_dir / "openml.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "stage": "prepare-openml",
        "created_utc": datetime.now(UTC).isoformat(),
        "config_openml": cfg.openml.__dict__,
        "sources": sources,
        "artifacts": {"npz": str(npz_path)},
        "note": (
            "optional external-dataset slice; it is outside the synthetic gate's pass/fail evidence"
        ),
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _stratified_indices(rng: np.random.Generator, y: np.ndarray, n_take: int) -> np.ndarray:
    """Take an even number of rows per class, then shuffle."""
    per_class = n_take // 2
    picks = []
    for c in (0, 1):
        pool = np.flatnonzero(y == c)
        if len(pool) < per_class:
            raise ValueError(f"class {c} has {len(pool)} rows, need {per_class}")
        picks.append(rng.choice(pool, size=per_class, replace=False))
    idx = np.concatenate(picks)
    rng.shuffle(idx)
    return idx
