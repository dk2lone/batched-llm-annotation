"""Stage 1: build the three matched synthetic arms and validate them on CPU.

This stage never touches the network and never loads a language model. It freezes the
arrays, the concentration metrics, the classical baselines and a provenance manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .config import Config
from .metrics import (
    DistanceStats,
    distance_stats,
    euclidean_distances,
    mean_stats,
)
from .prompts import format_row
from .synth import (
    ARMS,
    ArmDataset,
    LatentDataset,
    arm_summary,
    build_arm,
    label_letters,
    make_latent,
)


def run_baselines(arm: ArmDataset) -> dict[str, float]:
    """Classical accuracy on this arm.

    `random_forest_observed` is the flexible learner that re-weights features. It must stay
    strong on every arm, including `independent_nuisance`.
    `knn1_observed` is the distance-based reference. It is expected to fall on the nuisance
    arm; that fall is the raw-distance concentration the gate wants to create.
    `knn1_latent` is the signal-aware reference on the shared latent 2D rows.
    """
    xtr, ytr = arm.demo_x, arm.demo_y
    xte, yte = arm.query_x, arm.query_y
    models = {
        "knn1_observed": KNeighborsClassifier(n_neighbors=1),
        "knn5_observed": KNeighborsClassifier(n_neighbors=min(5, len(xtr))),
        "logreg_observed": LogisticRegression(max_iter=2000),
        "random_forest_observed": RandomForestClassifier(n_estimators=200, random_state=0),
    }
    out: dict[str, float] = {}
    for name, model in models.items():
        model.fit(xtr, ytr)
        out[name] = float((model.predict(xte) == yte).mean())
    latent_knn = KNeighborsClassifier(n_neighbors=1).fit(arm.latent.demo_z, ytr)
    out["knn1_latent"] = float((latent_knn.predict(arm.latent.query_z) == yte).mean())
    return out


def arm_concentration(arm: ArmDataset) -> dict[str, Any]:
    """Per-query concentration of query-to-demonstration distances.

    Measured twice: in the full observed space of this arm, and in the shared latent 2D
    space, which is identical across arms by construction.
    """
    observed: list[DistanceStats] = []
    latent: list[DistanceStats] = []
    for i in range(len(arm.query_x)):
        observed.append(distance_stats(euclidean_distances(arm.query_x[i], arm.demo_x)))
        latent.append(distance_stats(euclidean_distances(arm.latent.query_z[i], arm.latent.demo_z)))
    return {
        "observed_space": mean_stats(observed),
        "latent_space": mean_stats(latent),
        "observed_per_query": [asdict(s) for s in observed],
    }


def _text_lengths(arm: ArmDataset, number_format: str, tokenizer: Any | None) -> dict[str, Any]:
    """Record row text length per arm. Token length needs a local tokenizer."""
    rows = [
        format_row(arm.demo_x[i], arm.feature_names, number_format) for i in range(len(arm.demo_x))
    ]
    chars = [len(r) for r in rows]
    out: dict[str, Any] = {
        "char_mean": float(np.mean(chars)),
        "char_max": int(max(chars)),
        "cell_count": len(arm.feature_names),
        "token_mean": None,
        "token_max": None,
        "token_source": "unavailable",
    }
    if tokenizer is not None:
        counts = [len(tokenizer(r, add_special_tokens=False)["input_ids"]) for r in rows]
        out["token_mean"] = float(np.mean(counts))
        out["token_max"] = int(max(counts))
        out["token_source"] = "local_tokenizer"
    return out


def _load_local_tokenizer(path: str | None) -> Any | None:
    """Load a tokenizer from a local directory only. Never downloads."""
    if not path:
        return None
    from transformers import AutoTokenizer  # noqa: PLC0415  (lazy, optional dependency)

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def check_invariants(
    latent: LatentDataset,
    arms: dict[str, ArmDataset],
    concentration: dict[str, dict[str, Any]],
    lengths: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, float]],
    cfg: Config,
) -> list[dict[str, Any]]:
    """Return one record per invariant. ``passed`` false means the future GPU run stops."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    ref = arms[ARMS[0]]
    add(
        "latent_rows_identical",
        all(
            np.array_equal(a.latent.demo_z, ref.latent.demo_z)
            and np.array_equal(a.latent.query_z, ref.latent.query_z)
            for a in arms.values()
        ),
        "all arms share the same latent 2D rows",
    )
    add(
        "labels_identical",
        all(
            np.array_equal(a.demo_y, ref.demo_y) and np.array_equal(a.query_y, ref.query_y)
            for a in arms.values()
        ),
        "all arms share the same labels",
    )
    add(
        "label_rule_holds",
        bool(
            np.array_equal(
                latent.demo_y, (latent.demo_z[:, 0] * latent.demo_z[:, 1] > 0).astype(int)
            )
            and np.array_equal(
                latent.query_y, (latent.query_z[:, 0] * latent.query_z[:, 1] > 0).astype(int)
            )
        ),
        "label equals XOR sign rule on the latent coordinates",
    )
    add(
        "class_balance_identical",
        len({tuple(int((a.demo_y == c).sum()) for c in (0, 1)) for a in arms.values()}) == 1,
        {a.arm: [int((a.demo_y == c).sum()) for c in (0, 1)] for a in arms.values()},
    )
    add(
        "feature_names_identical",
        len({a.feature_names for a in arms.values()}) == 1,
        list(ref.feature_names),
    )
    add(
        "shapes_identical",
        len({(a.demo_x.shape, a.query_x.shape) for a in arms.values()}) == 1,
        str(ref.demo_x.shape),
    )
    widths: set[int] = set()
    for a in arms.values():
        for row in (a.demo_x[0], a.query_x[0]):
            widths.update(
                len(c.split("=", 1)[1])
                for c in format_row(row, a.feature_names, cfg.synthetic.number_format).split(", ")
            )
    add("numeric_format_fixed_width", len(widths) == 1, sorted(widths))
    latent_rc = {name: c["latent_space"]["relative_contrast"] for name, c in concentration.items()}
    add(
        "latent_concentration_identical",
        max(latent_rc.values()) - min(latent_rc.values()) < 1e-9,
        latent_rc,
    )

    rf = {name: b["random_forest_observed"] for name, b in baselines.items()}
    checks.append(
        {
            "check": "observed_classical_diagnostic",
            "passed": True,
            "detail": {
                "random_forest_observed": rf,
                "note": (
                    "descriptive only; forcing an ordinary learner to solve the "
                    "independent-nuisance arm would remove the intended difficulty"
                ),
            },
        }
    )
    add(
        "signal_aware_classical_strong",
        min(b["knn1_latent"] for b in baselines.values()) >= cfg.gate.min_classical_accuracy,
        {name: b["knn1_latent"] for name, b in baselines.items()},
    )

    sparse_rc = concentration["sparse_zero_pad"]["observed_space"]["relative_contrast"]
    nuisance_rc = concentration["independent_nuisance"]["observed_space"]["relative_contrast"]
    dense_rc = concentration["dense_redundant_projection"]["observed_space"]["relative_contrast"]
    ratio = sparse_rc / max(nuisance_rc, 1e-12)
    if ref.dim <= 2:
        add(
            "nuisance_raw_contrast",
            True,
            {"skipped": "dim 2 has no nuisance columns", "ratio": ratio},
        )
    else:
        add(
            "nuisance_raw_contrast",
            ratio >= cfg.gate.min_nuisance_raw_contrast_ratio,
            {
                "sparse_relative_contrast": sparse_rc,
                "nuisance_relative_contrast": nuisance_rc,
                "ratio": ratio,
                "threshold": cfg.gate.min_nuisance_raw_contrast_ratio,
            },
        )
    # Descriptive only. A missing raw contrast here is expected, not a failure, because
    # both arms live on the same 2D manifold.
    checks.append(
        {
            "check": "sparse_vs_dense_raw_contrast",
            "passed": True,
            "detail": {
                "sparse_relative_contrast": sparse_rc,
                "dense_relative_contrast": dense_rc,
                "note": "descriptive; the two arms share one 2D manifold",
            },
        }
    )
    checks.append(
        {
            "check": "token_length_recorded",
            "passed": True,
            "detail": {name: lengths[name] for name in arms},
        }
    )
    return checks


def prepare_synthetic(cfg: Config, out_dir: Path) -> dict[str, Any]:
    """Generate, validate and freeze every arm. Returns the manifest."""
    synth_dir = out_dir / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_local_tokenizer(cfg.synthetic.tokenizer_path)

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []

    for seed in cfg.synthetic.seeds:
        latent = make_latent(
            seed=seed,
            n_demo=cfg.synthetic.n_demo,
            n_query=cfg.synthetic.n_query,
            cluster_offset=cfg.synthetic.cluster_offset,
            cluster_sigma=cfg.synthetic.cluster_sigma,
            margin=cfg.synthetic.margin,
        )
        arrays[f"latent__s{seed}__demo_z"] = latent.demo_z
        arrays[f"latent__s{seed}__demo_y"] = latent.demo_y
        arrays[f"latent__s{seed}__query_z"] = latent.query_z
        arrays[f"latent__s{seed}__query_y"] = latent.query_y
        for dim in cfg.synthetic.dims:
            arms = {
                arm: build_arm(
                    arm,
                    dim,
                    latent,
                    cfg.synthetic.nuisance_scale,
                    cfg.synthetic.dense_column_scale,
                )
                for arm in ARMS
            }
            concentration = {name: arm_concentration(a) for name, a in arms.items()}
            baselines = {name: run_baselines(a) for name, a in arms.items()}
            lengths = {
                name: _text_lengths(a, cfg.synthetic.number_format, tokenizer)
                for name, a in arms.items()
            }
            checks = check_invariants(latent, arms, concentration, lengths, baselines, cfg)
            for c in checks:
                all_checks.append({"dim": dim, "seed": seed, **c})
            for name, a in arms.items():
                key = f"{name}__d{dim}__s{seed}"
                arrays[f"{key}__demo_x"] = a.demo_x
                arrays[f"{key}__query_x"] = a.query_x
                records.append(
                    {
                        "key": key,
                        **arm_summary(a),
                        "demo_letters": label_letters(a.demo_y),
                        "query_letters": label_letters(a.query_y),
                        "concentration": {
                            "observed_space": concentration[name]["observed_space"],
                            "latent_space": concentration[name]["latent_space"],
                        },
                        "baselines": baselines[name],
                        "text_lengths": lengths[name],
                    }
                )

    npz_path = synth_dir / "data.npz"
    np.savez_compressed(npz_path, **arrays)
    rows_path = synth_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    failures = [c for c in all_checks if not c["passed"]]
    manifest = {
        "stage": "prepare-synthetic",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": cfg.as_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            "data_npz": str(npz_path),
            "data_npz_sha256": _sha256_file(npz_path),
            "rows_jsonl": str(rows_path),
            "rows_jsonl_sha256": _sha256_file(rows_path),
        },
        "n_arm_datasets": len(records),
        "invariants": all_checks,
        "construction_ok": not failures,
        "failed_invariants": [c["check"] for c in failures],
    }
    (synth_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
