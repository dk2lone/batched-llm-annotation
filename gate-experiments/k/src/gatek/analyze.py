"""Stage 5: join the paired jobs, report the metrics, and apply the pre-registered gate.

This stage never loads a model. Terminology follows `metrics.py`:
lower relative contrast = `distance_concentrated`, higher = `distance_deconcentrated`.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, GateConfig
from .metrics import CONCENTRATED, DECONCENTRATED, concentration_label
from .synth import ARMS

VERDICTS = (
    "construction_failure",
    "model_capability_failure",
    "insufficient_results",
    "mechanism_unsupported",
    "mechanism_supported_enough_to_expand",
    "ambiguous",
)


def load_results(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "run" / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no results at {path}; run `gatek run` first")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(statistics.fmean(clean)) if clean else None


def cell_summary(rows: list[dict[str, Any]], repr_key: str) -> dict[str, Any]:
    """Summarize one (arm, dim, job_type) cell."""
    return {
        "n": len(rows),
        "accuracy": _mean([float(r["correct"]) for r in rows]),
        "mean_margin": _mean([r["margin"] for r in rows]),
        "parse_failure_rate": _mean([float(r["parse_status"] != "ok") for r in rows]),
        "observed_relative_contrast": _mean(
            [r["observed_stats"]["relative_contrast"] for r in rows]
        ),
        "observed_nearest_farthest_ratio": _mean(
            [r["observed_stats"]["nearest_farthest_ratio"] for r in rows]
        ),
        "observed_coefficient_of_variation": _mean(
            [r["observed_stats"]["coefficient_of_variation"] for r in rows]
        ),
        "latent_relative_contrast": _mean([r["latent_stats"]["relative_contrast"] for r in rows]),
        "repr_relative_contrast": _mean([r[repr_key]["relative_contrast"] for r in rows]),
        "repr_nearest_farthest_ratio": _mean([r[repr_key]["nearest_farthest_ratio"] for r in rows]),
        "repr_coefficient_of_variation": _mean(
            [r[repr_key]["coefficient_of_variation"] for r in rows]
        ),
        "spearman_latent_vs_repr": _mean([r["spearman_latent_vs_repr"] for r in rows]),
        "spearman_observed_vs_repr": _mean([r["spearman_observed_vs_repr"] for r in rows]),
        "mean_prompt_tokens": _mean([float(r["n_prompt_tokens"]) for r in rows]),
    }


def _by(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def paired_arm_contrast(
    rows: list[dict[str, Any]], arm_a: str, arm_b: str, repr_key: str
) -> dict[int, dict[str, Any]]:
    """Compare two arms at each dimension, paired on (seed, query_index).

    Only `order_random` jobs are used, so demonstration order is matched too.
    """
    base = [r for r in rows if r["job_type"] == "order_random"]
    index = {(r["arm"], r["dim"], r["seed"], r["query_index"]): r for r in base}
    out: dict[int, dict[str, Any]] = {}
    dims = sorted({r["dim"] for r in base})
    for dim in dims:
        keys_a = {
            (r["seed"], r["query_index"]) for r in base if r["arm"] == arm_a and r["dim"] == dim
        }
        keys_b = {
            (r["seed"], r["query_index"]) for r in base if r["arm"] == arm_b and r["dim"] == dim
        }
        pairs = [
            (index[(arm_a, dim, s, q)], index[(arm_b, dim, s, q)])
            for s, q in sorted(keys_a & keys_b)
        ]
        if not pairs:
            continue
        acc_a = _mean([float(a["correct"]) for a, _ in pairs]) or 0.0
        acc_b = _mean([float(b["correct"]) for _, b in pairs]) or 0.0
        rc_a = _mean([a[repr_key]["relative_contrast"] for a, _ in pairs]) or 0.0
        rc_b = _mean([b[repr_key]["relative_contrast"] for _, b in pairs]) or 0.0
        raw_a = _mean([a["observed_stats"]["relative_contrast"] for a, _ in pairs]) or 0.0
        raw_b = _mean([b["observed_stats"]["relative_contrast"] for _, b in pairs]) or 0.0
        deltas = [float(a["correct"]) - float(b["correct"]) for a, b in pairs]
        out[dim] = {
            "n_pairs": len(pairs),
            "accuracy_a": acc_a,
            "accuracy_b": acc_b,
            "accuracy_delta": acc_a - acc_b,
            "parse_failure_rate_a": _mean([float(a["parse_status"] != "ok") for a, _ in pairs]),
            "parse_failure_rate_b": _mean([float(b["parse_status"] != "ok") for _, b in pairs]),
            "accuracy_delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
            "repr_relative_contrast_a": rc_a,
            "repr_relative_contrast_b": rc_b,
            "repr_contrast_ratio_a_over_b": rc_a / max(rc_b, 1e-12),
            "repr_label_b_vs_a": concentration_label(rc_b, rc_a),
            "raw_relative_contrast_a": raw_a,
            "raw_relative_contrast_b": raw_b,
            "raw_contrast_ratio_a_over_b": raw_a / max(raw_b, 1e-12),
            "raw_label_b_vs_a": concentration_label(raw_b, raw_a),
            "margin_delta": (
                None
                if any(a["margin"] is None or b["margin"] is None for a, b in pairs)
                else _mean([a["margin"] - b["margin"] for a, b in pairs])
            ),
        }
    return out


def ablation_effects(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    """Nearest-row removal against matched random removal, paired on ``pair_id``."""
    index = {(r["pair_id"], r["job_type"]): r for r in rows}
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        per_dim: dict[int, dict[str, Any]] = {}
        pair_ids = sorted(
            {r["pair_id"] for r in rows if r["arm"] == arm and r["job_type"] == "ablate_nearest"}
        )
        for pid in pair_ids:
            base = index.get((pid, "order_random"))
            near = index.get((pid, "ablate_nearest"))
            rand = index.get((pid, "ablate_random"))
            if base is None or near is None or rand is None:
                continue
            bucket = per_dim.setdefault(
                base["dim"],
                {"drop_nearest": [], "drop_random": [], "flip_nearest": [], "flip_random": []},
            )
            bucket["drop_nearest"].append(float(base["correct"]) - float(near["correct"]))
            bucket["drop_random"].append(float(base["correct"]) - float(rand["correct"]))
            bucket["flip_nearest"].append(float(base["pred_letter"] != near["pred_letter"]))
            bucket["flip_random"].append(float(base["pred_letter"] != rand["pred_letter"]))
        summarized: dict[int, dict[str, Any]] = {}
        for dim, bucket in sorted(per_dim.items()):
            diff = [
                a - b for a, b in zip(bucket["drop_nearest"], bucket["drop_random"], strict=True)
            ]
            summarized[dim] = {
                "n_pairs": len(diff),
                "accuracy_drop_nearest_removal": _mean(bucket["drop_nearest"]),
                "accuracy_drop_random_removal": _mean(bucket["drop_random"]),
                "drop_difference": _mean(diff),
                "drop_difference_sd": float(np.std(diff, ddof=1)) if len(diff) > 1 else None,
                "prediction_flip_rate_nearest": _mean(bucket["flip_nearest"]),
                "prediction_flip_rate_random": _mean(bucket["flip_random"]),
            }
        out[arm] = summarized
    return out


def ordering_effects(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    """Nearest-first ordering against random ordering. A behavioural control only."""
    index = {(r["pair_id"], r["job_type"]): r for r in rows}
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        per_dim: dict[int, list[float]] = defaultdict(list)
        for pid in sorted({r["pair_id"] for r in rows if r["arm"] == arm}):
            base = index.get((pid, "order_random"))
            near = index.get((pid, "order_nearest_first"))
            if base is None or near is None:
                continue
            per_dim[base["dim"]].append(float(near["correct"]) - float(base["correct"]))
        out[arm] = {
            dim: {"n_pairs": len(v), "accuracy_delta_nearest_first_minus_random": _mean(v)}
            for dim, v in sorted(per_dim.items())
        }
    return out


def attention_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Descriptive only. Attention never decides the gate."""
    have = [r for r in rows if r.get("attention_spearman_vs_latent") is not None]
    if not have:
        return None
    per_dim: dict[int, list[float]] = defaultdict(list)
    for r in have:
        per_dim[r["dim"]].append(float(r["attention_spearman_vs_latent"]))
    return {
        "note": "weak evidence; reported for description only",
        "n_jobs": len(have),
        "mean_spearman_row_mass_vs_latent_distance_by_dim": {
            str(d): _mean(v) for d, v in sorted(per_dim.items())
        },
    }


def _high_dim_mean(per_dim: dict[int, dict[str, Any]], key: str, dims: list[int]) -> float | None:
    values = [per_dim[d][key] for d in dims if d in per_dim and per_dim[d].get(key) is not None]
    return _mean(values)


def anchor_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the standalone sparse-2D capability checkpoint."""
    anchor = [
        row
        for row in rows
        if row["arm"] == "sparse_zero_pad" and row["dim"] == 2 and row["job_type"] == "order_random"
    ]
    return {
        "n": len(anchor),
        "accuracy": _mean([float(row["correct"]) for row in anchor]),
        "parse_failure_rate": _mean([float(row["parse_status"] != "ok") for row in anchor]),
    }


def evaluate_gate(
    sparse_vs_dense: dict[int, dict[str, Any]],
    sparse_vs_nuisance: dict[int, dict[str, Any]],
    ablation: dict[str, dict[int, dict[str, Any]]],
    construction_ok: bool,
    failed_invariants: list[str],
    gate: GateConfig,
    anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered thresholds. Thresholds live in config, not here."""
    dims = gate.high_dims
    repr_ratio_dense = _high_dim_mean(sparse_vs_dense, "repr_contrast_ratio_a_over_b", dims)
    repr_ratio_nuisance = _high_dim_mean(sparse_vs_nuisance, "repr_contrast_ratio_a_over_b", dims)
    acc_delta_dense = _high_dim_mean(sparse_vs_dense, "accuracy_delta", dims)
    acc_delta_nuisance = _high_dim_mean(sparse_vs_nuisance, "accuracy_delta", dims)
    sparse_ablation = _high_dim_mean(ablation.get("sparse_zero_pad", {}), "drop_difference", dims)
    dense_ablation = _high_dim_mean(
        ablation.get("dense_redundant_projection", {}), "drop_difference", dims
    )
    ablation_selectivity = (
        None
        if sparse_ablation is None or dense_ablation is None
        else sparse_ablation - dense_ablation
    )
    if anchor is None:
        paired_anchor = sparse_vs_dense.get(2, {})
        anchor_accuracy = paired_anchor.get("accuracy_a")
        anchor_parse_failure = paired_anchor.get("parse_failure_rate_a")
    else:
        anchor_accuracy = anchor.get("accuracy")
        anchor_parse_failure = anchor.get("parse_failure_rate")

    conditions = {
        "model_solves_sparse_2d_anchor": {
            "value": anchor_accuracy,
            "threshold": gate.min_anchor_accuracy,
            "passed": anchor_accuracy is not None and anchor_accuracy >= gate.min_anchor_accuracy,
        },
        "anchor_parse_quality": {
            "value": anchor_parse_failure,
            "threshold": gate.max_anchor_parse_failure_rate,
            "passed": anchor_parse_failure is not None
            and anchor_parse_failure <= gate.max_anchor_parse_failure_rate,
        },
        "repr_contrast_sparse_over_dense": {
            "value": repr_ratio_dense,
            "threshold": gate.min_repr_contrast_ratio,
            "passed": repr_ratio_dense is not None
            and repr_ratio_dense >= gate.min_repr_contrast_ratio,
        },
        "repr_contrast_sparse_over_nuisance": {
            "value": repr_ratio_nuisance,
            "threshold": gate.min_repr_contrast_ratio,
            "passed": repr_ratio_nuisance is not None
            and repr_ratio_nuisance >= gate.min_repr_contrast_ratio,
        },
        "accuracy_sparse_minus_dense": {
            "value": acc_delta_dense,
            "threshold": gate.min_sparse_minus_dense_accuracy,
            "passed": acc_delta_dense is not None
            and acc_delta_dense >= gate.min_sparse_minus_dense_accuracy,
        },
        "accuracy_sparse_minus_nuisance": {
            "value": acc_delta_nuisance,
            "threshold": gate.min_sparse_minus_dense_accuracy,
            "passed": acc_delta_nuisance is not None
            and acc_delta_nuisance >= gate.min_sparse_minus_dense_accuracy,
        },
        "nearest_removal_selectivity_sparse_minus_dense": {
            "value": ablation_selectivity,
            "sparse_value": sparse_ablation,
            "dense_value": dense_ablation,
            "threshold": gate.min_sparse_minus_dense_ablation_delta,
            "passed": ablation_selectivity is not None
            and ablation_selectivity >= gate.min_sparse_minus_dense_ablation_delta,
        },
    }

    if not construction_ok:
        verdict = "construction_failure"
        reason = f"failed construction invariants: {failed_invariants}"
    elif not (
        conditions["model_solves_sparse_2d_anchor"]["passed"]
        and conditions["anchor_parse_quality"]["passed"]
    ):
        verdict = "model_capability_failure"
        reason = (
            "the model did not clear the 2D accuracy and parse-quality floor, so any "
            "high-dimensional failure is uninterpretable"
        )
    elif any(
        conditions[name]["value"] is None
        for name in (
            "repr_contrast_sparse_over_dense",
            "accuracy_sparse_minus_dense",
            "accuracy_sparse_minus_nuisance",
            "nearest_removal_selectivity_sparse_minus_dense",
        )
    ):
        verdict = "insufficient_results"
        reason = "the capability anchor passed, but the high-dimensional jobs are incomplete"
    else:
        repr_ok = (
            conditions["repr_contrast_sparse_over_dense"]["passed"]
            or conditions["repr_contrast_sparse_over_nuisance"]["passed"]
        )
        accuracy_ok = conditions["accuracy_sparse_minus_dense"]["passed"]
        nuisance_ok = conditions["accuracy_sparse_minus_nuisance"]["passed"]
        ablation_ok = conditions["nearest_removal_selectivity_sparse_minus_dense"]["passed"]
        if (
            conditions["repr_contrast_sparse_over_dense"]["passed"]
            and accuracy_ok
            and nuisance_ok
            and ablation_ok
        ):
            verdict = "mechanism_supported_enough_to_expand"
            reason = (
                "representation contrast, accuracy direction, nuisance arm and ablation all agree"
            )
        elif repr_ok and not accuracy_ok and not ablation_ok:
            verdict = "mechanism_unsupported"
            reason = (
                "representation concentration differs but accuracy and ablation do not follow it"
            )
        elif not repr_ok and not accuracy_ok:
            verdict = "mechanism_unsupported"
            reason = (
                "sparse padding does no better than dense projection and no "
                "representation contrast exists"
            )
        else:
            verdict = "ambiguous"
            reason = "mixed pattern across the pre-registered conditions"
    return {"verdict": verdict, "reason": reason, "conditions": conditions}


def gate_sensitivity(
    sparse_vs_dense: dict[int, dict[str, Any]],
    sparse_vs_nuisance: dict[int, dict[str, Any]],
    ablation: dict[str, dict[int, dict[str, Any]]],
    construction_ok: bool,
    failed_invariants: list[str],
    gate: GateConfig,
    anchor: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Re-apply the gate with every threshold scaled up and down."""
    out: list[dict[str, Any]] = []
    scale = gate.sensitivity_scale
    for factor, label in ((1.0 - scale, "loosened"), (1.0 + scale, "tightened")):
        perturbed = GateConfig(
            min_classical_accuracy=gate.min_classical_accuracy,
            min_anchor_accuracy=gate.min_anchor_accuracy,
            max_anchor_parse_failure_rate=gate.max_anchor_parse_failure_rate,
            min_nuisance_raw_contrast_ratio=gate.min_nuisance_raw_contrast_ratio,
            min_repr_contrast_ratio=gate.min_repr_contrast_ratio * factor,
            min_sparse_minus_dense_accuracy=gate.min_sparse_minus_dense_accuracy * factor,
            min_sparse_minus_dense_ablation_delta=(
                gate.min_sparse_minus_dense_ablation_delta * factor
            ),
            high_dims=gate.high_dims,
            sensitivity_scale=scale,
        )
        result = evaluate_gate(
            sparse_vs_dense,
            sparse_vs_nuisance,
            ablation,
            construction_ok,
            failed_invariants,
            perturbed,
            anchor,
        )
        out.append({"thresholds": label, "factor": factor, "verdict": result["verdict"]})
    return out


def analyze(cfg: Config, out_dir: Path) -> dict[str, Any]:
    rows = load_results(out_dir)
    repr_key = f"repr_stats_{cfg.run.repr_distance}"
    manifest_path = out_dir / "synthetic" / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    construction_ok = bool(manifest.get("construction_ok", False))
    failed = list(manifest.get("failed_invariants", ["synthetic manifest missing"]))

    cells = {
        f"{arm}|d{dim}|{jt}": cell_summary(rs, repr_key)
        for (arm, dim, jt), rs in sorted(_by(rows, "arm", "dim", "job_type").items())
    }
    sparse_vs_dense = paired_arm_contrast(
        rows, "sparse_zero_pad", "dense_redundant_projection", repr_key
    )
    sparse_vs_nuisance = paired_arm_contrast(
        rows, "sparse_zero_pad", "independent_nuisance", repr_key
    )
    ablation = ablation_effects(rows)
    ordering = ordering_effects(rows)
    attention = attention_summary(rows)
    anchor = anchor_summary(rows)
    gate = evaluate_gate(
        sparse_vs_dense,
        sparse_vs_nuisance,
        ablation,
        construction_ok,
        failed,
        cfg.gate,
        anchor,
    )
    sensitivity = gate_sensitivity(
        sparse_vs_dense,
        sparse_vs_nuisance,
        ablation,
        construction_ok,
        failed,
        cfg.gate,
        anchor,
    )

    analysis = {
        "stage": "analyze",
        "created_utc": datetime.now(UTC).isoformat(),
        "n_results": len(rows),
        "backend": sorted({r["backend"] for r in rows}),
        "model_id": sorted({r["model_id"] for r in rows}),
        "repr_distance_primary": cfg.run.repr_distance,
        "terminology": {
            CONCENTRATED: "distances are more alike; relative contrast is lower",
            DECONCENTRATED: "distances are less alike; relative contrast is higher",
        },
        "construction": {"ok": construction_ok, "failed_invariants": failed},
        "cells": cells,
        "paired_sparse_vs_dense": {str(k): v for k, v in sparse_vs_dense.items()},
        "paired_sparse_vs_nuisance": {str(k): v for k, v in sparse_vs_nuisance.items()},
        "ablation": {a: {str(k): v for k, v in d.items()} for a, d in ablation.items()},
        "ordering_control": {a: {str(k): v for k, v in d.items()} for a, d in ordering.items()},
        "attention": attention,
        "anchor": anchor,
        "gate": gate,
        "gate_thresholds": cfg.gate.__dict__,
        "gate_sensitivity": sensitivity,
    }
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    (analysis_dir / "REPORT.md").write_text(render_report(analysis), encoding="utf-8")
    return analysis


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Gate K report")
    lines.append("")
    lines.append(f"Generated {analysis['created_utc']}.")
    lines.append(
        f"Backend {analysis['backend']}, model {analysis['model_id']}, "
        f"{analysis['n_results']} results, primary representation distance "
        f"`{analysis['repr_distance_primary']}`."
    )
    if analysis["backend"] == ["mock"]:
        lines.append("")
        lines.append(
            "**These numbers come from the deterministic mock backend. They test the code path "
            "only. They are not evidence about any language model.**"
        )
    lines.append("")
    lines.append(f"## Verdict: `{analysis['gate']['verdict']}`")
    lines.append("")
    lines.append(analysis["gate"]["reason"])
    lines.append("")
    lines.append("| condition | value | threshold | passed |")
    lines.append("|---|---|---|---|")
    for name, c in analysis["gate"]["conditions"].items():
        lines.append(f"| {name} | {_fmt(c['value'])} | {_fmt(c['threshold'])} | {c['passed']} |")
    lines.append("")
    lines.append("Threshold sensitivity:")
    lines.append("")
    for s in analysis["gate_sensitivity"]:
        lines.append(
            f"- thresholds {s['thresholds']} by factor {s['factor']:.2f}: `{s['verdict']}`"
        )
    lines.append("")
    lines.append("## Construction")
    lines.append("")
    lines.append(
        f"Invariants passed: {analysis['construction']['ok']}. "
        f"Failed: {analysis['construction']['failed_invariants'] or 'none'}."
    )
    lines.append("")
    lines.append("## Paired arms at fixed ambient dimension (`order_random` jobs)")
    for title, key in (
        ("sparse_zero_pad vs dense_redundant_projection", "paired_sparse_vs_dense"),
        ("sparse_zero_pad vs independent_nuisance", "paired_sparse_vs_nuisance"),
    ):
        lines.append("")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(
            "| dim | n | acc A | acc B | acc delta | repr RC A | repr RC B | repr ratio | "
            "raw RC A | raw RC B | B vs A (repr) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for dim, v in sorted(analysis[key].items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {dim} | {v['n_pairs']} | {_fmt(v['accuracy_a'])} | {_fmt(v['accuracy_b'])} | "
                f"{_fmt(v['accuracy_delta'])} | {_fmt(v['repr_relative_contrast_a'])} | "
                f"{_fmt(v['repr_relative_contrast_b'])} | "
                f"{_fmt(v['repr_contrast_ratio_a_over_b'])} | "
                f"{_fmt(v['raw_relative_contrast_a'])} | "
                f"{_fmt(v['raw_relative_contrast_b'])} | {v['repr_label_b_vs_a']} |"
            )
    lines.append("")
    lines.append("## Causal row ablation")
    lines.append("")
    lines.append(
        "| arm | dim | n | drop nearest | drop random | difference | flip nearest | flip random |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm, per_dim in analysis["ablation"].items():
        for dim, v in sorted(per_dim.items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {arm} | {dim} | {v['n_pairs']} | {_fmt(v['accuracy_drop_nearest_removal'])} | "
                f"{_fmt(v['accuracy_drop_random_removal'])} | {_fmt(v['drop_difference'])} | "
                f"{_fmt(v['prediction_flip_rate_nearest'])} | "
                f"{_fmt(v['prediction_flip_rate_random'])} |"
            )
    lines.append("")
    lines.append("## Ordering control (behavioural, not a metric mechanism)")
    lines.append("")
    lines.append("| arm | dim | n | nearest-first minus random |")
    lines.append("|---|---|---|---|")
    for arm, per_dim in analysis["ordering_control"].items():
        for dim, v in sorted(per_dim.items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {arm} | {dim} | {v['n_pairs']} | "
                f"{_fmt(v['accuracy_delta_nearest_first_minus_random'])} |"
            )
    lines.append("")
    lines.append("## Per-cell detail")
    lines.append("")
    lines.append(
        "| cell | n | accuracy | margin | raw RC | repr RC | repr NF ratio | repr CoV | "
        "rho(latent, repr) | tokens |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, c in analysis["cells"].items():
        lines.append(
            f"| {name} | {c['n']} | {_fmt(c['accuracy'])} | {_fmt(c['mean_margin'])} | "
            f"{_fmt(c['observed_relative_contrast'])} | {_fmt(c['repr_relative_contrast'])} | "
            f"{_fmt(c['repr_nearest_farthest_ratio'])} | "
            f"{_fmt(c['repr_coefficient_of_variation'])} | "
            f"{_fmt(c['spearman_latent_vs_repr'])} | {_fmt(c['mean_prompt_tokens'], 0)} |"
        )
    if analysis["attention"]:
        lines.append("")
        lines.append("## Attention (descriptive only)")
        lines.append("")
        lines.append(json.dumps(analysis["attention"], indent=2, sort_keys=True))
    lines.append("")
    return "\n".join(lines)
