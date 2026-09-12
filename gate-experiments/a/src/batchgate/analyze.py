"""Stage 4: turn raw prediction rows into a machine-readable result and a report.

No model is loaded. The only input is the raw JSONL written by `run`.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .stats import (
    bootstrap_mean,
    cluster_bootstrap_icc,
    design_effect,
    effective_n,
    group_stats,
    icc_from_group_stats,
    icc_oneway,
    percentile_interval,
    permutation_null_icc,
    slot_effect,
    upper_p_value,
)
from .util import read_jsonl, write_json

AGGREGATE_ICC_NOTE = (
    "Every target document appears once per assignment, so the three per-assignment ICCs "
    "are computed on disjoint rows but on the same documents. Pooling all assignments into "
    "one ANOVA would count each document three times and understate the standard error. "
    "The aggregate point estimate is therefore the unweighted mean of the per-assignment "
    "ICCs. Its bootstrap resamples batches inside each assignment independently, which "
    "ignores the dependence between assignments induced by the shared document set, so the "
    "aggregate interval is optimistic. The reported decision interval is the conservative "
    "envelope: the lowest lower bound and the highest upper bound across the aggregate "
    "interval and the three per-assignment intervals."
)


def _clean(value: float) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path, tolerate_truncated_tail=True))
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["job_id"]), str(row["doc_id"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _parsed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["parse_status"] == "ok" and r["error"] is not None]


def _by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return dict(out)


def _doc_mean_error(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        totals[str(row["doc_id"])].append(int(row["error"]))
    return {doc: float(np.mean(errs)) for doc, errs in totals.items()}


def parse_section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition, group in sorted(_by(rows, "condition").items()):
        statuses: dict[str, int] = defaultdict(int)
        for row in group:
            statuses[str(row["parse_status"])] += 1
        ok = statuses.get("ok", 0)
        by_assignment: dict[str, float] = {}
        for assignment, sub in sorted(_by(group, "assignment").items()):
            good = sum(1 for r in sub if r["parse_status"] == "ok")
            by_assignment[str(assignment)] = 1.0 - good / len(sub)
        out[str(condition)] = {
            "n_rows": len(group),
            "n_parsed": ok,
            "parse_failure_rate": 1.0 - ok / len(group),
            "status_counts": dict(sorted(statuses.items())),
            "parse_failure_rate_by_assignment": by_assignment,
            "n_rows_with_unknown_ids": sum(1 for r in group if r.get("n_unknown_ids", 0)),
        }
    return out


def accuracy_section(
    rows: list[dict[str, Any]], cfg: Config, rng: np.random.Generator
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition, group in sorted(_by(_parsed(rows), "condition").items()):
        doc_means = _doc_mean_error(group)
        values = np.array(list(doc_means.values()), dtype=float)
        interval = bootstrap_mean(values, cfg.analyze.bootstrap, rng, cfg.analyze.alpha)
        by_assignment: dict[str, Any] = {}
        for assignment, sub in sorted(_by(group, "assignment").items()):
            errs = np.array([r["error"] for r in sub], dtype=float)
            by_assignment[str(assignment)] = {
                "n": int(errs.size),
                "accuracy": float(1.0 - errs.mean()),
                "error_rate": float(errs.mean()),
            }
        out[str(condition)] = {
            "n_rows": len(group),
            "n_documents": len(doc_means),
            "error_rate_document_mean": _clean(interval.point),
            "accuracy_document_mean": _clean(1.0 - interval.point),
            "error_rate_ci": [_clean(interval.low), _clean(interval.high)],
            "by_assignment": by_assignment,
            "note": "The bootstrap resamples documents, not rows, so repeats do not inflate n.",
        }
    return out


def paired_section(
    rows: list[dict[str, Any]], cfg: Config, rng: np.random.Generator
) -> dict[str, Any]:
    parsed = _parsed(rows)
    per_condition = {c: _doc_mean_error(g) for c, g in _by(parsed, "condition").items()}
    out: dict[str, Any] = {}
    for left, right in (("joint", "solo"), ("filler", "solo"), ("joint", "filler")):
        if left not in per_condition or right not in per_condition:
            continue
        shared = sorted(set(per_condition[left]) & set(per_condition[right]))
        if not shared:
            continue
        deltas = np.array(
            [per_condition[left][d] - per_condition[right][d] for d in shared], dtype=float
        )
        interval = bootstrap_mean(deltas, cfg.analyze.bootstrap, rng, cfg.analyze.alpha)
        out[f"{left}_minus_{right}"] = {
            "n_documents": len(shared),
            "mean_error_delta": _clean(interval.point),
            "ci": [_clean(interval.low), _clean(interval.high)],
            "accuracy_delta": _clean(-interval.point),
            "excludes_zero": bool(
                not math.isnan(interval.low) and (interval.low > 0 or interval.high < 0)
            ),
            "unit": "positive means more errors in the first condition",
        }
    return out


def _icc_block(
    values: np.ndarray, groups: np.ndarray, k: float, cfg: Config, rng: np.random.Generator
) -> dict[str, Any]:
    point = icc_oneway(values, groups)
    boot = cluster_bootstrap_icc(values, groups, cfg.analyze.bootstrap, rng)
    interval = percentile_interval(boot, point, cfg.analyze.alpha)
    null = permutation_null_icc(values, groups, cfg.analyze.permutations, rng)
    n = int(values.size)
    return {
        **interval.as_dict(),
        "n_rows": n,
        "n_batches": int(np.unique(groups).size),
        "n_errors": int(values.sum()),
        "n_correct": int(values.size - values.sum()),
        "mean_batch_size": float(n / max(1, np.unique(groups).size)),
        "design_effect": _clean(design_effect(point, k)),
        "effective_n": _clean(effective_n(n, point, k)),
        "permutation_p": _clean(upper_p_value(point, null)),
        "permutation_null_mean": _clean(float(np.nanmean(null)) if null.size else math.nan),
        "n_permutations": int(cfg.analyze.permutations),
    }


def _condition_arrays(rows: list[dict[str, Any]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per assignment: error values and their real batch labels."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for assignment, group in _by(rows, "assignment").items():
        errors = np.array([r["error"] for r in group], dtype=float)
        batches = np.array([str(r["real_batch_id"]) for r in group])
        out[int(assignment)] = (errors, batches)
    return out


def _solo_placebo_arrays(
    rows: list[dict[str, Any]], batch_of: dict[str, dict[str, str]]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Solo rows borrow each assignment's batch labels. Solo shares no context, so
    a non-zero ICC here is leftover item structure inside a batch, not batching."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    per_doc = {str(r["doc_id"]): r for r in rows}
    for assignment, mapping in batch_of.items():
        pairs = [(per_doc[d], b) for d, b in mapping.items() if d in per_doc]
        if not pairs:
            continue
        out[int(assignment)] = (
            np.array([r["error"] for r, _ in pairs], dtype=float),
            np.array([b for _, b in pairs]),
        )
    return out


def icc_section(
    rows: list[dict[str, Any]], cfg: Config, rng: np.random.Generator
) -> dict[str, Any]:
    parsed = _parsed(rows)
    by_condition = _by(parsed, "condition")
    joint = by_condition.get("joint", [])
    if not joint:
        return {"error": "no parsed joint rows"}

    # Document to real batch map, taken from the joint rows themselves.
    batch_of: dict[str, dict[str, str]] = defaultdict(dict)
    for row in joint:
        batch_of[str(row["assignment"])][str(row["doc_id"])] = str(row["real_batch_id"])

    k = float(cfg.plan.k)
    blocks: dict[str, dict[str, Any]] = {}
    per_assignment: dict[str, dict[str, Any]] = {}
    stats_for_aggregate: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for assignment, (errors, batches) in sorted(_condition_arrays(joint).items()):
        block = _icc_block(errors, batches, k, cfg, rng)
        per_assignment[str(assignment)] = block
        stats_for_aggregate.append(group_stats(errors, batches))

    points = np.array([b["point"] for b in per_assignment.values() if b["point"] is not None])
    aggregate_point = float(points.mean()) if points.size else math.nan
    boot = np.empty(cfg.analyze.bootstrap, dtype=float)
    for i in range(cfg.analyze.bootstrap):
        values = []
        for counts, sums, sumsq in stats_for_aggregate:
            idx = rng.integers(0, counts.size, counts.size)
            values.append(icc_from_group_stats(counts[idx], sums[idx], sumsq[idx]))
        with np.errstate(invalid="ignore"):
            boot[i] = float(np.nanmean(values)) if values else math.nan
    aggregate_interval = percentile_interval(boot, aggregate_point, cfg.analyze.alpha)

    lows = [aggregate_interval.low] + [
        b["ci_low"] for b in per_assignment.values() if b["ci_low"] is not None
    ]
    highs = [aggregate_interval.high] + [
        b["ci_high"] for b in per_assignment.values() if b["ci_high"] is not None
    ]
    lows = [x for x in lows if x is not None and not math.isnan(x)]
    highs = [x for x in highs if x is not None and not math.isnan(x)]

    n_rows = sum(b["n_rows"] for b in per_assignment.values())
    blocks["joint"] = {
        "per_assignment": per_assignment,
        "aggregate": {
            **aggregate_interval.as_dict(),
            "strategy": "unweighted mean of per-assignment ICCs",
            "note": AGGREGATE_ICC_NOTE,
            "between_assignment_sd": float(points.std(ddof=1)) if points.size > 1 else None,
            "design_effect": _clean(design_effect(aggregate_point, k)),
            "effective_n_per_assignment": _clean(
                effective_n(n_rows // max(1, len(per_assignment)), aggregate_point, k)
            ),
        },
        "decision_interval": {
            "ci_low": min(lows) if lows else None,
            "ci_high": max(highs) if highs else None,
            "rule": "conservative envelope over the aggregate and per-assignment intervals",
        },
    }

    # Placebo blocks. Filler and solo share no context across a joint batch, so a
    # non-zero ICC there is residual item structure, not batching.
    for condition in ("filler", "solo"):
        group = by_condition.get(condition, [])
        if not group:
            continue
        arrays = (
            _solo_placebo_arrays(group, dict(batch_of))
            if condition == "solo"
            else _condition_arrays(group)
        )
        placebo: dict[str, dict[str, Any]] = {}
        for assignment, (errors, batches) in sorted(arrays.items()):
            if errors.size == 0:
                continue
            placebo[str(assignment)] = _icc_block(errors, batches, k, cfg, rng)
        points_p = np.array([b["point"] for b in placebo.values() if b["point"] is not None])
        blocks[condition] = {
            "per_assignment": placebo,
            "aggregate_point": float(points_p.mean()) if points_p.size else None,
            "grouping": (
                "corresponding joint batch id; the 16 targets were labelled in 16 separate "
                "prompts, so any positive value is not a batching effect"
            ),
        }

    # Sensitivity: parse failures are not missing at random if they cluster by batch.
    failures_as_errors: dict[str, Any] = {}
    joint_all = _by(rows, "condition").get("joint", [])
    for assignment, group in sorted(_by(joint_all, "assignment").items()):
        errors = np.array(
            [1.0 if r["error"] is None else float(r["error"]) for r in group], dtype=float
        )
        batches = np.array([str(r["real_batch_id"]) for r in group])
        failures_as_errors[str(assignment)] = _clean(icc_oneway(errors, batches))
    blocks["sensitivity_failures_as_errors"] = {
        "joint_icc_by_assignment": failures_as_errors,
        "note": "Parse failures are dropped elsewhere. This treats each one as an error instead.",
    }
    return blocks


def position_section(
    rows: list[dict[str, Any]], cfg: Config, rng: np.random.Generator
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition, group in sorted(_by(_parsed(rows), "condition").items()):
        if condition == "solo":
            continue
        errors = np.array([r["error"] for r in group], dtype=float)
        slots = np.array([int(r["slot"]) for r in group])
        if np.unique(slots).size < 2:
            continue
        out[str(condition)] = slot_effect(errors, slots, cfg.analyze.permutations, rng)
    return out


def verdict_section(
    icc: dict[str, Any],
    paired: dict[str, Any],
    parse: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    a = cfg.analyze
    joint = icc.get("joint", {})
    point = joint.get("aggregate", {}).get("point")
    decision = joint.get("decision_interval", {})
    low = decision.get("ci_low")
    high = decision.get("ci_high")
    p_values = [
        b.get("permutation_p")
        for b in joint.get("per_assignment", {}).values()
        if b.get("permutation_p") is not None
    ]
    p_max = max(p_values) if p_values else None
    error_counts = [
        int(block.get("n_errors", 0)) for block in joint.get("per_assignment", {}).values()
    ]
    enough_errors = bool(error_counts) and min(error_counts) >= a.min_errors_per_assignment
    joint_parse_failure = parse.get("joint", {}).get("parse_failure_rate")
    parse_ok = joint_parse_failure is not None and joint_parse_failure <= a.max_parse_failure_rate
    filler_available = "filler_minus_solo" in paired and "filler" in icc

    filler_gap = paired.get("filler_minus_solo", {})
    joint_gap = paired.get("joint_minus_solo", {})
    filler_delta = filler_gap.get("mean_error_delta")
    joint_delta = joint_gap.get("mean_error_delta")
    material_filler_gap = bool(
        filler_delta is not None
        and abs(filler_delta) >= a.material_accuracy_gap
        and filler_gap.get("excludes_zero")
    )
    filler_explains = bool(
        material_filler_gap
        and joint_delta is not None
        and joint_delta > 0
        and filler_delta is not None
        and filler_delta >= a.filler_explain_fraction * joint_delta
    )
    placebo_point = icc.get("filler", {}).get("aggregate_point")
    placebo_matches = bool(
        placebo_point is not None and point is not None and placebo_point >= point
    )

    reasons: list[str] = []
    if point is None:
        label = "UNDETERMINED"
        reasons.append(
            "the aggregate ICC is undefined, usually because a condition has no variance"
        )
    elif not enough_errors:
        label = "UNDETERMINED"
        reasons.append(
            "too few joint-condition errors to estimate a stable within-batch correlation: "
            f"counts by assignment are {error_counts}, requirement is "
            f"{a.min_errors_per_assignment} each"
        )
    elif not parse_ok:
        label = "UNDETERMINED"
        reasons.append(
            f"joint parse-failure rate {_fmt(joint_parse_failure)} exceeds the "
            f"pre-registered maximum {a.max_parse_failure_rate:.4f}"
        )
    elif not filler_available and point >= a.proceed_rho:
        label = "NEEDS_FILLER_CONTROL"
        reasons.append(
            "the primary point estimate clears the threshold, but the length-matched "
            "filler control has not run"
        )
    elif filler_explains or placebo_matches:
        label = "FILLER_CONFOUNDED"
        if filler_explains:
            reasons.append(
                f"the length-matched filler reproduces {filler_delta:.4f} of the "
                f"{joint_delta:.4f} joint error increase"
            )
        if placebo_matches:
            reasons.append("the filler placebo ICC is at least as large as the joint ICC")
        reasons.append("do not claim a batching effect from these numbers")
    elif (
        point >= a.proceed_rho
        and low is not None
        and low > 0
        and p_max is not None
        and p_max < a.alpha
    ):
        label = "PROCEED"
        reasons.append(f"point estimate {point:.4f} >= {a.proceed_rho}")
        reasons.append(f"conservative CI lower bound {low:.4f} excludes zero")
        reasons.append(f"largest fake-batch p-value {p_max:.4f} < {a.alpha}")
    elif point >= a.proceed_rho:
        label = "PROCEED_POINT_ONLY"
        reasons.append(f"point estimate {point:.4f} >= {a.proceed_rho}")
        if low is None or low <= 0:
            reasons.append("the conservative CI does not exclude zero")
        if p_max is None or p_max >= a.alpha:
            reasons.append("at least one fake-batch permutation test is not significant")
        reasons.append("this is not evidence of a non-zero effect; treat it as a point estimate")
    elif (
        low is not None
        and high is not None
        and low >= 0.0
        and high <= a.bounded_null_high
        and not material_filler_gap
    ):
        label = "BOUNDED_NEGATIVE_CANDIDATE"
        reasons.append(
            f"the conservative CI [{low:.4f}, {high:.4f}] lies inside [0, {a.bounded_null_high}]"
        )
        reasons.append("the filler control shows no material accuracy gap")
    else:
        label = "AMBIGUOUS"
        reasons.append(
            f"point estimate {point:.4f} is below {a.proceed_rho} and the conservative CI "
            f"[{low}, {high}] is not inside [0, {a.bounded_null_high}]"
        )
        reasons.append("no gate decision follows from this result")
    return {
        "label": label,
        "reasons": reasons,
        "aggregate_icc": point,
        "decision_ci": [low, high],
        "max_permutation_p": p_max,
        "filler_minus_solo_error_delta": filler_delta,
        "joint_minus_solo_error_delta": joint_delta,
        "material_filler_accuracy_gap": material_filler_gap,
        "filler_explains_accuracy_loss": filler_explains,
        "filler_placebo_icc": placebo_point,
        "joint_errors_by_assignment": error_counts,
        "enough_errors": enough_errors,
        "joint_parse_failure_rate": joint_parse_failure,
        "parse_quality_passed": parse_ok,
        "filler_control_available": filler_available,
        "thresholds": {
            "proceed_rho": a.proceed_rho,
            "bounded_null_high": a.bounded_null_high,
            "alpha": a.alpha,
            "material_accuracy_gap": a.material_accuracy_gap,
            "filler_explain_fraction": a.filler_explain_fraction,
            "min_errors_per_assignment": a.min_errors_per_assignment,
            "max_parse_failure_rate": a.max_parse_failure_rate,
        },
        "labels": {
            "PROCEED": "point estimate clears the threshold and the interval excludes zero",
            "PROCEED_POINT_ONLY": "point estimate clears the threshold, interval evidence does not",
            "BOUNDED_NEGATIVE_CANDIDATE": "interval inside the pre-registered null band",
            "FILLER_CONFOUNDED": "context length explains the effect",
            "AMBIGUOUS": "no decision",
            "UNDETERMINED": "the estimand could not be computed",
            "NEEDS_FILLER_CONTROL": "the primary signal requires the context-length control",
        },
    }


def analyze(cfg: Config) -> dict[str, Any]:
    raw_path = cfg.path(cfg.run.out_path)
    rows = load_rows(raw_path)
    if not rows:
        raise ValueError(f"{raw_path} has no rows")
    rng = np.random.default_rng(cfg.analyze.seed)

    backends = sorted({str(r.get("backend")) for r in rows})
    icc = icc_section(rows, cfg, rng)
    paired = paired_section(rows, cfg, rng)
    parse = parse_section(rows)
    result: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "config_name": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "source": {
            "raw_path": str(raw_path),
            "n_rows": len(rows),
            "backends": backends,
            "is_mock": "mock" in backends,
            "model_ids": sorted({str(r.get("model_id")) for r in rows}),
            "run_ids": sorted({str(r.get("run_id")) for r in rows}),
            "n_jobs": len({str(r["job_id"]) for r in rows}),
            "n_documents": len({str(r["doc_id"]) for r in rows}),
            "total_generation_seconds": float(sum(float(r.get("gen_seconds", 0.0)) for r in rows)),
        },
        "parse": parse,
        "accuracy": accuracy_section(rows, cfg, rng),
        "paired_deltas": paired,
        "icc": icc,
        "position": position_section(rows, cfg, rng),
    }
    result["verdict"] = verdict_section(icc, paired, parse, cfg)
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(result: dict[str, Any]) -> str:
    source = result["source"]
    verdict = result["verdict"]
    lines: list[str] = []
    lines.append("# Gate A result: within-batch error correlation")
    lines.append("")
    if source["is_mock"]:
        lines.append(
            "> **MOCK DATA. NOT A RESULT.** These rows come from the deterministic mock "
            "backend, which synthesises errors from the gold labels. They check the "
            "pipeline, never the model. No pilot has run."
        )
        lines.append("")
    lines.append(
        f"- Config: `{result['config_name']}` (fingerprint `{result['config_fingerprint']}`)"
    )
    lines.append(
        f"- Backend: {', '.join(source['backends'])}; model: {', '.join(source['model_ids'])}"
    )
    lines.append(
        f"- Rows: {source['n_rows']} over {source['n_jobs']} jobs "
        f"and {source['n_documents']} documents"
    )
    lines.append("")
    lines.append(f"## Verdict: {verdict['label']}")
    lines.append("")
    for reason in verdict["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(
        f"Aggregate ICC {_fmt(verdict['aggregate_icc'])}, "
        f"decision interval [{_fmt(verdict['decision_ci'][0])}, "
        f"{_fmt(verdict['decision_ci'][1])}], "
        f"largest fake-batch p {_fmt(verdict['max_permutation_p'])}."
    )
    lines.append("")

    lines.append("## Accuracy and parsing by condition")
    lines.append("")
    lines.append("| condition | rows | docs | accuracy | error rate CI | parse failure rate |")
    lines.append("|---|---|---|---|---|---|")
    for condition, block in sorted(result["accuracy"].items()):
        parse = result["parse"].get(condition, {})
        ci = block["error_rate_ci"]
        lines.append(
            f"| {condition} | {block['n_rows']} | {block['n_documents']} | "
            f"{_fmt(block['accuracy_document_mean'])} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(parse.get('parse_failure_rate'))} |"
        )
    lines.append("")

    lines.append("## Paired error deltas, per document")
    lines.append("")
    lines.append("| contrast | docs | mean error delta | CI | excludes zero |")
    lines.append("|---|---|---|---|---|")
    for name, block in sorted(result["paired_deltas"].items()):
        lines.append(
            f"| {name} | {block['n_documents']} | {_fmt(block['mean_error_delta'])} | "
            f"[{_fmt(block['ci'][0])}, {_fmt(block['ci'][1])}] | {block['excludes_zero']} |"
        )
    lines.append("")

    joint = result["icc"].get("joint", {})
    if joint:
        lines.append("## Within-real-batch ICC, joint condition")
        lines.append("")
        lines.append("| assignment | ICC | CI | batches | DEFF | n_eff | fake-batch p |")
        lines.append("|---|---|---|---|---|---|---|")
        for assignment, block in sorted(joint["per_assignment"].items()):
            lines.append(
                f"| {assignment} | {_fmt(block['point'])} | "
                f"[{_fmt(block['ci_low'])}, {_fmt(block['ci_high'])}] | {block['n_batches']} | "
                f"{_fmt(block['design_effect'], 3)} | {_fmt(block['effective_n'], 1)} | "
                f"{_fmt(block['permutation_p'])} |"
            )
        agg = joint["aggregate"]
        lines.append(
            f"| aggregate | {_fmt(agg['point'])} | [{_fmt(agg['ci_low'])}, {_fmt(agg['ci_high'])}] "
            f"| - | {_fmt(agg['design_effect'], 3)} | "
            f"{_fmt(agg['effective_n_per_assignment'], 1)} | - |"
        )
        lines.append("")
        lines.append(f"Aggregate strategy: {agg['note']}")
        lines.append("")

    for placebo in ("filler", "solo"):
        block = result["icc"].get(placebo)
        if not block or "per_assignment" not in block:
            continue
        lines.append(f"## Placebo ICC, {placebo} condition")
        lines.append("")
        lines.append(f"{block['grouping']}")
        lines.append("")
        lines.append("| assignment | ICC | CI | fake-batch p |")
        lines.append("|---|---|---|---|")
        for assignment, sub in sorted(block["per_assignment"].items()):
            lines.append(
                f"| {assignment} | {_fmt(sub['point'])} | "
                f"[{_fmt(sub['ci_low'])}, {_fmt(sub['ci_high'])}] | "
                f"{_fmt(sub['permutation_p'])} |"
            )
        lines.append("")

    if result["position"]:
        lines.append("## Output position")
        lines.append("")
        lines.append("| condition | p any slot difference | slot-error correlation | p trend |")
        lines.append("|---|---|---|---|")
        for condition, block in sorted(result["position"].items()):
            lines.append(
                f"| {condition} | {_fmt(block['p_any_slot_difference'])} | "
                f"{_fmt(block['slot_error_correlation'])} | {_fmt(block['p_monotone_trend'])} |"
            )
        lines.append("")

    lines.append("## Caveats carried by this report")
    lines.append("")
    lines.append(
        "- Parse failures are dropped from the ICC. If they cluster by batch, "
        "that deletion is not ignorable. See `icc.sensitivity_failures_as_errors`."
    )
    lines.append(
        "- The aggregate interval treats the three assignments as independent. "
        "The decision interval widens it to the conservative envelope."
    )
    lines.append("- A single gate on one model and one dataset bounds nothing beyond that cell.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(cfg: Config, result: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = cfg.path(cfg.analyze.out_dir)
    json_path = out_dir / "analysis.json"
    md_path = out_dir / "report.md"
    write_json(json_path, result)
    md_path.write_text(render_report(result), encoding="utf-8")
    return json_path, md_path
