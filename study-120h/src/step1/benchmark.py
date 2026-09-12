"""Throughput measurement and the forecast for the remaining frozen call ledger.

Every number here is a performance fact. None of it is a scientific finding, and none of it
may enter a claim about dependence, covariance, variance, or coverage. The screen report and
this report are written to separate files for that reason.

Step 1 measures only solo calls, so a forecast for 16-label joint calls is an extrapolation.
The extrapolation range is reported with every projection instead of being hidden inside it.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

#: Conditions of the frozen plan, and the number of labels each call requests.
LABELS_PER_CALL = {
    "joint": 16,
    "serial_joint": 16,
    "exact_context": 1,
    "solo": 1,
    "filler": 1,
    "serial_exact": 1,
}


def fit_cost_model(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit chunk_seconds = a * prefill_tokens + b * decode_tokens + c * n_calls.

    Non-negative least squares, because a negative token cost is not physical. With few
    chunks or a narrow token range the fit is weak, and `n_chunks` plus `observed_prompt_tokens`
    let the reader see that directly.
    """
    usable = [c for c in chunks if float(c["chunk_seconds"]) > 0 and int(c["n_calls"]) > 0]
    if len(usable) < 3:
        return {
            "ok": False,
            "reason": "fewer than 3 timed chunks; no cost model is fitted",
            "n_chunks": len(usable),
        }
    design = np.asarray(
        [
            [
                float(c["prompt_tokens_sum"]),
                float(c["output_tokens_sum"]),
                float(c["n_calls"]),
            ]
            for c in usable
        ]
    )
    seconds = np.asarray([float(c["chunk_seconds"]) for c in usable])
    from scipy.optimize import nnls  # noqa: PLC0415 - only needed on this path

    coefficients, residual_norm = nnls(design, seconds)
    predicted = design @ coefficients
    per_call_tokens = design[:, 0] / design[:, 2]
    return {
        "ok": True,
        "n_chunks": len(usable),
        "seconds_per_prefill_token": float(coefficients[0]),
        "seconds_per_decode_token": float(coefficients[1]),
        "seconds_per_call_fixed": float(coefficients[2]),
        "residual_norm_seconds": float(residual_norm),
        "relative_residual": float(residual_norm / max(np.linalg.norm(seconds), 1e-9)),
        "mean_absolute_error_seconds": float(np.abs(predicted - seconds).mean()),
        "observed_prompt_tokens_per_call": {
            "min": float(per_call_tokens.min()),
            "max": float(per_call_tokens.max()),
            "mean": float(per_call_tokens.mean()),
        },
        "observed_seconds_total": float(seconds.sum()),
        "observed_calls_total": int(design[:, 2].sum()),
        "observed_calls_per_second": float(design[:, 2].sum() / seconds.sum()),
        "observed_prefill_tokens_per_second": float(design[:, 0].sum() / seconds.sum()),
        "observed_decode_tokens_per_second": float(design[:, 1].sum() / seconds.sum()),
    }


def measured(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the Step 1 solo measurements of one worker or one model."""
    chunks = [chunk for run in runs for chunk in run.get("chunks", [])]
    calls = sum(int(c["n_calls"]) for c in chunks)
    generation = sum(float(c["chunk_seconds"]) for c in chunks)
    wall = sum(float(run.get("wall_seconds", 0.0)) for run in runs)
    output_tokens = sum(int(c["output_tokens_sum"]) for c in chunks)
    return {
        "n_runs": len(runs),
        "n_chunks": len(chunks),
        "n_calls": calls,
        "generation_seconds": generation,
        "wall_seconds": wall,
        "non_generation_seconds": max(0.0, wall - generation),
        "calls_per_second_generation_only": calls / generation if generation else 0.0,
        "calls_per_second_including_overhead": calls / wall if wall else 0.0,
        "mean_output_tokens_per_label": output_tokens / calls if calls else 0.0,
        "cost_model": fit_cost_model(chunks),
    }


def _condition_seconds(
    stats: dict[str, Any],
    n_calls: int,
    coefficients: dict[str, Any],
    output_tokens_per_call: float,
) -> float:
    prefill = float(stats["prompt_tokens_mean"]) * n_calls
    decode = output_tokens_per_call * n_calls
    return (
        float(coefficients["seconds_per_prefill_token"]) * prefill
        + float(coefficients["seconds_per_decode_token"]) * decode
        + float(coefficients["seconds_per_call_fixed"]) * n_calls
    )


def forecast_cell(
    preflight_report: dict[str, Any],
    measurement: dict[str, Any],
    *,
    remaining_calls: dict[str, int] | None = None,
    session_overhead_seconds: float = 0.0,
    n_sessions: int = 8,
) -> dict[str, Any]:
    """Project the remaining frozen calls of one cell from the measured cost model.

    Three output-token assumptions bracket the answer: the measured per-label rate scaled by
    the number of labels, the same rate with no scaling, and the frozen output cap. The cap is
    the hard upper bound, because no call may generate past it.
    """
    cost = measurement["cost_model"]
    if not cost.get("ok"):
        return {
            "ok": False,
            "reason": cost.get("reason", "no cost model"),
            "cell_id": preflight_report.get("cell_id"),
        }

    per_label = float(measurement["mean_output_tokens_per_label"])
    conditions = preflight_report["by_condition"]
    counts = remaining_calls or {name: int(stats["n_calls"]) for name, stats in conditions.items()}
    observed_max = float(cost["observed_prompt_tokens_per_call"]["max"])

    variants = {
        "measured_proportional": lambda labels: per_label * labels,
        "measured_flat": lambda labels: per_label,
        "frozen_cap": None,
    }
    projections: dict[str, dict[str, Any]] = {}
    for name in variants:
        by_condition: dict[str, float] = {}
        for condition, n_calls in counts.items():
            if condition not in conditions or n_calls <= 0:
                continue
            stats = conditions[condition]
            labels = LABELS_PER_CALL.get(condition, 1)
            if name == "frozen_cap":
                output_tokens = float(stats["max_tokens"])
            else:
                helper = variants[name]
                assert helper is not None
                output_tokens = helper(labels)
            by_condition[condition] = _condition_seconds(stats, n_calls, cost, output_tokens)
        total = sum(by_condition.values()) + session_overhead_seconds * n_sessions
        projections[name] = {
            "seconds_by_condition": by_condition,
            "session_overhead_seconds": session_overhead_seconds * n_sessions,
            "total_seconds": total,
            "total_a100_hours": total / 3600.0,
        }

    return {
        "ok": True,
        "cell_id": preflight_report.get("cell_id"),
        "model_id": preflight_report.get("model_id"),
        "frame_task_id": preflight_report.get("frame_task_id"),
        "remaining_calls": dict(sorted(counts.items())),
        "projections": projections,
        "extrapolation": {
            "measured_prompt_tokens_per_call_max": observed_max,
            "forecast_prompt_tokens_per_call_max": max(
                float(conditions[c]["prompt_tokens_mean"]) for c in counts if c in conditions
            ),
            "warning": (
                "Step 1 measures solo calls only. A 16-label joint call carries far more prefill "
                "tokens than any measured call, so the fitted prefill coefficient is extrapolated "
                "well beyond its observed range. Treat every joint projection as an order of "
                "magnitude, not a runtime prediction."
            ),
        },
        "is_a_performance_fact_not_a_finding": True,
    }


def remaining_ledger(protocol: dict[str, Any], *, solo_done_per_cell: int = 0) -> dict[str, Any]:
    """The frozen call ledger that Step 1 leaves for Steps 2 to 5."""
    counts = dict(protocol["execution_ledger"]["planned_call_counts"])
    n_cells = len(protocol["panel"]["cells"])
    solo_credit = solo_done_per_cell * n_cells
    return {
        "frozen_planned_calls": counts,
        "steps_2_and_3_total": int(counts["steps_2_and_3_total"]),
        "solo_calls_already_produced_in_step_1": solo_credit,
        "steps_2_and_3_remaining_after_step_1": int(counts["steps_2_and_3_total"]) - solo_credit,
        "step_4_minimum_two_cells": int(counts["step_4_minimum_two_cells"]),
        "step_4_maximum_two_cells": int(counts["step_4_maximum_two_cells"]),
        "ceilings_hours": dict(protocol["execution_ledger"]["a100_ceilings_hours"]),
        "trim_order": list(protocol["execution_ledger"]["trim_order_if_the_budget_is_exceeded"]),
        "never_trim": list(protocol["execution_ledger"]["never_trim"]),
    }


def render_report(
    stamp: str,
    per_model: dict[str, dict[str, Any]],
    forecasts: list[dict[str, Any]],
    ledger: dict[str, Any],
    meter_report: dict[str, Any],
) -> str:
    """Markdown throughput report. Performance facts only, and it says so."""
    lines = [
        "# Step 1 throughput benchmark and forecast",
        "",
        f"Freeze: `{stamp}`",
        f"Written: {dt.datetime.now(dt.UTC).isoformat()}",
        "",
        "Every number in this file is a performance fact measured on solo calls. None of it is",
        "a scientific finding. No number here may enter a claim about error covariance,",
        "variance, or interval coverage. Those endpoints belong to Steps 2 to 4.",
        "",
        "## Measured solo throughput",
        "",
        "| model | calls | generation s | calls/s generation | calls/s with overhead | "
        "output tokens/label |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, record in sorted(per_model.items()):
        lines.append(
            f"| `{model}` | {record['n_calls']} | {record['generation_seconds']:.1f} | "
            f"{record['calls_per_second_generation_only']:.3f} | "
            f"{record['calls_per_second_including_overhead']:.3f} | "
            f"{record['mean_output_tokens_per_label']:.1f} |"
        )
    lines += [
        "",
        "## Fitted cost model",
        "",
        "| model | s/prefill token | s/decode token | s/call fixed | mean abs error s | "
        "measured prompt tokens/call |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model, record in sorted(per_model.items()):
        cost = record["cost_model"]
        if not cost.get("ok"):
            lines.append(f"| `{model}` | not fitted | | | | {cost.get('reason', '')} |")
            continue
        observed = cost["observed_prompt_tokens_per_call"]
        lines.append(
            f"| `{model}` | {cost['seconds_per_prefill_token']:.3e} | "
            f"{cost['seconds_per_decode_token']:.3e} | {cost['seconds_per_call_fixed']:.4f} | "
            f"{cost['mean_absolute_error_seconds']:.3f} | "
            f"{observed['min']:.0f} to {observed['max']:.0f} |"
        )

    lines += [
        "",
        "## Forecast for the remaining frozen calls",
        "",
        "| cell | calls | measured-proportional h | measured-flat h | frozen-cap h |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in forecasts:
        if not record.get("ok"):
            lines.append(
                f"| `{record.get('cell_id')}` | | not forecast | | {record.get('reason', '')} |"
            )
            continue
        projections = record["projections"]
        total_calls = sum(record["remaining_calls"].values())
        lines.append(
            f"| `{record['cell_id']}` | {total_calls} | "
            f"{projections['measured_proportional']['total_a100_hours']:.2f} | "
            f"{projections['measured_flat']['total_a100_hours']:.2f} | "
            f"{projections['frozen_cap']['total_a100_hours']:.2f} |"
        )
    lines += [
        "",
        "The frozen-cap column is the hard upper bound: no call may generate past its cap.",
        "",
        "## Extrapolation limit",
        "",
        "Step 1 generates solo calls only, by protocol. A joint call carries about sixteen times",
        "the prefill tokens of a solo call, so the fitted prefill coefficient is applied far",
        "outside its measured range. Read every joint projection as an order of magnitude.",
        "",
        "## Metered A100 hours so far",
        "",
        f"- runs recorded: {meter_report['n_runs']}",
        f"- total metered A100 hours: {meter_report['a100_hours_total']:.3f}",
        f"- longest single worker: {meter_report['longest_single_worker_hours']:.3f} h",
        "",
        meter_report["accounting_rule"],
        "",
        "## Frozen call ledger that remains",
        "",
        f"- Steps 2 and 3 planned calls: {ledger['steps_2_and_3_total']}",
        "- solo calls already produced in Step 1: "
        f"{ledger['solo_calls_already_produced_in_step_1']}",
        f"- Steps 2 and 3 remaining: {ledger['steps_2_and_3_remaining_after_step_1']}",
        f"- Step 4 minimum across the two frozen cells: {ledger['step_4_minimum_two_cells']}",
        f"- Step 4 maximum across the two frozen cells: {ledger['step_4_maximum_two_cells']}",
        "",
        "Trim order if the projection exceeds the budget:",
        "",
    ]
    lines += [f"{i}. {text}" for i, text in enumerate(ledger["trim_order"], start=1)]
    lines += ["", "Never trimmed:", ""] + [f"- {text}" for text in ledger["never_trim"]] + [""]
    return "\n".join(lines)
