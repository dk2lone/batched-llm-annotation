"""Deterministic Freeze B forecasts from the completed synthetic benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .runtime_benchmark import FORECAST_MULTIPLIERS

TASKS = (
    "civil_comments",
    "ag_news",
    "multi_nli",
    "imdb",
    "go_emotions",
    "multinerd",
)
ITEM_COUNTS = (2048, 4096, 8192)
HISTORICAL_STEP1_A100_HOURS = 0.190
TARGET_A100_HOURS = 120.0
STOP_A100_HOURS = 150.0
OPERATING_RESERVE_A100_HOURS = 10.0


def _forecast(base_hours: float) -> dict[str, float]:
    return {f"{m:g}x": base_hours * m for m in FORECAST_MULTIPLIERS}


def _event_groups(worker: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in worker["events"]:
        groups.setdefault(str(event["case_id"]), []).append(event)
    return groups


def _rate(groups: dict[str, list[dict[str, Any]]], case_id: str, factor: float) -> float | None:
    rows = [
        row
        for row in groups.get(case_id, [])
        if row.get("phase") == "measure" and row.get("status") == "ok"
    ]
    if len(rows) != 2:
        return None
    return max(float(row["latency_seconds"]) for row in rows) * factor


def _interpolate(one: float | None, sixteen: float | None, k: int) -> float | None:
    if one is None or sixteen is None:
        return None
    return one + (sixteen - one) * (k - 1) / 15


def _required_rate(value: float | None) -> float:
    if value is None:
        raise ValueError("required runtime rate is missing")
    return value


def _block(
    block_id: str,
    stage: str,
    rates: dict[str, float | None],
    counts: dict[str, int],
    load_hours: float,
    **metadata: Any,
) -> dict[str, Any]:
    missing = sorted(key for key, count in counts.items() if count and rates.get(key) is None)
    result: dict[str, Any] = {
        "block_id": block_id,
        "stage": stage,
        "call_counts": dict(sorted(counts.items())),
        "load_a100_hours": load_hours,
        **metadata,
    }
    if missing:
        result.update({"status": "unforecastable", "missing_rates": missing})
        return result
    call_seconds = sum(_required_rate(rates[key]) * count for key, count in counts.items())
    base = load_hours + call_seconds / 3600
    result.update(
        {
            "status": "forecastable",
            "base_a100_hours": base,
            "forecasts_a100_hours": _forecast(base),
        }
    )
    return result


def _range(rows: Iterable[dict[str, Any]], count: int) -> dict[str, Any]:
    available = [row for row in rows if row["status"] == "forecastable"]
    if len(available) < count:
        return {"status": "unforecastable", "needed": count, "available": len(available)}
    ordered = sorted(available, key=lambda row: float(row["base_a100_hours"]))
    low = sum(float(row["base_a100_hours"]) for row in ordered[:count])
    high = sum(float(row["base_a100_hours"]) for row in ordered[-count:])
    return {
        "status": "conditional_range",
        "cells": count,
        "base_a100_hours": {"low": low, "high": high},
        "forecasts_a100_hours": {
            f"{m:g}x": {"low": low * m, "high": high * m} for m in FORECAST_MULTIPLIERS
        },
        "note": (
            "Range over forecastable candidate cells; admission and item counts are not known yet."
        ),
    }


def build_ledger(
    report: dict[str, Any],
    workers: list[dict[str, Any]],
    plan: dict[str, Any],
    observations: dict[str, Any],
) -> dict[str, Any]:
    """Validate the run and build cell-level and conditional queue forecasts."""
    if not report.get("synthetic_only") or report["run_id"] != observations.get(
        "run_id", report["run_id"]
    ):
        raise ValueError("benchmark identity or synthetic-only status is invalid")
    if report["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("benchmark plan hash does not match report")
    if len(workers) != 6 or len({row["model_id"] for row in workers}) != 6:
        raise ValueError("six unique completed model workers are required")

    billing = observations["billing"]
    hourly_price = float(observations["a100_40gb_hourly_usd"])
    billed_a100_hours = float(billing["a100_40gb_usd"]) / hourly_price
    proxy_hours = sum(float(worker["metered_proxy_seconds"]) for worker in workers) / 3600
    meter_factor = billed_a100_hours / proxy_hours
    all_in_per_a100_hour = float(billing["total_usd"]) / billed_a100_hours

    rate_rows: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    per_cell: dict[tuple[str, str], dict[str, float | None]] = {}
    load_hours_by_model: dict[str, float] = {}
    context_exclusions = 0
    successful_measures = 0

    for worker in workers:
        model = str(worker["model_id"])
        groups = _event_groups(worker)
        if len(groups) != 30:
            raise ValueError(f"{model} does not account for all 30 benchmark cases")
        ok_seconds = sum(
            float(event["latency_seconds"])
            for event in worker["events"]
            if event.get("status") == "ok"
        )
        load_seconds = max(0.0, float(worker["metered_proxy_seconds"]) - ok_seconds)
        load_hours = load_seconds * meter_factor / 3600
        load_hours_by_model[model] = load_hours
        for task in TASKS:
            one = _rate(groups, f"{task}::short::joint", meter_factor)
            joint16 = _rate(groups, f"{task}::size16::joint", meter_factor)
            exact16 = _rate(groups, f"{task}::size16::exact_context", meter_factor)
            joint32 = _rate(groups, f"{task}::size32::joint", meter_factor)
            exact32 = _rate(groups, f"{task}::size32::exact_context", meter_factor)
            rates: dict[str, float | None] = {
                "solo": one,
                "joint_k1": one,
                "exact_k1": one,
                "joint_k2": _interpolate(one, joint16, 2),
                "joint_k4": _interpolate(one, joint16, 4),
                "joint_k8": _interpolate(one, joint16, 8),
                "joint_k16": joint16,
                "joint_k32": joint32,
                "exact_k2": _interpolate(one, exact16, 2),
                "exact_k4": _interpolate(one, exact16, 4),
                "exact_k8": _interpolate(one, exact16, 8),
                "exact_k16": exact16,
                "exact_k32": exact32,
            }
            per_cell[(model, task)] = rates
            excluded = sum(
                1
                for rows in groups.values()
                for event in rows
                if event.get("status") == "context_excluded"
            )
            successful = sum(
                1
                for rows in groups.values()
                for event in rows
                if event.get("phase") == "measure" and event.get("status") == "ok"
            )
            # Count each worker once, outside the task-specific rate row.
            if task == TASKS[0]:
                context_exclusions += excluded
                successful_measures += successful
            rate_rows.append(
                {
                    "model_id": model,
                    "task_id": task,
                    "seconds_per_call": rates,
                    "size16_forecastable": joint16 is not None and exact16 is not None,
                    "size32_forecastable": joint32 is not None and exact32 is not None,
                }
            )
            blocks.append(
                _block(
                    f"screen::{model}::{task}",
                    "solo_screen_cell",
                    rates,
                    {"solo": 2048},
                    load_hours / len(TASKS),
                    model_id=model,
                    task_id=task,
                    item_count=2048,
                    note="One-sixth of one model load allocated to each task in the model bundle.",
                )
            )
            for n_items in ITEM_COUNTS:
                e2_counts = {
                    "joint_k16": 3 * n_items // 16,
                    "exact_k16": 3 * n_items,
                    "solo": n_items,
                }
                blocks.append(
                    _block(
                        f"e2::{model}::{task}::n{n_items}",
                        "e2_cell",
                        rates,
                        e2_counts,
                        load_hours,
                        model_id=model,
                        task_id=task,
                        item_count=n_items,
                    )
                )
                e3_counts: dict[str, int] = {}
                for k in (2, 4, 8, 32):
                    e3_counts[f"joint_k{k}"] = 3 * n_items // k
                    e3_counts[f"exact_k{k}"] = 3 * n_items
                # Two additional size-16 variants, each with joint and exact arms.
                e3_counts["joint_k16"] = 2 * 3 * n_items // 16
                e3_counts["exact_k16"] = 2 * 3 * n_items
                blocks.append(
                    _block(
                        f"e3::{model}::{task}::n{n_items}",
                        "e3_incremental_cell",
                        rates,
                        e3_counts,
                        load_hours,
                        model_id=model,
                        task_id=task,
                        item_count=n_items,
                        note=(
                            "Excludes reusable E2 size-16 baseline; k=2/4/8 rates "
                            "are linear interpolations."
                        ),
                    )
                )
            blocks.append(
                _block(
                    f"e4::{model}::{task}",
                    "e4_cell",
                    rates,
                    {"exact_k16": 480},
                    load_hours,
                    model_id=model,
                    task_id=task,
                    note=(
                        "Three assignments x (32 mixed calls x four rotations + "
                        "32 homogeneous calls)."
                    ),
                )
            )
            for label, calls in (
                ("through400", (250 + 400) * 32),
                ("400to1000", (1000 - 400) * 32),
                ("1000to2500", (2500 - 1000) * 32),
                ("2500to5000", (5000 - 2500) * 32),
                ("capacity1000", 1000 * 32),
                ("capacity250", 250 * 32),
            ):
                blocks.append(
                    _block(
                        f"e5::{model}::{task}::{label}",
                        f"e5_{label}",
                        rates,
                        {"joint_k16": calls},
                        load_hours,
                        model_id=model,
                        task_id=task,
                        fresh_joint_calls=calls,
                    )
                )

    # Model bundles use one load per model, not six allocated loads rounded independently.
    screen_base = sum(
        load_hours_by_model[model]
        + sum(_required_rate(per_cell[(model, task)]["solo"]) * 2048 for task in TASKS) / 3600
        for model in load_hours_by_model
    )
    qwen_civil = per_cell[("Qwen/Qwen3-8B", "civil_comments")]
    e1 = _block(
        "e1::qwen3-8b::civil_comments",
        "historical_replication_anchor",
        qwen_civil,
        {"joint_k16": 384, "exact_k16": 12_288},
        load_hours_by_model["Qwen/Qwen3-8B"],
        model_id="Qwen/Qwen3-8B",
        task_id="civil_comments",
        fresh_calls=12_672,
        note="384 joint + 6,144 exact + 6,144 filler-as-exact; 2,048 solo calls already exist.",
    )

    def stage_rows(stage: str, n_items: int | None = None) -> list[dict[str, Any]]:
        return [
            row
            for row in blocks
            if row["stage"] == stage and (n_items is None or row.get("item_count") == n_items)
        ]

    queue_envelopes: dict[str, Any] = {
        "all_36_solo_screen_cells": {
            "status": "forecastable",
            "cells": 36,
            "base_a100_hours": screen_base,
            "forecasts_a100_hours": _forecast(screen_base),
        },
        "historical_replication_anchor": e1,
        "e4_eight_cells": _range(stage_rows("e4_cell"), 8),
        "e5_first_two_through400": _range(stage_rows("e5_through400"), 2),
        "e5_six_cells_through5000": _range(
            [
                {
                    **row,
                    "base_a100_hours": sum(
                        float(
                            next(
                                x
                                for x in blocks
                                if x["block_id"]
                                == f"e5::{row['model_id']}::{row['task_id']}::{suffix}"
                            )["base_a100_hours"]
                        )
                        for suffix in ("through400", "400to1000", "1000to2500", "2500to5000")
                    ),
                }
                for row in stage_rows("e5_through400")
                if all(
                    next(
                        x
                        for x in blocks
                        if x["block_id"] == f"e5::{row['model_id']}::{row['task_id']}::{suffix}"
                    )["status"]
                    == "forecastable"
                    for suffix in ("through400", "400to1000", "1000to2500", "2500to5000")
                )
            ],
            6,
        ),
    }
    for n_items in ITEM_COUNTS:
        queue_envelopes[f"e2_primary_16_n{n_items}"] = _range(stage_rows("e2_cell", n_items), 16)
        queue_envelopes[f"e2_heldout_12_n{n_items}"] = _range(stage_rows("e2_cell", n_items), 12)
        queue_envelopes[f"e3_six_n{n_items}"] = _range(
            stage_rows("e3_incremental_cell", n_items), 6
        )

    current_benchmark_hours = billed_a100_hours
    cumulative = HISTORICAL_STEP1_A100_HOURS + current_benchmark_hours
    pre_run_budget = float(observations["user_stated_pre_run_account_budget_usd"])
    remaining_budget = max(0.0, pre_run_budget - float(billing["total_usd"]))
    aggregate_capacity = observations["user_stated_aggregate_a100_capacity_hours"]
    aggregate_minimum = float(aggregate_capacity["minimum"])
    aggregate_maximum = float(aggregate_capacity["maximum"])
    return {
        "schema_version": "freezeb-runtime-ledger-v1",
        "status": "sealed_with_declared_instrumentation_gap",
        "benchmark_run_id": report["run_id"],
        "benchmark_plan_sha256": report["plan_sha256"],
        "synthetic_only": True,
        "measurement_summary": {
            "models": len(workers),
            "planned_model_cases": len(workers) * 30,
            "successful_measured_requests": successful_measures,
            "failed_measured_requests": 0,
            "context_excluded_cases": context_exclusions,
            "proxy_a100_hours": proxy_hours,
            "billed_a100_hours": billed_a100_hours,
            "meter_calibration_factor": meter_factor,
            "billed_total_usd": float(billing["total_usd"]),
            "all_in_usd_per_billed_a100_hour": all_in_per_a100_hour,
        },
        "forecast_policy": {
            "base": "maximum of two measured request wall times, calibrated to billed A100 time",
            "multipliers": list(FORECAST_MULTIPLIERS),
            "k2_k4_k8": "linear interpolation in batch size between measured k=1 and k=16",
            "load_overhead": "observed worker time not attributed to request events",
            "real_call_reforecast": "mandatory after first 1,000 calls per model policy",
            "instrumentation_gap": (
                "prefill and decode were not timed separately; do not claim either component"
            ),
        },
        "rate_rows": rate_rows,
        "blocks": [e1, *blocks],
        "queue_envelopes": queue_envelopes,
        "budget": {
            "historical_step1_a100_hours": HISTORICAL_STEP1_A100_HOURS,
            "benchmark_billed_a100_hours": current_benchmark_hours,
            "cumulative_a100_hours_after_benchmark": cumulative,
            "target_a100_hours": TARGET_A100_HOURS,
            "stop_a100_hours": STOP_A100_HOURS,
            "operating_reserve_a100_hours": OPERATING_RESERVE_A100_HOURS,
            "estimated_all_in_usd_at_120h_current_rate": TARGET_A100_HOURS * all_in_per_a100_hour,
            "estimated_all_in_usd_at_150h_current_rate": STOP_A100_HOURS * all_in_per_a100_hour,
            "user_stated_pre_run_budget_usd": pre_run_budget,
            "estimated_remaining_after_benchmark_usd": remaining_budget,
            "estimated_remaining_a100_hours_current_rate": remaining_budget / all_in_per_a100_hour,
            "ten_hour_reserve_usd_current_rate": OPERATING_RESERVE_A100_HOURS
            * all_in_per_a100_hour,
            "current_account_can_support_full_plan": False,
            "aggregate_capacity_a100_hours": {
                "minimum": aggregate_minimum,
                "maximum": aggregate_maximum,
                "distributed_across_multiple_accounts": bool(
                    aggregate_capacity["distributed_across_multiple_accounts"]
                ),
            },
            "aggregate_capacity_can_support_120h_target": aggregate_minimum >= TARGET_A100_HOURS,
            "aggregate_capacity_reaches_150h_stop": aggregate_maximum >= STOP_A100_HOURS,
        },
        "gpu_memory": observations["gpu_memory_from_vllm_startup_logs"],
    }


def render_markdown(ledger: dict[str, Any]) -> str:
    """Render a concise human-readable companion to the machine ledger."""
    m = ledger["measurement_summary"]
    b = ledger["budget"]
    q = ledger["queue_envelopes"]
    lines = [
        "# Freeze B runtime and compute ledger",
        "",
        f"Benchmark run: `{ledger['benchmark_run_id']}`. Status: **{ledger['status']}**.",
        "Synthetic outputs are runtime evidence only and never enter scientific analyses.",
        "",
        "## Measured benchmark",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| model workers | {m['models']} |",
        f"| planned model/case checks | {m['planned_model_cases']} |",
        f"| successful measured requests | {m['successful_measured_requests']} |",
        f"| failed measured requests | {m['failed_measured_requests']} |",
        f"| context-excluded model/cases | {m['context_excluded_cases']} |",
        f"| billed A100 hours | {m['billed_a100_hours']:.3f} |",
        f"| total benchmark cost | ${m['billed_total_usd']:.2f} |",
        "",
        "## Immediate queue forecast",
        "",
        "| block | base h | 1.6x h | 3x h |",
        "|---|---:|---:|---:|",
    ]
    for key in ("all_36_solo_screen_cells", "historical_replication_anchor"):
        row = q[key]
        lines.append(
            f"| `{key}` | {row['base_a100_hours']:.2f} | "
            f"{row['forecasts_a100_hours']['1.6x']:.2f} | "
            f"{row['forecasts_a100_hours']['3x']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The screen estimate covers 36 runnable model-task candidates and one "
            "model load per six-task bundle.",
            "The anchor estimate covers the 12,672 still-unrun Qwen3-8B/Civil Comments calls.",
            "",
            "## Conditional panel ranges",
            "",
            "Admission and solo-only item-count selection have not happened. These "
            "are low/high ranges over",
            "currently forecastable candidate cells, not promises about the final selected panel.",
            "",
            "| block | base h | 1.6x h | 3x h |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in (
        "e2_primary_16_n2048",
        "e2_primary_16_n4096",
        "e2_primary_16_n8192",
        "e5_first_two_through400",
        "e5_six_cells_through5000",
        "e4_eight_cells",
    ):
        row = q[key]
        if row["status"] != "conditional_range":
            lines.append(f"| `{key}` | unavailable | unavailable | unavailable |")
            continue
        base = row["base_a100_hours"]
        one6 = row["forecasts_a100_hours"]["1.6x"]
        three = row["forecasts_a100_hours"]["3x"]
        lines.append(
            f"| `{key}` | {base['low']:.2f}–{base['high']:.2f} | "
            f"{one6['low']:.2f}–{one6['high']:.2f} | {three['low']:.2f}–{three['high']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Account decision",
            "",
            (
                f"At this run's all-in rate, 120 A100-hours is about "
                f"**${b['estimated_all_in_usd_at_120h_current_rate']:.0f}** and "
                "150 hours is about "
                f"**${b['estimated_all_in_usd_at_150h_current_rate']:.0f}**."
            ),
            (
                "If the stated $14 was available before this benchmark, about "
                f"**${b['estimated_remaining_after_benchmark_usd']:.2f}** remains, "
                "equivalent to roughly "
                f"**{b['estimated_remaining_a100_hours_current_rate']:.2f} "
                "A100-hours** at the observed all-in rate."
            ),
            (
                "The required 10-hour operating reserve alone is about "
                f"**${b['ten_hour_reserve_usd_current_rate']:.2f}** at that rate."
            ),
            (
                "That single account cannot fund the study by itself. This is not "
                "the overall funding decision."
            ),
            (
                "Across the user's accounts, declared capacity is "
                f"**{b['aggregate_capacity_a100_hours']['minimum']:.0f}–"
                f"{b['aggregate_capacity_a100_hours']['maximum']:.0f} A100-hours**. "
                "That covers the 120-hour target and reaches the 150-hour hard stop."
            ),
            "",
            "## Limits and next action",
            "",
            (
                "The base rate uses the slower of two measured repetitions and "
                "reconciles request time to Modal's billed A100 time. The 1.6x case "
                "is the start-decision forecast; 3x is the incident case. Batch "
                "sizes 2, 4, and 8 are interpolated because only 1, 16, and 32 were "
                "measured."
            ),
            "",
            (
                "The benchmark did not separately capture time-to-first-token/prefill "
                "and decode time. That gap is sealed here rather than reconstructed. "
                "Capture those components in the mandatory first-1,000-real-calls "
                "reforecast, and do not describe this ledger as a component-level "
                "timing measurement."
            ),
            "",
            (
                "Next: run the solo-only screens within the declared aggregate "
                "capacity. The admission lock will replace conditional ranges with "
                "the exact selected cells and item counts before any joint scientific "
                "output."
            ),
            "",
        ]
    )
    return "\n".join(lines)
