"""CPU-only tests for benchmark-derived Freeze B runtime forecasts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from freezeb.runtime_ledger import build_ledger, render_markdown

STUDY = Path(__file__).resolve().parents[1]
RUN = STUDY / "freezeb-runtime/20260912T233301Z-5738ea84"


def _inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return tuple(
        json.loads((RUN / name).read_text())
        for name in ("report.json", "workers.json", "plan.json", "runtime-observations.json")
    )


def test_ledger_accounts_for_the_complete_benchmark() -> None:
    ledger = build_ledger(*_inputs())
    measured = ledger["measurement_summary"]
    assert measured["planned_model_cases"] == 180
    assert measured["successful_measured_requests"] == 316
    assert measured["failed_measured_requests"] == 0
    assert measured["context_excluded_cases"] == 22
    assert measured["billed_total_usd"] == pytest.approx(2.01044056)


def test_forecasts_have_frozen_multipliers_and_missing_context_stays_missing() -> None:
    ledger = build_ledger(*_inputs())
    screen = ledger["queue_envelopes"]["all_36_solo_screen_cells"]
    assert screen["forecasts_a100_hours"]["1.6x"] == pytest.approx(screen["base_a100_hours"] * 1.6)
    assert screen["forecasts_a100_hours"]["3x"] == pytest.approx(screen["base_a100_hours"] * 3)
    olmo_imdb = next(
        row
        for row in ledger["blocks"]
        if row["block_id"] == "e2::allenai/OLMo-2-1124-7B-Instruct::imdb::n2048"
    )
    assert olmo_imdb["status"] == "unforecastable"
    assert set(olmo_imdb["missing_rates"]) == {"exact_k16", "joint_k16"}


def test_account_decision_and_markdown_are_explicit() -> None:
    ledger = build_ledger(*_inputs())
    budget = ledger["budget"]
    assert budget["current_account_can_support_full_plan"] is False
    assert budget["aggregate_capacity_can_support_120h_target"] is True
    assert budget["aggregate_capacity_reaches_150h_stop"] is True
    assert budget["estimated_remaining_after_benchmark_usd"] == pytest.approx(11.98955944)
    rendered = render_markdown(ledger)
    assert "declared capacity is **120–150 A100-hours**" in rendered
    assert "prefill and decode" in rendered
