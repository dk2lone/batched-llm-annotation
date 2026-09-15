"""The A8 budget release must use only sealed technical timing evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "seal_extension_full_grid_budget.py"
SPEC = importlib.util.spec_from_file_location("seal_extension_full_grid_budget", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


def test_forecast_covers_fixed_grid_and_passes_guard() -> None:
    payload = budget.build()
    assert payload["technical_only"] is True
    assert payload["scientific_outputs_inspected"] is False
    assert payload["gpu_work_launched"] is False
    assert len(payload["cell_forecasts"]) == 16
    assert payload["target"]["items_per_cell"] == 8192
    assert payload["target"]["total_calls"] == 548864
    assert payload["budget_gate"]["base_forecast_a100_hours"] == pytest.approx(27.301587972486917)
    assert payload["budget_gate"]["passed"] is True
    assert payload["budget_gate"]["projected_total_including_reserve_a100_hours"] < 80


def test_release_stops_at_first_1000_attempts_per_model() -> None:
    release = budget.build()["release"]
    assert release["budget_gate_passed"] is True
    assert release["paid_run_authorized_by_user"] is False
    assert release["released_attempts"] == 4000
    assert "no remaining attempt" in release["released_scope"]


def test_gate_blocks_a_forecast_over_the_stop() -> None:
    gate = budget.budget_gate(3.0, 50.0, multiplier=1.6, reserve_hours=10.0, stop_hours=80.0)
    assert gate["passed"] is False
    assert gate["headroom_after_guard_and_reserve_a100_hours"] == pytest.approx(-13.0)


def test_runtime_case_needs_two_measurements() -> None:
    evidence = {
        "events": [{"phase": "measure", "status": "ok", "case_id": "x", "latency_seconds": 1.0}]
    }
    with pytest.raises(ValueError, match="two positive measurements"):
        budget.median_measurement(evidence, "x")
