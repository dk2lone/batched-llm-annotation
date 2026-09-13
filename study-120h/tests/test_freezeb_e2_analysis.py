"""CPU-only checks for the first-cell Freeze B E2 analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from freezeb.e2_analysis import analyze, render_report

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "e2-out" / "qwen3-8b--multi_nli"


def test_completed_cell_validates_with_reduced_replay() -> None:
    result = analyze(
        CELL / "events.jsonl",
        ROOT / "freeze-b-work" / "20260912-cpu-draft",
        ROOT / "freeze" / "20260912T222618Z-B1",
        ROOT / "freeze" / "20260913T022442Z-B2" / "ADMISSION_LOCK.json",
        CELL / "plan.json",
        CELL / "SUMMARY.json",
        n_draws=7,
    )
    assert result["source"]["n_first_calls"] == 17_152
    assert result["source"]["n_retry_calls"] == 0
    assert result["data_quality"]["unresolved_target_rows"] == {
        "joint": 0,
        "exact_context": 3,
    }
    assert result["accuracy"]["solo"]["n_unresolved"] == 5
    assert result["primary_sensitivity"]["cell_mean"][
        "delta_unresolved_as_correct"
    ] == pytest.approx(0.007959270477294923)
    assert result["covariance"]["exact_context"]["mean_covariance"] == pytest.approx(
        0.00010837713877360029
    )
    assert result["panel_status"]["claim_available"] is False
    assert result["coverage_status"]["measured"] is False
    assert "No confidence-interval coverage was measured" in render_report(result)


def test_analysis_refuses_zero_replays() -> None:
    with pytest.raises(ValueError, match="positive"):
        analyze(
            Path("missing"),
            Path("missing"),
            Path("missing"),
            Path("missing"),
            Path("missing"),
            Path("missing"),
            n_draws=0,
        )
