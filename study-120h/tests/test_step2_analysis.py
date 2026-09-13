"""CPU-only checks for the historical bridge analyzer."""

from __future__ import annotations

from pathlib import Path

import pytest

from step2.analysis import analyze, render_report

ROOT = Path(__file__).resolve().parents[1]


def test_completed_bridge_passes_with_a_reduced_replay() -> None:
    result = analyze(
        ROOT / "step2-out" / "bridge" / "calls.jsonl",
        ROOT / "freeze" / "20260912T041633Z" / "data" / "civil_comments" / "items.jsonl",
        ROOT / "step1-out" / "results" / "qwen_civil__civil_comments__Qwen_Qwen3-8B" / "solo.jsonl",
        n_draws=99,
    )
    assert result["source"]["target_rows"] == 18_432
    assert result["source"]["unique_first_calls"] == 12_672
    assert result["primary_sensitivity"]["cell_mean"][
        "delta_unresolved_as_correct"
    ] == pytest.approx(0.005451382531060113)
    assert result["fake_batch_falsification"]["observed_fake_batch_covariance"] == pytest.approx(
        -0.0004205067952473959
    )
    assert result["verdict"]["label"] == "REPLICATION_SUPPORTED"
    assert "No confidence-interval coverage result" in render_report(result)


def test_analysis_refuses_zero_replays() -> None:
    with pytest.raises(ValueError, match="positive"):
        analyze(Path("missing"), Path("missing"), Path("missing"), n_draws=0)
