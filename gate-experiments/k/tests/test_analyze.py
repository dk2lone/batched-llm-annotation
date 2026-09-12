"""Gate verdict logic, exercised on constructed result records."""

from __future__ import annotations

from typing import Any

from gatek.analyze import evaluate_gate, gate_sensitivity, render_report
from gatek.config import GateConfig

GATE = GateConfig(
    high_dims=[32],
    min_repr_contrast_ratio=1.2,
    min_sparse_minus_dense_accuracy=0.05,
    min_sparse_minus_dense_ablation_delta=0.05,
)


def pair(acc_delta: float, ratio: float) -> dict[int, dict[str, Any]]:
    return {
        2: {
            "accuracy_a": 0.9,
            "parse_failure_rate_a": 0.0,
            "accuracy_delta": 0.0,
            "repr_contrast_ratio_a_over_b": 1.0,
        },
        32: {"accuracy_delta": acc_delta, "repr_contrast_ratio_a_over_b": ratio},
    }


def ablation(diff: float) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        "sparse_zero_pad": {32: {"drop_difference": diff}},
        "dense_redundant_projection": {32: {"drop_difference": 0.0}},
    }


def test_construction_failure_wins_over_everything() -> None:
    out = evaluate_gate(
        pair(0.3, 2.0), pair(0.3, 2.0), ablation(0.3), False, ["nuisance_raw_contrast"], GATE
    )
    assert out["verdict"] == "construction_failure"
    assert "nuisance_raw_contrast" in out["reason"]


def test_all_conditions_met_gives_support() -> None:
    out = evaluate_gate(pair(0.20, 2.0), pair(0.20, 2.0), ablation(0.20), True, [], GATE)
    assert out["verdict"] == "mechanism_supported_enough_to_expand"


def test_contrast_without_accuracy_or_ablation_is_unsupported() -> None:
    out = evaluate_gate(pair(0.00, 2.0), pair(0.00, 2.0), ablation(0.00), True, [], GATE)
    assert out["verdict"] == "mechanism_unsupported"


def test_no_contrast_and_no_accuracy_gain_is_unsupported() -> None:
    out = evaluate_gate(pair(0.00, 1.0), pair(0.00, 1.0), ablation(0.00), True, [], GATE)
    assert out["verdict"] == "mechanism_unsupported"


def test_mixed_pattern_is_ambiguous() -> None:
    out = evaluate_gate(pair(0.20, 2.0), pair(0.00, 1.0), ablation(0.00), True, [], GATE)
    assert out["verdict"] == "ambiguous"


def test_missing_cells_do_not_crash_the_gate() -> None:
    out = evaluate_gate({}, {}, {}, True, [], GATE)
    assert out["verdict"] == "model_capability_failure"
    assert out["conditions"]["accuracy_sparse_minus_dense"]["value"] is None


def test_sensitivity_reports_both_directions() -> None:
    rows = gate_sensitivity(pair(0.06, 1.25), pair(0.06, 1.25), ablation(0.06), True, [], GATE)
    assert {r["thresholds"] for r in rows} == {"loosened", "tightened"}
    assert rows[0]["verdict"] == "mechanism_supported_enough_to_expand"
    assert rows[1]["verdict"] != "mechanism_supported_enough_to_expand"


def test_report_names_the_verdict_and_the_mock_warning() -> None:
    analysis = {
        "created_utc": "now",
        "backend": ["mock"],
        "model_id": ["mock"],
        "n_results": 4,
        "repr_distance_primary": "cosine",
        "gate": evaluate_gate(pair(0.2, 2.0), pair(0.2, 2.0), ablation(0.2), True, [], GATE),
        "gate_sensitivity": gate_sensitivity(
            pair(0.2, 2.0), pair(0.2, 2.0), ablation(0.2), True, [], GATE
        ),
        "construction": {"ok": True, "failed_invariants": []},
        "paired_sparse_vs_dense": {},
        "paired_sparse_vs_nuisance": {},
        "ablation": {},
        "ordering_control": {},
        "cells": {},
        "attention": None,
    }
    text = render_report(analysis)
    assert "mechanism_supported_enough_to_expand" in text
    assert "not evidence" in text
