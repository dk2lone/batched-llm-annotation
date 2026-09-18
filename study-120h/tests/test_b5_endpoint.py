"""Frozen endpoint pilot schedule and decision checks."""

from __future__ import annotations

import numpy as np

from freezeb import b5_endpoint as pilot
from freezeb import b5_endpoint_analysis as analysis


def test_schedule_counts_and_nesting() -> None:
    assert pilot.expected_counts() == {
        "joint_per_cell": 4352,
        "exact_context_per_cell": 16384,
        "total_per_cell": 20736,
        "cells": 4,
        "joint": 17408,
        "exact_context": 65536,
        "total": 82944,
    }
    pilot.verify_counts(pilot.expected_counts())
    for cell in pilot.CELLS:
        permutation = pilot.canonical_permutation(cell, 1)
        pairs = pilot.group_positions(permutation, 2)
        supergroups = pilot.group_positions(permutation, 32)
        assert len(pairs) == 4096 and len(supergroups) == 256
        assert all(
            group == tuple(item for pair in pairs[16 * i : 16 * i + 16] for item in pair)
            for i, group in enumerate(supergroups)
        )
        assert sorted(permutation.tolist()) == list(range(8192))
        assert pilot.nesting_report(cell)["nested_pairs_checked"] == 256


def test_unique_call_ids_and_schema() -> None:
    ids = [f"doc-{i}" for i in range(8192)]
    for cell in pilot.CELLS:
        calls = list(pilot.cell_calls(cell, ids))
        assert len(calls) == 20736
        assert len({call["call_id"] for call in calls}) == len(calls)
        for call in (calls[0], calls[1], calls[-1]):
            schema = call["json_schema"]
            assert schema["additionalProperties"] is False
            assert schema["properties"]["labels"]["additionalProperties"] is False
            assert set(schema["properties"]["labels"]["required"]) == set(call["target_ids"])


def test_decision_rule() -> None:
    cells = [
        {"by_size": [{"standardized_contrast": 0.02}, {"standardized_contrast": 0.03}]}
        for _ in range(4)
    ]
    intervals = {
        "2": {"low": 0.01, "high": 0.03},
        "32": {"low": 0.02, "high": 0.04},
        "delta": {"low": 0.001, "high": 0.02},
    }
    assert analysis.decide(0.02, 0.03, 0.01, intervals, [0.01] * 4, cells)["verdict"] == "GO"
    null = {key: {"low": -0.005, "high": 0.005} for key in intervals}
    assert analysis.decide(0, 0, 0, null, [0] * 4, cells)["verdict"] == "NO_GO"
    wide = {key: {"low": -0.03, "high": 0.03} for key in intervals}
    assert analysis.decide(0, 0, 0, wide, [0] * 4, cells)["verdict"] == "INCONCLUSIVE"


def test_bootstrap_preserves_supergroups() -> None:
    assert analysis.SUPER_SIZE == 32
    blocks = np.arange(8192).reshape(256, 32)
    picks = np.array([3, 1, 3])
    assert blocks[picks].reshape(-1)[:32].tolist() == list(range(96, 128))
