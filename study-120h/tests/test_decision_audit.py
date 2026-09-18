from __future__ import annotations

import numpy as np
import pytest

from freezeb import assign
from freezeb import extension as ext
from freezeb.decision_audit import (
    _dependence,
    _homogeneity,
    clean_replicates,
    coverage_variant,
    replay_from_observed,
)


def test_repair_classification_excludes_whole_replicate() -> None:
    planned = [
        {"call_id": f"r{replicate}:c{call}", "replicate": replicate}
        for replicate in range(3)
        for call in range(32)
    ]
    keep = clean_replicates(planned, {"r1:c17": {}})
    assert keep.tolist() == [0, 2]
    with pytest.raises(ValueError, match="foreign call"):
        clean_replicates(planned, {"unplanned": {}})


def test_standardized_dependence_uses_binary_marginal() -> None:
    vector = np.asarray([0, 0, 1, 1], dtype=float)
    matrix = np.asarray([[0, 2], [1, 3]])
    result = _dependence(vector, matrix)
    assert result["marginal_error"] == 0.5
    assert result["covariance"] == 0.25
    assert result["standardized"] == 1.0
    assert result["within_group_error_homogeneity"] == 1.0
    assert _dependence(np.zeros(4), matrix)["standardized"] is None


def test_cached_replay_matches_sealed_assignment_code() -> None:
    frame = assign.Frame(tuple(str(index) for index in range(256)), tuple(range(256)))
    observed = assign.observed_stat_matrices(
        frame, "test::cell", 16, namespace=ext.ASSIGN_NAMESPACE
    )
    expected = assign.replay_stat_matrices(
        frame, "test::cell", 16, 7, namespace=ext.ASSIGN_NAMESPACE
    )
    actual = replay_from_observed(observed, "test::cell", 7)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_post_hoc_regeneration_is_deterministic() -> None:
    blocks = np.tile(np.asarray([[0] * 8 + [1] * 8] * 32, dtype=float), (12, 1, 1))
    first = coverage_variant(blocks, 0.5, 0.01, "test-cell")
    assert first == coverage_variant(blocks, 0.5, 0.01, "test-cell")
    original = [{"maximum_label_share": 0.5}] * 30
    repaired = [{"maximum_label_share": 0.75}] * 12
    assert _homogeneity(original, repaired, "test-seed") == _homogeneity(
        original, repaired, "test-seed"
    )
