"""CPU checks for the prospectively sealed X1R4 A9 analysis."""

from __future__ import annotations

import numpy as np
import pytest

from freezeb.extension_b3_analysis import (
    _accuracy,
    _final_events,
    _groups,
    _sensitivities,
)
from step0.hashing import sha256_json


def _expected(call_id: str) -> dict[str, dict[str, object]]:
    return {
        call_id: {
            "call_id": call_id,
            "cell_key": "model::task",
            "doc_order_sha256": sha256_json([]),
            "target_ids_sha256": sha256_json([]),
        }
    }


def _attempt(call_id: str, attempt: int, status: str) -> dict[str, object]:
    return {
        "call_id": call_id,
        "attempt_id": f"{call_id}::attempt::{attempt}",
        "attempt": attempt,
        "status": status,
        "cell_key": "model::task",
        "model_id": "model",
        "task_id": "task",
        "doc_order": [],
        "target_ids": [],
    }


def test_retry_replaces_a_retryable_first_attempt() -> None:
    call_id = "cell::call"
    expected = _expected(call_id)
    first = _attempt(call_id, 1, "truncated")
    retry = _attempt(call_id, 2, "ok")
    retry["parent_attempt_id"] = first["attempt_id"]
    final, counts = _final_events([first, retry], expected)
    assert final[call_id] == retry
    assert counts == {"stored_attempts": 2, "first_attempts": 1, "retry_attempts": 1}


def test_retry_after_success_is_rejected() -> None:
    call_id = "cell::call"
    expected = _expected(call_id)
    first = _attempt(call_id, 1, "ok")
    retry = _attempt(call_id, 2, "ok")
    retry["parent_attempt_id"] = first["attempt_id"]
    with pytest.raises(ValueError, match="unpermitted retry"):
        _final_events([first, retry], expected)


def test_retryable_first_attempt_requires_its_terminal_retry() -> None:
    call_id = "cell::call"
    with pytest.raises(ValueError, match="missing required retry"):
        _final_events([_attempt(call_id, 1, "truncated")], _expected(call_id))


def test_accuracy_reports_resolved_denominator_and_parse_rate() -> None:
    result = _accuracy(np.asarray([0.0, 1.0, np.nan, 0.0]))
    assert result["resolved"] == 3
    assert result["parse_rate"] == 0.75
    assert result["accuracy_resolved"] == pytest.approx(2 / 3)


def test_malformed_treatments_match_frozen_statistics() -> None:
    matrix = np.asarray([[0, 2], [1, 3]])
    groups = _groups(matrix, 4)
    assert groups.tolist() == [0, 0, 1, 1]
    joint = [np.asarray([1.0, 1.0, 0.0, np.nan])] * 3
    exact = [np.asarray([1.0, 0.0, 0.0, 0.0])] * 3
    result = _sensitivities(joint, exact, [matrix] * 3)
    means = result["cell_mean"]
    assert set(means) >= {
        "delta_unresolved_as_correct",
        "delta_unresolved_as_error",
        "delta_unresolved_dropped",
        "delta_whole_call_discarded",
    }
    assert result["n_calls_discarded_total"] == 3
