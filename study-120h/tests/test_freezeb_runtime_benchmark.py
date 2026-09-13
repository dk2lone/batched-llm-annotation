"""CPU-only checks for the synthetic Freeze B runtime benchmark."""

from __future__ import annotations

import pytest

from freezeb.runtime_benchmark import (
    MODEL_POLICIES,
    build_cases,
    context_fit,
    forecast_block,
    plan_payload,
    summarize_measurements,
)


def test_benchmark_matrix_is_complete_deterministic_and_synthetic() -> None:
    first = build_cases()
    second = build_cases()
    assert first == second
    assert len(first) == 30
    assert len({case.case_id for case in first}) == 30
    for task_id in {case.task_id for case in first}:
        task_cases = [case for case in first if case.task_id == task_id]
        assert [(case.shape, case.condition) for case in task_cases] == [
            ("short", "joint"),
            ("size16", "joint"),
            ("size16", "exact_context"),
            ("size32", "joint"),
            ("size32", "exact_context"),
        ]
    rendered = "\n".join(
        message["content"] for case in first for message in case.messages
    )
    assert "synthetic record" in rendered
    assert "civil-comments-" not in rendered


def test_exact_controls_keep_full_context_but_request_one_target() -> None:
    cases = {case.case_id: case for case in build_cases()}
    joint = cases["imdb::size16::joint"]
    exact = cases["imdb::size16::exact_context"]
    assert joint.batch_size == exact.batch_size == 16
    assert joint.target_count == 16
    assert exact.target_count == 1
    for index in range(16):
        marker = f'<doc id="synthetic-{index:02d}">'
        assert marker in joint.messages[1]["content"]
        assert marker in exact.messages[1]["content"]


def test_plan_records_exclusions_and_has_a_stable_self_hash() -> None:
    plan = plan_payload()
    assert plan == plan_payload()
    assert plan["synthetic_only"] is True
    assert len(plan["plan_sha256"]) == 64
    policies = {policy.model_id: policy for policy in MODEL_POLICIES}
    assert policies["allenai/OLMo-2-1124-7B"].runnable is False
    assert policies["google/gemma-3-4b-it"].runnable is False
    assert sum(policy.runnable for policy in MODEL_POLICIES) == 6


def test_context_fit_includes_the_output_reserve() -> None:
    assert context_fit(3000, 1000, 4096)["fits"] is True
    excluded = context_fit(3000, 1097, 4096)
    assert excluded["fits"] is False
    assert excluded["headroom_tokens"] == -1


def test_summary_keeps_failures_separate_from_latency() -> None:
    rows = [
        {"model_id": "m", "case_id": "c", "status": "ok", "latency_seconds": 2.0,
         "prompt_tokens": 10, "output_tokens": 2},
        {"model_id": "m", "case_id": "c", "status": "failed", "latency_seconds": 99.0,
         "prompt_tokens": 10, "output_tokens": None},
        {"model_id": "m", "case_id": "c", "status": "ok", "latency_seconds": 4.0,
         "prompt_tokens": 11, "output_tokens": 3},
    ]
    summary = summarize_measurements(rows)["case_summaries"][0]
    assert summary["attempts"] == 3
    assert summary["successes"] == 2
    assert summary["failures"] == 1
    assert summary["median_latency_seconds"] == 3.0


def test_forecast_requires_every_supplied_case_rate() -> None:
    result = forecast_block({"joint": 2.0, "exact": 0.5}, {"joint": 10, "exact": 20})
    assert result["base_a100_hours"] == pytest.approx(30 / 3600)
    assert result["forecasts_a100_hours"]["1.6x"] == pytest.approx(48 / 3600)
    assert result["forecasts_a100_hours"]["3x"] == pytest.approx(90 / 3600)
    with pytest.raises(ValueError, match="missing benchmark rates"):
        forecast_block({"joint": 2.0}, {"joint": 10, "exact": 20})
