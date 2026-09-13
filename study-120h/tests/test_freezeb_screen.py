"""CPU-only tests for the Freeze B solo-screen execution boundary."""

from __future__ import annotations

import json
from typing import Any

import pytest

from freezeb.frames import N_SCREEN
from freezeb.runtime_benchmark import policy_for
from freezeb.screen import (
    RUNNABLE_MODELS,
    TASK_KEYS,
    build_calls,
    build_part2_payload,
    chunks,
    completed_call_ids,
    pending_calls,
    summarize_cell,
    technical_reforecast,
    validate_worker_provenance,
)


def _rows(task: str = "ag_news") -> list[dict[str, object]]:
    return [
        {
            "doc_id": f"x{i:04d}",
            "text": f"document {i}",
            "reference": i % 4,
            "reference_normalized": i % 4,
            "n_tokens": None,
        }
        for i in range(N_SCREEN)
    ]


def _provenance(model_id: str) -> dict[str, Any]:
    policy = policy_for(model_id)
    return {
        "worker_host": {
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "installed_engine_version_string": "0.28.0",
        },
        "worker_backend": {
            "backend": "vllm",
            "model_id": model_id,
            "model_revision": policy.revision,
            "tokenizer_revision": policy.revision,
            "dtype": "bfloat16",
            "max_model_len": policy.max_model_len,
            "max_num_seqs": 32,
            "tensor_parallel_size": 1,
        },
    }


def test_screen_calls_are_solo_only_and_first_1000_is_a_boundary() -> None:
    calls = build_calls("Qwen/Qwen3-8B", "ag_news", _rows())
    assert len(calls) == N_SCREEN
    assert all(call["condition"] == "solo" for call in calls)
    assert all(call["messages"][1]["content"].count("<doc id=") == 1 for call in calls)
    groups = chunks(calls, 32)
    assert sum(map(len, groups)) == N_SCREEN
    assert sum(map(len, groups[:32])) == 1000


def test_completed_ids_require_terminal_flag() -> None:
    assert completed_call_ids(
        [
            {"call_id": "a", "terminal": True, "final_for_call": True},
            {"call_id": "b", "terminal": True, "final_for_call": False},
        ]
    ) == {"a"}


def test_failed_attempt_is_preserved_and_resumed_with_a_new_attempt_id() -> None:
    calls = build_calls("Qwen/Qwen3-8B", "ag_news", _rows())
    failed = [
        {
            **{key: calls[0][key] for key in ("call_id", "attempt_id", "attempt")},
            "terminal": True,
            "final_for_call": False,
            "status": "worker_failed",
        }
    ]
    resumed = pending_calls(calls, failed)
    assert resumed[0]["call_id"] == calls[0]["call_id"]
    assert resumed[0]["attempt"] == 2
    assert resumed[0]["attempt_id"].endswith("::attempt::2")
    assert len(resumed) == N_SCREEN


def test_first_1000_reforecast_rebuilds_from_ledger() -> None:
    events = [
        {
            "status": "ok",
            "attempt_id": f"attempt-{index}",
            "prompt_tokens": 20,
            "output_tokens": 2,
            "worker_session_id": "worker-1",
            "worker_load_seconds": 10.0,
            "metered_a100_seconds": 0.4,
            **(
                {
                    "timing_probe_method": "vllm_request_metrics_single_call",
                    "probe_prompt_plus_first_token_seconds": 0.2,
                    "probe_remaining_generation_seconds": 0.1,
                }
                if index == 0
                else {}
            ),
        }
        for index in range(1000)
    ]
    result = technical_reforecast("Qwen/Qwen3-8B", events, 1.0)
    assert result is not None
    assert result["real_calls"] == 1000
    assert result["loading_a100_seconds"] == 10.0
    assert result["all_in_a100_seconds"] == pytest.approx(400.0)
    assert result["component_timing"]["scope"] == "single_call_probe_only"
    assert result["component_timing"]["prompt_plus_first_token_seconds"] == 0.2
    assert result["projected_model_screen_base_a100_hours"] == pytest.approx(
        (10.0 + 390.0 * 12.288) / 3600
    )
    assert technical_reforecast("Qwen/Qwen3-8B", events[:999], 1.0) is None


def test_summary_parses_once_and_rejects_duplicate_events() -> None:
    rows = _rows()
    calls = build_calls("Qwen/Qwen3-8B", "ag_news", rows)
    events = []
    for index, call in enumerate(calls):
        label = (index % 4 + (1 if index % 10 == 0 else 0)) % 4
        events.append(
            {
                "call_id": call["call_id"],
                "attempt_id": call["attempt_id"],
                "attempt": 1,
                "terminal": True,
                "final_for_call": True,
                "status": "ok",
                "first_response": json.dumps({"labels": {call["doc_id"]: label}}),
                **_provenance("Qwen/Qwen3-8B"),
            }
        )
    result = summarize_cell("Qwen/Qwen3-8B", "ag_news", rows, events, bootstrap_draws=20)
    assert result["technical_complete"] is True
    assert result["parse_rate"] == 1.0
    assert result["accuracy"] == pytest.approx(0.89990234375)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_cell("Qwen/Qwen3-8B", "ag_news", rows, [events[0], events[0]])


def test_worker_provenance_rejects_wrong_engine_or_model_revision() -> None:
    model_id = "Qwen/Qwen3-8B"
    provenance = _provenance(model_id)
    validate_worker_provenance(model_id, provenance["worker_host"], provenance["worker_backend"])
    bad_host = {**provenance["worker_host"], "installed_engine_version_string": "0.27.0"}
    with pytest.raises(ValueError, match="vLLM version"):
        validate_worker_provenance(model_id, bad_host, provenance["worker_backend"])
    bad_backend = {**provenance["worker_backend"], "model_revision": "main"}
    with pytest.raises(ValueError, match="model_revision"):
        validate_worker_provenance(model_id, provenance["worker_host"], bad_backend)


def test_part2_contains_every_required_lock_component() -> None:
    summaries = [
        {
            "cell_key": f"{model}::{task}",
            "model_id": model,
            "task_id": task,
            "technical_complete": True,
            "status_counts": {"ok": N_SCREEN},
            "admitted": True,
            "error_rate": 0.1,
        }
        for model in RUNNABLE_MODELS
        for task in TASK_KEYS
    ]
    power_rows = [
        {"p": p, "n": float(n), "sims": 800.0, "power": 1.0, "power_mc_se": 0.0}
        for p in (0.05, 0.10, 0.20, 0.35, 0.50)
        for n in (2048, 4096, 8192)
    ]
    payload = build_part2_payload(summaries, power_rows)
    assert len(payload["attempted_candidates"]) == 36
    assert len(payload["ordered_cells"]["primary"]) == 16
    assert len(payload["ordered_cells"]["held_out"]) == 12
    assert set(payload) >= {
        "attempted_candidates",
        "technical_status",
        "solo_metrics",
        "ordered_cells",
        "item_counts",
        "complete_block_queue",
        "claim_scope",
    }
