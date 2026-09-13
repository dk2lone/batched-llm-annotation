"""CPU-only checks for the extension batch-invariance gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from freezeb import batch_invariance as bi
from step1.backends import VLLMBackend

ROOT = Path(__file__).resolve().parents[1]


def test_matrix_changes_request_batch_size_and_anchor_position() -> None:
    cases = bi.cases()
    assert len(cases) == len(bi.TASKS) * (1 + 2 * (len(bi.BATCH_SIZES) - 1))
    assert {case["batch_size"] for case in cases} == set(bi.BATCH_SIZES)
    for case in cases:
        anchor_id = f"batch-invariance::{case['task']}::00"
        prompt_ids = [prompt["prompt_id"] for prompt in case["requests"]]
        assert len(prompt_ids) == case["batch_size"]
        assert anchor_id in prompt_ids
        if case["order"] == "forward":
            assert prompt_ids[0] == anchor_id
        else:
            assert prompt_ids[-1] == anchor_id


def test_plan_records_required_environment_and_is_deterministic() -> None:
    models = [{"model_id": "example/model", "revision": "a" * 40}]
    first = bi.plan_payload(models)
    second = bi.plan_payload(models)
    assert first == second
    assert first["required_environment"] == {"VLLM_BATCH_INVARIANT": "1"}
    assert first["models"] == models
    assert first["engine_image"] == {"vllm": "0.28.0"}
    assert first["serving"]["max_num_seqs"] == 32
    assert first["plan_sha256"]


def test_plan_refuses_unpinned_or_duplicate_models() -> None:
    with pytest.raises(ValueError, match="exactly model_id and revision"):
        bi.plan_payload([{"model_id": "example/model"}])
    with pytest.raises(ValueError, match="unique"):
        bi.plan_payload(
            [
                {"model_id": "example/model", "revision": "a" * 40},
                {"model_id": "example/model", "revision": "b" * 40},
            ]
        )
    with pytest.raises(ValueError, match="40-character Git SHA"):
        bi.plan_payload([{"model_id": "example/model", "revision": "not-a-pin"}])


def _models(*model_ids: str) -> list[dict[str, str]]:
    return [
        {"model_id": model_id, "revision": f"{index + 1:040x}"}
        for index, model_id in enumerate(model_ids)
    ]


def _valid_events(models: list[dict[str, str]]) -> list[dict[str, Any]]:
    events = []
    revisions = {model["model_id"]: model["revision"] for model in models}
    prompt_hashes = {
        task: prompts[0]["prompt_sha256"] for task, prompts in bi.prompt_bank().items()
    }
    for model_id in revisions:
        for session_index in bi.SESSION_INDICES:
            for case in bi.cases():
                events.append(
                    {
                        "model_id": model_id,
                        "session_index": session_index,
                        "case_id": case["case_id"],
                        "task": case["task"],
                        "batch_size": case["batch_size"],
                        "order": case["order"],
                        "status": "ok",
                        "finish_reason": "stop",
                        "anchor_prompt_id": f"batch-invariance::{case['task']}::00",
                        "anchor_prompt_sha256": prompt_hashes[case["task"]],
                        "anchor_output_sha256": f"stable-{model_id}-{case['task']}",
                        "backend": {
                            "model_id": model_id,
                            "model_revision": revisions[model_id],
                            "tokenizer_revision": revisions[model_id],
                            "batch_invariant": True,
                            "batch_invariance_required": True,
                            "dtype": "bfloat16",
                            "max_num_seqs": 32,
                            "tensor_parallel_size": 1,
                            "seed": 20260912,
                        },
                        "host": {
                            "gpu_name": "NVIDIA A100-SXM4-40GB",
                            "installed_engine_version_string": "0.28.0",
                        },
                    }
                )
    return events


def test_complete_identical_matrix_passes() -> None:
    models = _models("model/a", "model/b")
    report = bi.validate_events(_valid_events(models), models)
    assert report["status"] == "passed"
    assert report["events_verified"] == 2 * len(bi.SESSION_INDICES) * len(bi.cases())


def test_changed_identical_prompt_fails() -> None:
    models = _models("model/a")
    events = _valid_events(models)
    events[-1]["anchor_output_sha256"] = "changed"
    with pytest.raises(ValueError, match="identical prompts changed"):
        bi.validate_events(events, models)


def test_missing_or_noninvariant_evidence_fails() -> None:
    models = _models("model/a")
    events = _valid_events(models)
    with pytest.raises(ValueError, match="expected"):
        bi.validate_events(events[:-1], models)
    events[0]["backend"]["batch_invariant"] = False
    with pytest.raises(ValueError, match="mode was not active"):
        bi.validate_events(events, models)


def test_checkpoint_anchor_and_layout_drift_fail() -> None:
    models = _models("model/a")
    events = _valid_events(models)
    events[0]["backend"]["model_revision"] = "f" * 40
    with pytest.raises(ValueError, match="checkpoint identity changed"):
        bi.validate_events(events, models)

    events = _valid_events(models)
    events[0]["batch_size"] = 99
    with pytest.raises(ValueError, match="request layout changed"):
        bi.validate_events(events, models)

    events = _valid_events(models)
    events[0]["anchor_prompt_id"] = "different-anchor"
    with pytest.raises(ValueError, match="anchor ID changed"):
        bi.validate_events(events, models)


def test_backend_rejects_required_mode_before_importing_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    with pytest.raises(RuntimeError, match="must be set before constructing"):
        VLLMBackend(
            "example/model",
            "revision",
            "revision",
            {"max_num_seqs": 32},
            {"seed": 1},
            require_batch_invariance=True,
        )


def test_modal_preflight_imports_without_launching_gpu() -> None:
    path = ROOT / "modal_extension_batch_invariance.py"
    spec = importlib.util.spec_from_file_location("modal_extension_batch_invariance", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.RUNTIME_ENV["VLLM_BATCH_INVARIANT"] == "1"
    assert runner.VLLM_VERSION == bi.VLLM_VERSION == "0.28.0"
