"""Synthetic batch-invariance preflight for the cross-family extension."""

from __future__ import annotations

from collections import defaultdict
from string import hexdigits
from typing import Any

from step0.hashing import sha256_json

from .prompts import build_messages

SCHEMA_VERSION = "ext-batch-invariance-v1"
ENGINE_IMAGE = {"vllm": "0.28.0"}
GPU = "A100"
SERVING = {
    "tensor_parallel_size": 1,
    "dtype": "bfloat16",
    "max_num_seqs": 32,
}
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 20260912}
BATCH_SIZES = (1, 2, 8, 16, 32)
ORDERS = ("forward", "reverse")
SESSION_INDICES = (0, 1)
TASKS = ("civil_comments", "multi_nli", "imdb", "ag_news")
N_PROMPTS_PER_TASK = max(BATCH_SIZES)
VLLM_VERSION = "0.28.0"

SYNTHETIC_TEXT = {
    "civil_comments": "This is a synthetic neutral comment used only for a runtime check.",
    "multi_nli": (
        "Premise: A synthetic person is standing outside. Hypothesis: A person is outdoors."
    ),
    "imdb": "This synthetic movie review says the acting was clear and the story was pleasant.",
    "ag_news": "A synthetic city council announced a routine public meeting for Tuesday.",
}


def required_environment() -> dict[str, str]:
    """Environment that must exist before vLLM is imported."""
    return {"VLLM_BATCH_INVARIANT": "1"}


def prompt_bank() -> dict[str, list[dict[str, Any]]]:
    """Build 32 deterministic single-target synthetic prompts for each extension task."""
    bank: dict[str, list[dict[str, Any]]] = {}
    for task in TASKS:
        prompts = []
        for index in range(N_PROMPTS_PER_TASK):
            prompt_id = f"batch-invariance::{task}::{index:02d}"
            text = f"{SYNTHETIC_TEXT[task]} Synthetic record {index:02d}."
            messages = build_messages(task, [(prompt_id, text)], [prompt_id])
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "task": task,
                    "messages": messages,
                    "prompt_sha256": sha256_json(messages),
                    "max_tokens": 84,
                }
            )
        bank[task] = prompts
    return bank


def cases() -> list[dict[str, Any]]:
    """Compare one anchor prompt alone and amid several request-batch layouts."""
    result = []
    for task, prompts in prompt_bank().items():
        for batch_size in BATCH_SIZES:
            order_names = ("forward",) if batch_size == 1 else ORDERS
            for order in order_names:
                selected = prompts[:batch_size]
                if order == "reverse":
                    selected = list(reversed(selected))
                result.append(
                    {
                        "case_id": f"{task}::n{batch_size:02d}::{order}",
                        "task": task,
                        "batch_size": batch_size,
                        "order": order,
                        "requests": selected,
                    }
                )
    return result


def plan_payload(models: list[dict[str, str]]) -> dict[str, Any]:
    """Bind checkpoint revisions and every runtime setting tested by the gate."""
    required = {"model_id", "revision"}
    if not models or any(set(model) != required for model in models):
        raise ValueError("each batch-invariance model needs exactly model_id and revision")
    if len({model["model_id"] for model in models}) != len(models):
        raise ValueError("batch-invariance model IDs must be unique")
    if any(
        len(model["revision"]) != 40
        or any(character not in hexdigits for character in model["revision"])
        for model in models
    ):
        raise ValueError("each batch-invariance revision must be a 40-character Git SHA")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "synthetic_only": True,
        "required_environment": required_environment(),
        "engine_image": ENGINE_IMAGE,
        "gpu": GPU,
        "serving": SERVING,
        "decoding": DECODING,
        "batch_sizes": list(BATCH_SIZES),
        "orders": list(ORDERS),
        "session_indices": list(SESSION_INDICES),
        "models": models,
        "cases": cases(),
    }
    payload["plan_sha256"] = sha256_json(payload)
    return payload


def validate_events(events: list[dict[str, Any]], models: list[dict[str, str]]) -> dict[str, Any]:
    """Reject missing runs, runtime drift, failures, or any changed anchor response."""
    expected_case_rows = {case["case_id"]: case for case in cases()}
    expected_cases = set(expected_case_rows)
    expected_prompt_hashes = {
        task: prompts[0]["prompt_sha256"] for task, prompts in prompt_bank().items()
    }
    expected_sessions = set(SESSION_INDICES)
    expected_revisions = {model["model_id"]: model["revision"] for model in models}
    expected_models = set(expected_revisions)
    expected_rows = len(expected_models) * len(expected_sessions) * len(expected_cases)
    if len(events) != expected_rows:
        raise ValueError(f"expected {expected_rows} completed cases, received {len(events)}")

    seen: set[tuple[str, int, str]] = set()
    anchor_hashes: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    hardware_names: defaultdict[str, set[str]] = defaultdict(set)
    for event in events:
        model_id = str(event.get("model_id"))
        session_index = int(event.get("session_index", -1))
        case_id = str(event.get("case_id"))
        key = (model_id, session_index, case_id)
        if key in seen:
            raise ValueError(f"duplicate batch-invariance case: {key}")
        seen.add(key)
        if model_id not in expected_models:
            raise ValueError(f"foreign model in batch-invariance evidence: {model_id}")
        if session_index not in expected_sessions or case_id not in expected_cases:
            raise ValueError(f"foreign session or case in batch-invariance evidence: {key}")
        if event.get("status") != "ok":
            raise ValueError(f"batch-invariance case failed: {key}")
        if event.get("finish_reason") != "stop":
            raise ValueError(f"batch-invariance anchor did not finish normally: {key}")
        backend = event.get("backend") or {}
        if backend.get("batch_invariant") is not True:
            raise ValueError(f"batch-invariant mode was not active: {key}")
        if backend.get("batch_invariance_required") is not True:
            raise ValueError(f"the backend did not require batch invariance: {key}")
        if (
            backend.get("model_id") != model_id
            or backend.get("model_revision") != expected_revisions[model_id]
            or backend.get("tokenizer_revision") != expected_revisions[model_id]
        ):
            raise ValueError(f"the pinned checkpoint identity changed: {key}")
        if (
            backend.get("dtype") != "bfloat16"
            or backend.get("max_num_seqs") != 32
            or backend.get("tensor_parallel_size") != 1
            or backend.get("seed") != 20260912
        ):
            raise ValueError(f"the frozen serving configuration changed: {key}")
        host = event.get("host") or {}
        if host.get("installed_engine_version_string") != VLLM_VERSION:
            raise ValueError(f"the pinned vLLM version changed: {key}")
        if "A100" not in str(host.get("gpu_name", "")):
            raise ValueError(f"the batch-invariance check did not run on an A100: {key}")
        hardware_names[model_id].add(str(host["gpu_name"]))
        expected_case = expected_case_rows[case_id]
        if (
            event.get("task") != expected_case["task"]
            or event.get("batch_size") != expected_case["batch_size"]
            or event.get("order") != expected_case["order"]
        ):
            raise ValueError(f"the frozen request layout changed: {key}")
        prompt_id = str(event.get("anchor_prompt_id"))
        output_sha256 = str(event.get("anchor_output_sha256"))
        if not prompt_id or not output_sha256:
            raise ValueError(f"batch-invariance case lacks its anchor evidence: {key}")
        task = str(event.get("task"))
        if prompt_id != f"batch-invariance::{task}::00":
            raise ValueError(f"the synthetic anchor ID changed: {key}")
        if event.get("anchor_prompt_sha256") != expected_prompt_hashes.get(task):
            raise ValueError(f"the synthetic anchor prompt changed: {key}")
        anchor_hashes[(model_id, prompt_id)].add(output_sha256)

    expected_keys = {
        (model_id, session_index, case_id)
        for model_id in expected_models
        for session_index in expected_sessions
        for case_id in expected_cases
    }
    if seen != expected_keys:
        raise ValueError("batch-invariance evidence does not cover the complete frozen matrix")
    changed = {key: hashes for key, hashes in anchor_hashes.items() if len(hashes) != 1}
    if changed:
        raise ValueError(f"identical prompts changed across request batches: {sorted(changed)}")
    mixed_hardware = {model: names for model, names in hardware_names.items() if len(names) != 1}
    if mixed_hardware:
        raise ValueError(f"fresh sessions used different A100 variants: {mixed_hardware}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "synthetic_only": True,
        "models_checked": sorted(expected_models),
        "sessions_per_model": len(expected_sessions),
        "cases_per_session": len(expected_cases),
        "events_verified": len(events),
        "anchors_verified": len(anchor_hashes),
    }
