"""Synthetic runtime benchmark plan and result validation for Freeze B.

The benchmark never reads study datasets or emits scientific labels.  It renders
fixed synthetic records through the frozen task templates so model access,
context fit, latency, token counts, and memory use can be checked before a solo
screen starts.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

from step0.hashing import sha256_json

from .frames import TASKS
from .models import MODELS, Model
from .prompts import build_messages, estimated_response_tokens

SCHEMA_VERSION = "freezeb-runtime-benchmark-v1"
ENGINE_IMAGE = {
    "python": "3.11",
    "huggingface-hub": "1.31.0",
    "numpy": "2.2.6",
    "scipy": "1.16.3",
    "vllm": "0.28.0",
}
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 20260912}
FORECAST_MULTIPLIERS = (1.0, 1.6, 3.0)


@dataclass(frozen=True)
class RuntimePolicy:
    model_id: str
    revision: str
    family: str
    max_model_len: int
    enable_thinking: bool
    runnable: bool
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    task_id: str
    shape: str
    condition: str
    batch_size: int
    target_count: int
    chars_per_document: int
    token_counts: tuple[int, ...]
    max_tokens: int
    messages: list[dict[str, str]]
    prompt_sha256: str


def _runtime_policy(model: Model) -> RuntimePolicy:
    # The benchmark caps all long-context candidates at 32k. OLMo's pinned
    # configuration is 4k. These are benchmark settings, not a scientific lock.
    max_model_len = 4096 if model.family == "olmo" else 32768
    runnable = True
    reason = None
    if model.repo == "allenai/OLMo-2-1124-7B":
        runnable = False
        reason = "base checkpoint has no pinned chat template policy"
    elif model.repo == "google/gemma-3-4b-it":
        runnable = False
        reason = "gated checkpoint requires separately verified access"
    return RuntimePolicy(
        model_id=model.repo,
        revision=model.revision,
        family=model.family,
        max_model_len=max_model_len,
        enable_thinking=False,
        runnable=runnable,
        exclusion_reason=reason,
    )


MODEL_POLICIES = tuple(_runtime_policy(model) for model in MODELS)


def policy_for(model_id: str) -> RuntimePolicy:
    try:
        return next(policy for policy in MODEL_POLICIES if policy.model_id == model_id)
    except StopIteration as exc:
        raise ValueError(f"unknown model {model_id}") from exc


def _synthetic_text(task_id: str, index: int, n_chars: int) -> str:
    # Spaces are intentional: MultiNERD's output cap depends on whitespace tokens.
    seed = f"synthetic record {index:02d} neutral context only "
    repeats = (n_chars // len(seed)) + 1
    return (seed * repeats)[:n_chars].rstrip()


def _case(task_id: str, shape: str, batch_size: int, exact: bool) -> BenchmarkCase:
    task = next((item for item in TASKS if item.key == task_id), None)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    chars = min(task.max_chars, 128) if shape == "short" else task.max_chars
    docs = [
        (f"synthetic-{index:02d}", _synthetic_text(task_id, index, chars))
        for index in range(batch_size)
    ]
    target_ids = [docs[batch_size // 2][0]] if exact else [doc_id for doc_id, _ in docs]
    token_counts = tuple(len(text.split()) for _, text in docs)
    messages = build_messages(task_id, docs, target_ids)
    condition = "exact_context" if exact else "joint"
    case_id = f"{task_id}::{shape}::{condition}"
    return BenchmarkCase(
        case_id=case_id,
        task_id=task_id,
        shape=shape,
        condition=condition,
        batch_size=batch_size,
        target_count=len(target_ids),
        chars_per_document=chars,
        token_counts=token_counts,
        max_tokens=estimated_response_tokens(
            task_id,
            len(target_ids),
            [token_counts[batch_size // 2]] if exact else list(token_counts),
        ),
        messages=messages,
        prompt_sha256=sha256_json(messages),
    )


def build_cases() -> tuple[BenchmarkCase, ...]:
    """Return the fixed 30-case matrix: five shapes for each of six tasks."""
    cases: list[BenchmarkCase] = []
    for task in TASKS:
        cases.extend(
            (
                _case(task.key, "short", 1, False),
                _case(task.key, "size16", 16, False),
                _case(task.key, "size16", 16, True),
                _case(task.key, "size32", 32, False),
                _case(task.key, "size32", 32, True),
            )
        )
    return tuple(cases)


def plan_payload() -> dict[str, Any]:
    policies = [asdict(policy) for policy in MODEL_POLICIES]
    cases = [asdict(case) for case in build_cases()]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "synthetic_only": True,
        "engine_image": ENGINE_IMAGE,
        "decoding": DECODING,
        "gpu": "A100",
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 32,
        "warmups_per_case": 1,
        "measured_repetitions_per_case": 2,
        "models": policies,
        "cases": cases,
    }
    payload["plan_sha256"] = sha256_json(payload)
    return payload


def context_fit(prompt_tokens: int, max_tokens: int, max_model_len: int) -> dict[str, Any]:
    required = prompt_tokens + max_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "required_tokens": required,
        "max_model_len": max_model_len,
        "fits": required <= max_model_len,
        "headroom_tokens": max_model_len - required,
    }


def summarize_measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate successful repetitions without converting failures to zero time."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model_id"]), str(row["case_id"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (model_id, case_id), group in sorted(grouped.items()):
        successful = [row for row in group if row.get("status") == "ok"]
        latencies = [float(row["latency_seconds"]) for row in successful]
        summaries.append(
            {
                "model_id": model_id,
                "case_id": case_id,
                "attempts": len(group),
                "successes": len(successful),
                "failures": len(group) - len(successful),
                "median_latency_seconds": statistics.median(latencies) if latencies else None,
                "max_prompt_tokens": max(
                    (int(row["prompt_tokens"]) for row in successful), default=None
                ),
                "max_output_tokens": max(
                    (int(row["output_tokens"]) for row in successful), default=None
                ),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "case_summaries": summaries}


def forecast_block(
    case_seconds: dict[str, float], call_counts: dict[str, int]
) -> dict[str, Any]:
    """Forecast one explicitly supplied complete block from measured case rates."""
    missing = sorted(set(call_counts) - set(case_seconds))
    if missing:
        raise ValueError(f"missing benchmark rates for cases: {missing}")
    if any(count < 0 for count in call_counts.values()):
        raise ValueError("call counts must be nonnegative")
    base_seconds = sum(case_seconds[case_id] * count for case_id, count in call_counts.items())
    return {
        "call_counts": dict(sorted(call_counts.items())),
        "base_a100_hours": base_seconds / 3600.0,
        "forecasts_a100_hours": {
            f"{multiplier:g}x": multiplier * base_seconds / 3600.0
            for multiplier in FORECAST_MULTIPLIERS
        },
    }
