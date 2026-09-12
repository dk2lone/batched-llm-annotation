"""Generation backends. vLLM is imported lazily and never during analysis or tests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Config
from .prompts import Message, prompt_text
from .util import uniform01


@dataclass(frozen=True)
class GenRequest:
    job_id: str
    condition: str
    assignment: int
    real_batch_id: str
    target_ids: tuple[str, ...]
    messages: list[Message]
    max_tokens: int


@dataclass(frozen=True)
class GenResult:
    job_id: str
    text: str
    gen_seconds: float
    prompt_tokens: int | None


class Backend(Protocol):
    name: str

    def generate(self, requests: list[GenRequest]) -> list[GenResult]: ...

    def count_tokens(self, messages: list[Message]) -> int | None: ...

    def describe(self) -> dict[str, Any]: ...


class CharEstimateCounter:
    """Token counter used on a machine that has no tokenizer. Deliberately coarse."""

    def __init__(self, chars_per_token: float) -> None:
        self.chars_per_token = chars_per_token

    def count(self, messages: list[Message]) -> int:
        return int(len(prompt_text(messages)) / self.chars_per_token) + 1


class MockBackend:
    """Deterministic stand-in for a model. It is not a model.

    It reads the gold labels to synthesise a controlled error process, so the
    pipeline, the parser and the estimators can be checked end to end on a CPU.
    It plants a shared batch effect in the joint condition only. No mock output
    is evidence about any real model.
    """

    name = "mock"

    def __init__(self, cfg: Config, gold: dict[str, int]) -> None:
        self.cfg = cfg
        self.gold = gold
        self.counter = CharEstimateCounter(cfg.run.chars_per_token)

    def _error_probability(self, request: GenRequest) -> float:
        base = self.cfg.run.mock_base_error
        if request.condition != "joint":
            return base
        shift = uniform01(f"{self.cfg.run.mock_seed}:batch:{request.real_batch_id}") - 0.5
        return min(0.95, max(0.01, base + self.cfg.run.mock_batch_effect * shift))

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        results: list[GenResult] = []
        for request in requests:
            start = time.perf_counter()
            seed = f"{self.cfg.run.mock_seed}:{request.job_id}"
            if uniform01(f"{seed}:malformed") < self.cfg.run.mock_parse_failure_rate:
                text = "Sure! Here are the labels: the first one looks toxic."
            else:
                probability = self._error_probability(request)
                labels: dict[str, int] = {}
                for doc_id in request.target_ids:
                    if uniform01(f"{seed}:{doc_id}:drop") < self.cfg.run.mock_missing_id_rate:
                        continue
                    gold = self.gold[doc_id]
                    wrong = uniform01(f"{seed}:{doc_id}:err") < probability
                    labels[doc_id] = 1 - gold if wrong else gold
                text = json.dumps({"labels": labels})
            results.append(
                GenResult(
                    job_id=request.job_id,
                    text=text,
                    gen_seconds=time.perf_counter() - start,
                    prompt_tokens=self.counter.count(request.messages),
                )
            )
        return results

    def count_tokens(self, messages: list[Message]) -> int | None:
        return self.counter.count(messages)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "mock_seed": self.cfg.run.mock_seed,
            "mock_base_error": self.cfg.run.mock_base_error,
            "mock_batch_effect": self.cfg.run.mock_batch_effect,
        }


class VLLMBackend:
    """Offline vLLM engine for a Linux CUDA host."""

    name = "vllm"

    def __init__(self, cfg: Config) -> None:
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        self.cfg = cfg
        self._sampling_cls = SamplingParams
        model = cfg.model
        self.llm = LLM(
            model=model.model_id,
            revision=model.revision,
            tensor_parallel_size=model.tensor_parallel_size,
            dtype=model.dtype,
            max_model_len=model.max_model_len,
            gpu_memory_utilization=model.gpu_memory_utilization,
            max_num_seqs=model.max_num_seqs,
            max_num_batched_tokens=model.max_num_batched_tokens,
            enforce_eager=model.enforce_eager,
            trust_remote_code=model.trust_remote_code,
            seed=model.seed,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _chat_kwargs(self) -> dict[str, Any]:
        # Qwen3 reads enable_thinking from the chat template. Other templates ignore it.
        return {"chat_template_kwargs": {"enable_thinking": self.cfg.model.enable_thinking}}

    def count_tokens(self, messages: list[Message]) -> int | None:
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.cfg.model.enable_thinking,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return len(self.tokenizer(rendered)["input_ids"])

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        model = self.cfg.model
        params = [
            self._sampling_cls(
                temperature=model.temperature,
                top_p=model.top_p,
                top_k=model.top_k,
                max_tokens=request.max_tokens,
                seed=model.seed,
            )
            for request in requests
        ]
        conversations = [request.messages for request in requests]
        start = time.perf_counter()
        try:
            outputs = self.llm.chat(conversations, params, use_tqdm=False, **self._chat_kwargs())
        except TypeError:
            outputs = self.llm.chat(conversations, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        share = elapsed / max(1, len(requests))
        return [
            GenResult(
                job_id=request.job_id,
                text=output.outputs[0].text,
                gen_seconds=share,
                prompt_tokens=len(output.prompt_token_ids),
            )
            for request, output in zip(requests, outputs, strict=True)
        ]

    def describe(self) -> dict[str, Any]:
        model = self.cfg.model
        return {
            "backend": self.name,
            "model_id": model.model_id,
            "revision": model.revision,
            "tensor_parallel_size": model.tensor_parallel_size,
            "dtype": model.dtype,
            "max_model_len": model.max_model_len,
            "max_num_seqs": model.max_num_seqs,
            "enable_thinking": model.enable_thinking,
        }


def make_backend(cfg: Config, gold: dict[str, int], name: str | None = None) -> Backend:
    chosen = name or cfg.run.backend
    if chosen == "mock":
        return MockBackend(cfg, gold)
    if chosen == "vllm":
        return VLLMBackend(cfg)
    raise ValueError(f"unknown backend {chosen!r}")
