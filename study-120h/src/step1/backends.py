"""Generation backends and token counters.

vLLM and transformers are imported lazily, so the tests, the linter, and the type checker
never need a CUDA host. `FakeBackend` is a deterministic stand-in used by the tests. It is
not a model and no output of it is evidence about any real model.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from step0.hashing import sha256_text

from .prompts import Message, flat_text


@dataclass(frozen=True)
class GenRequest:
    call_id: str
    condition: str
    target_ids: tuple[str, ...]
    messages: list[Message]
    max_tokens: int


@dataclass(frozen=True)
class GenResult:
    call_id: str
    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    queue_seconds: float | None = None
    prefill_seconds: float | None = None
    decode_seconds: float | None = None


class TokenCounter(Protocol):
    name: str

    def count_tokens(self, messages: list[Message]) -> int: ...


def token_ids_length(encoded: Any) -> int:
    """Return one prompt length from list- or BatchEncoding-style tokenizer output."""
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer output has no input_ids")
        encoded = encoded["input_ids"]
    shape = getattr(encoded, "shape", None)
    if shape is not None:
        if not shape:
            raise ValueError("tokenizer returned scalar input_ids")
        return int(shape[-1])
    if (
        isinstance(encoded, (list, tuple))
        and len(encoded) == 1
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = encoded[0]
    return len(encoded)


class Backend(Protocol):
    name: str

    def generate(self, requests: list[GenRequest]) -> list[GenResult]: ...

    def count_tokens(self, messages: list[Message]) -> int: ...

    def describe(self) -> dict[str, Any]: ...


def request_timing(metrics: Any) -> tuple[float | None, float | None, float | None]:
    """Read both legacy and v1 vLLM per-request timestamp field names."""
    if metrics is None:
        return None, None, None
    scheduled = getattr(metrics, "first_scheduled_time", None)
    scheduled = getattr(metrics, "scheduled_ts", None) if scheduled is None else scheduled
    first = getattr(metrics, "first_token_time", None)
    first = getattr(metrics, "first_token_ts", None) if first is None else first
    finished = getattr(metrics, "finished_time", None)
    finished = getattr(metrics, "last_token_ts", None) if finished is None else finished
    queue = getattr(metrics, "time_in_queue", None)
    if queue is None:
        queued = getattr(metrics, "queued_ts", None)
        queue = scheduled - queued if scheduled is not None and queued is not None else None
    prefill = first - scheduled if first is not None and scheduled is not None else None
    decode = finished - first if finished is not None and first is not None else None
    return (
        float(queue) if queue is not None else None,
        float(prefill) if prefill is not None else None,
        float(decode) if decode is not None else None,
    )


# ------------------------------------------------------------------------------------
# Token counters
# ------------------------------------------------------------------------------------


class CharEstimateCounter:
    """Coarse counter for tests only. The frozen context-fit check never uses it."""

    name = "char_estimate"
    exact = False

    def __init__(self, chars_per_token: float = 3.4) -> None:
        self.chars_per_token = chars_per_token

    def count_tokens(self, messages: list[Message]) -> int:
        return int(len(flat_text(messages)) / self.chars_per_token) + 1


class PinnedTokenizerCounter:
    """Exact counts from the pinned tokenizer revision, through the model's chat template."""

    name = "pinned_tokenizer"
    exact = True

    def __init__(
        self,
        model_id: str,
        tokenizer_revision: str,
        *,
        cache_dir: str | None = None,
        enable_thinking: bool = False,
        local_files_only: bool = True,
    ) -> None:
        from transformers import AutoTokenizer  # noqa: PLC0415 - CUDA-host-only dependency

        self.model_id = model_id
        self.tokenizer_revision = tokenizer_revision
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=tokenizer_revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    def render(self, messages: list[Message]) -> str:
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return str(rendered)

    def count_tokens(self, messages: list[Message]) -> int:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }
        try:
            token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        return token_ids_length(token_ids)

    def describe(self) -> dict[str, Any]:
        return {
            "counter": self.name,
            "model_id": self.model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "enable_thinking": self.enable_thinking,
        }


# ------------------------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------------------------


def _unit(*parts: object) -> float:
    """Deterministic value in [0, 1) from a key tuple."""
    return int(sha256_text("|".join(str(p) for p in parts))[:13], 16) / float(1 << 52)


class FakeBackend:
    """Deterministic stand-in for a model. It is not a model.

    It reads the reference labels to synthesise a controlled error process, so the runner,
    the parser, the ledger, and the screen can be checked end to end on a CPU. No output of
    this class is evidence about any real model.
    """

    name = "fake"

    def __init__(
        self,
        reference: dict[str, int],
        *,
        seed: int = 11,
        error_rate: float = 0.15,
        parse_failure_rate: float = 0.0,
        truncation_rate: float = 0.0,
        missing_id_rate: float = 0.0,
        n_classes: int = 2,
        counter: TokenCounter | None = None,
        seconds_per_call: float = 0.0,
    ) -> None:
        self.reference = reference
        self.seed = seed
        self.error_rate = error_rate
        self.parse_failure_rate = parse_failure_rate
        self.truncation_rate = truncation_rate
        self.missing_id_rate = missing_id_rate
        self.n_classes = n_classes
        self.counter = counter or CharEstimateCounter()
        self.seconds_per_call = seconds_per_call

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        results: list[GenResult] = []
        for request in requests:
            key = f"{self.seed}:{request.call_id}"
            if _unit(key, "truncated") < self.truncation_rate:
                results.append(self._result(request, '{"labels": {"cc', "length"))
                continue
            if _unit(key, "malformed") < self.parse_failure_rate:
                results.append(self._result(request, "Sure! The first one looks toxic.", "stop"))
                continue
            labels: dict[str, int] = {}
            for doc_id in request.target_ids:
                if _unit(key, doc_id, "drop") < self.missing_id_rate:
                    continue
                reference = self.reference[doc_id]
                if _unit(key, doc_id, "err") < self.error_rate:
                    labels[doc_id] = (reference + 1) % self.n_classes
                else:
                    labels[doc_id] = reference
            results.append(self._result(request, json.dumps({"labels": labels}), "stop"))
        if self.seconds_per_call:
            time.sleep(self.seconds_per_call * len(requests))
        return results

    def _result(self, request: GenRequest, text: str, finish_reason: str) -> GenResult:
        return GenResult(
            call_id=request.call_id,
            text=text,
            prompt_tokens=self.counter.count_tokens(request.messages),
            output_tokens=max(1, len(text) // 4),
            finish_reason=finish_reason,
        )

    def count_tokens(self, messages: list[Message]) -> int:
        return self.counter.count_tokens(messages)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "is_a_model": False,
            "warning": "synthetic output; never evidence about any real model",
            "seed": self.seed,
            "error_rate": self.error_rate,
            "parse_failure_rate": self.parse_failure_rate,
        }


class VLLMBackend:
    """Offline vLLM engine on one A100. Only the frozen serving fields are passed."""

    name = "vllm"

    def __init__(
        self,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        serving: dict[str, Any],
        decoding: dict[str, Any],
        *,
        enable_thinking: bool = False,
        max_num_seqs: int | None = None,
        require_batch_invariance: bool = False,
        require_prefix_caching: bool = False,
    ) -> None:
        batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT") == "1"
        if require_batch_invariance and not batch_invariant:
            raise RuntimeError(
                "VLLM_BATCH_INVARIANT=1 must be set before constructing this vLLM backend"
            )
        from vllm import LLM, SamplingParams  # noqa: PLC0415 - CUDA-host-only dependency

        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.serving = serving
        self.decoding = decoding
        self.enable_thinking = enable_thinking
        self.batch_invariant = batch_invariant
        self.batch_invariance_required = require_batch_invariance
        self.prefix_caching_required = require_prefix_caching
        self.max_num_seqs = int(max_num_seqs or serving["max_num_seqs"])
        self._sampling_cls = SamplingParams
        llm_kwargs: dict[str, Any] = dict(
            model=model_id,
            revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            tensor_parallel_size=int(serving["tensor_parallel_size"]),
            dtype=str(serving["dtype"]),
            max_model_len=int(serving["max_model_len"]),
            gpu_memory_utilization=float(serving["gpu_memory_utilization"]),
            max_num_seqs=self.max_num_seqs,
            seed=int(decoding["seed"]),
        )
        if "enable_prefix_caching" in serving:
            llm_kwargs["enable_prefix_caching"] = bool(serving["enable_prefix_caching"])
        self.llm = LLM(**llm_kwargs)
        cache_config = getattr(
            getattr(getattr(self.llm, "llm_engine", None), "vllm_config", None),
            "cache_config",
            None,
        )
        self.prefix_caching_enabled = getattr(
            cache_config, "enable_prefix_caching", serving.get("enable_prefix_caching")
        )
        if require_prefix_caching and self.prefix_caching_enabled is not True:
            raise RuntimeError("vLLM prefix caching was required but is not enabled")
        self.tokenizer = self.llm.get_tokenizer()

    def count_tokens(self, messages: list[Message]) -> int:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }
        try:
            token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        return token_ids_length(token_ids)

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        params = [
            self._sampling_cls(
                temperature=float(self.decoding["temperature"]),
                top_p=float(self.decoding["top_p"]),
                top_k=int(self.decoding["top_k"]),
                max_tokens=request.max_tokens,
                seed=int(self.decoding["seed"]),
            )
            for request in requests
        ]
        conversations = [request.messages for request in requests]
        kwargs = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        try:
            outputs = self.llm.chat(conversations, params, use_tqdm=False, **kwargs)
        except TypeError:
            outputs = self.llm.chat(conversations, params, use_tqdm=False)
        results: list[GenResult] = []
        for request, output in zip(requests, outputs, strict=True):
            queue, prefill, decode = request_timing(getattr(output, "metrics", None))
            results.append(
                GenResult(
                    call_id=request.call_id,
                    text=output.outputs[0].text,
                    prompt_tokens=len(output.prompt_token_ids),
                    output_tokens=len(output.outputs[0].token_ids),
                    finish_reason=str(output.outputs[0].finish_reason),
                    queue_seconds=queue,
                    prefill_seconds=prefill,
                    decode_seconds=decode,
                )
            )
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": str(self.serving["dtype"]),
            "max_model_len": int(self.serving["max_model_len"]),
            "gpu_memory_utilization": float(self.serving["gpu_memory_utilization"]),
            "max_num_seqs": self.max_num_seqs,
            "tensor_parallel_size": int(self.serving["tensor_parallel_size"]),
            "enable_thinking": self.enable_thinking,
            "batch_invariant": self.batch_invariant,
            "enable_prefix_caching": self.prefix_caching_enabled,
            "prefix_caching_required": self.prefix_caching_required,
            "batch_invariance_required": self.batch_invariance_required,
            "seed": int(self.decoding["seed"]),
        }


def host_facts() -> dict[str, Any]:
    """The six host facts the freeze allows Step 1 to record, plus the engine version.

    Every field is a property of the machine, not a design choice. A field that cannot be
    read is recorded as null and never guessed.
    """
    facts: dict[str, Any] = {
        "cuda_version": None,
        "driver_version": None,
        "gpu_name": None,
        "gpu_uuid": None,
        "torch_version": None,
        "installed_engine_version_string": None,
    }
    try:
        import torch  # noqa: PLC0415 - CUDA-host-only dependency

        facts["torch_version"] = str(torch.__version__)
        facts["cuda_version"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            facts["gpu_name"] = str(torch.cuda.get_device_name(0))
            properties = torch.cuda.get_device_properties(0)
            uuid = getattr(properties, "uuid", None)
            facts["gpu_uuid"] = str(uuid) if uuid is not None else None
    except Exception:  # noqa: BLE001 - a missing fact stays null and is never invented
        pass
    try:
        import subprocess  # noqa: PLC0415 - only needed on a CUDA host

        out = subprocess.run(  # noqa: S603 - fixed command, no shell
            ["nvidia-smi", "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            driver, name, uuid = (part.strip() for part in out.stdout.splitlines()[0].split(","))
            facts["driver_version"] = driver
            facts["gpu_name"] = facts["gpu_name"] or name
            facts["gpu_uuid"] = uuid
    except Exception:  # noqa: BLE001 - a missing fact stays null and is never invented
        pass
    try:
        from importlib.metadata import version  # noqa: PLC0415

        facts["installed_engine_version_string"] = version("vllm")
    except Exception:  # noqa: BLE001 - a missing fact stays null and is never invented
        pass
    return facts
