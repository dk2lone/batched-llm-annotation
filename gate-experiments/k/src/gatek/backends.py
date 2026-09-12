"""Inference backends.

`MockBackend` is deterministic, runs on CPU and never touches the network. It exists to
exercise the plumbing, not to produce evidence.

`TransformersBackend` imports torch and transformers lazily, so the analysis stages and
the tests never pull in a model stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .prompts import (
    RenderedPrompt,
    SpanError,
    TokenSpan,
    prompt_sha256,
    relocate_spans,
    render_prompt,
    token_spans,
)

MOCK_MAX_DIM = 64


@dataclass(frozen=True)
class BackendResult:
    demo_reps: np.ndarray  # (n_demo_kept, hidden)
    query_rep: np.ndarray  # (hidden,)
    letter: str | None
    parse_status: str
    label_scores: dict[str, float]
    margin: float | None
    n_prompt_tokens: int
    token_source: str
    prompt_sha256: str
    prediction_method: str
    attention_row_mass: list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ContextBudgetError(RuntimeError):
    """Raised before inference when a prompt exceeds the configured token budget."""


def load_tokenizer(
    model_id: str,
    revision: str | None,
    cache_dir: str | None,
    *,
    local_files_only: bool = False,
) -> Any:
    """Load a tokenizer and recover an official standalone chat template if needed."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer

    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import (  # noqa: PLC0415
        EntryNotFoundError,
        LocalEntryNotFoundError,
    )

    try:
        template_path = hf_hub_download(
            repo_id=model_id,
            filename="chat_template.json",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return tokenizer
    payload = json.loads(Path(template_path).read_text(encoding="utf-8"))
    template = payload.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"{model_id} has no usable chat_template in chat_template.json")
    tokenizer.chat_template = template
    return tokenizer


def _body(
    demo_x: np.ndarray,
    demo_letters: list[str],
    query_x: np.ndarray,
    names: tuple[str, ...],
    number_format: str,
) -> RenderedPrompt:
    return render_prompt(demo_x, demo_letters, query_x, names, number_format)


class MockBackend:
    """Deterministic stand-in. Representations are a fixed nonlinear map of the raw row."""

    name = "mock"

    def __init__(self, seed: int, repr_dim: int, number_format: str) -> None:
        rng = np.random.default_rng(seed)
        self._w = rng.normal(0.0, 1.0 / np.sqrt(MOCK_MAX_DIM), size=(repr_dim, MOCK_MAX_DIM))
        self._number_format = number_format

    def _rep(self, rows: np.ndarray) -> np.ndarray:
        padded = np.zeros((len(rows), MOCK_MAX_DIM), dtype=np.float64)
        padded[:, : rows.shape[1]] = rows
        return np.tanh(padded @ self._w.T)

    def infer(
        self,
        demo_x: np.ndarray,
        demo_letters: list[str],
        query_x: np.ndarray,
        names: tuple[str, ...],
        capture_attention: bool = False,
    ) -> BackendResult:
        body = _body(demo_x, demo_letters, query_x, names, self._number_format)
        demo_reps = self._rep(demo_x)
        query_rep = self._rep(query_x[None, :])[0]
        d = np.linalg.norm(demo_reps - query_rep[None, :], axis=1)
        nearest = int(np.argmin(d))
        letter = demo_letters[nearest]
        gap = float(np.sort(d)[1] - d[nearest]) if len(d) > 1 else 0.0
        p = 1.0 / (1.0 + np.exp(-gap))
        scores = {"A": p if letter == "A" else 1.0 - p, "B": p if letter == "B" else 1.0 - p}
        mass = None
        if capture_attention:
            w = np.exp(-d)
            mass = list((w / w.sum()).astype(float))
        return BackendResult(
            demo_reps=demo_reps,
            query_rep=query_rep,
            letter=letter,
            parse_status="ok",
            label_scores=scores,
            margin=float(scores["A"] - scores["B"]),
            n_prompt_tokens=len(body.text) // 4 + 1,
            token_source="char_proxy",
            prompt_sha256=prompt_sha256(body.text),
            prediction_method="mock_nearest_neighbour",
            attention_row_mass=mass,
            meta={"backend": "mock"},
        )


class TransformersBackend:
    """Open-weight causal model through transformers, one prompt at a time."""

    name = "transformers"

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        dtype: str,
        device_map: str,
        attn_implementation: str,
        hidden_layers: list[int],
        label_tokens: list[str],
        max_prompt_tokens: int,
        number_format: str,
        enable_thinking: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        # transformers >= 4.56 takes `dtype`; the older `torch_dtype` name is deprecated.
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )

        self._torch = torch
        self._number_format = number_format
        self._hidden_layers = hidden_layers
        self._max_prompt_tokens = max_prompt_tokens
        self._enable_thinking = enable_thinking
        self._model_id = model_id
        self.tokenizer = load_tokenizer(
            model_id,
            revision,
            cache_dir,
            local_files_only=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise SpanError(
                f"{model_id} did not load a fast tokenizer, so offset mappings are not "
                "available and row spans cannot be verified. Refusing to guess."
            )
        model_config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        model_loader = (
            AutoModelForImageTextToText
            if getattr(model_config, "model_type", None) == "mistral3"
            else AutoModelForCausalLM
        )
        self.model = model_loader.from_pretrained(
            model_id,
            revision=revision,
            config=model_config,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self.model.eval()
        backbone: Any = getattr(self.model, "model", None)
        if hasattr(backbone, "language_model"):
            backbone = backbone.language_model
        self._decoder_layers: Any = getattr(backbone, "layers", None)
        if self._decoder_layers is None:
            raise RuntimeError(f"cannot locate decoder layers for {model_id}")
        n_layers = len(self._decoder_layers)
        invalid_layers = [layer for layer in hidden_layers if layer < 0 or layer >= n_layers]
        if invalid_layers:
            raise ValueError(
                f"hidden layer indices {invalid_layers} are outside decoder range 0..{n_layers - 1}"
            )
        self._attn_implementation = attn_implementation
        self._label_ids = {tok: self._single_token_id(tok) for tok in label_tokens}
        self._thinking_supported = True

    def _single_token_id(self, text: str) -> int:
        """Resolve a label letter to exactly one token id, trying the space-prefixed form."""
        for candidate in (text, f" {text}"):
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        raise ValueError(f"label {text!r} is not a single token for this tokenizer")

    def _chat_text(self, body_text: str) -> str:
        if getattr(self.tokenizer, "chat_template", None) is None:
            self._thinking_supported = False
            return body_text
        messages = [{"role": "user", "content": body_text}]
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self._enable_thinking,
                )
            )
        except TypeError:
            self._thinking_supported = False
            return str(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

    def _pool(self, hidden: dict[int, Any], spans: list[TokenSpan]) -> np.ndarray:
        """Mean-pool the chosen layers over each row's feature tokens."""
        vectors = []
        for span in spans:
            per_layer = [
                hidden[layer][0, span.start : span.end, :].float().mean(dim=0)
                for layer in self._hidden_layers
            ]
            vectors.append(self._torch.stack(per_layer).mean(dim=0))
        return self._torch.stack(vectors).cpu().numpy().astype(np.float64)

    def infer(
        self,
        demo_x: np.ndarray,
        demo_letters: list[str],
        query_x: np.ndarray,
        names: tuple[str, ...],
        capture_attention: bool = False,
    ) -> BackendResult:
        torch = self._torch
        if capture_attention and self._attn_implementation != "eager":
            raise RuntimeError(
                "attention capture needs attn_implementation='eager'; sdpa and flash kernels "
                "do not return attention weights. Re-run the attention jobs with "
                "--attn-implementation eager."
            )
        body = _body(demo_x, demo_letters, query_x, names, self._number_format)
        text = self._chat_text(body.text)
        spans = relocate_spans(text, body)
        tspans, input_ids = token_spans(self.tokenizer, text, spans)
        if len(input_ids) > self._max_prompt_tokens:
            raise ContextBudgetError(
                f"prompt is {len(input_ids)} tokens, above max_prompt_tokens="
                f"{self._max_prompt_tokens}; reduce n_demo or dim"
            )
        ids = torch.tensor([input_ids], device=self.model.device)
        captured: dict[int, Any] = {}
        handles = []
        for layer in self._hidden_layers:
            def capture(
                _module: Any, args: tuple[Any, ...], *, layer_index: int = layer
            ) -> None:
                captured[layer_index] = args[0]

            handles.append(self._decoder_layers[layer].register_forward_pre_hook(capture))
        with torch.no_grad():
            try:
                out = self.model(
                    ids,
                    output_hidden_states=False,
                    output_attentions=capture_attention,
                    use_cache=False,
                    logits_to_keep=1,
                )
            finally:
                for handle in handles:
                    handle.remove()
        missing_layers = [layer for layer in self._hidden_layers if layer not in captured]
        if missing_layers:
            raise RuntimeError(f"failed to capture hidden layers {missing_layers}")
        demo_spans = [s for s in tspans if s.role == "demo"]
        query_spans = [s for s in tspans if s.role == "query"]
        demo_reps = self._pool(captured, demo_spans)
        query_rep = self._pool(captured, query_spans)[0]

        logits = out.logits[0, -1, :].float()
        label_ids = list(self._label_ids.values())
        restricted = torch.softmax(logits[label_ids], dim=-1).cpu().numpy()
        scores = {tok: float(p) for tok, p in zip(self._label_ids, restricted, strict=True)}
        greedy = self.tokenizer.decode([int(logits.argmax().item())]).strip()
        parse_status = "ok" if greedy in self._label_ids else "off_label"
        letter = greedy if parse_status == "ok" else max(scores, key=lambda k: scores[k])
        keys = list(scores)
        margin = float(scores[keys[0]] - scores[keys[1]]) if len(keys) == 2 else None

        mass = None
        if capture_attention and out.attentions is not None:
            attn = out.attentions[-1][0].mean(dim=0)[-1].float().cpu().numpy()
            mass = [float(attn[s.start : s.end].sum()) for s in demo_spans]

        return BackendResult(
            demo_reps=demo_reps,
            query_rep=query_rep,
            letter=letter,
            parse_status=parse_status,
            label_scores=scores,
            margin=margin,
            n_prompt_tokens=len(input_ids),
            token_source="tokenizer",
            prompt_sha256=prompt_sha256(text),
            prediction_method="restricted_label_token_softmax+greedy_strict",
            attention_row_mass=mass,
            meta={
                "backend": "transformers",
                "model_id": self._model_id,
                "revision": self.model.config._commit_hash,
                "hidden_layers": self._hidden_layers,
                "representation_capture": "decoder_input_pre_hook",
                "thinking_flag_supported": self._thinking_supported,
                "greedy_token": greedy,
            },
        )
