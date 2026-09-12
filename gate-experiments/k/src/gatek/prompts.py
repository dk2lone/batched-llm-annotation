"""Prompt rendering and auditable row-span extraction.

Each demonstration row carries a unique marker such as ``[ROW 007]``. The feature text
of that row is the substring between the marker and the ``=>`` that introduces the gold
label, so a demonstration representation never includes its own label token.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

DEMO_SEPARATOR = " => "
QUERY_MARKER = "[QUERY]"
ANSWER_SUFFIX = "\nAnswer with exactly one letter, A or B.\nAnswer:"


class SpanError(RuntimeError):
    """Raised when a span cannot be located or verified. Never guess a span."""


@dataclass(frozen=True)
class CharSpan:
    role: str  # "demo" or "query"
    row_index: int  # index into the job's demonstration list, or -1 for the query
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "row_index": self.row_index,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class TokenSpan:
    role: str
    row_index: int
    start: int
    end: int  # exclusive

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "row_index": self.row_index,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    spans: tuple[CharSpan, ...]
    markers: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class _Tokenizer(Protocol):
    is_fast: bool

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


def format_row(values: np.ndarray, names: tuple[str, ...], number_format: str) -> str:
    """Render one row as ``f0=+000.123, f1=-001.045``."""
    cells = [
        f"{name}={number_format.format(float(v))}" for name, v in zip(names, values, strict=True)
    ]
    widths = {len(c.split("=", 1)[1]) for c in cells}
    if len(widths) != 1:
        raise ValueError(f"number_format {number_format!r} is not fixed width: widths {widths}")
    return ", ".join(cells)


def demo_marker(index: int) -> str:
    return f"[ROW {index:03d}]"


def render_prompt(
    demo_x: np.ndarray,
    demo_letters: list[str],
    query_x: np.ndarray,
    names: tuple[str, ...],
    number_format: str,
) -> RenderedPrompt:
    """Build the user-message body and the feature-only character spans inside it."""
    if len(demo_x) != len(demo_letters):
        raise ValueError("demonstration rows and labels disagree in length")
    header = (
        f"Task: binary classification. Each row has {len(names)} numeric features "
        f"named {names[0]}..{names[-1]}.\n"
        "Each labelled example ends with its class letter, A or B.\n"
        "Predict the class letter of the final query row.\n\n"
    )
    parts = [header]
    spans: list[CharSpan] = []
    markers: list[str] = []
    cursor = len(header)
    for i, (row, letter) in enumerate(zip(demo_x, demo_letters, strict=True)):
        marker = demo_marker(i)
        markers.append(marker)
        feature_text = format_row(row, names, number_format)
        line = f"{marker} {feature_text}{DEMO_SEPARATOR}{letter}\n"
        start = cursor + len(marker) + 1
        spans.append(CharSpan("demo", i, start, start + len(feature_text)))
        parts.append(line)
        cursor += len(line)
    query_text = format_row(query_x, names, number_format)
    query_line = f"\n{QUERY_MARKER} {query_text}"
    markers.append(QUERY_MARKER)
    qstart = cursor + 1 + len(QUERY_MARKER) + 1
    spans.append(CharSpan("query", -1, qstart, qstart + len(query_text)))
    parts.append(query_line)
    parts.append(ANSWER_SUFFIX)
    text = "".join(parts)
    for span in spans:
        if not text[span.start : span.end].startswith(f"{names[0]}="):
            raise SpanError(f"character span {span} does not start at a feature cell")
    return RenderedPrompt(text=text, spans=tuple(spans), markers=tuple(markers))


def relocate_spans(final_text: str, body: RenderedPrompt) -> tuple[CharSpan, ...]:
    """Find the feature spans again inside a chat-template-wrapped prompt.

    Each marker must occur exactly once, and the text after it must match the body.
    """
    out: list[CharSpan] = []
    for span, marker in zip(body.spans, body.markers, strict=True):
        if final_text.count(marker) != 1:
            raise SpanError(
                f"marker {marker!r} occurs {final_text.count(marker)} times; expected 1"
            )
        marker_end = final_text.index(marker) + len(marker)
        start = marker_end + 1
        expected = body.text[span.start : span.end]
        got = final_text[start : start + len(expected)]
        if got != expected:
            raise SpanError(f"feature text after {marker!r} changed under the chat template")
        out.append(CharSpan(span.role, span.row_index, start, start + len(expected)))
    return tuple(out)


def token_spans(
    tokenizer: _Tokenizer, text: str, spans: tuple[CharSpan, ...]
) -> tuple[list[TokenSpan], list[int]]:
    """Map character spans to token spans through the tokenizer offset mapping.

    Returns the token spans and the full input id list.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise SpanError(
            "this tokenizer is not a fast tokenizer and cannot return offset mappings; "
            "install `tokenizers`-backed weights or set a tokenizer that supports "
            "return_offsets_mapping. Refusing to guess row spans."
        )
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    input_ids = list(enc["input_ids"])
    if offsets is None or len(offsets) != len(input_ids):
        raise SpanError("tokenizer returned no usable offset mapping; refusing to guess row spans")
    out: list[TokenSpan] = []
    for span in spans:
        idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < span.end and b > span.start]
        if not idx:
            raise SpanError(f"no tokens found for {span}")
        if idx != list(range(idx[0], idx[-1] + 1)):
            raise SpanError(f"token span for {span} is not contiguous")
        out.append(TokenSpan(span.role, span.row_index, idx[0], idx[-1] + 1))
    check_spans_disjoint(out)
    return out, input_ids


def check_spans_disjoint(spans: list[TokenSpan]) -> None:
    """Verify that every token span is nonempty and that no two of them overlap."""
    ordered = sorted(spans, key=lambda s: s.start)
    previous_end = -1
    for span in ordered:
        if span.end <= span.start:
            raise SpanError(f"empty token span {span}")
        if span.start < previous_end:
            raise SpanError(f"overlapping token spans near {span}")
        previous_end = span.end


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
