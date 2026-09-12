"""Prompt rendering and span auditing."""

from __future__ import annotations

import re

import numpy as np
import pytest

from gatek.prompts import (
    DEMO_SEPARATOR,
    CharSpan,
    SpanError,
    TokenSpan,
    check_spans_disjoint,
    format_row,
    relocate_spans,
    render_prompt,
    token_spans,
)

NAMES = ("f0", "f1", "f2", "f3")
DEMO = np.array([[1.5, -2.25, 0.0, 0.0], [-0.5, 3.0, 0.0, 0.0]])
QUERY = np.array([0.25, 0.75, 0.0, 0.0])
FMT = "{:+08.3f}"


class FakeFastTokenizer:
    """Splits on whitespace runs and reports real character offsets."""

    is_fast = True

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [(m.start(), m.end()) for m in matches],
        }


class FakeSlowTokenizer:
    is_fast = False

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        raise AssertionError("should never be called")


def test_format_row_is_fixed_width() -> None:
    row = format_row(DEMO[0], NAMES, FMT)
    widths = {len(cell.split("=", 1)[1]) for cell in row.split(", ")}
    assert widths == {8}


def test_format_row_rejects_a_variable_width_format() -> None:
    with pytest.raises(ValueError):
        format_row(np.array([1.0, -22.0]), ("f0", "f1"), "{:g}")


def test_demo_span_excludes_the_gold_label() -> None:
    rendered = render_prompt(DEMO, ["A", "B"], QUERY, NAMES, FMT)
    for span in rendered.spans:
        text = rendered.text[span.start : span.end]
        assert DEMO_SEPARATOR not in text
        assert not text.endswith("A")
        assert text.startswith("f0=")


def test_query_span_covers_only_the_query_features() -> None:
    rendered = render_prompt(DEMO, ["A", "B"], QUERY, NAMES, FMT)
    query_span = [s for s in rendered.spans if s.role == "query"][0]
    assert rendered.text[query_span.start : query_span.end] == format_row(QUERY, NAMES, FMT)


def test_markers_are_unique() -> None:
    rendered = render_prompt(DEMO, ["A", "B"], QUERY, NAMES, FMT)
    for marker in rendered.markers:
        assert rendered.text.count(marker) == 1


def test_relocate_spans_survives_a_chat_wrapper() -> None:
    rendered = render_prompt(DEMO, ["A", "B"], QUERY, NAMES, FMT)
    wrapped = f"<|im_start|>user\n{rendered.text}<|im_end|>\n<|im_start|>assistant\n"
    moved = relocate_spans(wrapped, rendered)
    for original, new in zip(rendered.spans, moved, strict=True):
        assert wrapped[new.start : new.end] == rendered.text[original.start : original.end]


def test_relocate_spans_fails_when_the_text_changed() -> None:
    rendered = render_prompt(DEMO, ["A", "B"], QUERY, NAMES, FMT)
    broken = rendered.text.replace("f0=", "f0:", 1)
    with pytest.raises(SpanError):
        relocate_spans(broken, rendered)


def test_token_spans_are_nonempty_and_disjoint() -> None:
    rendered = render_prompt(DEMO, ["A", "B"], QUERY, NAMES, FMT)
    spans, input_ids = token_spans(FakeFastTokenizer(), rendered.text, rendered.spans)
    assert len(spans) == len(rendered.spans)
    assert all(s.end > s.start for s in spans)
    assert len(input_ids) > 0
    check_spans_disjoint(spans)


def test_slow_tokenizer_raises_an_actionable_error() -> None:
    rendered = render_prompt(DEMO, ["A", "B"], QUERY, NAMES, FMT)
    with pytest.raises(SpanError, match="offset"):
        token_spans(FakeSlowTokenizer(), rendered.text, rendered.spans)


def test_overlapping_spans_are_rejected() -> None:
    with pytest.raises(SpanError, match="overlapping"):
        check_spans_disjoint([TokenSpan("demo", 0, 0, 5), TokenSpan("demo", 1, 3, 8)])


def test_empty_span_is_rejected() -> None:
    with pytest.raises(SpanError, match="empty"):
        check_spans_disjoint([TokenSpan("demo", 0, 4, 4)])


def test_char_span_dict_is_serializable() -> None:
    assert CharSpan("demo", 0, 1, 2).as_dict() == {
        "role": "demo",
        "row_index": 0,
        "start": 1,
        "end": 2,
    }
