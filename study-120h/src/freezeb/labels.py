"""Strict first-response parsers and reference-relative item errors for Freeze B."""

from __future__ import annotations

import json
from typing import Any

NER_TYPES = (
    "PER", "ORG", "LOC", "ANIM", "BIO", "CEL", "DIS", "EVE", "FOOD",
    "INST", "MEDIA", "MYTH", "PLANT", "TIME", "VEHI",
)


class ParseError(ValueError):
    """The first response does not satisfy the frozen grammar."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ParseError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def entity_spans(tags: list[int]) -> tuple[tuple[int, int, str], ...]:
    """Convert integer BIO tags to half-open token spans.

    An I-tag without a matching open entity starts a new span. This handles malformed
    BIO sequences deterministically without making the entire call disappear.
    """
    spans: list[tuple[int, int, str]] = []
    start = -1
    kind = ""
    for index, tag in enumerate([*tags, 0]):
        if type(tag) is not int or not 0 <= tag <= 30:
            raise ParseError("NER tags must be integers from 0 through 30")
        next_kind = NER_TYPES[(tag - 1) // 2] if tag else ""
        continues = tag != 0 and tag % 2 == 0 and next_kind == kind
        if start >= 0 and not continues:
            spans.append((start, index, kind))
            start = -1
            kind = ""
        if tag and start < 0:
            start, kind = index, next_kind
    return tuple(spans)


def normalize_label(value: Any, task: str, *, n_tokens: int | None = None) -> Any:
    if task in {"civil_comments", "imdb"}:
        classes = 2
    elif task == "ag_news":
        classes = 4
    elif task == "multi_nli":
        classes = 3
    else:
        classes = 0
    if classes:
        if type(value) is not int or not 0 <= value < classes:
            raise ParseError(f"{task} label must be an integer from 0 through {classes - 1}")
        return value
    if task == "go_emotions":
        if not isinstance(value, list) or not value or any(
            type(label) is not int or not 0 <= label <= 27 for label in value
        ):
            raise ParseError("GoEmotions requires a nonempty list of label IDs 0 through 27")
        if len(set(value)) != len(value):
            raise ParseError("GoEmotions labels must be unique")
        return tuple(sorted(value))
    if task == "multinerd":
        if n_tokens is None:
            raise ValueError("MultiNERD parsing requires the reference token count")
        if not isinstance(value, list) or len(value) != n_tokens:
            raise ParseError("MultiNERD requires one integer BIO tag per input token")
        return entity_spans(value)
    raise ValueError(f"unknown task {task}")


def parse_first_response(
    raw: str, doc_ids: list[str], task: str, *, token_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    """Require exactly {"labels": {doc_id: label}} and exactly the requested IDs."""
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise ParseError("response is not a single JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"labels"}:
        raise ParseError("response must contain only the labels key")
    labels = payload["labels"]
    if (
        not isinstance(labels, dict)
        or set(labels) != set(doc_ids)
        or len(doc_ids) != len(set(doc_ids))
    ):
        raise ParseError("response IDs must match requested IDs exactly")
    return {
        doc_id: normalize_label(
            labels[doc_id], task, n_tokens=None if token_counts is None else token_counts[doc_id]
        )
        for doc_id in doc_ids
    }


def disagreement(predicted: Any, reference: Any, task: str, *, n_tokens: int | None = None) -> int:
    return int(
        normalize_label(predicted, task, n_tokens=n_tokens)
        != normalize_label(reference, task, n_tokens=n_tokens)
    )
