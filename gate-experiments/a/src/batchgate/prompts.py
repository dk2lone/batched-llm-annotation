"""Prompt construction and the strict label parser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

Message = dict[str, str]

SYSTEM = (
    "You are a careful content moderation annotator. "
    "You answer with one JSON object and nothing else."
)

DEFINITION = (
    "A document is toxic if a reasonable reader would call it rude, disrespectful, "
    "or likely to make someone leave the conversation. Use 1 for toxic and 0 for not toxic."
)


def render_doc(doc_id: str, text: str) -> str:
    return f'<doc id="{doc_id}">\n{text}\n</doc>'


def _documents_block(docs: list[tuple[str, str]]) -> str:
    return "\n".join(render_doc(doc_id, text) for doc_id, text in docs)


def build_joint_prompt(docs: list[tuple[str, str]]) -> list[Message]:
    """Ask for one label per document id. This is the batched condition."""
    ids = [doc_id for doc_id, _ in docs]
    _reject_duplicates(ids)
    user = (
        f"Label each of the {len(docs)} documents below for toxicity.\n\n"
        f"{DEFINITION}\n\n"
        "Documents:\n"
        f"{_documents_block(docs)}\n\n"
        "Return one label for every document id listed above, and for no other id.\n"
        "Answer with this JSON object and nothing else:\n"
        '{"labels": {' + ", ".join(f'"{i}": 0' for i in ids) + "}}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build_single_prompt(docs: list[tuple[str, str]], target_id: str) -> list[Message]:
    """Ask for exactly one label. Used for solo (1 doc) and filler (k docs)."""
    ids = [doc_id for doc_id, _ in docs]
    _reject_duplicates(ids)
    if target_id not in ids:
        raise ValueError("target_id is not present in docs")
    user = (
        "Label one document for toxicity.\n\n"
        f"{DEFINITION}\n\n"
        "Documents:\n"
        f"{_documents_block(docs)}\n\n"
        f'Label only the document with id "{target_id}". '
        "Ignore every other document. Return no other id.\n"
        "Answer with this JSON object and nothing else:\n"
        '{"labels": {"' + target_id + '": 0}}'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _reject_duplicates(ids: list[str]) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError("a prompt cannot contain the same document twice")


def prompt_text(messages: list[Message]) -> str:
    """Flat rendering used for hashing and for the character-based token estimate."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


@dataclass(frozen=True)
class ParseResult:
    status: str
    labels: dict[str, int]
    missing_ids: tuple[str, ...] = ()
    unknown_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()
    bad_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _strip_fence(text: str) -> tuple[str, tuple[str, ...]]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped, ()
    body = stripped[3:]
    if body.lower().startswith("json"):
        body = body[4:]
    end = body.rfind("```")
    if end == -1:
        return stripped, ()
    return body[:end].strip(), ("stripped_code_fence",)


def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateKey(key)
        seen[key] = value
    return seen


class _DuplicateKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def parse_labels(raw: str, requested_ids: list[str]) -> ParseResult:
    """Parse model output strictly.

    Malformed output is a parse failure. The only repair is removal of a single
    Markdown code fence, and that is recorded in notes. Duplicate ids anywhere in
    the JSON fail the whole job, because the intended mapping is not recoverable.
    """
    body, notes = _strip_fence(raw)
    if not body.startswith("{"):
        return ParseResult(status="no_json", labels={}, notes=notes)
    try:
        parsed = json.loads(body, object_pairs_hook=_pairs_hook)
    except _DuplicateKey as exc:
        return ParseResult(status="duplicate_ids", labels={}, duplicate_ids=(exc.key,), notes=notes)
    except json.JSONDecodeError:
        return ParseResult(status="json_error", labels={}, notes=notes)
    if not isinstance(parsed, dict):
        return ParseResult(status="not_object", labels={}, notes=notes)
    raw_labels = parsed.get("labels")
    if not isinstance(raw_labels, dict):
        return ParseResult(status="missing_labels", labels={}, notes=notes)

    labels: dict[str, int] = {}
    bad: list[str] = []
    extra_notes = list(notes)
    for key, value in raw_labels.items():
        if isinstance(value, bool):
            bad.append(str(key))
        elif isinstance(value, int) and value in (0, 1):
            labels[str(key)] = value
        elif isinstance(value, str) and value in ("0", "1"):
            labels[str(key)] = int(value)
            if "string_label" not in extra_notes:
                extra_notes.append("string_label")
        else:
            bad.append(str(key))
    if bad:
        return ParseResult(
            status="bad_value",
            labels={},
            bad_ids=tuple(sorted(bad)),
            notes=tuple(extra_notes),
        )

    requested = list(requested_ids)
    missing = tuple(i for i in requested if i not in labels)
    unknown = tuple(sorted(set(labels) - set(requested)))
    if unknown:
        extra_notes.append("unknown_ids_present")
    status = "ok" if len(missing) < len(requested) else "missing_labels"
    return ParseResult(
        status=status,
        labels=labels,
        missing_ids=missing,
        unknown_ids=unknown,
        notes=tuple(extra_notes),
    )


def target_status(result: ParseResult, doc_id: str) -> str:
    """Per-document status. A job-level failure applies to every target in the job."""
    if not result.ok:
        return result.status
    if doc_id in result.missing_ids:
        return "missing_id"
    return "ok"
