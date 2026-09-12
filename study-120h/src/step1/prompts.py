"""Frozen prompt construction and the frozen strict parser.

Every string comes from the sealed protocol, except the four fixed scaffold lines below.
Those lines are not in the lock, but the protocol declares the Civil Comments prompt
"byte-identical to gate-experiments/a/src/batchgate/prompts.py at the pilot revision", so
they are copied from the pilot and pinned by a test. The same scaffold is reused for the
other tasks because the protocol calls them `parallel_to_civil`.

The parser is the pilot's `parse_labels` with one frozen extension: the permitted integer
set widens to the task's class set. No other rule changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

Message = dict[str, str]

DOCUMENTS_HEADER = "Documents:"
JOINT_CLOSING = "Return one label for every document id listed above, and for no other id."
ANSWER_LINE = "Answer with this JSON object and nothing else:"
SINGLE_TARGET_INSTRUCTION = (
    'Label only the document with id "{target_doc_id}". '
    "Ignore every other document. Return no other id."
)

#: Conditions that request exactly one label. From the protocol `single_target_applies_to`.
SINGLE_TARGET_CONDITIONS = frozenset({"exact_context", "solo", "filler", "serial_exact"})
JOINT_CONDITIONS = frozenset({"joint", "serial_joint"})


def render_doc(doc_id: str, text: str) -> str:
    return f'<doc id="{doc_id}">\n{text}\n</doc>'


def _documents_block(docs: list[tuple[str, str]]) -> str:
    return "\n".join(render_doc(doc_id, text) for doc_id, text in docs)


def _reject_duplicates(ids: list[str]) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError("a prompt cannot contain the same document twice")


def build_joint(spec: dict[str, Any], docs: list[tuple[str, str]]) -> list[Message]:
    """One call that shows every document and requests one label per document id."""
    ids = [doc_id for doc_id, _ in docs]
    _reject_duplicates(ids)
    instruction = str(spec["joint_instruction"]).format(n=len(docs))
    user = (
        f"{instruction}\n\n"
        f"{spec['definition']}\n\n"
        f"{DOCUMENTS_HEADER}\n"
        f"{_documents_block(docs)}\n\n"
        f"{JOINT_CLOSING}\n"
        f"{ANSWER_LINE}\n"
        '{"labels": {' + ", ".join(f'"{i}": 0' for i in ids) + "}}"
    )
    return [{"role": "system", "content": str(spec["system"])}, {"role": "user", "content": user}]


def build_single(
    spec: dict[str, Any], docs: list[tuple[str, str]], target_id: str
) -> list[Message]:
    """One call that shows every document and requests exactly one label."""
    ids = [doc_id for doc_id, _ in docs]
    _reject_duplicates(ids)
    if target_id not in ids:
        raise ValueError("target_id is not present in docs")
    user = (
        f"{spec['single_instruction']}\n\n"
        f"{spec['definition']}\n\n"
        f"{DOCUMENTS_HEADER}\n"
        f"{_documents_block(docs)}\n\n"
        f"{SINGLE_TARGET_INSTRUCTION.format(target_doc_id=target_id)}\n"
        f"{ANSWER_LINE}\n"
        '{"labels": {"' + target_id + '": 0}}'
    )
    return [{"role": "system", "content": str(spec["system"])}, {"role": "user", "content": user}]


def build_messages(
    spec: dict[str, Any], condition: str, docs: list[tuple[str, str]], target_ids: list[str]
) -> list[Message]:
    """Dispatch on the frozen condition name. An unknown condition is an error."""
    if condition in JOINT_CONDITIONS:
        if len(target_ids) != len(docs):
            raise ValueError(f"{condition} must request one label per shown document")
        return build_joint(spec, docs)
    if condition in SINGLE_TARGET_CONDITIONS:
        if len(target_ids) != 1:
            raise ValueError(f"{condition} must request exactly one label")
        return build_single(spec, docs, target_ids[0])
    raise ValueError(f"unknown condition {condition!r}")


def flat_text(messages: list[Message]) -> str:
    """The frozen flat rendering used for hashing: `role: content`, joined by a blank line."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


# ------------------------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------------------------


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


class _DuplicateKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateKey(key)
        seen[key] = value
    return seen


def parse_labels(raw: str, requested_ids: list[str], permitted: tuple[int, ...]) -> ParseResult:
    """Parse one model output strictly.

    Malformed output is a parse failure. The only permitted repair is removal of a single
    Markdown code fence, and it is recorded in `notes`. Duplicate JSON keys fail the whole
    call, because the intended mapping is unrecoverable. Booleans are rejected. A digit
    string is accepted and noted. Unknown ids are recorded and never scored.
    """
    allowed = set(permitted)
    allowed_text = {str(value) for value in permitted}
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
        elif isinstance(value, int) and value in allowed:
            labels[str(key)] = value
        elif isinstance(value, str) and value in allowed_text:
            labels[str(key)] = int(value)
            if "string_label" not in extra_notes:
                extra_notes.append("string_label")
        else:
            bad.append(str(key))
    if bad:
        return ParseResult(
            status="bad_value", labels={}, bad_ids=tuple(sorted(bad)), notes=tuple(extra_notes)
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
    """Per-row status. A call-level failure applies to every target of the call."""
    if not result.ok:
        return result.status
    if doc_id in result.missing_ids:
        return "missing_id"
    return "ok"
