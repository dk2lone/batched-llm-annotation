"""Hashing, JSON and JSONL helpers shared by every stage."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def doc_id_for(text: str) -> str:
    """Stable per-document id. Depends on the text only, never on row order."""
    return "cc" + sha256_text(text)[:14]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform01(key: str) -> float:
    """Deterministic uniform(0,1) draw from a string key."""
    return int(sha256_text(key)[:13], 16) / float(1 << 52)


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def read_jsonl(path: Path, tolerate_truncated_tail: bool = False) -> Iterator[dict[str, Any]]:
    """Read a JSONL file.

    A run that is killed mid-write can leave one partial trailing line. With
    tolerate_truncated_tail the reader drops that line instead of failing, so a
    resumed run recomputes only the job that was in flight.
    """
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            if tolerate_truncated_tail and index == len(lines) - 1:
                return
            raise
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{index + 1} is not a JSON object")
        yield parsed


def append_jsonl(path: Path, rows: Iterable[Any]) -> None:
    """Append rows and force them to disk, so an interrupted run resumes cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
