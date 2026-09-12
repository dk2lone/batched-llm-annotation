"""Hashing and refuse-to-overwrite file helpers.

Kept local instead of importing `batchgate`: the Gate A package is immutable
historical evidence and this package must not depend on its import path.
The canonical-JSON and doc-id conventions match Gate A on purpose, so document
ids computed here are comparable with the pilot's ids.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class RefusedOverwriteError(RuntimeError):
    """Raised instead of destroying an existing frozen artifact."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_id_for(prefix: str, text: str) -> str:
    """Stable per-document id. Same construction as Gate A's `doc_id_for`."""
    return prefix + sha256_text(text)[:14]


def order_key(seed: int, doc_id: str) -> str:
    """Seeded hash order. Same construction as Gate A's `_order_key`."""
    return sha256_text(f"{seed}:{doc_id}")


def refuse_overwrite(path: Path) -> Path:
    if path.exists():
        raise RefusedOverwriteError(
            f"{path} already exists. Frozen artifacts are never overwritten. "
            "Use a new freeze directory."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any, *, allow_existing: bool = False) -> Path:
    if not allow_existing:
        refuse_overwrite(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_text(path: Path, text: str, *, allow_existing: bool = False) -> Path:
    if not allow_existing:
        refuse_overwrite(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Iterable[Any], *, allow_existing: bool = False) -> int:
    if not allow_existing:
        refuse_overwrite(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{index + 1} is not a JSON object")
            yield parsed


def derived_seed(*parts: object) -> int:
    """Deterministic 64-bit seed from any key tuple. Stable across Python versions."""
    return int(sha256_text("|".join(str(p) for p in parts))[:16], 16)
