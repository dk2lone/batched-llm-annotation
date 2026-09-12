"""Seal a freeze directory and validate every frozen hash.

`seal` writes MANIFEST.sha256 and makes every file read-only. `verify` recomputes
every hash, rejects any added or removed file, and rejects any writable file.
Neither command ever repairs a mismatch; a mismatch means the freeze is void and a
new freeze directory is required.
"""

from __future__ import annotations

import datetime as dt
import os
import stat
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_text, write_json, write_text

MANIFEST_NAME = "MANIFEST.sha256"
SEAL_NAME = "SEAL.json"
READ_ONLY = 0o444


def _walk(freeze_dir: Path) -> list[Path]:
    skip = {MANIFEST_NAME, SEAL_NAME}
    return sorted(
        path
        for path in freeze_dir.rglob("*")
        if path.is_file() and path.name not in skip and "__pycache__" not in path.parts
    )


def _manifest_lines(freeze_dir: Path) -> list[str]:
    return [
        f"{sha256_file(path)}  {path.relative_to(freeze_dir).as_posix()}"
        for path in _walk(freeze_dir)
    ]


def seal(freeze_dir: Path) -> dict[str, Any]:
    """Write the hash manifest, then remove write permission from every sealed file."""
    if (freeze_dir / MANIFEST_NAME).exists():
        raise FileExistsError(
            f"{freeze_dir / MANIFEST_NAME} exists. A sealed freeze is never resealed."
        )
    lines = _manifest_lines(freeze_dir)
    body = "\n".join(lines) + "\n"
    write_text(freeze_dir / MANIFEST_NAME, body)
    seal_record = {
        "sealed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "n_files": len(lines),
        "manifest_sha256": sha256_text(body),
        "permissions": "0444 on every sealed file",
    }
    write_json(freeze_dir / SEAL_NAME, seal_record)
    for path in _walk(freeze_dir) + [freeze_dir / MANIFEST_NAME, freeze_dir / SEAL_NAME]:
        os.chmod(path, READ_ONLY)
    return seal_record


def verify(freeze_dir: Path) -> dict[str, Any]:
    """Recompute every hash. Return a report whose `ok` field is the exit condition."""
    manifest = freeze_dir / MANIFEST_NAME
    problems: list[str] = []
    if not manifest.exists():
        return {
            "ok": False,
            "freeze_dir": str(freeze_dir),
            "problems": [f"{MANIFEST_NAME} is missing; the freeze was never sealed"],
            "n_files_checked": 0,
        }

    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        expected[rel] = digest

    present = {path.relative_to(freeze_dir).as_posix(): path for path in _walk(freeze_dir)}

    for rel in sorted(set(expected) - set(present)):
        problems.append(f"missing file: {rel}")
    for rel in sorted(set(present) - set(expected)):
        problems.append(f"unlisted file added after sealing: {rel}")

    mutated: list[str] = []
    writable: list[str] = []
    for rel in sorted(set(expected) & set(present)):
        path = present[rel]
        if sha256_file(path) != expected[rel]:
            mutated.append(rel)
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            writable.append(rel)
    for control_name in (MANIFEST_NAME, SEAL_NAME):
        control = freeze_dir / control_name
        if not control.exists():
            problems.append(f"missing seal control file: {control_name}")
        elif stat.S_IMODE(control.stat().st_mode) & 0o222:
            writable.append(control_name)
    problems += [f"hash mismatch: {rel}" for rel in mutated]
    problems += [f"file is writable, so the freeze is not immutable: {rel}" for rel in writable]

    manifest_hash = sha256_text(manifest.read_text(encoding="utf-8"))
    seal_path = freeze_dir / SEAL_NAME
    if seal_path.exists():
        import json  # noqa: PLC0415

        recorded = json.loads(seal_path.read_text(encoding="utf-8")).get("manifest_sha256")
        if recorded != manifest_hash:
            problems.append("MANIFEST.sha256 was edited after sealing")

    problems += [
        f"file is writable, so the freeze is not immutable: {rel}"
        for rel in writable
        if f"file is writable, so the freeze is not immutable: {rel}" not in problems
    ]

    return {
        "ok": not problems,
        "freeze_dir": str(freeze_dir),
        "n_files_checked": len(expected),
        "n_mutated": len(mutated),
        "n_writable": len(writable),
        "manifest_sha256": manifest_hash,
        "problems": problems,
        "checked_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
