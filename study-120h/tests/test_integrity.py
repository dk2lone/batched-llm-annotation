"""Seal and verify tests. A sealed freeze must detect every kind of change."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from step0 import integrity


def make_freeze(tmp_path: Path) -> Path:
    freeze_dir = tmp_path / "20260912T000000Z"
    (freeze_dir / "data" / "demo").mkdir(parents=True)
    (freeze_dir / "PROTOCOL.md").write_text("protocol\n", encoding="utf-8")
    (freeze_dir / "data" / "demo" / "items.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    return freeze_dir


def unlock(path: Path) -> None:
    os.chmod(path, 0o644)


def test_seal_writes_a_manifest_and_removes_write_permission(tmp_path: Path) -> None:
    freeze_dir = make_freeze(tmp_path)
    record = integrity.seal(freeze_dir)
    assert record["n_files"] == 2
    assert (freeze_dir / integrity.MANIFEST_NAME).exists()
    for path in freeze_dir.rglob("*"):
        if path.is_file():
            assert not stat.S_IMODE(path.stat().st_mode) & 0o222, path
    assert integrity.verify(freeze_dir)["ok"]


def test_a_sealed_freeze_is_never_resealed(tmp_path: Path) -> None:
    freeze_dir = make_freeze(tmp_path)
    integrity.seal(freeze_dir)
    with pytest.raises(FileExistsError):
        integrity.seal(freeze_dir)


def test_verify_fails_before_the_freeze_is_sealed(tmp_path: Path) -> None:
    report = integrity.verify(make_freeze(tmp_path))
    assert not report["ok"]
    assert "never sealed" in report["problems"][0]


def test_verify_detects_an_edited_file(tmp_path: Path) -> None:
    freeze_dir = make_freeze(tmp_path)
    integrity.seal(freeze_dir)
    target = freeze_dir / "PROTOCOL.md"
    unlock(target)
    target.write_text("edited\n", encoding="utf-8")
    report = integrity.verify(freeze_dir)
    assert not report["ok"]
    assert report["n_mutated"] == 1
    assert any("hash mismatch" in p for p in report["problems"])


def test_verify_detects_an_added_or_removed_file(tmp_path: Path) -> None:
    freeze_dir = make_freeze(tmp_path)
    integrity.seal(freeze_dir)
    (freeze_dir / "EXTRA.md").write_text("new\n", encoding="utf-8")
    report = integrity.verify(freeze_dir)
    assert any("unlisted file" in p for p in report["problems"])
    os.remove(freeze_dir / "EXTRA.md")
    removed = freeze_dir / "PROTOCOL.md"
    unlock(removed)
    os.remove(removed)
    report = integrity.verify(freeze_dir)
    assert any("missing file" in p for p in report["problems"])


def test_verify_rejects_a_writable_sealed_file(tmp_path: Path) -> None:
    freeze_dir = make_freeze(tmp_path)
    integrity.seal(freeze_dir)
    unlock(freeze_dir / "PROTOCOL.md")
    report = integrity.verify(freeze_dir)
    assert not report["ok"]
    assert report["n_writable"] == 1


def test_verify_detects_an_edited_manifest(tmp_path: Path) -> None:
    freeze_dir = make_freeze(tmp_path)
    integrity.seal(freeze_dir)
    manifest = freeze_dir / integrity.MANIFEST_NAME
    unlock(manifest)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join(lines[:1]) + "\n", encoding="utf-8")
    report = integrity.verify(freeze_dir)
    assert not report["ok"]
    assert any("edited after sealing" in p for p in report["problems"])


def test_verify_rejects_writable_or_missing_seal_controls(tmp_path: Path) -> None:
    freeze_dir = make_freeze(tmp_path)
    integrity.seal(freeze_dir)
    unlock(freeze_dir / integrity.SEAL_NAME)
    report = integrity.verify(freeze_dir)
    assert not report["ok"]
    assert any(integrity.SEAL_NAME in p and "writable" in p for p in report["problems"])

    os.remove(freeze_dir / integrity.SEAL_NAME)
    report = integrity.verify(freeze_dir)
    assert not report["ok"]
    assert any("missing seal control file" in p for p in report["problems"])
