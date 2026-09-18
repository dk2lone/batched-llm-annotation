"""Seal the B5 batch-size dose-response protocol, code and CPU preparation artifacts.

The seal hashes every decision-relevant input in place and keeps a snapshot beside the
manifest. `verify` recomputes the live hashes, so editing any covered file invalidates
the seal and the scientific launcher refuses to run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402

FREEZE = STUDY / "freeze"
SCHEMA = "b5-seal-v1"
PREPARATION_FILES = (
    "SCHEDULE_DIGEST.json",
    "TOKEN_EVIDENCE.json",
    "RUNTIME_EVIDENCE.json",
    "CONTEXT_AUDIT.json",
    "FORECAST.json",
    "POWER.json",
    "MANIFEST.json",
)
CODE_FILES = (
    "B5_PROTOCOL.md",
    "prepare_b5.py",
    "seal_b5.py",
    "run_b5.py",
    "b5_worker.py",
    "analyze_b5.py",
    "src/freezeb/b5.py",
    "src/freezeb/b5_analysis.py",
    "src/freezeb/b5_power.py",
    "src/freezeb/prompts.py",
    "src/freezeb/labels.py",
    "src/freezeb/extension_b4.py",
    "src/step0/stats.py",
    "src/step0/hashing.py",
    "src/step1/backends.py",
    "tests/test_b5.py",
    "tests/test_b5_analysis.py",
    "tests/test_b5_runner.py",
    "tests/test_batch_invariance.py",
    "pyproject.toml",
)


def covered(preparation: Path) -> dict[str, Path]:
    preparation = preparation.resolve()
    files = {name: STUDY / name for name in CODE_FILES}
    for name in PREPARATION_FILES:
        files[str((preparation / name).relative_to(STUDY))] = preparation / name
    missing = sorted(str(path) for path in files.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"B5 seal is missing: {missing}")
    return files


def seal(preparation: Path) -> Path:
    preparation = preparation.resolve()
    files = covered(preparation)
    prior = sorted(FREEZE.glob("*-B5"))
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = FREEZE / f"{stamp}-B5"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for relative, source in files.items():
        target = destination / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "sealed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "preparation_directory": str(preparation.relative_to(STUDY)),
        "files_sha256": {relative: sha256_file(source) for relative, source in files.items()},
        "schedule_sha256": json.loads((preparation / "SCHEDULE_DIGEST.json").read_text())[
            "schedule_sha256"
        ],
        "gpu_work_launched": False,
        "supersedes": str(prior[-1].relative_to(STUDY)) if prior else None,
        "scope": (
            "B5 batch-size dose-response. It adds a new protocol and never edits the "
            "sealed X1R4, Freeze B or output-recovery artifacts."
        ),
    }
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    write_json(destination / "MANIFEST.json", manifest)
    return destination


def verify(seal_directory: Path) -> dict[str, Any]:
    seal_directory = seal_directory.resolve()
    manifest = json.loads((seal_directory / "MANIFEST.json").read_text())
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if sha256_json(core) != manifest.get("manifest_sha256"):
        raise ValueError(f"B5 manifest digest changed: {seal_directory}")
    changed = []
    for relative, digest in cast(dict[str, str], manifest["files_sha256"]).items():
        live = STUDY / relative
        if not live.is_file() or sha256_file(live) != digest:
            changed.append(relative)
        snapshot = seal_directory / "snapshot" / relative
        if not snapshot.is_file() or sha256_file(snapshot) != digest:
            changed.append(f"snapshot/{relative}")
    if changed:
        raise ValueError(f"B5 sealed files changed: {sorted(changed)}")
    return {
        "seal": str(seal_directory.relative_to(STUDY)),
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "files_verified": len(manifest["files_sha256"]),
        "preparation_directory": manifest["preparation_directory"],
        "confirmation_token": str(manifest["manifest_sha256"])[:12],
    }


def latest() -> Path:
    seals = sorted(FREEZE.glob("*-B5"))
    if not seals:
        raise FileNotFoundError("no B5 seal exists")
    return seals[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        print(json.dumps(verify(args.verify), indent=2))
        return
    if args.preparation is None:
        raise SystemExit("pass --preparation to seal or --verify to check a seal")
    destination = seal(args.preparation)
    print(json.dumps(verify(destination), indent=2))


if __name__ == "__main__":
    main()
