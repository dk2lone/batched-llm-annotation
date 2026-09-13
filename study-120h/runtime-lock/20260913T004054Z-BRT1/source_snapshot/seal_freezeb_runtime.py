"""Seal a benchmark-derived Freeze B runtime ledger without rerunning a model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path

from freezeb.runtime_ledger import build_ledger, render_markdown
from step0.hashing import sha256_file, write_json, write_text

STUDY = Path(__file__).resolve().parent
RUNTIME_ROOT = STUDY / "freezeb-runtime"
LOCK_ROOT = STUDY / "runtime-lock"


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def seal(run_id: str, stamp: str | None = None) -> Path:
    source = RUNTIME_ROOT / run_id
    inputs = {
        name: source / name
        for name in (
            "plan.json",
            "report.json",
            "workers.json",
            "checkpoint-manifests.json",
            "runtime-observations.json",
        )
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    stamp = stamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = LOCK_ROOT / f"{stamp}-BRT1"
    if destination.exists():
        raise FileExistsError(destination)
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".runtime-stage-", dir=LOCK_ROOT))
    try:
        copied: dict[str, str] = {}
        for name, path in inputs.items():
            target = stage / "benchmark" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied[str(target.relative_to(stage))] = sha256_file(target)
        snapshots = (
            STUDY / "src/freezeb/runtime_benchmark.py",
            STUDY / "src/freezeb/runtime_ledger.py",
            STUDY / "seal_freezeb_runtime.py",
            STUDY / "tests/test_freezeb_runtime_benchmark.py",
            STUDY / "tests/test_freezeb_runtime_ledger.py",
        )
        for path in snapshots:
            target = stage / "source_snapshot" / path.relative_to(STUDY)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied[str(target.relative_to(stage))] = sha256_file(target)
        report = _read(inputs["report.json"])
        workers = _read(inputs["workers.json"])
        plan = _read(inputs["plan.json"])
        observations = _read(inputs["runtime-observations.json"])
        if (
            not isinstance(report, dict)
            or not isinstance(plan, dict)
            or not isinstance(observations, dict)
        ):
            raise TypeError("benchmark JSON roots must be objects")
        if not isinstance(workers, list):
            raise TypeError("workers JSON root must be a list")
        ledger = build_ledger(report, workers, plan, observations)
        write_json(stage / "LEDGER.json", ledger)
        write_text(stage / "RUNTIME_LEDGER.md", render_markdown(ledger))
        copied["LEDGER.json"] = sha256_file(stage / "LEDGER.json")
        copied["RUNTIME_LEDGER.md"] = sha256_file(stage / "RUNTIME_LEDGER.md")
        write_json(
            stage / "MANIFEST.json",
            {
                "seal_kind": "Freeze B benchmark-derived runtime ledger",
                "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "benchmark_run_id": run_id,
                "contains_scientific_model_outputs": False,
                "contains_synthetic_model_outputs": False,
                "status": ledger["status"],
                "files_sha256": copied,
            },
        )
        stage.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stamp")
    args = parser.parse_args()
    print(seal(args.run_id, args.stamp))


if __name__ == "__main__":
    main()
