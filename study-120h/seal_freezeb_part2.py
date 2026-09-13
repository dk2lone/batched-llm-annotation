"""Seal a completed 36-cell solo screen as Freeze B admission lock part 2."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from freezeb.frames import N_SCREEN  # noqa: E402
from freezeb.screen import (  # noqa: E402
    RUNNABLE_MODELS,
    TASK_KEYS,
    build_calls,
    build_part2_payload,
    model_key,
    summarize_cell,
    verify_screen_frame,
)
from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402

FRAME_ROOT = ROOT / "freeze-b-work/20260912-cpu-draft"
PART1_ROOT = ROOT / "freeze/20260912T222618Z-B1"
SOURCE_FILES = (
    ROOT / "modal_freezeb_screen.py",
    ROOT / "seal_freezeb_part2.py",
    ROOT / "src/freezeb/screen.py",
    ROOT / "src/freezeb/admission.py",
    ROOT / "src/freezeb/selection.py",
    ROOT / "src/freezeb/power.py",
    ROOT / "src/step1/backends.py",
    ROOT / "tests/test_freezeb_screen.py",
    ROOT / "tests/test_step1.py",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _verify_completed_screen(screen_run: Path) -> list[dict[str, Any]]:
    plan_path = screen_run / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != sha256_json(plan_core):
        raise ValueError("screen plan hash does not match its contents")
    if (
        plan.get("solo_only") is not True
        or plan.get("models") != list(RUNNABLE_MODELS)
        or plan.get("tasks") != list(TASK_KEYS)
        or plan.get("calls") != len(RUNNABLE_MODELS) * len(TASK_KEYS) * N_SCREEN
    ):
        raise ValueError("part 2 requires the frozen full 36-cell solo-only plan")
    frames = {task: verify_screen_frame(FRAME_ROOT, PART1_ROOT, task) for task in TASK_KEYS}
    summaries: list[dict[str, Any]] = []
    for model in RUNNABLE_MODELS:
        for task in TASK_KEYS:
            path = screen_run / "cells" / model_key(model) / f"{task}.jsonl"
            events = _read_jsonl(path)
            expected = {row["call_id"] for row in build_calls(model, task, frames[task])}
            actual = [str(row.get("call_id", "")) for row in events]
            attempt_ids = [str(row.get("attempt_id", "")) for row in events]
            final_ids = {
                str(row.get("call_id", ""))
                for row in events
                if row.get("terminal") is True and row.get("final_for_call") is True
            }
            if (
                len(events) < N_SCREEN
                or not set(actual) <= expected
                or len(attempt_ids) != len(set(attempt_ids))
                or "" in attempt_ids
                or final_ids != expected
                or any(row.get("condition") != "solo" for row in events)
                or any(row.get("terminal") is not True for row in events)
            ):
                raise ValueError(f"incomplete or non-solo event ledger for {model}::{task}")
            recomputed = summarize_cell(model, task, frames[task], events)
            summary_path = screen_run / "summaries" / model_key(model) / f"{task}.json"
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if saved != recomputed:
                raise ValueError(f"stale or altered summary for {model}::{task}")
            summaries.append(recomputed)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("screen_run", type=Path)
    parser.add_argument("--seal-id", default="")
    args = parser.parse_args()
    summaries = _verify_completed_screen(args.screen_run)
    power_path = PART1_ROOT / "cpu-audits/power-v2.json"
    power_rows = json.loads(power_path.read_text(encoding="utf-8"))
    payload = build_part2_payload(summaries, power_rows)
    payload.update(
        {
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "screen_plan_sha256": json.loads(
                (args.screen_run / "plan.json").read_text(encoding="utf-8")
            )["plan_sha256"],
            "power_table_sha256": sha256_file(power_path),
        }
    )
    seal_id = args.seal_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ-B2")
    destination = ROOT / "freeze" / seal_id
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".freezeb-part2-stage-", dir=destination.parent))
    try:
        screen_files = sorted(path for path in args.screen_run.rglob("*") if path.is_file())
        screen_hashes = {
            str(path.relative_to(args.screen_run)): sha256_file(path) for path in screen_files
        }
        source_hashes: dict[str, str] = {}
        for source in SOURCE_FILES:
            relative = source.relative_to(ROOT)
            target = stage / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[str(relative)] = sha256_file(target)
        write_json(stage / "ADMISSION_LOCK.json", payload)
        write_json(
            stage / "MANIFEST.json",
            {
                "seal_kind": "Freeze B admission lock part 2",
                "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "source_screen_run": str(args.screen_run),
                "contains_joint_outputs": False,
                "admission_lock_sha256": sha256_file(stage / "ADMISSION_LOCK.json"),
                "screen_files_sha256": screen_hashes,
                "source_snapshot_sha256": source_hashes,
            },
        )
        stage.rename(destination)
    except Exception:
        shutil.rmtree(stage)
        raise
    print(destination)


if __name__ == "__main__":
    main()
