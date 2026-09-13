"""Seal the CPU scientific protocol without copying restricted dataset text."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path

from freezeb.frames import N_CONFIRM, N_FILLER, N_SCREEN, TASKS
from step0.hashing import read_jsonl, sha256_file, write_json, write_jsonl

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
WORK = STUDY / "freeze-b-work/20260912-cpu-draft"
FREEZE = STUDY / "freeze"
HISTORICAL = FREEZE / "20260912T041633Z"
SNAPSHOT_FILES = [
    STUDY / "FREEZE_B_PROTOCOL.md",
    STUDY / "FREEZE_B_PREP.md",
    STUDY / "pyproject.toml",
    STUDY / "uv.lock",
    STUDY / "seal_freezeb.py",
    STUDY / "tests/test_freezeb.py",
    *sorted((STUDY / "src/freezeb").glob("*.py")),
    *[STUDY / f"src/step0/{name}.py" for name in ("assign", "hashing", "lock", "stats")],
]


def seal() -> Path:
    if not HISTORICAL.is_dir():
        raise FileNotFoundError("the historical Step 0 seal must be preserved")
    for path in SNAPSHOT_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = FREEZE / f"{stamp}-B1"
    if destination.exists():
        raise FileExistsError(destination)
    stage = Path(tempfile.mkdtemp(prefix=".freezeb-stage-", dir=FREEZE))
    try:
        hashes: dict[str, str] = {}
        for source in SNAPSHOT_FILES:
            relative = source.relative_to(STUDY)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = sha256_file(target)
        for filename in ("model-audit.json", "data-audit.json", "power-v2.json"):
            source = WORK / filename
            target = stage / "cpu-audits" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(target.relative_to(stage))] = sha256_file(target)
        frame_summary: dict[str, object] = {}
        for task in TASKS:
            task_dir = WORK / task.key
            source_manifest = json.loads((task_dir / "manifest.json").read_text())
            role_ids: dict[str, set[str]] = {}
            role_info: dict[str, object] = {}
            for role, expected in (("screen", N_SCREEN), ("confirm", N_CONFIRM),
                                   ("filler", N_FILLER)):
                source = task_dir / f"{role}.jsonl"
                recorded = source_manifest["files"][source.name]
                if sha256_file(source) != recorded["sha256"]:
                    raise ValueError(f"source frame hash changed: {source}")
                rows = list(read_jsonl(source))
                if len(rows) != expected:
                    raise ValueError(f"wrong {task.key} {role} count")
                ids = {str(row["doc_id"]) for row in rows}
                if len(ids) != expected:
                    raise ValueError(f"duplicate {task.key} {role} IDs")
                role_ids[role] = ids
                identifier_rows = [{
                    "doc_id": row["doc_id"], "source_index": row["source_index"],
                    "source_id": row["source_id"],
                    "length_stratum": row.get("length_stratum"),
                } for row in rows]
                target = stage / "frame-identifiers" / task.key / f"{role}.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(target, identifier_rows)
                hashes[str(target.relative_to(stage))] = sha256_file(target)
                role_info[role] = {"rows": expected, "source_file_sha256": recorded["sha256"],
                                   "identifier_file_sha256": sha256_file(target)}
            if any(role_ids[a] & role_ids[b] for a, b in
                   (("screen", "confirm"), ("screen", "filler"), ("confirm", "filler"))):
                raise ValueError(f"overlapping {task.key} roles")
            target_manifest = stage / "frame-identifiers" / task.key / "source-manifest.json"
            write_json(target_manifest, source_manifest)
            hashes[str(target_manifest.relative_to(stage))] = sha256_file(target_manifest)
            frame_summary[task.key] = role_info
        expected_outputs = {
            "scientific_model_outputs_at_part1_seal": 0,
            "runtime_lock_required_before_solo": ["engine_image", "checkpoint_and_tokenizer_hashes",
                                                  "context_fit", "metered_cost_by_block"],
            "admission_lock_required_before_joint": ["attempted_candidates", "technical_status",
                                                     "solo_metrics", "ordered_cells",
                                                     "item_counts", "complete_block_queue"],
            "first_joint_output": "historical Qwen/Civil Step 2 after part 2 is sealed",
        }
        write_json(stage / "EXPECTED_OUTPUTS.json", expected_outputs)
        hashes["EXPECTED_OUTPUTS.json"] = sha256_file(stage / "EXPECTED_OUTPUTS.json")
        write_json(stage / "MANIFEST.json", {
            "seal_kind": "Freeze B scientific protocol part 1",
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "historical_step0_seal": str(HISTORICAL.relative_to(ROOT)),
            "contains_model_outputs": False,
            "contains_dataset_text": False,
            "files_sha256": hashes,
            "frames": frame_summary,
            "power_table_status": "planning proxy; not final design-test power calibration",
            "remaining_before_paid_science": (
                "synthetic runtime lock, then solo-only admission part 2"
            ),
        })
        stage.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(stage)
        raise


if __name__ == "__main__":
    print(seal())
