"""Seal the batch-invariant cross-family extension (X1R4) without dataset text.

The seal records identifiers, hashes, frozen settings, the exact M1 call order, the
conditional B3/B4 order, the budget, and the disclosure of prior opened families.
It contains no model output, no dataset text, and no credential. It is written into
a new timestamped directory and never overwrites an existing artifact.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from freezeb import batch_invariance as bi  # noqa: E402
from freezeb import extension as ext  # noqa: E402
from freezeb.frames import N_CONFIRM, N_FILLER, N_SCREEN  # noqa: E402
from step0.hashing import (  # noqa: E402
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
WORK = STUDY / "ext-work/20260913-cpu-draft"
FREEZE = STUDY / "freeze"
PRIOR_SEAL = FREEZE / "20260913T181826Z-X1R2"
AUDITS = ("model-audit.json", "disjointness-audit.json", "power-x1.json")
SCHEDULES = {
    "M1_CALL_ORDER.json": "m1-call-order.json",
    "B3_PLANS.json": "b3-plans.json",
    "B4_SCHEDULES.json": "b4-schedules.json",
}
SNAPSHOT_FILES = [
    STUDY / "EXTENSION_PROTOCOL.md",
    STUDY / "pyproject.toml",
    STUDY / "uv.lock",
    STUDY / "build_extension_frames.py",
    STUDY / "modal_extension_batch_invariance.py",
    STUDY / "seal_extension.py",
    STUDY / "tests/test_batch_invariance.py",
    STUDY / "tests/test_extension.py",
    *[
        STUDY / f"src/freezeb/{name}.py"
        for name in (
            "__init__",
            "admission",
            "assign",
            "batch_invariance",
            "claims",
            "extension",
            "frames",
            "labels",
            "meter",
            "models",
            "power",
            "prompts",
            "repeated",
            "runtime_benchmark",
            "screen",
            "selection",
        )
    ],
    STUDY / "src/step1/backends.py",
    *[STUDY / f"src/step0/{name}.py" for name in ("assign", "hashing", "lock", "stats")],
]
ROLES = (("screen", N_SCREEN), ("confirm", N_CONFIRM), ("filler", N_FILLER))


def _frames(stage: Path, hashes: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    every_id: dict[str, set[str]] = {}
    for task in ext.EXT_TASKS:
        task_dir = WORK / task.key
        source_manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        if source_manifest["selection_seed"] != ext.EXT_SEED:
            raise ValueError(f"{task.key} was not built with the extension seed")
        if source_manifest["task"]["split"] != task.split:
            raise ValueError(f"{task.key} was not built from the frozen split")
        role_ids: dict[str, set[str]] = {}
        role_info: dict[str, Any] = {}
        for role, expected in ROLES:
            source = task_dir / f"{role}.jsonl"
            recorded = source_manifest["files"][source.name]
            if sha256_file(source) != recorded["sha256"]:
                raise ValueError(f"source frame hash changed since the build: {source}")
            rows = list(read_jsonl(source))
            ids = {str(row["doc_id"]) for row in rows}
            if len(rows) != expected or len(ids) != expected:
                raise ValueError(f"wrong or duplicated {task.key} {role} rows")
            role_ids[role] = ids
            identifier_rows = [
                {
                    "doc_id": row["doc_id"],
                    "source_index": row["source_index"],
                    "source_id": row["source_id"],
                    "n_chars": row["n_chars"],
                    "length_stratum": row.get("length_stratum"),
                }
                for row in rows
            ]
            target = stage / "frame-identifiers" / task.key / f"{role}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(target, identifier_rows)
            hashes[str(target.relative_to(stage))] = sha256_file(target)
            role_info[role] = {
                "rows": expected,
                "source_file_sha256": recorded["sha256"],
                "row_ids_sha256": recorded["row_ids_sha256"],
                "identifier_file_sha256": sha256_file(target),
            }
        for a, b in (("screen", "confirm"), ("screen", "filler"), ("confirm", "filler")):
            if role_ids[a] & role_ids[b]:
                raise ValueError(f"{task.key} roles overlap")
        prior = ext.ext_exclusions(ROOT, task)
        combined = role_ids["screen"] | role_ids["confirm"] | role_ids["filler"]
        overlap = combined & (prior["historical"] | prior["freeze_b1"])
        if overlap:
            raise ValueError(f"{task.key} reuses {len(overlap)} prior rows")
        every_id[task.key] = combined
        target_manifest = stage / "frame-identifiers" / task.key / "source-manifest.json"
        write_json(target_manifest, source_manifest)
        hashes[str(target_manifest.relative_to(stage))] = sha256_file(target_manifest)
        role_info["disjointness"] = {
            "prior_gate_a_and_step0_ids": len(prior["historical"]),
            "prior_freeze_b1_ids": len(prior["freeze_b1"]),
            "overlap_with_prior": 0,
            "dataset": task.dataset,
            "dataset_revision": task.revision,
            "split": task.split,
        }
        summary[task.key] = role_info
    keys = list(every_id)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if every_id[a] & every_id[b]:
                raise ValueError(f"{a} and {b} share document IDs")
    return summary


def seal() -> Path:
    for path in SNAPSHOT_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in AUDITS:
        if not (WORK / name).is_file():
            raise FileNotFoundError(WORK / name)
    for name in SCHEDULES.values():
        if not (WORK / name).is_file():
            raise FileNotFoundError(WORK / name)
    if not (ROOT / ext.FREEZE_B1).is_dir():
        raise FileNotFoundError("the Freeze B part-1 seal must be preserved")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = FREEZE / f"{stamp}-X1R4"
    if destination.exists():
        raise FileExistsError(destination)
    stage = Path(tempfile.mkdtemp(prefix=".ext-stage-", dir=FREEZE))
    try:
        hashes: dict[str, str] = {}
        for source in SNAPSHOT_FILES:
            relative = source.relative_to(STUDY)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = sha256_file(target)
        for name in AUDITS:
            target = stage / "cpu-audits" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(WORK / name, target)
            hashes[str(target.relative_to(stage))] = sha256_file(target)
        prior_manifest = json.loads((PRIOR_SEAL / "MANIFEST.json").read_text(encoding="utf-8"))
        for target_name, source_name in SCHEDULES.items():
            if sha256_file(WORK / source_name) != prior_manifest["files_sha256"][target_name]:
                raise ValueError(f"{target_name} differs from the active X1R2 scientific seal")
            target = stage / target_name
            shutil.copy2(WORK / source_name, target)
            hashes[target_name] = sha256_file(target)
        frames = _frames(stage, hashes)

        batch_invariance_plan = bi.plan_payload(
            [{"model_id": model.repo, "revision": model.revision} for model in ext.EXT_MODELS]
        )
        write_json(stage / "BATCH_INVARIANCE_PLAN.json", batch_invariance_plan)
        hashes["BATCH_INVARIANCE_PLAN.json"] = sha256_file(stage / "BATCH_INVARIANCE_PLAN.json")

        model_audit = json.loads((WORK / "model-audit.json").read_text(encoding="utf-8"))
        candidates = []
        for record in model_audit["models"]:
            if not record["pinned_sha_matches"]:
                raise ValueError(f"{record['repo']} does not resolve to its pinned revision")
            candidates.append(
                {
                    "repo": record["repo"],
                    "family": record["family"],
                    "revision": record["revision"],
                    "gated": record["gated"],
                    "license": record["license"],
                    "n_files": record["n_files"],
                    "file_fingerprints_sha256": record["file_fingerprints_sha256"],
                    "tokenizer_vocabulary": record["tokenizer_vocabulary"],
                    "chat_template": record["chat_template"],
                    "attempted_in_frozen_m1_schedule": True,
                    "access_observation": record["access_observation"],
                    "runtime_policy": record["runtime_policy"],
                }
            )

        for name, payload in (
            ("DISCLOSURE.json", ext.disclosure()),
            (
                "CANDIDATES.json",
                {
                    "schema_version": ext.SCHEMA_VERSION,
                    "candidates": candidates,
                    "retrieved_at_utc": model_audit["created_at_utc"],
                },
            ),
            (
                "EXPECTED_OUTPUTS.json",
                {
                    "scientific_model_outputs_at_x1_seal": 0,
                    "extension_joint_outputs_seen": False,
                    "prior_opened_joint_families": list(ext.OPENED_JOINT_FAMILIES),
                    "required_before_solo_screen": [
                        "signed runtime and access lock covering all five candidates",
                        "synthetic runtime lock for every attempted candidate",
                        "passing batch-invariance report for every attempted candidate",
                        "measured guarded cost inside the 80-hour extension ceiling",
                    ],
                    "required_before_joint": [
                        "complete solo screen for every attempted candidate",
                        "sealed X1 admission lock built from solo evidence only",
                    ],
                    "conditional_order": ext.b3_b4_order(
                        {
                            "primary": ["<admitted cells in balanced order>"],
                            "family_core": ["<first cell of each admitted family>"],
                            "e5_cells": ["<one cell per admitted family>"],
                        }
                    ),
                },
            ),
        ):
            write_json(stage / name, payload)
            hashes[name] = sha256_file(stage / name)

        manifest: dict[str, Any] = {
            "seal_kind": "cross-family extension X1R4 CPU protocol",
            "schema_version": ext.SCHEMA_VERSION,
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "supersedes": ext.SUPERSEDES,
            "supersession_reason": (
                "X1R3 pinned revisions and layouts in its batch-invariance plan but did not "
                "validate the reported worker revision, anchor ID, or request layout against "
                "those pins; X1R4 adds those fail-closed checks"
            ),
            "scientific_schedules_inherited_from": str(PRIOR_SEAL.relative_to(ROOT)),
            "batch_invariance_plan_sha256": batch_invariance_plan["plan_sha256"],
            "separate_from_freeze_b": True,
            "freeze_b_part1_seal": ext.FREEZE_B1,
            "contains_model_outputs": False,
            "contains_dataset_text": False,
            "contains_credentials": False,
            "identity_sha256": ext.disclosure()["identity_sha256"],
            "budget": ext.disclosure()["budget"],
            "seeds": ext.disclosure()["seeds"],
            "statistics": ext.disclosure()["statistics"],
            "frames": frames,
            "files_sha256": hashes,
            "power_table_status": "planning proxy; not final design-test calibration",
            "remaining_before_paid_science": (
                "Llama access decision, authorized synthetic runtime and batch-invariance "
                "checks, signed runtime/access lock, then the blinded solo screen and X1R4 "
                "admission lock"
            ),
        }
        manifest["manifest_sha256"] = sha256_json(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        )
        write_json(stage / "MANIFEST.json", manifest)
        stage.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(seal())
