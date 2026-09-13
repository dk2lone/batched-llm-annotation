"""CPU-only extension (X1) preparation: frames, disjointness audit, model audit, power.

Nothing here loads model weights, starts a GPU, or reads a scientific model output.
It downloads pinned dataset rows and small Hub metadata only. Every output is written
into a working directory that the separate seal step copies into an immutable freeze.

Usage:
    python build_extension_frames.py frames
    python build_extension_frames.py disjointness
    python build_extension_frames.py models
    python build_extension_frames.py power
    python build_extension_frames.py schedules
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from freezeb import extension as ext  # noqa: E402
from freezeb.models import audit_model  # noqa: E402
from freezeb.power import ERROR_GRID, SIZES, simulate  # noqa: E402
from step0.hashing import read_jsonl, sha256_json, write_json  # noqa: E402

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
WORK = STUDY / "ext-work/20260913-cpu-draft"


def _frame_ids(task_key: str) -> dict[str, list[str]]:
    return {
        role: [str(row["doc_id"]) for row in read_jsonl(WORK / task_key / f"{role}.jsonl")]
        for role in ext.FRAME_ROLES
    }


def cmd_frames() -> None:
    for task in ext.EXT_TASKS:
        if (WORK / task.key / "manifest.json").exists():
            print(f"{task.key}: already materialized, left untouched")
            continue
        manifest = ext.build_ext_task(ROOT, task, WORK)
        rows = manifest["extension"]
        print(
            f"{task.key}: {manifest['raw_candidates_after_length_filter']} eligible, "
            f"{rows['excluded_total_ids']} excluded "
            f"({rows['excluded_historical_ids']} historical, "
            f"{rows['excluded_freeze_b1_ids']} Freeze B)"
        )


def cmd_disjointness() -> None:
    """Prove that no extension row appears in Gate A, Step 0, or Freeze B."""
    report: dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "schema_version": ext.SCHEMA_VERSION,
        "tasks": {},
    }
    all_ext: set[str] = set()
    for task in ext.EXT_TASKS:
        ids = _frame_ids(task.key)
        flat = [doc_id for role in ext.FRAME_ROLES for doc_id in ids[role]]
        sources = ext.ext_exclusions(ROOT, task)
        prior = sources["historical"] | sources["freeze_b1"]
        overlap_by_role = {
            role: sorted(set(ids[role]) & prior)[:5] for role in ext.FRAME_ROLES
        }
        report["tasks"][task.key] = {
            "split": task.split,
            "dataset_revision": task.revision,
            "rows": {role: len(ids[role]) for role in ext.FRAME_ROLES},
            "unique_rows": len(set(flat)),
            "total_rows": len(flat),
            "prior_ids_gate_a_and_step0": len(sources["historical"]),
            "prior_ids_freeze_b1": len(sources["freeze_b1"]),
            "prior_ids_total": len(prior),
            "overlap_with_prior": len(set(flat) & prior),
            "overlap_examples": overlap_by_role,
            "internal_role_overlap": len(flat) - len(set(flat)),
            "row_ids_sha256": {
                role: sha256_json(ids[role]) for role in ext.FRAME_ROLES
            },
        }
        all_ext |= set(flat)
    report["cross_task_unique_ids"] = len(all_ext)
    report["all_tasks_disjoint_from_prior"] = all(
        row["overlap_with_prior"] == 0 and row["internal_role_overlap"] == 0
        for row in report["tasks"].values()
    )
    write_json(WORK / "disjointness-audit.json", report, allow_existing=True)
    print(json.dumps(
        {k: {"rows": v["rows"], "overlap_with_prior": v["overlap_with_prior"]}
         for k, v in report["tasks"].items()},
        indent=1,
    ))
    print("all disjoint:", report["all_tasks_disjoint_from_prior"])


def cmd_models() -> None:
    audits = []
    for model in ext.EXT_MODELS:
        record = audit_model(model)
        policy = ext.ext_policy_for(model.repo)
        record["runtime_policy"] = {
            "max_model_len": policy.max_model_len,
            "enable_thinking": policy.enable_thinking,
            "attempted_in_m1": policy.runnable,
            "note": "the immutable policy attempts every candidate; access is separate",
        }
        # Access is an observation, not part of the frozen policy. A config read that
        # succeeds proves current read access; it does not prove weight access.
        readable = "sha256" in record["files"].get("config.json", {})
        record["access_observation"] = {
            "access_verified_now": readable,
            "status": "verified" if readable else "gated_pending_access",
            "detail": (
                "config and tokenizer readable at the pinned revision"
                if readable
                else str(record["files"].get("config.json", {}).get("error", "unreadable"))
            ),
            "binding": (
                "not binding; only a signed runtime and access lock decides execution"
            ),
        }
        audits.append(record)
        print(
            f"{model.repo}: sha_ok={record['pinned_sha_matches']} gated={record['gated']} "
            f"files={record['n_files']} template={record['chat_template']['present']} "
            f"access_now={readable}"
        )
    write_json(
        WORK / "model-audit.json",
        {
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "schema_version": ext.SCHEMA_VERSION,
            "method": "hub metadata and small pinned config files only; no weights read",
            "models": audits,
        },
        allow_existing=True,
    )


def cmd_power() -> None:
    """Planning power at the extension Holm family size, not the final test."""
    rows = ext.ext_power_table(simulate, ERROR_GRID, SIZES)
    write_json(
        WORK / "power-x1.json",
        {
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "schema_version": ext.SCHEMA_VERSION,
            "alpha": ext.PLANNING_ALPHA,
            "namespaces": list(ext.POWER_REPLICATE_NAMESPACES),
            "sims_per_replicate": ext.POWER_SIMS_PER_REPLICATE,
            "alpha_basis": f"Bonferroni proxy for a {ext.MAX_PRIMARY_CELLS}-cell Holm family",
            "practical_effect": 0.004,
            "limitation": (
                "conservative planning proxy for choosing the item count; "
                "the reported test is the design-respecting replay randomization"
            ),
            "rows": rows,
        },
        allow_existing=True,
    )
    for row in rows:
        print(
            f"p={row['p']:.2f} n={int(row['n']):5d} power={row['power']:.4f} "
            f"spread={row['power_batch_spread']:.4f}"
        )


def _rows(task_key: str, role: str) -> list[dict[str, Any]]:
    return list(read_jsonl(WORK / task_key / f"{role}.jsonl"))


def cmd_schedules() -> None:
    """Freeze the exact M1, B3, and B4 schedules by digest; no text is written."""
    screens = {task.key: _rows(task.key, "screen") for task in ext.EXT_TASKS}
    m1 = ext.m1_call_order(screens)
    write_json(WORK / "m1-call-order.json", m1, allow_existing=True)
    print(f"M1: {m1['n_calls']:,} calls over {m1['n_cells']} cells")
    print(f"    global call-ID digest {m1['global_call_id_order_sha256'][:16]}")
    print(f"    global prompt digest  {m1['global_prompt_sha256_order_sha256'][:16]}")

    confirms = {task.key: _rows(task.key, "confirm") for task in ext.EXT_TASKS}
    b3: list[dict[str, Any]] = []
    for model in ext.EXT_MODELS:
        for task_key in ext.EXT_TASK_KEYS:
            for n_items in ext.B3_SIZES:
                b3.append(
                    ext.b3_plan_digest(model.repo, task_key, confirms[task_key], n_items)
                )
        print(f"B3: {model.repo} done")
    write_json(
        WORK / "b3-plans.json",
        {
            "schema_version": ext.B3_SCHEMA,
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "sizes": list(ext.B3_SIZES),
            "conditions": list(ext.B3_CONDITIONS),
            "n_plans": len(b3),
            "total_calls": sum(int(row["counts"]["total"]) for row in b3),
            "plans_sha256": sha256_json(
                [
                    [
                        row["cell_key"], row["n_items"], row["call_id_order_sha256"],
                        row["record_stream_sha256"],
                    ]
                    for row in b3
                ]
            ),
            "plans": b3,
        },
        allow_existing=True,
    )
    print(f"B3: {len(b3)} plans, {sum(int(r['counts']['total']) for r in b3):,} calls")

    b4: list[dict[str, Any]] = []
    for model in ext.EXT_MODELS:
        for task_key in ext.EXT_TASK_KEYS:
            doc_ids = tuple(str(row["doc_id"]) for row in confirms[task_key])
            b4.append(ext.b4_schedule_digest(model.repo, task_key, doc_ids))
        print(f"B4: {model.repo} done")
    write_json(
        WORK / "b4-schedules.json",
        {
            "schema_version": ext.B4_SCHEMA,
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "reference_replicates": ext.E5_N_REFERENCE,
            "evaluation_prefixes": list(ext.E5_EVALUATION_PREFIXES),
            "n_schedules": len(b4),
            "total_calls": sum(
                int(row["reference"]["n_calls"]) + int(row["evaluation"]["n_calls"])
                for row in b4
            ),
            "schedules_sha256": sha256_json(
                [
                    [
                        row["cell_key"],
                        row["reference"]["call_id_order_sha256"],
                        row["reference"]["record_stream_sha256"],
                        row["evaluation"]["call_id_order_sha256"],
                        row["evaluation"]["doc_order_sha256_order_sha256"],
                    ]
                    for row in b4
                ]
            ),
            "schedules": b4,
        },
        allow_existing=True,
    )
    print(f"B4: {len(b4)} schedules")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=("frames", "disjointness", "models", "power", "schedules")
    )
    step = parser.parse_args().step
    WORK.mkdir(parents=True, exist_ok=True)
    {"frames": cmd_frames, "disjointness": cmd_disjointness, "models": cmd_models,
     "power": cmd_power, "schedules": cmd_schedules}[step]()


if __name__ == "__main__":
    main()
