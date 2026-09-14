"""Seal the X1R4 A7 choices from completed solo evidence, without running a model."""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from freezeb import extension as ext  # noqa: E402
from freezeb import extension_screen as screen  # noqa: E402
from freezeb.power import choose_n  # noqa: E402
from step0.hashing import read_jsonl, sha256_file, sha256_json, write_json  # noqa: E402

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
SEAL = ROOT / screen.X1R4_SEAL
RUNTIME_LOCK = ROOT / screen.X1R4_RUNTIME_LOCK
M1 = STUDY / "ext-work/extension-m1/20260914T003735Z-5a8c63f2"
OUTPUT = STUDY / "admission-lock"
A5_BATCH = STUDY / "extension-batch-invariance/20260913T191010Z-4807e7d1/report.json"
A5_RUNTIME = STUDY / "extension-runtime/20260913T200450Z-a1f3bd0f/report.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def selected_schedule(
    payload: dict[str, Any], b3: dict[str, Any], b4: dict[str, Any]
) -> dict[str, Any]:
    """Bind each selected cell to its exact frozen schedule digests and call count."""
    b3_by_key = {(r["cell_key"], r["n_items"]): r for r in b3["plans"]}
    b4_by_key = {r["cell_key"]: r for r in b4["schedules"]}
    primary = payload["ordered_cells"]["primary"]
    e5 = payload["ordered_cells"]["e5_cells"]
    if len(b3_by_key) != b3["n_plans"] or len(b4_by_key) != b4["n_schedules"]:
        raise ValueError("frozen schedules contain duplicate keys")
    chosen_b3 = {}
    for key in primary:
        n = payload["item_counts"][key]["n_items"]
        if n not in (2048, 4096, 8192) or (key, n) not in b3_by_key:
            raise ValueError(f"no frozen B3 schedule for {key} at {n} items")
        row = b3_by_key[key, n]
        if row["counts"] != ext.b3_expected_counts(n):
            raise ValueError(f"B3 call counts disagree for {key}")
        chosen_b3[key] = {
            "n_items": n,
            "counts": row["counts"],
            "call_id_order_sha256": row["call_id_order_sha256"],
            "record_stream_sha256": row["record_stream_sha256"],
            "selected_plan_sha256": sha256_json(row),
        }
    chosen_b4 = {}
    for key in e5:
        if key not in b4_by_key:
            raise ValueError(f"no frozen B4 schedule for {key}")
        row = b4_by_key[key]
        if row["reference"]["replicates"] != ext.E5_N_REFERENCE:
            raise ValueError(f"B4 reference count disagrees for {key}")
        if set(row["evaluation"]["prefixes"]) != {str(n) for n in ext.E5_EVALUATION_PREFIXES}:
            raise ValueError(f"B4 evaluation prefixes disagree for {key}")
        for n in ext.E5_EVALUATION_PREFIXES:
            prefix = row["evaluation"]["prefixes"][str(n)]
            if prefix["replicates"] != n or prefix["n_calls"] != n * row["calls_per_replicate"]:
                raise ValueError(f"B4 evaluation prefix count disagrees for {key}")
        chosen_b4[key] = {
            "reference_calls": row["reference"]["n_calls"],
            "evaluation_calls": row["evaluation"]["n_calls"],
            "reference_call_id_order_sha256": row["reference"]["call_id_order_sha256"],
            "reference_record_stream_sha256": row["reference"]["record_stream_sha256"],
            "evaluation_call_id_order_sha256": row["evaluation"]["call_id_order_sha256"],
            "evaluation_doc_order_sha256_order_sha256": row["evaluation"][
                "doc_order_sha256_order_sha256"
            ],
            "evaluation_prefixes": row["evaluation"]["prefixes"],
            "selected_schedule_sha256": sha256_json(row),
        }
    return {
        "b3": chosen_b3,
        "b4": chosen_b4,
        "b3_total_calls": sum(r["counts"]["total"] for r in chosen_b3.values()),
        "b4_total_calls": sum(
            r["reference_calls"] + r["evaluation_calls"] for r in chosen_b4.values()
        ),
    }


def build() -> dict[str, Any]:
    seal = screen.verify_seal(SEAL)
    runtime = screen.verify_runtime_lock(RUNTIME_LOCK, seal)
    inputs = load(M1 / "admission-inputs.json")
    summary = load(M1 / "SUMMARY.json")
    plan = load(M1 / "plan.json")
    if not inputs["screen_complete"] or inputs["extension_joint_outputs_seen"]:
        raise ValueError("M1 is incomplete or extension joint output has been seen")
    if not summary["screen_complete"] or summary["stopped"] or summary["quarantined_payloads"]:
        raise ValueError("M1 completion record is not clean")
    if inputs["seal_manifest_sha256"] != seal["manifest_sha256"]:
        raise ValueError("M1 used a different scientific seal")
    if inputs["access_lock_sha256"] != runtime["access_lock_sha256"]:
        raise ValueError("M1 used a different access lock")
    if inputs["runtime_lock_sha256"] != runtime["runtime_lock_sha256"]:
        raise ValueError("M1 used a different runtime lock")
    if inputs["attempted_models"] != runtime["attempted_models"]:
        raise ValueError("M1 attempted model list changed")
    if plan["plan_sha256"] != summary["plan_sha256"]:
        raise ValueError("M1 plan and summary disagree")
    meter = screen.fold_receipts(read_jsonl(M1 / "receipts.jsonl"))
    if meter["receipts"] != summary["meter_receipts"]:
        raise ValueError("M1 receipt count changed")
    if abs(meter["cumulative_a100_hours"] - summary["cumulative_a100_hours"]) > 1e-9:
        raise ValueError("M1 receipt meter changed")
    if abs(inputs["metered_a100_hours"] - meter["cumulative_a100_hours"]) > 1e-9:
        raise ValueError("M1 admission meter changed")
    b3_path = SEAL / "B3_PLANS.json"
    b4_path = SEAL / "B4_SCHEDULES.json"
    power_path = SEAL / "cpu-audits/power-x1.json"
    base = ext.build_x1_admission_payload(
        inputs["summaries"], load(power_path)["rows"], runtime["access_lock"], choose_n=choose_n
    )
    admitted = {r["cell_key"] for r in inputs["summaries"] if r["admitted"]}
    if admitted != set(inputs["admitted_cells"]) or admitted != set(
        base["ordered_cells"]["primary"]
    ):
        raise ValueError("M1 admitted panel changed")
    if set(summary["admitted_cells"]) != admitted:
        raise ValueError("M1 summary admitted panel changed")
    if any(not base["item_counts"][k]["power_target_met"] for k in admitted):
        raise ValueError("an admitted cell misses the frozen proxy power target")
    schedules = selected_schedule(base, load(b3_path), load(b4_path))
    family_slots = {
        family: f"a100_slot_{i + 1}" for i, family in enumerate(base["ordered_cells"]["families"])
    }
    owners = {
        r["cell_key"]: family_slots[r["family"]]
        for r in inputs["summaries"]
        if r["cell_key"] in admitted
    }
    batch = load(A5_BATCH)
    synthetic = load(A5_RUNTIME)
    prior = (
        batch["metered_a100_hours"]
        + synthetic["metered_proxy_a100_hours"]
        + meter["cumulative_a100_hours"]
    )
    if prior + ext.RESERVE_HOURS >= ext.STOP_HOURS:
        raise ValueError("no guarded extension budget remains")
    base.update(
        {
            "schema_version": "ext-x1r4-a7-admission-lock-v1",
            "source_hashes": {
                "scientific_seal_manifest_sha256": seal["manifest_sha256"],
                "runtime_access_lock_manifest_sha256": runtime["manifest_sha256"],
                "m1_admission_inputs_file_sha256": sha256_file(M1 / "admission-inputs.json"),
                "m1_summary_file_sha256": sha256_file(M1 / "SUMMARY.json"),
                "m1_plan_file_sha256": sha256_file(M1 / "plan.json"),
                "m1_receipts_file_sha256": sha256_file(M1 / "receipts.jsonl"),
                "power_file_sha256": sha256_file(power_path),
                "b3_plans_file_sha256": sha256_file(b3_path),
                "b4_schedules_file_sha256": sha256_file(b4_path),
                "a5_batch_report_file_sha256": sha256_file(A5_BATCH),
                "a5_runtime_report_file_sha256": sha256_file(A5_RUNTIME),
            },
            "selected_schedules": schedules,
            "worker_ownership": {
                "hardware": "A100-40GB",
                "max_parallel_workers": len(family_slots),
                "family_to_logical_slot": family_slots,
                "cell_to_logical_slot": owners,
                "binding_rule": (
                    "one model family per logical slot; A8 must verify actual worker GPU "
                    "and checkpoint"
                ),
            },
            "budget_lock": {
                "meter_origin_a100_hours": ext.METER_ORIGIN_HOURS,
                "hard_stop_a100_hours": ext.STOP_HOURS,
                "protected_reserve_a100_hours": ext.RESERVE_HOURS,
                "complete_block_guard_multiplier": ext.FORECAST_MULTIPLIER,
                "a5_batch_invariance_metered_a100_hours": batch["metered_a100_hours"],
                "a5_synthetic_runtime_metered_proxy_a100_hours": synthetic[
                    "metered_proxy_a100_hours"
                ],
                "a6_receipt_metered_a100_hours": meter["cumulative_a100_hours"],
                "conservative_prior_use_including_proxy_a100_hours": prior,
                "guarded_headroom_after_reserve_a100_hours": ext.STOP_HOURS
                - ext.RESERVE_HOURS
                - prior,
                "start_rule": (
                    "prior_use + 1.6 * complete_block_base_forecast + 10 <= 80; "
                    "reforecast after each model's first 1000 attempts"
                ),
                "note": "A5 runtime is a metered proxy, not an authoritative Modal billing value",
            },
        }
    )
    return base


def report(payload: dict[str, Any]) -> str:
    cells = payload["solo_metrics"]
    sizes = payload["item_counts"]
    lines = [
        "# X1R4 A7 admission decision",
        "",
        "The completed solo screen admitted seven of twenty cells in four model families. "
        "This is an admission decision, not a joint-dependence or coverage result. "
        "No extension joint output was used.",
        "",
        "| Family | Task | Solo status | Accuracy | Parse rate | Decision | Items |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for key in sorted(cells):
        row = cells[key]
        decision = "admitted" if row["admitted"] else ", ".join(row["reasons"])
        n = str(sizes[key]["n_items"]) if key in sizes else "—"
        accuracy = f"{row['accuracy']:.3f}" if "accuracy" in row else "—"
        lines.append(
            f"| {row['family']} | {row['task_id']} | "
            f"{','.join(f'{k}:{v}' for k, v in row['status_counts'].items())} | "
            f"{accuracy} | {row['parse_rate']:.3f} | {decision} | {n} |"
        )
    lines.extend(
        [
            "",
            "The B3 family core is " + ", ".join(payload["ordered_cells"]["family_core"]) + ".",
            "The B4 repeated-sampling cells are "
            + ", ".join(payload["ordered_cells"]["e5_cells"])
            + ".",
            "",
            f"Frozen selected calls: {payload['selected_schedules']['b3_total_calls']:,} B3 "
            f"and {payload['selected_schedules']['b4_total_calls']:,} B4 across all prefixes. "
            "Each selected schedule and the full source files have SHA-256 bindings in "
            "ADMISSION_LOCK.json.",
            "",
            "Four logical A100-40GB worker slots own one family each. These are slot "
            "assignments, not a record of provisioned GPUs. A8 must verify actual workers.",
            "",
            f"The conservative recorded prior use is "
            f"{payload['budget_lock']['conservative_prior_use_including_proxy_a100_hours']:.6f} "
            "A100-hours, including a synthetic-runtime proxy. The extension stop remains "
            "80 hours with a protected 10-hour reserve and a 1.6× complete-block start guard. "
            "A8 must reforecast after each model's first 1,000 attempts.",
            "",
        ]
    )
    return "\n".join(lines)


def seal() -> Path:
    payload = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = OUTPUT / f"{stamp}-X1R4-A7"
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".a7-stage-", dir=OUTPUT) as temporary:
        stage = Path(temporary)
        write_json(stage / "ADMISSION_LOCK.json", payload)
        (stage / "REPORT.md").write_text(report(payload), encoding="utf-8")
        manifest = {
            "schema_version": "ext-x1r4-a7-manifest-v1",
            "status": "sealed",
            "scientific_model_outputs": False,
            "files_sha256": {
                "ADMISSION_LOCK.json": sha256_file(stage / "ADMISSION_LOCK.json"),
                "REPORT.md": sha256_file(stage / "REPORT.md"),
            },
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        write_json(stage / "MANIFEST.json", manifest)
        stage.rename(destination)
    return destination


if __name__ == "__main__":
    print(seal())
