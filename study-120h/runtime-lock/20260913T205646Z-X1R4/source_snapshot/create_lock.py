"""Create the X1R4 runtime/access lock from completed synthetic M0 evidence."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
import tempfile
from pathlib import Path

STUDY = Path("/Users/dz/bigdata/study-120h")
SEAL = STUDY / "freeze/20260913T183353Z-X1R4"
BATCH = STUDY / "extension-batch-invariance/20260913T191010Z-4807e7d1"
RUNTIME = STUDY / "extension-runtime/20260913T200450Z-a1f3bd0f"
STAGING = STUDY / "extension-staging/20260913T191025Z-x1r4"
AMENDMENT = STUDY / "amendments/20260913T190555Z-x1r4-modal-scaledown"
sys.path.insert(0, str(STUDY / "src"))

from freezeb import batch_invariance as bi
from freezeb.extension import (
    EXT_MODELS,
    AccessRecord,
    runtime_access_lock,
)
from step0.hashing import sha256_file, sha256_json


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checked_plan(path: Path) -> dict:
    plan = read(path)
    assert isinstance(plan, dict)
    digest = plan.pop("plan_sha256")
    assert sha256_json(plan) == digest, path
    plan["plan_sha256"] = digest
    return plan


def main() -> None:
    manifest = read(SEAL / "MANIFEST.json")
    assert isinstance(manifest, dict)
    assert len(manifest["files_sha256"]) == 55
    for name, digest in manifest["files_sha256"].items():
        assert sha256_file(SEAL / name) == digest, name
    expected_models = [
        {"model_id": model.repo, "revision": model.revision} for model in EXT_MODELS
    ]

    batch_plan = checked_plan(BATCH / "plan.json")
    batch_report = read(BATCH / "report.json")
    batch_events = read(BATCH / "events.json")
    assert isinstance(batch_report, dict) and isinstance(batch_events, list)
    assert batch_plan["models"] == expected_models
    assert batch_plan["plan_sha256"] == manifest["batch_invariance_plan_sha256"]
    assert batch_report["plan_sha256"] == batch_plan["plan_sha256"]
    validation = bi.validate_events(batch_events, expected_models)
    assert batch_report["status"] == validation["status"] == "passed"
    for field in ("events_verified", "anchors_verified", "sessions_per_model", "cases_per_session"):
        assert batch_report[field] == validation[field], field
    batch_report_sha = sha256_file(BATCH / "report.json")

    runtime_plan = checked_plan(RUNTIME / "plan.json")
    runtime_report = read(RUNTIME / "report.json")
    assert isinstance(runtime_report, dict)
    assert runtime_plan["models"] == expected_models
    assert runtime_plan["seal_manifest_sha256"] == manifest["manifest_sha256"]
    assert runtime_plan["runtime_helper_sha256"] == sha256_file(RUNTIME / "runtime_helper.py")
    assert runtime_plan["batch_invariant"] is True and runtime_plan["vllm"] == "0.28.0"
    assert runtime_report["complete"] is True
    assert runtime_report["plan_sha256"] == runtime_plan["plan_sha256"]
    assert [row["model_id"] for row in runtime_report["models"]] == [
        row["model_id"] for row in expected_models
    ]
    expected_cases = {case["case_id"] for case in runtime_plan["cases"]}
    assert len(expected_cases) == 20
    maxima = []
    candidate_rows = []
    runtime_seconds = 0.0
    for expected, summary in zip(expected_models, runtime_report["models"], strict=True):
        model_id = expected["model_id"]
        worker = read(RUNTIME / (model_id.replace("/", "--") + ".json"))
        assert isinstance(worker, dict) and summary["status"] == "passed"
        assert worker["model_id"] == model_id and worker["revision"] == expected["revision"]
        backend = worker["backend"]
        assert backend["model_id"] == model_id
        assert backend["model_revision"] == backend["tokenizer_revision"] == expected["revision"]
        assert backend["batch_invariant"] is backend["batch_invariance_required"] is True
        assert backend["dtype"] == "bfloat16" and backend["max_num_seqs"] == 32
        assert backend["max_model_len"] == 32768 and backend["tensor_parallel_size"] == 1
        assert backend["seed"] == 20260912
        assert worker["host"]["installed_engine_version_string"] == "0.28.0"
        assert "A100" in worker["host"]["gpu_name"]
        events = worker["events"]
        assert len(events) == 80 and all(event["status"] == "ok" for event in events)
        assert {event["case_id"] for event in events} == expected_cases
        fits = [event for event in events if event["phase"] == "fit"]
        warmups = [event for event in events if event["phase"] == "warmup"]
        measures = [event for event in events if event["phase"] == "measure"]
        assert len(fits) == len(warmups) == 20 and len(measures) == 40
        assert all(event["finish_reason"] == "stop" for event in warmups + measures)
        maximum = max(event["required_tokens"] for event in fits)
        assert maximum <= 32768
        maxima.append(maximum)
        assert summary["fit_cases"] == 20 and summary["measured_calls"] == 40
        assert abs(summary["metered_a100_hours"] * 3600 - worker["metered_proxy_seconds"]) < 1e-5
        runtime_seconds += worker["metered_proxy_seconds"]
        candidate_rows.append({
            "model_id": model_id,
            "revision": expected["revision"],
            "status": "attempted",
            "access_status": "verified",
            "batch_invariance_status": "passed",
            "synthetic_runtime_status": "passed",
            "max_required_tokens": maximum,
            "context_window_tokens": 32768,
            "runtime_worker_sha256": sha256_file(RUNTIME / (model_id.replace("/", "--") + ".json")),
        })
    assert abs(runtime_seconds / 3600 - runtime_report["metered_proxy_a100_hours"]) < 1e-8
    staging = read(STAGING / "report.json")
    assert isinstance(staging, dict) and staging["gpu_used"] is False
    assert staging["seal_manifest_sha256"] == manifest["manifest_sha256"]
    for row in staging["models"]:
        assert row["revision"] == next(
            model["revision"] for model in expected_models if model["model_id"] == row["model_id"]
        )

    lock_root = STUDY / "runtime-lock"
    lock_root.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = lock_root / f"{stamp}-X1R4"
    if destination.exists():
        raise FileExistsError(destination)
    stage = Path(tempfile.mkdtemp(prefix=".x1r4-lock-stage-", dir=lock_root))
    try:
        sources = {
            "evidence/batch/plan.json": BATCH / "plan.json",
            "evidence/batch/report.json": BATCH / "report.json",
            "evidence/batch/events.json": BATCH / "events.json",
            "evidence/batch/workers.json": BATCH / "workers.json",
            "evidence/runtime/plan.json": RUNTIME / "plan.json",
            "evidence/runtime/report.json": RUNTIME / "report.json",
            "evidence/runtime/runtime_helper.py": RUNTIME / "runtime_helper.py",
            "evidence/staging/report.json": STAGING / "report.json",
            "evidence/amendment.json": AMENDMENT.with_suffix(".json"),
            "evidence/amendment.py": AMENDMENT.with_suffix(".py"),
            "source_snapshot/create_lock.py": Path(__file__),
        }
        for model in expected_models:
            name = model["model_id"].replace("/", "--") + ".json"
            sources[f"evidence/runtime/{name}"] = RUNTIME / name
        files_sha = {}
        for name, source in sources.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files_sha[name] = sha256_file(target)
            assert files_sha[name] == sha256_file(source)

        technical = {
            "schema_version": "ext-x1r4-synthetic-runtime-lock-v1",
            "status": "passed",
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "x1r4_manifest_identifier": manifest["manifest_sha256"],
            "sealed_batch_plan_sha256": batch_plan["plan_sha256"],
            "batch_report_sha256": batch_report_sha,
            "runtime_report_sha256": sha256_file(RUNTIME / "report.json"),
            "staging_report_sha256": sha256_file(STAGING / "report.json"),
            "batch_events_verified": validation["events_verified"],
            "runtime_fit_checks": 100,
            "runtime_warmup_calls": 100,
            "runtime_measured_calls": 200,
            "maximum_required_tokens": max(maxima),
            "context_window_tokens": 32768,
            "batch_metered_a100_hours": batch_report["metered_a100_hours"],
            "runtime_metered_proxy_a100_hours": runtime_report["metered_proxy_a100_hours"],
            "combined_a100_hours_proxy": batch_report["metered_a100_hours"] + runtime_report["metered_proxy_a100_hours"],
            "candidates": candidate_rows,
            "limitations": [
                "Synthetic prompts and two fresh batch sessions do not prove universal bit-exact GPU behavior.",
                "The runtime peak_gpu_memory_bytes field is invalid because vLLM allocates in a child process.",
                "Runtime metering is a worker/invocation proxy, not a Modal billing invoice.",
            ],
            "scientific_model_outputs": False,
        }
        write(stage / "RUNTIME_LOCK.json", technical)
        files_sha["RUNTIME_LOCK.json"] = sha256_file(stage / "RUNTIME_LOCK.json")
        access = runtime_access_lock(
            [AccessRecord(row["model_id"], True, "verified", "Pinned checkpoint loaded offline and passed both synthetic M0 checks.") for row in candidate_rows],
            signed_by="Codex, on Daniel's 2026-09-13 instruction",
            runtime_lock_sha256=files_sha["RUNTIME_LOCK.json"],
            batch_invariance_report_sha256=batch_report_sha,
        )
        access["signature_kind"] = "attributed agent attestation; no cryptographic or personal signature"
        write(stage / "ACCESS_LOCK.json", access)
        files_sha["ACCESS_LOCK.json"] = sha256_file(stage / "ACCESS_LOCK.json")
        write(stage / "MANIFEST.json", {
            "schema_version": "ext-x1r4-runtime-access-manifest-v1",
            "status": "sealed",
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "x1r4_manifest_identifier": manifest["manifest_sha256"],
            "n_candidates": 5,
            "n_attempted": 5,
            "contains_scientific_model_outputs": False,
            "contains_credentials": False,
            "files_sha256": files_sha,
        })
        for name, digest in files_sha.items():
            assert sha256_file(stage / name) == digest
        stage.rename(destination)
    except Exception:
        shutil.rmtree(stage)
        raise
    print(json.dumps({"lock_directory": str(destination), "runtime_lock_sha256": files_sha["RUNTIME_LOCK.json"], "access_lock_sha256": files_sha["ACCESS_LOCK.json"], "manifest_file_sha256": sha256_file(destination / "MANIFEST.json"), "files_verified": len(files_sha)}))


if __name__ == "__main__":
    main()
