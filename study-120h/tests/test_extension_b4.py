"""B4 operational checks use synthetic responses and never import a GPU transport."""

from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from freezeb import e2
from freezeb import extension as ext
from freezeb import extension_b4 as b4
from step0.hashing import sha256_json, write_json

STUDY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b4_cpu_runner", STUDY / "modal_extension_b4.py")
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    return [
        {"doc_id": f"d{i}", "text": "A neutral comment.", "n_chars": 18, "reference_normalized": 0}
        for i in range(8192)
    ]


def response(job: dict[str, Any]) -> dict[str, Any]:
    return e2.event_for_result(
        job,
        raw=json.dumps({"labels": {doc_id: 0 for doc_id in job["target_ids"]}}),
        finish_reason="stop",
    )


def budget(plan: dict[str, Any]) -> dict[str, Any]:
    value = {
        "plan_sha256": plan["plan_sha256"],
        "profile": "cpu-test",
        "workspace_enabled": True,
        "exclusive_gpu_work": True,
        "verified_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "prior_extension_a100_hours": 0,
        "spend_limit_usd": 1000,
        "spend_used_usd": 0,
        "a100_usd_per_hour": 4,
        "base_forecast_hours": {
            f"{cell}|{stream}": 1 for cell in b4.CELLS for stream in ("reference", "evaluation")
        },
    }
    return {**value, "record_sha256": sha256_json(value)}


def payload(planned: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    model = planned[0]["model_id"]
    revision = ext.ext_policy_for(model).revision
    return {
        "chunk_id": sha256_json([job["attempt_id"] for job in planned]),
        "events": events,
        "host": {"gpu_name": "A100", "installed_engine_version_string": "0.28.0"},
        "backend": {
            "backend": "vllm",
            "model_id": model,
            "model_revision": revision,
            "tokenizer_revision": revision,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_num_seqs": 32,
            "seed": 20260912,
            "batch_invariant": True,
            "batch_invariance_required": True,
        },
    }


def test_default_runner_does_not_import_modal_transport() -> None:
    assert "extension_b4_transport" not in sys.modules
    assert runner.phases(400) == [(cell, "reference", 250) for cell in b4.CELLS] + [
        (cell, "evaluation", 400) for cell in b4.CELLS
    ]
    assert [count for _, stream, count in runner.phases(2500) if stream == "evaluation"] == (
        [400] * 4 + [1000] * 4 + [2500] * 4
    )


def test_schedules_are_fresh_disjoint_streams_and_nested(rows: list[dict[str, Any]]) -> None:
    cell = b4.CELLS[-1]
    first = list(b4.jobs(cell, rows, "reference", 2))
    evaluation = list(b4.jobs(cell, rows, "evaluation", 2))
    assert len(first) == 64
    assert len({doc_id for job in first[:32] for doc_id in job["target_ids"]}) == 512
    assert not {job["call_id"] for job in first} & {job["call_id"] for job in evaluation}
    assert list(b4.jobs(cell, rows, "evaluation", 1)) == evaluation[:32]
    sealed = list(
        ext.b4_calls(
            cell.rsplit("::", 1)[0],
            "civil_comments",
            tuple(row["doc_id"] for row in rows),
            "evaluation",
            2,
        )
    )
    for job, record in zip(evaluation, sealed, strict=True):
        for key, value in record.items():
            assert job[key] == value


def test_exact_duplicate_is_idempotent_conflicting_duplicate_fails(
    rows: list[dict[str, Any]],
) -> None:
    planned = list(b4.jobs(b4.CELLS[-1], rows, "reference", 1))
    event = response(planned[0])
    assert len(b4.validate_events(planned, [event, event])) == 1
    bad = {**event, "queue_seconds": 99}
    with pytest.raises(ValueError, match="conflicting"):
        b4.validate_events(planned, [event, bad])


@pytest.mark.parametrize("field", ["doc_order", "prompt_sha256", "output_sha256", "predictions"])
def test_tampered_identity_or_parse_is_rejected(rows: list[dict[str, Any]], field: str) -> None:
    planned = list(b4.jobs(b4.CELLS[-1], rows, "reference", 1))
    event = response(planned[0])
    event[field] = "tampered"
    with pytest.raises(ValueError):
        b4.validate_events(planned, [event])


def test_one_retry_only_and_no_retry_for_malformed(rows: list[dict[str, Any]]) -> None:
    planned = list(b4.jobs(b4.CELLS[-1], rows, "reference", 1))
    first = e2.event_for_result(planned[0], generation_error="synthetic failure")
    retry = e2.retry_job(planned[0])
    assert b4.pending(planned, [first])[0] == retry
    terminal = e2.event_for_result(retry, raw="unfinished", finish_reason="length")
    assert len(b4.pending(planned, [first, terminal])) == 31
    malformed = e2.event_for_result(planned[0], raw="not json", finish_reason="stop")
    assert len(b4.pending(planned, [malformed])) == 31
    with pytest.raises(ValueError, match="unpermitted"):
        b4.validate_events(planned, [malformed, response(retry)])
    with pytest.raises(ValueError, match="unpermitted"):
        b4.validate_events(planned, [response(retry)])


def test_resolved_arrays_keep_call_and_replicate_units(rows: list[dict[str, Any]]) -> None:
    planned = list(b4.jobs(b4.CELLS[-1], rows, "reference", 2))
    stored = [response(job) for job in planned]
    array, report = b4.stream_errors(planned, stored, rows)
    assert array.shape == (2, 32, 16)
    assert np.all(array == 0) and report["parsed_complete"]
    stored[0] = e2.event_for_result(planned[0], raw="bad json", finish_reason="stop")
    array, report = b4.stream_errors(planned, stored, rows)
    assert np.isnan(array[0, 0]).all()
    assert report["unresolved_calls"] == 1 and not report["parsed_complete"]


def test_budget_requires_hash_freshness_profile_and_finite_values() -> None:
    plan = {"plan_sha256": "cpu-plan"}
    original = budget(plan)
    b4.verify_budget(original, plan, "cpu-test")
    for key, value in (
        ("workspace_enabled", False),
        ("prior_extension_a100_hours", float("nan")),
        ("verified_at_utc", "2020-01-01T00:00:00+00:00"),
        ("profile", "other"),
    ):
        changed = {**original, key: value}
        changed["record_sha256"] = sha256_json(
            {k: v for k, v in changed.items() if k != "record_sha256"}
        )
        with pytest.raises(ValueError):
            b4.verify_budget(changed, plan, "cpu-test")
    with pytest.raises(ValueError, match="hash"):
        b4.verify_budget({**original, "spend_limit_usd": 900}, plan, "cpu-test")


def test_receipts_are_retained_and_recovery_is_idempotent(
    tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    cell = b4.CELLS[-1]
    planned = list(b4.jobs(cell, rows, "reference", 1))
    plan = {
        "serving_policy": {
            planned[0]["model_id"]: ext.ext_policy_for(planned[0]["model_id"]).revision
        }
    }
    receipt = payload(planned, [response(job) for job in planned])
    path = b4.stream_path(tmp_path, cell, "reference")
    runner.ingest(tmp_path, path, planned, receipt, plan, cell)
    before = path.read_bytes()
    runner.ingest(tmp_path, path, planned, receipt, plan, cell)
    assert path.read_bytes() == before
    rejected = copy.deepcopy(receipt)
    rejected["backend"]["seed"] = 0
    with pytest.raises(ValueError, match="serving"):
        runner.ingest(tmp_path, path, planned, rejected, plan, cell)
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 2


def test_unsettled_meter_fails_closed(tmp_path: Path) -> None:
    record = budget({"plan_sha256": "cpu-plan"})
    b4.append(
        tmp_path / "meter.jsonl",
        [{"kind": "start", "phase_id": "lost", "budget_sha256": record["record_sha256"]}],
    )
    with pytest.raises(RuntimeError, match="unsettled"):
        runner.meter(tmp_path, record)


def test_remote_recovery_orders_first_attempts_before_retries(
    tmp_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    cell = b4.CELLS[-1]
    planned = list(b4.jobs(cell, rows, "reference", 1))
    first = e2.event_for_result(planned[0], generation_error="synthetic failure")
    retry = e2.retry_job(planned[0])
    plan = {
        "serving_policy": {
            planned[0]["model_id"]: ext.ext_policy_for(planned[0]["model_id"]).revision
        }
    }
    receipts = [payload([retry], [response(retry)]), payload([planned[0]], [first])]
    path = b4.stream_path(tmp_path, cell, "reference")
    cache: dict[str, dict[str, Any]] = {}
    runner.recover_chunks(tmp_path, path, planned, receipts, plan, cell, cache)
    assert [event["attempt"] for event in b4.events(path)] == [1, 2]
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 2
    before = path.read_bytes()
    runner.recover_chunks(tmp_path, path, planned, receipts, plan, cell, cache)
    assert before == path.read_bytes()


def test_full_synthetic_stream_to_frozen_coverage(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = b4.CELLS[-1]
    plan = {
        "plan_sha256": "cpu-plan",
        "serving_policy": {
            key.rsplit("::", 1)[0]: ext.ext_policy_for(key.rsplit("::", 1)[0]).revision
            for key in b4.CELLS
        },
    }
    monkeypatch.setattr(b4, "build_plan", lambda study, prefix: plan)
    monkeypatch.setattr(b4, "frame", lambda study, task: rows)
    write_json(tmp_path / "PLAN-400.json", plan)
    rng = np.random.default_rng(20260915)
    for stream, count in (("reference", 250), ("evaluation", 400)):
        planned = list(b4.jobs(cell, rows, stream, count, render=False))
        stored = []
        for job in planned:
            label = int(rng.random() < 0.2)
            stored.append(
                e2.event_for_result(
                    job,
                    raw=json.dumps({"labels": dict.fromkeys(job["target_ids"], label)}),
                    finish_reason="stop",
                )
            )
        receipt = payload(planned, stored)
        write_json(tmp_path / "receipts" / f"{sha256_json(receipt)}.json", receipt)
        b4.append(b4.stream_path(tmp_path, cell, stream), stored)
    result = b4.analyze(STUDY, tmp_path)
    measured = result["cells"][-1]
    assert measured["status"] == "analyzed"
    assert measured["analysis"]["n_evaluation"] == 400
    assert measured["analysis"]["coverage"] < 0.90
    assert result["coverage_gate"]["passed"] is False  # only one complete family
    assert "Frozen C2 gate" in b4.render_report(result)


def test_remote_failure_blocks_next_cell_and_requires_reconciliation(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_jobs = b4.jobs
    monkeypatch.setattr(b4, "frame", lambda study, task: rows)
    monkeypatch.setattr(
        b4,
        "jobs",
        lambda cell, data, stream, count, **kwargs: original_jobs(cell, data, stream, 1, **kwargs),
    )
    plan = {
        "plan_sha256": "cpu-plan",
        "evaluation_prefix": 400,
        "schedules": dict.fromkeys(b4.CELLS, {}),
    }
    launches = []

    class Failed:
        @contextmanager
        def phase(self, cell: str, stream: str, schedule: Any) -> Iterator[Any]:
            launches.append(cell)
            raise RuntimeError("synthetic startup failure")
            yield self

    with pytest.raises(RuntimeError, match="startup"):
        runner.dispatch(STUDY, tmp_path, plan, budget(plan), Failed())
    assert launches == [b4.CELLS[0]]
    with pytest.raises(RuntimeError, match="unsettled"):
        runner.meter(tmp_path, budget(plan))


def test_cpu_budget_preparation_and_meter_reconciliation_clis(tmp_path: Path) -> None:
    core = {"schema_version": "cpu-test-plan"}
    plan = {**core, "plan_sha256": sha256_json(core)}
    write_json(tmp_path / "plan.json", plan)
    evidence = budget(plan)
    evidence.pop("record_sha256")
    write_json(tmp_path / "evidence.json", evidence)
    result = subprocess.run(
        [
            sys.executable,
            str(STUDY / "prepare_extension_b4_budget.py"),
            "--plan",
            str(tmp_path / "plan.json"),
            "--input",
            str(tmp_path / "evidence.json"),
            "--profile",
            "cpu-test",
            "--output",
            str(tmp_path / "budget.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["gpu_work_launched"] is False
    record = b4.load(tmp_path / "budget.json")
    b4.verify_budget(record, plan, "cpu-test")
    b4.append(
        tmp_path / "meter.jsonl",
        [{"kind": "start", "phase_id": "interrupted", "budget_sha256": record["record_sha256"]}],
    )
    command = [
        sys.executable,
        str(STUDY / "reconcile_extension_b4_meter.py"),
        "--run",
        str(tmp_path),
        "--phase-id",
        "interrupted",
        "--billed-a100-hours",
        "0.25",
        "--billing-evidence",
        str(tmp_path / "evidence.json"),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    assert runner.meter(tmp_path, record) == 0.25
    before = (tmp_path / "meter.jsonl").read_bytes()
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0
    assert (tmp_path / "meter.jsonl").read_bytes() == before


def test_dispatch_reference_barrier_resume_and_budget_stop(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep real identities/32-call replicate units, shorten only this fake orchestration test.
    original_jobs = b4.jobs
    monkeypatch.setattr(b4, "frame", lambda study, task: rows)
    monkeypatch.setattr(
        b4,
        "jobs",
        lambda cell, data, stream, count, **kwargs: original_jobs(cell, data, stream, 1, **kwargs),
    )
    monkeypatch.setattr(
        runner,
        "phases",
        lambda prefix: (
            [(cell, "reference", 1) for cell in b4.CELLS]
            + [(cell, "evaluation", 1) for cell in b4.CELLS]
        ),
    )
    plan = {
        "plan_sha256": "cpu-plan",
        "evaluation_prefix": 400,
        "schedules": dict.fromkeys(b4.CELLS, {}),
        "serving_policy": {
            cell.rsplit("::", 1)[0]: ext.ext_policy_for(cell.rsplit("::", 1)[0]).revision
            for cell in b4.CELLS
        },
    }
    launches = []

    class Fake:
        @contextmanager
        def phase(self, cell: str, stream: str, schedule: Any) -> Iterator[Any]:
            launches.append((cell, stream))
            yield self

        def recover(self, namespace: str) -> list[Any]:
            return []

        def run(
            self, namespace: str, chunk_id: str, group: list[Any], remaining: float
        ) -> dict[str, Any]:
            result = payload(group, [response(job) for job in group])
            requested_model = launches[-1][0].rsplit("::", 1)[0]
            result["backend"]["model_id"] = requested_model
            return result

    record = budget(plan)
    runner.dispatch(STUDY, tmp_path, plan, record, Fake())
    assert [stream for _, stream in launches] == ["reference"] * 4 + ["evaluation"] * 4
    runner.dispatch(STUDY, tmp_path, plan, record, Fake())
    assert len(launches) == 8
    assert runner.meter(tmp_path, record) >= 8 * 60 / 3600
    blocked = {**record, "prior_extension_a100_hours": 70}
    with pytest.raises(RuntimeError, match="budget"):
        runner.dispatch(STUDY, tmp_path / "blocked", plan, blocked, Fake())


def test_empty_ledgers_cannot_make_coverage_claim(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"plan_sha256": "cpu-plan"}
    monkeypatch.setattr(b4, "build_plan", lambda study, prefix: plan)
    monkeypatch.setattr(b4, "frame", lambda study, task: rows)
    write_json(tmp_path / "PLAN-400.json", plan)
    result = b4.analyze(STUDY, tmp_path)
    assert all(cell["status"] == "coverage_unavailable" for cell in result["cells"])
    assert result["coverage_gate"]["passed"] is False


@pytest.mark.skipif(
    not (STUDY / b4.FRAME / "imdb/confirm.jsonl").exists(), reason="private frames absent"
)
def test_real_a7_schedule_and_source_bindings() -> None:
    plan = b4.build_plan(STUDY)
    assert plan["cells"] == list(b4.CELLS)
    assert plan["first_calls"] == 83200
    assert plan["stop_a100_hours"] == 80
