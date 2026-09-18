"""Launcher, ledger, receipt and seal checks. Nothing here contacts Runpod."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from freezeb import b5
from step0.hashing import read_jsonl, sha256_file, sha256_json, write_json

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))

import run_b5  # noqa: E402
import seal_b5  # noqa: E402
from run_output_recovery import (  # noqa: E402
    ACCOUNT_ID,
    DATA_CENTER,
    GPU_ID,
    VOLUMES,
    append,
)

CELL = b5.CELLS[0]
TOKEN = ("c" * 64)[:12]


def frame_ids() -> list[str]:
    return [f"cc{index:014x}" for index in range(b5.N_ITEMS)]


class StubPlan(run_b5.Plan):
    """A plan built from synthetic documents. It reads no sealed frame and no network."""

    def __init__(self, cells: tuple[str, ...] = b5.CELLS) -> None:
        self.cells = cells
        self.ids = frame_ids()
        self.text_map = {doc_id: f"synthetic document {doc_id}" for doc_id in self.ids}
        self.seal = {
            "seal": "freeze/20260918T000000Z-B5",
            "manifest_sha256": "c" * 64,
            "schedule_sha256": "d" * 64,
            "files_verified": len(seal_b5.CODE_FILES),
            "preparation_directory": "extension-b5/stub",
            "confirmation_token": TOKEN,
        }
        self.schedule = {
            "schedule_sha256": "d" * 64,
            "observed_counts": {
                "joint": b5.EXPECTED_JOINT,
                "exact_context": b5.EXPECTED_EXACT,
                "total": b5.EXPECTED_TOTAL,
            },
        }
        self.forecast = {
            "point_metered_a100_hours": 18.0,
            "upper_metered_a100_hours": 70.0,
            "planning_point_balance_usd": 30.76,
            "full_launch_required_balance_usd": None,
            "smoke_required_balance_usd": 5.18,
            "retry_fraction": 0.02,
            "boot_hours_per_model": 0.25,
            "coefficients": {
                "seconds_per_prompt_token": 0.001,
                "seconds_per_output_token": 0.002,
                "seconds_per_chunk": 1.0,
            },
            "scenarios": {
                "prefix_reuse": {
                    "blocks": [
                        {"cell_key": key, "batch_size": k, "generation_hours": 0.5}
                        for key in cells
                        for k in b5.SIZES
                    ]
                }
            },
        }
        self.context = {
            "cells": [
                {"cell_key": key, "worst_case": {"required_tokens": 13_500}, "fits": True}
                for key in cells
            ]
        }

    def doc_ids(self, cell_key: str) -> list[str]:
        return self.ids

    def texts(self, cell_key: str) -> dict[str, str]:
        return self.text_map

    def blocks(self) -> list[tuple[str, int]]:
        return [(cell_key, k) for cell_key in self.cells for k in b5.SIZES]


def one_cell_plan() -> StubPlan:
    return StubPlan((CELL,))


def first_chunk(plan: StubPlan, k: int = 2) -> dict[str, Any]:
    return plan.chunks(CELL, k, 64)[0]


def make_event(
    plan: StubPlan,
    call: dict[str, Any],
    *,
    attempt: int = 1,
    status: str = "ok",
    parse_status: str = "ok",
) -> dict[str, Any]:
    """Exactly what the worker must return for `verify_event` to accept the call."""
    targets = [str(value) for value in call["target_ids"]]
    resolved = status == "ok" and parse_status == "ok"
    event = {
        "call_id": call["call_id"],
        "attempt": attempt,
        "attempt_id": f"{call['call_id']}::attempt::{attempt}",
        "cell_key": call["cell_key"],
        "condition": call["condition"],
        "batch_size": call["batch_size"],
        "assignment": call["assignment"],
        "target_ids": targets,
        "target_ids_sha256": call["target_ids_sha256"],
        "doc_order_sha256": call["doc_order_sha256"],
        "json_schema_sha256": call["json_schema_sha256"],
        "max_tokens": call["max_tokens"],
        "prompt_sha256": sha256_json(
            b5.render_messages(str(call["task_id"]), plan.texts(str(call["cell_key"])), call)
        ),
        "structured_output": True,
        "status": status,
        "parse_status": parse_status,
        "predictions": dict.fromkeys(targets, 0) if resolved else None,
        "prompt_tokens": 100,
        "output_tokens": 5,
        "num_cached_tokens": 50 if call["condition"] == "exact_context" else 0,
        "num_cache_creation_tokens": 50,
    }
    if attempt == 2:
        event["parent_attempt_id"] = f"{call['call_id']}::attempt::1"
    return event


def chunk_events(plan: StubPlan, chunk: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
    return [make_event(plan, call, **kwargs) for call in run_b5.expected_calls(chunk).values()]


def write_receipt(
    output: Path, chunk: dict[str, Any], events: list[dict[str, Any]], attempt: int = 1
) -> Path:
    chunk_id = sha256_json([str(event["attempt_id"]) for event in events])
    return write_json(
        output / "receipts" / f"{chunk_id}.json",
        {
            "chunk_id": chunk_id,
            "source_chunk_id": chunk["chunk_id"],
            "attempt": attempt,
            "events": events,
        },
    )


def refuse_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a read-only B5 mode contacted Runpod")

    monkeypatch.setattr(run_b5, "cli", blocked)
    monkeypatch.setattr(run_b5, "command", blocked)


# --- read-only modes create nothing ------------------------------------------


def test_dry_run_creates_no_pod_and_touches_no_run_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refuse_network(monkeypatch)
    plan = one_cell_plan()
    chunks = run_b5.selected_chunks(plan, 512, False)
    record = run_b5.dry_run(plan, tmp_path, chunks, False)
    assert record["pod_created"] is False
    assert record["gpu_work_launched"] is False
    assert record["calls"] == b5.expected_counts()["total_per_cell"]
    assert record["expected_counts"]["total"] == b5.EXPECTED_TOTAL
    assert sorted(path.name for path in tmp_path.iterdir()) == ["DRY_RUN-full.json"]


def test_verify_only_and_status_create_no_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refuse_network(monkeypatch)
    plan = one_cell_plan()
    chunks = [first_chunk(plan)]
    assert run_b5.verify_only(plan, tmp_path, chunks)["pod_created"] is False
    assert run_b5.status(plan, tmp_path, chunks)["pod_created"] is False
    assert list(tmp_path.iterdir()) == []


def test_preflight_creates_no_pod_when_the_account_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refuse_network(monkeypatch)
    plan = one_cell_plan()
    record = run_b5.preflight(
        plan,
        tmp_path,
        [first_chunk(plan)],
        ceiling_hours=35.0,
        usd_per_hour=1.59,
        reserve_usd=2.0,
        live=False,
        confirm=TOKEN,
        smoke=True,
    )
    assert record["pod_created"] is False
    assert record["live_account_checks"] is None
    assert record["gates"]["confirmation_token_supplied"] is True


def test_smoke_selects_whole_chunks_of_the_same_schedule() -> None:
    plan = one_cell_plan()
    smoke = run_b5.selected_chunks(plan, run_b5.SMOKE_CHUNK_CALLS, True)
    assert len(smoke) == len(b5.SIZES)
    assert {int(chunk["batch_size"]) for chunk in smoke} == set(b5.SIZES)
    for chunk in smoke:
        block = plan.chunks(CELL, int(chunk["batch_size"]), run_b5.SMOKE_CHUNK_CALLS)
        assert chunk == block[0]
        assert int(chunk["calls"]) <= run_b5.SMOKE_CHUNK_CALLS
    assert sum(int(chunk["calls"]) for chunk in smoke) < 400


def test_full_selection_covers_every_scheduled_call_once() -> None:
    plan = one_cell_plan()
    chunks = run_b5.selected_chunks(plan, 512, False)
    call_ids = [call_id for chunk in chunks for call_id in run_b5.expected_calls(chunk)]
    assert len(call_ids) == len(set(call_ids)) == b5.expected_counts()["total_per_cell"]


# --- event identity ----------------------------------------------------------


def test_verify_event_accepts_the_planned_call() -> None:
    plan = one_cell_plan()
    call = next(iter(run_b5.expected_calls(first_chunk(plan)).values()))
    run_b5.verify_event(plan.texts(CELL), make_event(plan, call), call, 1)


@pytest.mark.parametrize(
    "field, value",
    [
        ("prompt_sha256", "0" * 64),
        ("json_schema_sha256", "0" * 64),
        ("target_ids_sha256", "0" * 64),
        ("doc_order_sha256", "0" * 64),
        ("max_tokens", 4096),
        ("condition", "exact_context"),
        ("batch_size", 32),
        ("assignment", 3),
        ("structured_output", False),
        ("attempt_id", "someone-else"),
    ],
)
def test_verify_event_refuses_any_changed_contract_field(field: str, value: Any) -> None:
    plan = one_cell_plan()
    call = next(iter(run_b5.expected_calls(first_chunk(plan)).values()))
    event = make_event(plan, call)
    event[field] = value
    with pytest.raises(RuntimeError, match="identity differs"):
        run_b5.verify_event(plan.texts(CELL), event, call, 1)


def test_verify_event_refuses_a_response_whose_ids_differ() -> None:
    plan = one_cell_plan()
    call = next(iter(run_b5.expected_calls(first_chunk(plan)).values()))
    event = make_event(plan, call)
    event["predictions"] = {"nobody": 0}
    with pytest.raises(RuntimeError, match="response IDs differ"):
        run_b5.verify_event(plan.texts(CELL), event, call, 1)


def test_a_retry_repeats_the_prompt_schema_and_limit_under_a_new_attempt_id() -> None:
    plan = one_cell_plan()
    call = next(iter(run_b5.expected_calls(first_chunk(plan)).values()))
    first = make_event(plan, call, status="truncated", parse_status="failed")
    retry = make_event(plan, call, attempt=2)
    run_b5.verify_event(plan.texts(CELL), retry, call, 2)
    for field in ("prompt_sha256", "json_schema_sha256", "max_tokens", "doc_order_sha256"):
        assert retry[field] == first[field]
    assert retry["attempt_id"] != first["attempt_id"]
    assert retry["parent_attempt_id"] == first["attempt_id"]


def test_a_retry_without_parent_linkage_is_refused() -> None:
    plan = one_cell_plan()
    call = next(iter(run_b5.expected_calls(first_chunk(plan)).values()))
    retry = make_event(plan, call, attempt=2)
    retry.pop("parent_attempt_id")
    with pytest.raises(RuntimeError, match="retry linkage differs"):
        run_b5.verify_event(plan.texts(CELL), retry, call, 2)


# --- ledger, receipts and resume ---------------------------------------------


def test_duplicate_first_responses_are_refused_when_they_conflict(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    events = chunk_events(plan, chunk)
    ledger = run_b5.Ledger(tmp_path, plan.texts)
    assert ledger.add(events) == len(events)
    assert ledger.add(events) == 0
    conflicting = dict(events[0])
    conflicting["predictions"] = {key: 1 for key in cast(list[str], conflicting["target_ids"])}
    with pytest.raises(RuntimeError, match="conflicting duplicate scientific call"):
        ledger.add([conflicting])


def test_receipts_rebuild_the_ledger_after_local_state_is_lost(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    events = chunk_events(plan, chunk)
    write_receipt(tmp_path, chunk, events)
    ledger = run_b5.Ledger(tmp_path, plan.texts)
    report = ledger.reconcile([chunk])
    assert report["events_imported"] == len(events)
    assert report["events_checked"] == len(events)
    assert ledger.pending(chunk) == []
    reopened = run_b5.Ledger(tmp_path, plan.texts)
    assert reopened.pending(chunk) == []
    assert reopened.reconcile([chunk])["events_imported"] == 0


def test_reconcile_refuses_a_receipt_from_another_schedule(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    other = plan.chunks(CELL, 4, 64)[0]
    write_receipt(tmp_path, other, chunk_events(plan, other))
    with pytest.raises(RuntimeError, match="does not belong to this schedule"):
        run_b5.Ledger(tmp_path, plan.texts).reconcile([chunk])


def test_reconcile_refuses_a_receipt_carrying_a_foreign_call(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    other = plan.chunks(CELL, 4, 64)[0]
    events = chunk_events(plan, other)
    chunk_id = sha256_json([str(event["attempt_id"]) for event in events])
    write_json(
        tmp_path / "receipts" / f"{chunk_id}.json",
        {
            "chunk_id": chunk_id,
            "source_chunk_id": chunk["chunk_id"],
            "attempt": 1,
            "events": events,
        },
    )
    with pytest.raises(RuntimeError, match="foreign call"):
        run_b5.Ledger(tmp_path, plan.texts).reconcile([chunk])


def test_reconcile_refuses_a_receipt_whose_prompt_was_changed(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    events = chunk_events(plan, chunk)
    events[0]["prompt_sha256"] = "0" * 64
    write_receipt(tmp_path, chunk, events)
    with pytest.raises(RuntimeError, match="identity differs"):
        run_b5.Ledger(tmp_path, plan.texts).reconcile([chunk])


def test_resume_reruns_only_the_calls_with_no_first_attempt(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    events = chunk_events(plan, chunk)
    ledger = run_b5.Ledger(tmp_path, plan.texts)
    ledger.add(events[:10])
    pending = ledger.pending(chunk)
    assert len(pending) == len(events) - 10
    assert set(pending).isdisjoint({str(event["call_id"]) for event in events[:10]})
    ledger.add(events[10:])
    assert ledger.pending(chunk) == []


def test_only_retryable_statuses_are_retried_and_only_once(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    calls = list(run_b5.expected_calls(chunk).values())
    events = [
        make_event(plan, calls[0], status="truncated", parse_status="failed"),
        make_event(plan, calls[1], status="ok", parse_status="failed"),
        *[make_event(plan, call) for call in calls[2:]],
    ]
    ledger = run_b5.Ledger(tmp_path, plan.texts)
    ledger.add(events)
    assert ledger.retryable(chunk) == [str(calls[0]["call_id"])]
    ledger.add([make_event(plan, calls[0], attempt=2, status="truncated", parse_status="failed")])
    assert ledger.retryable(chunk) == []
    assert b5.MAX_ATTEMPTS == 2


def test_terminal_failures_stay_in_the_ledger(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    calls = list(run_b5.expected_calls(chunk).values())
    events = [
        make_event(plan, calls[0], status="ok", parse_status="failed"),
        *[make_event(plan, call) for call in calls[1:]],
    ]
    ledger = run_b5.Ledger(tmp_path, plan.texts)
    ledger.add(events)
    progress = ledger.progress([chunk], plan.block_hours())
    assert progress["terminal_unresolved_attempts"] == 1
    assert progress["completed_calls"] == len(calls)
    assert len(list(read_jsonl(tmp_path / "events.jsonl"))) == len(calls)


def test_progress_and_percentage_track_the_selected_work(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunk = first_chunk(plan)
    events = chunk_events(plan, chunk)
    run_b5.Ledger(tmp_path, plan.texts).add(events[: len(events) // 2])
    report = run_b5.status(plan, tmp_path, [chunk])
    assert report["planned_calls"] == len(events)
    assert report["completed_calls"] == len(events) // 2
    assert 40.0 < float(report["percent_complete"]) < 60.0
    assert report["pod_created"] is False


# --- metering and projection -------------------------------------------------


def test_metered_hours_sums_settled_lanes_and_flags_an_open_lane(tmp_path: Path) -> None:
    assert run_b5.metered_hours(tmp_path) == {"settled_hours": 0.0, "open_lanes": 0.0}
    append(
        tmp_path / "meter.jsonl",
        [
            {"kind": "start", "phase_id": "a"},
            {"kind": "settled", "phase_id": "a", "charged_hours": 2.5},
            {"kind": "start", "phase_id": "b"},
        ],
    )
    measured = run_b5.metered_hours(tmp_path)
    assert measured == {"settled_hours": 2.5, "open_lanes": 1.0}


def test_projection_scales_the_forecast_by_measured_throughput() -> None:
    progress = {
        "blocks": [
            {"complete": True, "forecast_generation_hours": 2.0},
            {"complete": False, "forecast_generation_hours": 6.0},
        ]
    }
    forward = run_b5.projection(progress, 3.0)
    assert forward["measured_over_forecast_ratio"] == pytest.approx(1.5)
    assert forward["projected_total_metered_hours"] == pytest.approx(12.0)
    empty = run_b5.projection(
        {"blocks": [{"complete": False, "forecast_generation_hours": 6.0}]}, 0.0
    )
    assert empty["projected_total_metered_hours"] is None


def test_completed_smoke_produces_a_full_launch_calibration(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunks = run_b5.selected_chunks(plan, run_b5.SMOKE_CHUNK_CALLS, True)
    ledger = run_b5.Ledger(tmp_path, plan.texts)
    for chunk in chunks:
        ledger.add(chunk_events(plan, chunk))
        append(
            tmp_path / "block-meter.jsonl",
            [
                {
                    "cell_key": chunk["cell_key"],
                    "batch_size": chunk["batch_size"],
                    "elapsed_seconds": 0.01,
                }
            ],
        )
    append(
        tmp_path / "meter.jsonl",
        [
            {"kind": "start", "phase_id": "smoke"},
            {"kind": "settled", "phase_id": "smoke", "charged_hours": 0.1},
        ],
    )
    record = run_b5.calibrate_smoke(plan, tmp_path, chunks, 35.0)
    assert record["passed"] is True
    assert record["gates"]["prefix_reuse_observed_in_every_block"] is True
    assert record["projected_full_metered_a100_hours"] < 35.0


def test_smoke_calibration_refuses_missing_cache_metrics(tmp_path: Path) -> None:
    plan = one_cell_plan()
    chunks = run_b5.selected_chunks(plan, run_b5.SMOKE_CHUNK_CALLS, True)
    ledger = run_b5.Ledger(tmp_path, plan.texts)
    for chunk in chunks:
        events = chunk_events(plan, chunk)
        events[0].pop("num_cached_tokens")
        ledger.add(events)
        append(
            tmp_path / "block-meter.jsonl",
            [
                {
                    "cell_key": chunk["cell_key"],
                    "batch_size": chunk["batch_size"],
                    "elapsed_seconds": 0.01,
                }
            ],
        )
    record = run_b5.calibrate_smoke(plan, tmp_path, chunks, 35.0)
    assert record["passed"] is False
    assert record["gates"]["cache_metrics_complete"] is False


def test_full_preflight_requires_matching_passing_calibration(tmp_path: Path) -> None:
    plan = one_cell_plan()
    record = run_b5.preflight(
        plan,
        tmp_path,
        [first_chunk(plan)],
        ceiling_hours=35.0,
        usd_per_hour=1.59,
        reserve_usd=2.0,
        live=False,
        confirm=TOKEN,
    )
    assert record["gates"]["smoke_calibration_passed_or_is_smoke"] is False
    assert record["passed"] is False


# --- refusals ----------------------------------------------------------------


def test_preflight_refuses_a_runtime_budget_above_the_ceiling(tmp_path: Path) -> None:
    plan = one_cell_plan()
    calibration = tmp_path / "calibration.json"
    calibration_record = {
        "schema_version": "b5-smoke-calibration-v1",
        "seal_manifest_sha256": plan.seal["manifest_sha256"],
        "schedule_sha256": plan.schedule["schedule_sha256"],
        "passed": True,
        "projected_full_metered_a100_hours": 12.0,
    }
    calibration_record["record_sha256"] = sha256_json(calibration_record)
    write_json(
        calibration,
        calibration_record,
    )
    record = run_b5.preflight(
        plan,
        tmp_path,
        [first_chunk(plan)],
        ceiling_hours=10.0,
        usd_per_hour=1.59,
        reserve_usd=2.0,
        live=False,
        confirm=TOKEN,
        calibration_path=calibration,
    )
    assert record["gates"]["ceiling_covers_runtime_budget"] is True
    assert record["gates"]["smoke_calibration_passed_or_is_smoke"] is False
    assert record["passed"] is False


def test_preflight_refuses_without_the_confirmation_token(tmp_path: Path) -> None:
    plan = one_cell_plan()
    for supplied in (None, "wrong-token"):
        record = run_b5.preflight(
            plan,
            tmp_path,
            [first_chunk(plan)],
            ceiling_hours=35.0,
            usd_per_hour=1.59,
            reserve_usd=2.0,
            live=False,
            confirm=supplied,
        )
        assert record["gates"]["confirmation_token_supplied"] is False
        assert record["passed"] is False


def test_preflight_refuses_while_a_metered_lane_is_still_open(tmp_path: Path) -> None:
    plan = one_cell_plan()
    append(tmp_path / "meter.jsonl", [{"kind": "start", "phase_id": "open"}])
    record = run_b5.preflight(
        plan,
        tmp_path,
        [first_chunk(plan)],
        ceiling_hours=35.0,
        usd_per_hour=1.59,
        reserve_usd=2.0,
        live=False,
        confirm=TOKEN,
    )
    assert record["gates"]["no_conflicting_active_run"] is False
    assert record["passed"] is False


def account(balance: float, running: bool = False) -> Any:
    volumes = [
        {"id": VOLUMES[b5.split_cell(key)[0]], "dataCenterId": DATA_CENTER} for key in b5.CELLS
    ]
    gpu = [
        {
            "gpuId": GPU_ID,
            "secureCloud": True,
            "securePricePerHr": 1.59,
            "dataCenterAvailability": [{"dataCenterId": DATA_CENTER, "stockStatus": "high"}],
        }
    ]
    pods = [{"desiredStatus": "RUNNING" if running else "EXITED"}]

    def stub(argv: list[str], *args: Any, **kwargs: Any) -> Any:
        if argv[0] == "user":
            return {"id": ACCOUNT_ID, "clientBalance": balance, "currentSpendPerHr": 0.031}
        if argv[:2] == ["pod", "list"]:
            return pods
        if argv[:2] == ["network-volume", "list"]:
            return volumes
        if argv[:2] == ["gpu", "list"]:
            return gpu
        raise AssertionError(f"unexpected read-only call: {argv}")

    return stub


def test_account_checks_pass_on_a_funded_idle_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_b5, "cli", account(100.0))
    checks = run_b5.account_checks(32.0)
    assert checks["running_pods"] == 0
    assert checks["balance_usd"] == 100.0
    assert checks["required_balance_usd"] == 32.0


def test_account_checks_refuse_a_balance_below_the_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_b5, "cli", account(8.94))
    with pytest.raises(RuntimeError, match="cannot fund the requested B5 run"):
        run_b5.account_checks(32.0)


def test_account_checks_refuse_a_missing_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_b5, "cli", account(cast(float, None)))
    with pytest.raises(RuntimeError, match="cannot fund the requested B5 run"):
        run_b5.account_checks(32.0)


def test_account_checks_refuse_an_active_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_b5, "cli", account(100.0, running=True))
    with pytest.raises(RuntimeError, match="active Pod prevents exclusive B5 metering"):
        run_b5.account_checks(32.0)


def test_no_credential_balance_or_pod_identifier_is_hard_coded() -> None:
    source = (STUDY / "run_b5.py").read_text()
    assert "clientBalance" in source
    for forbidden in ("RUNPOD_API_KEY=", "api_key", "8.947", 'pod_id = "'):
        assert forbidden not in source


def test_open_state_refuses_a_directory_from_another_run(tmp_path: Path) -> None:
    plan = one_cell_plan()
    state = run_b5.open_state(tmp_path, plan, 512, False)
    assert state["schema_version"] == run_b5.RUN_SCHEMA
    assert run_b5.open_state(tmp_path, plan, 512, False) == state
    with pytest.raises(ValueError, match="different B5 run"):
        run_b5.open_state(tmp_path, plan, 64, True)


# --- the seal ----------------------------------------------------------------


def build_seal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated seal over one covered file, so the real freeze directory is untouched."""
    study = tmp_path / "study"
    (study / "freeze").mkdir(parents=True)
    covered = study / "covered.toml"
    covered.write_text("[covered]\nvalue = 1\n")
    directory = study / "freeze" / "20260918T000000Z-B5"
    snapshot = directory / "snapshot" / "covered.toml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(covered.read_bytes())
    manifest: dict[str, Any] = {
        "schema_version": seal_b5.SCHEMA,
        "sealed_at_utc": "2026-09-18T00:00:00+00:00",
        "preparation_directory": "extension-b5/stub",
        "files_sha256": {"covered.toml": sha256_file(covered)},
        "schedule_sha256": "d" * 64,
        "gpu_work_launched": False,
        "supersedes": None,
        "scope": "test",
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(directory / "MANIFEST.json", manifest)
    monkeypatch.setattr(seal_b5, "STUDY", study)
    monkeypatch.setattr(seal_b5, "FREEZE", study / "freeze")
    return directory


def test_seal_verifies_and_prints_a_confirmation_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = build_seal(tmp_path, monkeypatch)
    record = seal_b5.verify(directory)
    assert record["files_verified"] == 1
    assert record["confirmation_token"] == record["manifest_sha256"][:12]
    assert record["preparation_directory"] == "extension-b5/stub"
    assert seal_b5.latest() == directory


def test_seal_fails_when_a_covered_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = build_seal(tmp_path, monkeypatch)
    (seal_b5.STUDY / "covered.toml").write_text("[covered]\nvalue = 2\n")
    with pytest.raises(ValueError, match="sealed files changed"):
        seal_b5.verify(directory)


def test_seal_fails_when_the_snapshot_is_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = build_seal(tmp_path, monkeypatch)
    (directory / "snapshot" / "covered.toml").write_text("# edited\n")
    with pytest.raises(ValueError, match="sealed files changed"):
        seal_b5.verify(directory)


def test_seal_fails_when_the_manifest_digest_is_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = build_seal(tmp_path, monkeypatch)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    manifest["scope"] = "something else"
    write_json(directory / "MANIFEST.json", manifest, allow_existing=True)
    with pytest.raises(ValueError, match="manifest digest changed"):
        seal_b5.verify(directory)


def test_seal_refuses_to_start_without_every_covered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_seal(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="missing"):
        seal_b5.covered(seal_b5.STUDY / "extension-b5" / "stub")


def test_the_seal_covers_every_decision_relevant_input() -> None:
    for name in (
        "B5_PROTOCOL.md",
        "prepare_b5.py",
        "run_b5.py",
        "b5_worker.py",
        "analyze_b5.py",
        "src/freezeb/b5.py",
        "src/freezeb/b5_analysis.py",
        "src/freezeb/b5_power.py",
        "src/freezeb/prompts.py",
        "src/step1/backends.py",
        "tests/test_b5.py",
        "tests/test_b5_analysis.py",
        "tests/test_b5_runner.py",
        "tests/test_batch_invariance.py",
    ):
        assert name in seal_b5.CODE_FILES
    assert "SCHEDULE_DIGEST.json" in seal_b5.PREPARATION_FILES
    assert "FORECAST.json" in seal_b5.PREPARATION_FILES
    assert "POWER.json" in seal_b5.PREPARATION_FILES
