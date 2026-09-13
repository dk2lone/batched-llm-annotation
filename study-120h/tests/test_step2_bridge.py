"""CPU-only tests for the historical Qwen3-8B/Civil Comments Step 2 bridge.

Nothing here starts Modal, loads a model, or generates real output. `FakeBackend` is a
deterministic stand-in and no output of it is evidence about any real model.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from step0.hashing import sha256_json
from step1.backends import FakeBackend, GenRequest
from step1.frozen import Frozen, load
from step1.ledger import LedgerError, RunMeta, append, rows_for
from step1.plan import build_call, doc_texts, retry_call_id
from step2 import amendment, bridge


@pytest.fixture(scope="module")
def frozen() -> Frozen:
    return load()


@pytest.fixture(scope="module")
def planned(frozen: Frozen) -> list[dict[str, Any]]:
    return bridge.planned_jobs(frozen)


def _mutate(frozen: Frozen, **protocol: Any) -> Frozen:
    return Frozen(freeze_dir=frozen.freeze_dir, protocol={**frozen.protocol, **protocol})


# ------------------------------------------------------------------------------------
# Plan
# ------------------------------------------------------------------------------------


def test_plan_covers_every_fresh_call_and_no_solo_call(planned: list[dict[str, Any]]) -> None:
    assert len(planned) == bridge.FRESH_CALLS == 12672
    counts: dict[str, int] = {}
    for job in planned:
        counts[str(job["condition"])] = counts.get(str(job["condition"]), 0) + 1
    assert counts == {"joint": 384, "exact_context": 6144, "filler": 6144}
    assert "solo" not in counts
    assert len({str(job["call_id"]) for job in planned}) == len(planned)


def test_plan_keeps_the_frozen_call_ids_and_interleaved_order(
    frozen: Frozen, planned: list[dict[str, Any]]
) -> None:
    sealed = {
        str(row["call_id"]): row
        for row in frozen.job_skeletons(bridge.CELL_ID)
        if row["condition"] in bridge.BRIDGE_CONDITIONS
    }
    assert {str(job["call_id"]) for job in planned} == set(sealed)
    for job in planned:
        assert job == sealed[str(job["call_id"])]
    keys = [(int(job["execution_block"]), str(job["execution_order_key"])) for job in planned]
    assert keys == sorted(keys)
    blocks = bridge.blocks_of(planned)
    assert [block for block, _ in blocks] == list(range(8))
    for _block, jobs in blocks:
        assert {str(job["condition"]) for job in jobs} == set(bridge.BRIDGE_CONDITIONS)


def test_plan_hash_is_stable_and_names_the_reused_solo_rows(frozen: Frozen) -> None:
    payload = _plan(frozen)
    assert payload["plan_sha256"] == _plan(frozen)["plan_sha256"]
    assert payload["solo_reuse"]["reused"] is True
    assert payload["solo_reuse"]["rerun"] is False
    assert payload["solo_reuse"]["n_calls"] == bridge.SOLO_CALLS
    assert payload["fresh_calls_total"] == bridge.FRESH_CALLS
    assert payload["serving_controls"]["applies_to_this_cell"] is False
    assert payload["fake_batch"]["creates_calls"] is False
    assert payload["fresh_serving_session_per_block"] is True
    assert payload["forecast"]["extra_serving_session_loads"] == 7


def _plan(frozen: Frozen, chunk_size: int = 32) -> dict[str, Any]:
    root = Path(bridge.__file__).resolve().parents[2]
    return bridge.plan(
        frozen,
        solo_ledger=root / bridge.SOLO_LEDGER,
        preflight_report=json.loads(
            (root / bridge.PREFLIGHT_REPORT).read_text(encoding="utf-8")
        ),
        runtime_lock=json.loads((root / bridge.RUNTIME_LOCK).read_text(encoding="utf-8")),
        chunk_size=chunk_size,
    )


# ------------------------------------------------------------------------------------
# Guards
# ------------------------------------------------------------------------------------


def test_a_changed_seal_stops_the_bridge(frozen: Frozen) -> None:
    with pytest.raises(bridge.BridgeGuardError, match="content hash"):
        bridge.verify_freeze(_mutate(frozen, content_hash="0" * 64))
    panel = {**frozen.protocol["panel"], "serving": {**frozen.serving, "max_model_len": 4096}}
    with pytest.raises(bridge.BridgeGuardError, match="max_model_len"):
        bridge.verify_freeze(_mutate(frozen, panel=panel))


def test_a_wrong_model_engine_or_gpu_stops_the_bridge() -> None:
    host = {"gpu_name": "NVIDIA A100-SXM4-40GB", "installed_engine_version_string": "0.28.0"}
    backend = {
        "backend": "vllm",
        "model_id": bridge.MODEL_ID,
        "model_revision": bridge.MODEL_REVISION,
        "tokenizer_revision": bridge.MODEL_REVISION,
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "max_num_seqs": 64,
        "tensor_parallel_size": 1,
        "seed": 0,
    }
    bridge.verify_worker(host, backend)
    with pytest.raises(bridge.BridgeGuardError, match="model_id"):
        bridge.verify_worker(host, {**backend, "model_id": "Qwen/Qwen3-4B-Instruct-2507"})
    with pytest.raises(bridge.BridgeGuardError, match="vLLM"):
        bridge.verify_worker({**host, "installed_engine_version_string": "0.29.0"}, backend)
    with pytest.raises(bridge.BridgeGuardError, match="A100"):
        bridge.verify_worker({**host, "gpu_name": "NVIDIA H100 80GB HBM3"}, backend)
    with pytest.raises(bridge.BridgeGuardError, match="provenance"):
        bridge.verify_worker(None, None)


def test_a_wrong_modal_profile_stops_the_bridge() -> None:
    assert bridge.verify_profile("daniel21cn2016", "daniel21cn2016") == "daniel21cn2016"
    with pytest.raises(bridge.BridgeGuardError, match="MODAL_PROFILE is"):
        bridge.verify_profile("daniel21cn2016", "other-account")
    with pytest.raises(bridge.BridgeGuardError, match="modal-profile"):
        bridge.verify_profile("", "daniel21cn2016")
    with pytest.raises(bridge.BridgeGuardError, match="MODAL_PROFILE in"):
        bridge.verify_profile("daniel21cn2016", "")


def test_an_unusable_preflight_stops_the_bridge(frozen: Frozen) -> None:
    root = Path(bridge.__file__).resolve().parents[2]
    report = json.loads((root / bridge.PREFLIGHT_REPORT).read_text(encoding="utf-8"))
    assert bridge.verify_preflight(report)["n_over_budget"] == 0
    with pytest.raises(bridge.BridgeGuardError, match="over the frozen budget"):
        bridge.verify_preflight({**report, "n_over_budget": 3})
    with pytest.raises(bridge.BridgeGuardError, match="exact pinned tokenizer"):
        bridge.verify_preflight({**report, "exact_tokenizer": False})


def test_an_incomplete_solo_ledger_stops_the_bridge(frozen: Frozen, tmp_path: Path) -> None:
    root = Path(bridge.__file__).resolve().parents[2]
    source = (root / bridge.SOLO_LEDGER).read_text(encoding="utf-8").splitlines()
    short = tmp_path / "solo.jsonl"
    short.write_text("\n".join(source[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(bridge.BridgeGuardError, match="cannot be reused"):
        bridge.verify_solo_reuse(frozen, short)


# ------------------------------------------------------------------------------------
# Resume and retry lineage
# ------------------------------------------------------------------------------------


def _run_fake_chunk(
    frozen: Frozen,
    jobs: list[dict[str, Any]],
    ledger: Path,
    *,
    truncation_rate: float = 0.0,
    parse_failure_rate: float = 0.0,
    chunk_id: str = "chunk-0",
) -> list[dict[str, Any]]:
    """Append one chunk of synthetic first responses through the frozen ledger path."""
    items = frozen.items(bridge.FRAME_TASK_ID)
    references = {item.doc_id: item.reference for item in items}
    strata = {item.doc_id: item.length_stratum for item in items}
    texts = doc_texts(frozen, bridge.FRAME_TASK_ID)
    calls = [build_call(frozen, job, bridge.FRAME_TASK_ID, texts) for job in jobs]
    backend = FakeBackend(
        references,
        truncation_rate=truncation_rate,
        parse_failure_rate=parse_failure_rate,
        n_classes=2,
    )
    results = {
        result.call_id: result
        for result in backend.generate(
            [
                GenRequest(
                    call_id=call.call_id,
                    condition=call.condition,
                    target_ids=tuple(call.target_ids),
                    messages=call.messages,
                    max_tokens=call.max_tokens,
                )
                for call in calls
            ]
        )
    }
    meta = RunMeta(
        run_id="test-run",
        serving_session="test-session",
        engine_version="0.28.0",
        step="step2_bridge",
        chunk_id=chunk_id,
        chunk_size=len(calls),
        chunk_seconds=1.0,
    )
    rows: list[dict[str, Any]] = []
    for call in calls:
        rows.extend(
            rows_for(
                frozen,
                call,
                results.get(call.call_id),
                bridge.FRAME_TASK_ID,
                references,
                strata,
                meta,
            )
        )
    report = append(ledger, rows)
    assert report["conflicts_rejected"] == 0
    return rows


def test_resume_skips_stored_calls_and_never_replaces_a_failure(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    subset = [job for job in planned if int(job["execution_block"]) == 0][:24]
    ledger = tmp_path / "calls.jsonl"
    assert len(bridge.pending_work(subset, [])["first_responses"]) == 24

    _run_fake_chunk(frozen, subset[:16], ledger, truncation_rate=1.0)
    stored = list(_rows(ledger))
    work = bridge.pending_work(subset, stored)
    assert [str(job["call_id"]) for job in work["first_responses"]] == [
        str(job["call_id"]) for job in subset[16:]
    ]
    assert len(work["retries"]) == 16
    for retry, parent in zip(work["retries"], subset[:16], strict=True):
        assert retry["parent_call_id"] == str(parent["call_id"])
        assert retry["retry_index"] == 1
        assert retry["call_id"] == retry_call_id(str(parent["call_id"]), 1)
        assert retry["condition"] == parent["condition"]
        assert retry["execution_block"] == parent["execution_block"]

    _run_fake_chunk(frozen, work["retries"], ledger, chunk_id="chunk-retry")
    after = bridge.pending_work(subset, list(_rows(ledger)))
    assert after["retries"] == []
    assert len(after["first_responses"]) == 8

    _run_fake_chunk(frozen, subset[16:], ledger, chunk_id="chunk-1")
    final = bridge.pending_work(subset, list(_rows(ledger)))
    assert final["first_responses"] == [] and final["retries"] == []
    assert bridge.completeness(subset, ledger)["complete"] is True


def test_a_valid_first_response_never_earns_a_retry(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    subset = [job for job in planned if int(job["execution_block"]) == 1][:8]
    ledger = tmp_path / "calls.jsonl"
    _run_fake_chunk(frozen, subset, ledger)
    stored = list(_rows(ledger))
    assert {str(row["first_response_status"]) for row in stored} == {"valid_first_response"}
    work = bridge.pending_work(subset, stored)
    assert work == {"first_responses": [], "retries": []}


def test_a_foreign_row_stops_the_bridge(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    subset = [job for job in planned if int(job["execution_block"]) == 2][:4]
    ledger = tmp_path / "calls.jsonl"
    _run_fake_chunk(frozen, subset, ledger)
    stored = list(_rows(ledger))
    with pytest.raises(bridge.BridgeGuardError, match="do not belong"):
        bridge.resume_state(subset[1:], stored)
    with pytest.raises(bridge.BridgeGuardError, match="do not belong"):
        bridge.resume_state(subset, [{**stored[0], "model_id": "other/model"}])


def test_an_unfinished_run_is_never_reported_as_complete(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    subset = [job for job in planned if int(job["execution_block"]) == 3][:12]
    ledger = tmp_path / "calls.jsonl"
    _run_fake_chunk(frozen, subset[:4], ledger)
    report = bridge.completeness(subset, ledger)
    assert report["complete"] is False
    assert report["n_missing_calls"] == 8
    assert bridge.remaining_counts(subset, list(_rows(ledger)))


def _rows(ledger: Path) -> list[dict[str, Any]]:
    from step0.hashing import read_jsonl

    return list(read_jsonl(ledger))


# ------------------------------------------------------------------------------------
# Timing checkpoint and reforecast
# ------------------------------------------------------------------------------------


def test_chunk_seconds_are_attributed_by_measured_request_time() -> None:
    shares, method = bridge.attribute_seconds(10.0, [1.0, 4.0])
    assert method == "per_request_engine_timing"
    assert shares == pytest.approx([2.0, 8.0])
    shares, method = bridge.attribute_seconds(10.0, [1.0, None])
    assert method == "equal_share_within_chunk"
    assert shares == pytest.approx([5.0, 5.0])


def _timing(condition: str, chunk_id: str, seconds: float, chunk_total: float) -> dict[str, Any]:
    return {
        "condition": condition,
        "chunk_id": chunk_id,
        "chunk_metered_a100_seconds": chunk_total,
        "metered_a100_seconds": seconds,
        "status": "ok",
        "attribution_method": "per_request_engine_timing",
        "prompt_tokens": 1600,
        "output_tokens": 20,
        "request_seconds": seconds,
        "prefill_seconds": seconds / 2,
        "decode_seconds": seconds / 2,
        "queue_seconds": 0.01,
    }


def test_timing_records_split_one_chunk_and_keep_the_condition(
    frozen: Frozen, planned: list[dict[str, Any]]
) -> None:
    jobs = [planned[0], next(job for job in planned if job["condition"] == "joint")]
    measurements = [
        {
            "status": "ok",
            "finish_reason": "stop",
            "prompt_tokens": 1600,
            "output_tokens": 20,
            "queue_seconds": 0.01,
            "prefill_seconds": 0.1,
            "decode_seconds": 0.1,
        },
        {
            "status": "ok",
            "finish_reason": "stop",
            "prompt_tokens": 1900,
            "output_tokens": 300,
            "queue_seconds": 0.02,
            "prefill_seconds": 0.2,
            "decode_seconds": 0.4,
        },
    ]
    rows = bridge.timing_records(
        jobs,
        measurements,
        run_id="r",
        chunk_id="c",
        serving_session="s",
        execution_block=0,
        chunk_wall_seconds=1.0,
        metered_a100_seconds=8.0,
        cumulative_a100_hours=3.0,
    )
    assert [row["condition"] for row in rows] == [str(job["condition"]) for job in jobs]
    assert sum(float(row["metered_a100_seconds"]) for row in rows) == pytest.approx(8.0)
    assert rows[1]["metered_a100_seconds"] > rows[0]["metered_a100_seconds"]
    assert {row["attribution_method"] for row in rows} == {"per_request_engine_timing"}

    checkpoint = bridge.checkpoint_by_condition(rows)
    assert checkpoint["n_calls"] == 2
    assert set(checkpoint["by_condition"]) == {str(job["condition"]) for job in jobs}
    joint = checkpoint["by_condition"]["joint"]
    assert joint["n_with_component_timing"] == 1
    assert joint["output_tokens_sum"] == 300
    assert joint["queue_seconds_mean"] == pytest.approx(0.02)


def test_least_squares_recovers_the_rate_of_each_condition() -> None:
    rates = {"joint": 2.0, "exact_context": 0.5, "filler": 0.25}
    rows: list[dict[str, Any]] = []
    for index in range(12):
        composition = {
            "joint": index % 3,
            "exact_context": 4 + index % 2,
            "filler": 3 + index % 4,
        }
        total = sum(rates[key] * count for key, count in composition.items())
        for condition, count in composition.items():
            for _ in range(count):
                rows.append(_timing(condition, f"c{index}", rates[condition], total))
    estimate = bridge.condition_rates(rows)
    assert estimate is not None
    assert estimate["all_positive"]
    for condition, expected in rates.items():
        assert estimate["seconds_per_call"][condition] == pytest.approx(expected, abs=1e-6)
    assert bridge.condition_rates(rows[:4]) is None


def test_a_condition_with_no_real_call_yet_blocks_a_guessed_rate() -> None:
    rows = [_timing("filler", "c0", 0.25, 10.0)]
    assert bridge.missing_rates(rows, {"joint": 10, "filler": 10}) == ["joint"]
    assert bridge.missing_rates(rows, {"joint": 0, "filler": 10}) == []
    with pytest.raises(ValueError, match="no measured rate yet"):
        bridge.reforecast(rows, {"joint": 10, "filler": 10}, cumulative_a100_hours=2.78)
    report = bridge.reforecast(rows, {"joint": 0, "filler": 10}, cumulative_a100_hours=2.78)
    assert report["projected_remaining_base_a100_hours"] == pytest.approx(2.5 / 3600.0)


def test_reforecast_projects_the_rest_and_obeys_the_150_hour_stop() -> None:
    rates = {"joint": 2.0, "exact_context": 0.5, "filler": 0.25}
    rows = [
        _timing(condition, f"c{index}", rates[condition], 10.0)
        for index in range(12)
        for condition in rates
    ]
    remaining = {"joint": 100, "exact_context": 1000, "filler": 1000}
    report = bridge.reforecast(rows, remaining, cumulative_a100_hours=2.78)
    expected_seconds = sum(rates[key] * count for key, count in remaining.items())
    assert report["projected_remaining_base_a100_hours"] == pytest.approx(
        expected_seconds / 3600.0, rel=1e-6
    )
    assert report["may_continue"] is True
    assert report["must_stop_now"] is False
    assert report["real_calls_used"] == len(rows)

    stopped = bridge.reforecast(rows, remaining, cumulative_a100_hours=149.9)
    assert stopped["may_continue"] is False
    assert stopped["must_stop_now"] is False
    assert bridge.reforecast(rows, remaining, cumulative_a100_hours=150.0)["must_stop_now"] is True

    loaded = bridge.reforecast(
        rows, remaining, cumulative_a100_hours=2.78, remaining_load_seconds=3600.0
    )
    assert loaded["projected_remaining_base_a100_hours"] == pytest.approx(
        report["projected_remaining_base_a100_hours"] + 1.0, rel=1e-6
    )


def test_call_ids_are_verified_without_a_gpu(frozen: Frozen, planned: list[dict[str, Any]]) -> None:
    payload = _plan(frozen)
    assert payload["fresh_call_ids_sha256"] == sha256_json(
        [str(job["call_id"]) for job in planned]
    )
    assert payload["preflight"]["n_calls_checked"] == 14720


# ------------------------------------------------------------------------------------
# Multi-target ledger rows
# ------------------------------------------------------------------------------------


def _joint_job(planned: list[dict[str, Any]]) -> dict[str, Any]:
    return next(job for job in planned if job["condition"] == "joint")


def test_a_joint_call_keeps_all_sixteen_target_rows(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    job = _joint_job(planned)
    assert len(job["target_ids"]) == 16
    ledger = tmp_path / "calls.jsonl"
    rows = _run_fake_chunk(frozen, [job], ledger)
    assert len(rows) == 16
    stored = _rows(ledger)
    assert len(stored) == 16
    assert {str(row["doc_id"]) for row in stored} == {str(t) for t in job["target_ids"]}
    assert len({str(row["output_sha256"]) for row in stored}) == 1
    assert len({int(row["slot"]) for row in stored}) == 16

    report = append(ledger, rows)
    assert report["appended"] == 0
    assert report["duplicates_skipped"] == 16
    assert len(_rows(ledger)) == 16


def test_a_joint_call_with_a_different_output_is_rejected_whole(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    job = _joint_job(planned)
    ledger = tmp_path / "calls.jsonl"
    rows = _run_fake_chunk(frozen, [job], ledger)
    changed = [{**row, "raw_text": "different", "output_sha256": "f" * 64} for row in rows]
    report = append(ledger, changed)
    assert report["appended"] == 0
    assert report["conflicts_rejected"] == 1
    assert report["conflicts"][0]["n_rows_rejected"] == 16
    assert len(_rows(ledger)) == 16
    assert bridge.conflicts_recorded(ledger) == 1


def test_a_joint_call_that_lost_target_rows_cannot_pass_resume(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    job = _joint_job(planned)
    ledger = tmp_path / "calls.jsonl"
    rows = _run_fake_chunk(frozen, [job], ledger)
    truncated = tmp_path / "lost.jsonl"
    append(truncated, rows[:1])
    assert len(_rows(truncated)) == 1

    state = bridge.resume_state([job], _rows(truncated))
    assert state["n_incomplete_first_responses"] == 1
    assert state["n_first_responses_stored"] == 0
    work = bridge.pending_work([job], _rows(truncated))
    assert [str(item["call_id"]) for item in work["first_responses"]] == [str(job["call_id"])]
    assert work["retries"] == []
    report = bridge.completeness([job], truncated)
    assert report["complete"] is False
    assert report["n_incomplete_calls"] == 1

    repaired = append(truncated, rows)
    assert repaired["appended"] == 15
    assert repaired["calls_completed"] == [str(job["call_id"])]
    assert len(_rows(truncated)) == 16
    assert bridge.completeness([job], truncated)["complete"] is True
    assert bridge.pending_work([job], _rows(truncated))["first_responses"] == []


def test_one_append_never_mixes_two_outputs_for_one_call(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    job = _joint_job(planned)
    ledger = tmp_path / "calls.jsonl"
    rows = _run_fake_chunk(frozen, [job], ledger)
    mixed = [rows[0], {**rows[1], "output_sha256": "a" * 64}]
    with pytest.raises(LedgerError, match="more than one output"):
        append(tmp_path / "other.jsonl", mixed)
    with pytest.raises(LedgerError, match="the same target twice"):
        append(tmp_path / "other.jsonl", [rows[0], dict(rows[0])])


# ------------------------------------------------------------------------------------
# Prompt retry scheduling
# ------------------------------------------------------------------------------------


def test_a_failed_first_result_earns_its_retry_in_the_same_chunk(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    subset = [job for job in planned if int(job["execution_block"]) == 4][:4]
    ledger = tmp_path / "calls.jsonl"
    rows = _run_fake_chunk(frozen, subset, ledger, truncation_rate=1.0)
    scheduled = bridge.retry_jobs_for(subset, rows)
    assert len(scheduled) == len(subset)
    assert [item["parent_call_id"] for item in scheduled] == [
        str(job["call_id"]) for job in subset
    ]
    assert {int(item["retry_index"]) for item in scheduled} == {1}
    # A retry is never retried again, because the seal permits one.
    retry_rows = _run_fake_chunk(frozen, scheduled, ledger, truncation_rate=1.0, chunk_id="r")
    assert bridge.retry_jobs_for(scheduled, retry_rows) == []


def test_a_malformed_first_response_is_never_retried(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    subset = [job for job in planned if int(job["execution_block"]) == 5][:4]
    ledger = tmp_path / "calls.jsonl"
    rows = _run_fake_chunk(frozen, subset, ledger, parse_failure_rate=1.0)
    assert {str(row["first_response_status"]) for row in rows} == {"malformed"}
    assert bridge.retry_jobs_for(subset, rows) == []
    assert bridge.pending_work(subset, rows)["retries"] == []
    assert "malformed" not in bridge.RETRY_STATUSES


# ------------------------------------------------------------------------------------
# Concurrency, recovery, and checkpoint guards
# ------------------------------------------------------------------------------------


def test_a_mismatched_chunk_payload_is_refused(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    job = _joint_job(planned)
    ledger = tmp_path / "calls.jsonl"
    rows = _run_fake_chunk(frozen, [job], ledger)
    payload = {
        "chunk_id": "b0-abcdef",
        "rows": rows,
        "measurements": [{"call_id": str(job["call_id"])}],
    }
    bridge.verify_chunk_payload(payload, [job], "b0-abcdef")
    with pytest.raises(bridge.BridgeGuardError, match="carries id"):
        bridge.verify_chunk_payload(payload, [job], "b0-other")
    with pytest.raises(bridge.BridgeGuardError, match="wrong target rows"):
        bridge.verify_chunk_payload({**payload, "rows": rows[:3]}, [job], "b0-abcdef")
    with pytest.raises(bridge.BridgeGuardError, match="different call order"):
        bridge.verify_chunk_payload({**payload, "measurements": []}, [job], "b0-abcdef")
    other = next(item for item in planned if item["call_id"] != job["call_id"])
    with pytest.raises(bridge.BridgeGuardError, match="not requested"):
        bridge.verify_chunk_payload(payload, [other], "b0-abcdef")


def test_the_checkpoint_is_verified_against_the_runtime_lock() -> None:
    root = Path(bridge.__file__).resolve().parents[2]
    manifests = json.loads((root / bridge.CHECKPOINT_MANIFESTS).read_text(encoding="utf-8"))
    expected = next(row for row in manifests if row["model_id"] == bridge.MODEL_ID)
    report = bridge.verify_checkpoint(expected, manifests)
    assert report["revision"] == bridge.MODEL_REVISION
    assert report["n_files_hash_verified"] == len(expected["files"]) == 15
    assert report["volume"] == "ieee-bigdata-freezeb-models"

    partial = {
        **expected,
        "files": [
            {**row, "sha256": None} if row["path"].endswith(".safetensors") else row
            for row in expected["files"]
        ],
    }
    relaxed = bridge.verify_checkpoint(partial, manifests)
    assert relaxed["n_files_size_verified_only"] > 0

    broken = {**expected, "files": [{**expected["files"][0], "sha256": "0" * 64}]}
    with pytest.raises(bridge.BridgeGuardError, match="missing"):
        bridge.verify_checkpoint(broken, manifests)
    wrong = {
        **expected,
        "files": [
            {**row, "sha256": "0" * 64} if index == 0 else row
            for index, row in enumerate(expected["files"])
        ],
    }
    with pytest.raises(bridge.BridgeGuardError, match="differs in content"):
        bridge.verify_checkpoint(wrong, manifests)
    with pytest.raises(bridge.BridgeGuardError, match="not the pinned model revision"):
        bridge.verify_checkpoint({**expected, "revision": "0" * 40}, manifests)


def test_the_stop_is_conservative_across_concurrent_workers() -> None:
    assert bridge.stop_now(149.0) is False
    assert bridge.stop_now(150.0) is True
    # Two workers may each still bill one chunk, so the reserve stops the run earlier.
    assert bridge.stop_now(149.5, in_flight_hours=0.2 * 2) is False
    assert bridge.stop_now(149.7, in_flight_hours=0.2 * 2) is True
    with pytest.raises(ValueError):
        bridge.stop_now(-1.0)


def test_execution_settings_stay_out_of_the_plan_identity(frozen: Frozen) -> None:
    one = _plan(frozen, chunk_size=32)
    other = bridge.plan(
        frozen,
        solo_ledger=Path(bridge.__file__).resolve().parents[2] / bridge.SOLO_LEDGER,
        preflight_report=json.loads(
            (Path(bridge.__file__).resolve().parents[2] / bridge.PREFLIGHT_REPORT).read_text(
                encoding="utf-8"
            )
        ),
        runtime_lock=json.loads(
            (Path(bridge.__file__).resolve().parents[2] / bridge.RUNTIME_LOCK).read_text(
                encoding="utf-8"
            )
        ),
        chunk_size=16,
        max_concurrent_workers=1,
    )
    assert one["plan_sha256"] == other["plan_sha256"]
    assert one["execution"]["max_concurrent_workers"] == 2
    assert other["execution"]["max_concurrent_workers"] == 1
    assert one["execution"]["results_namespace"] == one["plan_sha256"][:16]
    assert one["execution"]["deviation_from_sealed_serving_hardware"] is True
    assert other["execution"]["deviation_from_sealed_serving_hardware"] is False
    with pytest.raises(bridge.BridgeGuardError, match="workers must be"):
        _plan_with_workers(frozen, 3)


def _plan_with_workers(frozen: Frozen, workers: int) -> dict[str, Any]:
    root = Path(bridge.__file__).resolve().parents[2]
    return bridge.plan(
        frozen,
        solo_ledger=root / bridge.SOLO_LEDGER,
        preflight_report=json.loads(
            (root / bridge.PREFLIGHT_REPORT).read_text(encoding="utf-8")
        ),
        runtime_lock=json.loads((root / bridge.RUNTIME_LOCK).read_text(encoding="utf-8")),
        chunk_size=32,
        max_concurrent_workers=workers,
    )


# ------------------------------------------------------------------------------------
# The recorded operational deviation
# ------------------------------------------------------------------------------------


def test_the_two_worker_deviation_is_recorded_before_any_output(
    frozen: Frozen, tmp_path: Path
) -> None:
    path, payload = amendment.write(frozen, out_dir=tmp_path)
    assert path.exists()
    assert payload["amendment_id"] == "step2-bridge-concurrency"
    assert payload["deviation"]["field"] == "panel.serving.hardware"
    assert payload["deviation"]["sealed_text"] == "one NVIDIA A100, one at a time"
    assert payload["deviation"]["is_a_deviation"] is True
    assert payload["deviation"]["affects_a_scientific_field"] is False
    assert payload["max_concurrent_workers"] == 2
    assert payload["tensor_parallel_size"] == 1
    assert payload["blocks_are_never_split_across_workers"] is True
    assert payload["no_scientific_field_is_changed"] is True
    assert payload["authorised_before_any_bridge_output"] is True
    body = path.read_text(encoding="utf-8")
    assert "one NVIDIA A100, one at a time" in body
    assert "B3" in body and "B4" in body

    found, reused = amendment.ensure(frozen, out_dir=tmp_path)
    assert reused["content_hash"] == payload["content_hash"]
    assert found.suffix == ".json"
    with pytest.raises(RuntimeError, match="permits 2 workers"):
        amendment.ensure(frozen, out_dir=tmp_path, max_concurrent_workers=6)


def test_an_amendment_for_another_protocol_is_refused(frozen: Frozen, tmp_path: Path) -> None:
    amendment.write(frozen, out_dir=tmp_path)
    other = Frozen(freeze_dir=frozen.freeze_dir, protocol={**frozen.protocol, "content_hash": "z"})
    with pytest.raises(RuntimeError, match="different protocol"):
        amendment.ensure(other, out_dir=tmp_path)


# ------------------------------------------------------------------------------------
# Two concurrent workers
# ------------------------------------------------------------------------------------


def _entrypoint() -> Any:
    """Load the Modal entrypoint by path. Importing it starts nothing."""
    path = Path(bridge.__file__).resolve().parents[2] / "modal_bridge.py"
    spec = importlib.util.spec_from_file_location("bridge_entrypoint", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_concurrent_appends_keep_every_row_exactly_once(
    frozen: Frozen, planned: list[dict[str, Any]], tmp_path: Path
) -> None:
    """Two workers share one ledger, so every append runs under one lock."""
    joint = [job for job in planned if job["condition"] == "joint"][:4]
    other = [job for job in planned if job["condition"] == "exact_context"][:8]
    ledger = tmp_path / "calls.jsonl"
    batches = [
        _run_fake_chunk(frozen, [job], tmp_path / "scratch.jsonl", chunk_id=f"c{index}")
        for index, job in enumerate(joint + other)
    ]
    lock = threading.Lock()

    def worker(rows: list[dict[str, Any]]) -> None:
        for _ in range(3):  # a resumed worker may offer the same chunk again
            with lock:
                append(ledger, rows)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker, rows) for rows in batches]:
            future.result()

    stored = _rows(ledger)
    assert len(stored) == 4 * 16 + 8
    assert len({(str(row["call_id"]), str(row["doc_id"])) for row in stored}) == len(stored)
    assert bridge.completeness(joint + other, ledger)["complete"] is True


def test_chunk_ids_survive_a_local_interruption(planned: list[dict[str, Any]]) -> None:
    """A crash must not re-chunk the queue, or recovered work would be paid for twice."""
    entrypoint = _entrypoint()
    block = [job for job in planned if int(job["execution_block"]) == 0][:96]

    def chunk_ids(queue: list[dict[str, Any]], size: int) -> list[str]:
        return [
            entrypoint._chunk_id(0, queue[start : start + size])
            for start in range(0, len(queue), size)
        ]

    first = chunk_ids(block, 32)
    assert len(first) == 3
    assert len(set(first)) == 3
    assert chunk_ids(block, 32) == first
    # The crash happened after chunk one was committed and appended: the rest re-chunks
    # identically, so the committed chunk two is recovered instead of regenerated.
    assert chunk_ids(block[32:], 32) == first[1:]
    # A different chunk size gives different ids, which is why a resume refuses to change it.
    assert chunk_ids(block, 16)[0] != first[0]


def test_the_entrypoint_uses_the_existing_freeze_b_model_volume() -> None:
    entrypoint = _entrypoint()
    assert entrypoint.model_volume.name == bridge.MODEL_VOLUME
    assert entrypoint.GPU_TYPE == "A100-40GB"


@pytest.mark.parametrize("interrupted_at", ["ledger", "timing", "meter"])
def test_local_interrupted_writes_refuse_paid_resume(
    tmp_path: Path, interrupted_at: str
) -> None:
    entrypoint = _entrypoint()
    ledger = tmp_path / "calls.jsonl"
    timing = tmp_path / "timing.jsonl"
    meter = tmp_path / "meter.jsonl"
    state = tmp_path / "meter-state.json"
    rows = [
        (ledger, {"chunk_id": "c1", "call_id": "a"}),
        (timing, {"chunk_id": "c1", "call_id": "a"}),
        (meter, {"chunk_id": "c1", "cumulative_a100_hours": 3.0}),
    ]
    for path, row in rows[: {"ledger": 1, "timing": 2, "meter": 3}[interrupted_at]]:
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        entrypoint._verify_local_commit_boundary(ledger, timing, meter, state)


def test_remote_only_commit_leaves_local_resume_open(tmp_path: Path) -> None:
    entrypoint = _entrypoint()
    entrypoint._verify_local_commit_boundary(
        tmp_path / "calls.jsonl",
        tmp_path / "timing.jsonl",
        tmp_path / "meter.jsonl",
        tmp_path / "meter-state.json",
    )


def test_complete_local_chunk_passes_resume_check(tmp_path: Path) -> None:
    entrypoint = _entrypoint()
    ledger = tmp_path / "calls.jsonl"
    timing = tmp_path / "timing.jsonl"
    meter = tmp_path / "meter.jsonl"
    state = tmp_path / "meter-state.json"
    ledger.write_text('{"chunk_id":"c1","call_id":"a"}\n', encoding="utf-8")
    timing.write_text('{"chunk_id":"c1","call_id":"a"}\n', encoding="utf-8")
    meter.write_text(
        '{"chunk_id":"c1","cumulative_a100_hours":3.0}\n', encoding="utf-8"
    )
    state.write_text('{"cumulative_a100_hours":3.0}\n', encoding="utf-8")
    entrypoint._verify_local_commit_boundary(ledger, timing, meter, state)
