"""CPU-only tests for implemented frozen Freeze B E2 cells."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest

from freezeb import e2, e2_amendment
from freezeb.runtime_benchmark import DECODING, policy_for

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "freeze-b-work/20260912-cpu-draft"
SEAL = ROOT / "freeze/20260912T222618Z-B1"
LOCK_PATH = ROOT / "freeze/20260913T022442Z-B2/ADMISSION_LOCK.json"
CHECKPOINTS_PATH = ROOT / "runtime-lock/20260913T004054Z-BRT1/benchmark/checkpoint-manifests.json"


@pytest.fixture(scope="module")
def admission_lock() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(LOCK_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def rows(admission_lock: dict[str, Any]) -> list[dict[str, Any]]:
    return e2.verify_confirm_frame(SOURCE, SEAL, admission_lock)


@pytest.fixture(scope="module")
def jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return e2.planned_jobs(rows)


@pytest.fixture(scope="module")
def rendered(rows: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return e2.render_jobs(rows, jobs)


def test_first_cell_and_frame_are_sealed(
    admission_lock: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    e2.verify_first_cell(admission_lock)
    assert len(rows) == e2.N_ITEMS
    assert len({row["doc_id"] for row in rows}) == e2.N_ITEMS


def test_complete_e2_plan_counts_and_appearances(jobs: list[dict[str, Any]]) -> None:
    assert len(jobs) == e2.TOTAL_CALLS == 17152
    assert Counter(job["condition"] for job in jobs) == Counter(
        joint=768, exact_context=12288, solo=4096
    )
    appearances: Counter[tuple[str, str]] = Counter()
    for job in jobs:
        for target_id in job["target_ids"]:
            appearances[(job["condition"], target_id)] += 1
    ids = {target_id for _condition, target_id in appearances}
    assert all(appearances[("joint", doc_id)] == 3 for doc_id in ids)
    assert all(appearances[("exact_context", doc_id)] == 3 for doc_id in ids)
    assert all(appearances[("solo", doc_id)] == 1 for doc_id in ids)


def test_exact_context_reuses_joint_documents_and_order(jobs: list[dict[str, Any]]) -> None:
    joint = {
        (job["assignment"], job["call_index"]): job for job in jobs if job["condition"] == "joint"
    }
    for job in jobs:
        if job["condition"] == "exact_context":
            source = joint[(job["assignment"], job["call_index"])]
            assert job["doc_order"] == source["doc_order"]
            assert job["target_ids"][0] in job["doc_order"]


def test_plan_is_deterministic_interleaved_and_block_owned(
    rows: list[dict[str, Any]], jobs: list[dict[str, Any]]
) -> None:
    assert jobs == e2.planned_jobs(rows)
    keys = [(job["execution_block"], job["execution_order_key"]) for job in jobs]
    assert keys == sorted(keys)
    for block in range(8):
        first = [job["condition"] for job in jobs if job["execution_block"] == block][:3]
        rotated = e2.CONDITIONS[block % 3 :] + e2.CONDITIONS[: block % 3]
        assert tuple(first) == rotated


def test_rendered_prompts_obey_condition_targets(
    rendered: list[dict[str, Any]],
) -> None:
    assert len(rendered) == e2.TOTAL_CALLS
    by_condition = {job["condition"]: job for job in rendered}
    for condition in e2.CONDITIONS:
        job = by_condition[condition]
        assert job["prompt_sha256"]
        assert job["max_tokens"] > 0
        assert job["messages"][1]["content"].count("<doc id=") == job["batch_size"]


def test_first_response_is_preserved_and_only_failures_retry(
    jobs: list[dict[str, Any]], rendered: list[dict[str, Any]]
) -> None:
    first_job = rendered[0]
    target = first_job["target_ids"][0]
    failed = e2.event_for_result(first_job, generation_error="worker exited")
    malformed_job = next(job for job in rendered if job["condition"] == "solo")
    malformed = e2.event_for_result(malformed_job, raw="not json", finish_reason="stop")

    assert failed["first_response"] == ""
    assert malformed["status"] == "ok"
    assert malformed["parse_status"] == "malformed"
    work = e2.pending_work(jobs, [failed, malformed])
    assert [job["call_id"] for job in work["retries"]] == [first_job["call_id"]]
    retry = work["retries"][0]
    assert retry["attempt_id"] == f"{first_job['call_id']}::attempt::2"
    assert retry["parent_attempt_id"] == first_job["attempt_id"]

    valid = e2.event_for_result(
        first_job,
        raw=json.dumps({"labels": {doc_id: 1 for doc_id in first_job["target_ids"]}}),
        finish_reason="stop",
    )
    assert valid["predictions"][target] == 1
    assert valid["output_sha256"]


def test_resume_rejects_conflicting_or_foreign_attempts(
    jobs: list[dict[str, Any]], rendered: list[dict[str, Any]]
) -> None:
    event = e2.event_for_result(rendered[0], generation_error="worker exited")
    conflict = {**event, "generation_error": "different failure"}
    with pytest.raises(ValueError, match="conflicting"):
        e2.resume_state(jobs, [event, conflict])
    with pytest.raises(ValueError, match="foreign"):
        e2.resume_state(jobs, [{**event, "model_id": "other/model"}])


def test_timing_attribution_and_measured_reforecast() -> None:
    events = [
        {"queue_seconds": None, "prefill_seconds": None, "decode_seconds": None},
        {"queue_seconds": None, "prefill_seconds": None, "decode_seconds": None},
    ]
    shares, method = e2.attribute_seconds(6.0, events)
    assert shares == [3.0, 3.0]
    assert method == "equal_share_within_chunk"

    timings = [
        {"condition": "joint", "metered_a100_seconds": 0.4},
        {"condition": "exact_context", "metered_a100_seconds": 0.3},
        {"condition": "solo", "metered_a100_seconds": 0.05},
    ]
    report = e2.reforecast(
        timings,
        {"joint": 10, "exact_context": 20, "solo": 30},
        cumulative_a100_hours=5.0,
        workers=1,
    )
    assert report["projected_remaining_base_a100_hours"] == pytest.approx(11.5 / 3600)
    assert report["may_continue"] is True
    assert report["n_concurrent_workers_when_measured"] == 1


def test_worker_provenance_is_runtime_locked() -> None:
    policy = policy_for(e2.MODEL_ID)
    backend = {
        "backend": "vllm",
        "model_id": e2.MODEL_ID,
        "model_revision": e2.MODEL_REVISION,
        "tokenizer_revision": e2.MODEL_REVISION,
        "dtype": "bfloat16",
        "max_model_len": policy.max_model_len,
        "max_num_seqs": 32,
        "tensor_parallel_size": 1,
        "enable_thinking": False,
        "seed": int(DECODING["seed"]),
    }
    e2.verify_worker(
        {"gpu_name": "NVIDIA A100-SXM4-40GB", "installed_engine_version_string": "0.28.0"},
        backend,
    )
    with pytest.raises(ValueError, match="different vLLM"):
        e2.verify_worker({"gpu_name": "A100", "installed_engine_version_string": "0.27.0"}, backend)


def test_parallel_amendment_is_written_before_smoke(tmp_path: Path) -> None:
    amendment_path, payload = e2_amendment.ensure(
        tmp_path, "admission-hash", workers=4, plan_sha256="plan-hash"
    )
    assert amendment_path.exists()
    assert payload["recorded_before_any_e2_model_output"] is True
    assert payload["equivalence_smoke"]["status"] == "pending"
    assert amendment_path.with_suffix(".md").exists()

    completed = e2_amendment.complete_smoke(
        amendment_path,
        payload,
        {"status": "passed", "synthetic_only": True, "response_sha256": "abc"},
    )
    assert completed["equivalence_smoke"]["status"] == "passed"
    assert amendment_path.with_suffix(".smoke.json").exists()
    assert json.loads(amendment_path.read_text(encoding="utf-8")) == payload
    _, resumed = e2_amendment.ensure(tmp_path, "admission-hash", workers=4, plan_sha256="plan-hash")
    assert resumed["equivalence_smoke"]["status"] == "passed"
    with pytest.raises(RuntimeError, match="different E2 execution plan"):
        e2_amendment.ensure(tmp_path, "admission-hash", workers=2, plan_sha256="plan-hash")


def test_modal_runner_builds_the_same_sealed_plan_without_execution() -> None:
    runner_path = ROOT / "modal_freezeb_e2.py"
    spec = importlib.util.spec_from_file_location("modal_freezeb_e2", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    first = runner.build_plan(chunk_size=32, workers=4)
    second = runner.build_plan(chunk_size=32, workers=4)
    assert first == second
    assert first["total_calls"] == e2.TOTAL_CALLS
    assert first["calls_by_condition"] == {
        "exact_context": e2.EXACT_CALLS,
        "joint": e2.JOINT_CALLS,
        "solo": e2.SOLO_CALLS,
    }
    assert first["block_owners"] == {str(block): block % 4 for block in range(8)}
    assert (
        first["plan_sha256"] == "d642f6c3fd5deaaefce76aadcaeeef9d577a20df513c999d715fae8e21672409"
    )
    assert first["results_namespace"] == first["plan_sha256"][:16]


def test_second_cell_runner_is_sealed_and_independent() -> None:
    runner_path = ROOT / "modal_freezeb_e2.py"
    module_spec = importlib.util.spec_from_file_location("modal_freezeb_e2_second", runner_path)
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)

    spec = e2.SECOND_CELL_SPEC
    plan = runner.build_plan(chunk_size=32, workers=4, cell=spec.name)
    jobs = runner._jobs(spec.name)
    assert plan["cell_key"] == spec.cell_key
    assert plan["plan_sha256"] == "81159b7ea6ebb3fcf9ecd5e0a5a1ea26730519c8200cda8ce54f7926c7178f31"
    assert plan["base_forecast_a100_hours"] == pytest.approx(0.8081930165939837)
    assert plan["prior_cumulative_a100_hours"] == pytest.approx(4.824000183954456)
    assert len(jobs) == e2.TOTAL_CALLS
    assert {job["cell_key"] for job in jobs} == {spec.cell_key}
    assert {job["model_id"] for job in jobs} == {spec.model_id}
    assert {job["task_id"] for job in jobs} == {spec.task_id}
    assert all("civil_comments" in job["call_id"] for job in jobs)

    sample = next(job for job in jobs if job["condition"] == "solo")
    target = sample["target_ids"][0]
    event = e2.event_for_result(
        sample,
        raw=json.dumps({"labels": {target: 0}}),
        finish_reason="stop",
    )
    assert event["predictions"] == {target: 0}


def test_second_cell_amendment_has_independent_identity(tmp_path: Path) -> None:
    amendment_path, payload = e2_amendment.ensure(
        tmp_path,
        "admission-hash",
        workers=4,
        plan_sha256="granite-plan-hash",
        amendment_id="freezeb-e2-granite-civil-concurrency",
        scope=f"sealed E2 cell {e2.SECOND_CELL_SPEC.cell_key} only",
    )
    assert amendment_path.name.endswith("-freezeb-e2-granite-civil-concurrency.json")
    assert payload["scope"] == f"sealed E2 cell {e2.SECOND_CELL_SPEC.cell_key} only"


def test_second_cell_checkpoint_uses_its_own_sealed_manifest() -> None:
    manifests = cast(list[dict[str, Any]], json.loads(CHECKPOINTS_PATH.read_text(encoding="utf-8")))
    spec = e2.SECOND_CELL_SPEC
    expected = next(row for row in manifests if row["model_id"] == spec.model_id)
    actual = {
        "model_id": spec.model_id,
        "revision": spec.model_revision,
        "files": expected["files"],
    }
    report = e2.verify_checkpoint(actual, manifests, spec=spec)
    assert report["model_id"] == spec.model_id
    assert report["revision"] == spec.model_revision
    assert report["n_files_hash_verified"] == 15

    with pytest.raises(ValueError, match="selected pinned model revision"):
        e2.verify_checkpoint({**actual, "model_id": e2.MODEL_ID}, manifests, spec=spec)


def test_saved_reforecast_must_authorize_the_exact_plan(tmp_path: Path) -> None:
    runner_path = ROOT / "modal_freezeb_e2.py"
    module_spec = importlib.util.spec_from_file_location("modal_freezeb_e2_barrier", runner_path)
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    plan = runner.build_plan(chunk_size=32, workers=4, cell=e2.SECOND_CELL_SPEC.name)
    barrier_path = tmp_path / "reforecast-1000.json"

    barrier_path.write_text(
        json.dumps(
            {
                "cell_key": plan["cell_key"],
                "plan_sha256": plan["plan_sha256"],
                "may_continue": True,
            }
        ),
        encoding="utf-8",
    )
    assert runner._validated_barrier(barrier_path, plan)["may_continue"] is True

    cases: list[tuple[dict[str, Any], str]] = [
        ({"may_continue": False}, "does not authorize"),
        ({"plan_sha256": "other-plan"}, "different E2 execution plan"),
        ({"cell_key": e2.FIRST_CELL}, "different E2 cell"),
    ]
    for changed, message in cases:
        payload = {
            "cell_key": plan["cell_key"],
            "plan_sha256": plan["plan_sha256"],
            "may_continue": True,
            **changed,
        }
        barrier_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match=message):
            runner._validated_barrier(barrier_path, plan)


def test_recovered_chunks_keep_original_session_metering() -> None:
    runner_path = ROOT / "modal_freezeb_e2.py"
    module_spec = importlib.util.spec_from_file_location("modal_freezeb_e2_meter", runner_path)
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    payload = {
        "meter_session_id": "worker-session",
        "meter_session_elapsed_seconds": 125.0,
        "chunk_wall_seconds": 5.0,
        "served_from_cache": True,
    }

    session, elapsed, delta = runner._session_meter_delta(payload, {}, 1.2)
    assert (session, elapsed, delta) == ("worker-session", 125.0, 150.0)
    session, elapsed, delta = runner._session_meter_delta(payload, {"worker-session": 100.0}, 1.2)
    assert (session, elapsed, delta) == ("worker-session", 125.0, 30.0)


def test_rejected_payload_meter_does_not_enter_scientific_ledger(tmp_path: Path) -> None:
    runner_path = ROOT / "modal_freezeb_e2.py"
    module_spec = importlib.util.spec_from_file_location("modal_freezeb_e2_quarantine", runner_path)
    assert module_spec is not None and module_spec.loader is not None
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    events = tmp_path / "events.jsonl"
    timings = tmp_path / "timing.jsonl"
    meter_path = tmp_path / "meter.jsonl"
    state = tmp_path / "meter-state.json"
    runner._append_jsonl(
        meter_path,
        [
            {
                "kind": "scientific_payload",
                "chunk_id": "rejected-chunk",
                "gpu_seconds_billed": 125.0,
                "cumulative_a100_hours": 4.5,
            }
        ],
    )
    runner._write_json(
        state,
        {"cumulative_a100_hours": 4.5, "session_elapsed_seconds": {"worker-session": 125.0}},
    )

    runner._verify_local_boundary(events, timings, meter_path, state)
    assert not events.exists()
    assert not timings.exists()
