"""Step 1 safety, reproducibility, ledger, and screen tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from step1.amendment import build as build_amendment
from step1.backends import CharEstimateCounter, FakeBackend, request_timing, token_ids_length
from step1.frozen import load
from step1.ledger import ROW_FIELDS, LedgerError, append, append_run, validate
from step1.paths import Paths
from step1.plan import build_call, doc_texts, job_rows, retry_job, solo_jobs
from step1.prompts import build_joint, build_single, parse_labels
from step1.report import verify_step0_seal
from step1.runner import SoloOnlyError, preflight, run_solo
from step1.screen import decide_replacement, decide_task, screen_cell


class ExactCounter(CharEstimateCounter):
    name = "test_exact_counter"
    exact = True


def test_token_count_handles_batch_encoding_shape() -> None:
    assert token_ids_length([1, 2, 3]) == 3
    assert token_ids_length({"input_ids": [1, 2, 3]}) == 3
    assert token_ids_length({"input_ids": [[1, 2, 3]]}) == 3


def test_request_timing_supports_vllm_v1_and_legacy_fields() -> None:
    v1 = SimpleNamespace(queued_ts=1.0, scheduled_ts=1.2,
                         first_token_ts=1.7, last_token_ts=2.5)
    legacy = SimpleNamespace(time_in_queue=0.2, first_scheduled_time=1.2,
                             first_token_time=1.7, finished_time=2.5)
    assert request_timing(v1) == pytest.approx((0.2, 0.5, 0.8))
    assert request_timing(legacy) == pytest.approx((0.2, 0.5, 0.8))


@pytest.fixture(scope="module")
def frozen() -> Any:
    return load()


def test_step0_seal_is_the_recorded_freeze(frozen: Any) -> None:
    report = verify_step0_seal(frozen)
    assert report["ok"]
    assert report["n_files_checked"] == 70
    assert report["manifest_matches_record"]


def test_sealed_solo_plan_is_reproduced_for_all_four_cells(frozen: Any) -> None:
    for cell in frozen.cells:
        cell_id = str(cell["cell_id"])
        frame = "civil_comments" if cell["task_id"] == "civil_comments" else "ag_news"
        sealed = [row for row in frozen.job_skeletons(cell_id) if str(row["condition"]) == "solo"]
        rebuilt = solo_jobs(frozen, cell_id, frame)
        assert len(sealed) == len(rebuilt) == 2048
        assert {row["call_id"] for row in sealed} == {row["call_id"] for row in rebuilt}


def _pilot_prompts_module() -> Any:
    path = Path(__file__).resolve().parents[2] / "gate-experiments/a/src/batchgate/prompts.py"
    spec = importlib.util.spec_from_file_location("preserved_gatea_prompts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_civil_prompt_is_byte_identical_to_preserved_gate_a(frozen: Any) -> None:
    pilot = _pilot_prompts_module()
    spec = frozen.protocol["prompts"]["civil_comments"]
    docs = [("cc_a", "first text"), ("cc_b", "second text")]
    assert build_joint(spec, docs) == pilot.build_joint_prompt(docs)
    assert build_single(spec, docs, "cc_b") == pilot.build_single_prompt(docs, "cc_b")


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ('{"labels":{"x":1}}', "ok"),
        ('```json\n{"labels":{"x":"0"}}\n```', "ok"),
        ('{"labels":{"x":true}}', "bad_value"),
        ('{"labels":{"x":0,"x":1}}', "duplicate_ids"),
        ('prefix {"labels":{"x":1}}', "no_json"),
        ('{"labels":{"y":1}}', "missing_labels"),
    ],
)
def test_parser_stays_strict(raw: str, status: str) -> None:
    assert parse_labels(raw, ["x"], (0, 1)).status == status


def test_preflight_checks_only_conditions_frozen_for_the_cell(frozen: Any) -> None:
    report = preflight(
        frozen,
        "mistral_civil",
        "civil_comments",
        ExactCounter(chars_per_token=1000),
    )
    expected = {
        str(row["condition"]) for row in job_rows(frozen, "mistral_civil", "civil_comments")
    }
    assert report["ok"]
    assert set(report["by_condition"]) == expected == {"joint", "exact_context", "solo"}


def test_coarse_counter_can_never_pass_preflight(frozen: Any) -> None:
    report = preflight(
        frozen,
        "mistral_civil",
        "civil_comments",
        CharEstimateCounter(chars_per_token=1000),
        conditions=("solo",),
    )
    assert not report["ok"]
    assert not report["exact_tokenizer"]


def test_solo_runner_refuses_non_solo_work(frozen: Any, tmp_path: Path) -> None:
    job = job_rows(
        frozen,
        "mistral_civil",
        "civil_comments",
        conditions=("joint",),
    )[0]
    with pytest.raises(SoloOnlyError, match="solo responses only"):
        run_solo(
            frozen,
            "mistral_civil",
            "civil_comments",
            FakeBackend({}),
            tmp_path / "ledger.jsonl",
            jobs=[job],
        )


def test_retry_gets_a_new_id_and_keeps_its_parent(frozen: Any) -> None:
    first = solo_jobs(frozen, "qwen_civil", "civil_comments")[0]
    retry = retry_job(first, 1)
    assert retry["call_id"] != first["call_id"]
    assert retry["parent_call_id"] == first["call_id"]
    assert retry["retry_index"] == 1


def test_append_rejects_a_conflicting_output(tmp_path: Path) -> None:
    row: dict[str, Any] = {field: None for field in ROW_FIELDS}
    row.update(
        {
            "call_id": "a",
            "parent_call_id": None,
            "retry_index": 0,
            "first_response_status": "valid_first_response",
            "output_sha256": "one",
        }
    )
    path = tmp_path / "ledger.jsonl"
    assert append(path, [row])["appended"] == 1
    changed = {**row, "output_sha256": "two"}
    report = append(path, [changed])
    assert report["conflicts_rejected"] == 1
    assert validate(path, [], rows_per_call=1)["n_rejected_conflicts"] == 1


def test_model_specific_paths_cannot_mix_replacement_outputs(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    first = paths.ledger("mistral_civil", "civil_comments", "mistralai/Mistral-7B")
    replacement = paths.ledger("mistral_civil", "civil_comments", "allenai/OLMo-2")
    assert first != replacement


def test_end_to_end_fake_solo_screen_passes(frozen: Any, tmp_path: Path) -> None:
    cell_id = "mistral_civil"
    frame = "civil_comments"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    items = frozen.items(frame)
    references = {item.doc_id: item.reference for item in items}
    ledger_path = tmp_path / "solo.jsonl"
    result = run_solo(
        frozen,
        cell_id,
        frame,
        FakeBackend(references, error_rate=0.15),
        ledger_path,
        model_id=model_id,
        engine_version="0.28.0",
        chunk_size=256,
    )
    assert result["n_attempted"] == 2048
    append_run(
        tmp_path / "runs.jsonl",
        {
            "run_id": "test",
            "worker_id": "fake",
            "model_id": model_id,
            "gpu_name": "NVIDIA A100",
            "gpu_seconds_billed": 1.0,
        },
    )
    all_conditions = {
        str(row["condition"]) for row in job_rows(frozen, cell_id, frame, model_id=model_id)
    }
    preflight_report = {
        "ok": True,
        "exact_tokenizer": True,
        "n_over_budget": 0,
        "by_condition": {name: {} for name in all_conditions},
    }
    host = {
        "cuda_version": "12.8",
        "driver_version": "570",
        "gpu_name": "NVIDIA A100",
        "gpu_uuid": "GPU-test",
        "torch_version": "2.10",
        "installed_engine_version_string": "0.28.0",
    }
    screen = screen_cell(
        frozen,
        ledger_path,
        cell_id,
        frame,
        model_id=model_id,
        preflight_report=preflight_report,
        host=host,
        runs_path=tmp_path / "runs.jsonl",
    )
    assert screen["outcome"] == "PASS"
    assert screen["solo_summary"]["n_solo_calls"] == 2048


def test_information_poor_triggers_task_fallback_but_not_model_replacement(frozen: Any) -> None:
    screens = [
        {
            "cell_id": "qwen_secondtask",
            "model_id": "Qwen/Qwen3-8B",
            "frame_task_id": "ag_news",
            "outcome": "INFORMATION_POOR",
        },
        {
            "cell_id": "mistral_secondtask",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "frame_task_id": "ag_news",
            "outcome": "PASS",
        },
    ]
    assert decide_task(frozen, screens)["chosen_second_task"] is None
    replacement = decide_replacement(frozen, screens)
    assert not replacement["replace"]
    assert replacement["information_poor_cells_left_in_panel"] == []


def test_execution_amendment_changes_budget_only(frozen: Any) -> None:
    record = build_amendment(frozen, "20260912T120000Z")
    assert record["amends_protocol_content_hash"] == frozen.content_hash
    assert record["freeze_is_unchanged"]
    assert record["overall_ceiling_a100_hours"] == 150
    assert record["max_concurrent_workers"] == 2
    assert record["tensor_parallel_size"] == 1


def test_call_building_uses_the_frozen_output_cap(frozen: Any) -> None:
    job = solo_jobs(frozen, "qwen_civil", "civil_comments")[0]
    call = build_call(frozen, job, "civil_comments", doc_texts(frozen, "civil_comments"))
    assert call.condition == "solo"
    assert call.max_tokens == 84


def test_append_rejects_missing_schema_fields(tmp_path: Path) -> None:
    with pytest.raises(LedgerError, match="missing mandatory"):
        append(tmp_path / "bad.jsonl", [{"call_id": "x"}])
