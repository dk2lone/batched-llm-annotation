"""CPU-only validation of the X1R4 A8 B3 Modal runner."""

import gc
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "modal_extension_b3.py"
SPEC = importlib.util.spec_from_file_location("modal_extension_b3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
b3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b3)

MODELS = (
    "Qwen/Qwen3-8B",
    "ibm-granite/granite-3.3-8b-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)


def test_dry_run_rebuilds_every_released_call_without_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU plan construction contacted Modal")

    monkeypatch.setattr(b3.inspect_checkpoint, "remote", forbidden)
    monkeypatch.setattr(b3.recover_payloads, "remote", forbidden)
    for model_id in MODELS:
        plan, jobs_by_task = b3.build_plan(model_id)
        assert plan["total_calls"] == 137_216
        assert plan["dataset_order"] == [
            "civil_comments",
            "multi_nli",
            "imdb",
            "ag_news",
        ]
        assert [len(jobs_by_task[task]) for task in plan["dataset_order"]] == [34_304] * 4
        assert [cell["output_path"] for cell in plan["cells"]] == [
            f"cells/{task}/events.jsonl" for task in plan["dataset_order"]
        ]
        assert all(len(cell["call_id_order_sha256"]) == 64 for cell in plan["cells"])
        assert all(len(cell["prompt_hash_order_sha256"]) == 64 for cell in plan["cells"])
        assert all(len(cell["item_id_order_sha256"]) == 64 for cell in plan["cells"])
        del jobs_by_task, plan
        gc.collect()


def _job() -> dict[str, object]:
    call_id = "extx1::b3::Qwen--Qwen3-8B::civil_comments::one"
    return {
        "call_id": call_id,
        "attempt_id": f"{call_id}::attempt::1",
        "attempt": 1,
        "model_id": "Qwen/Qwen3-8B",
        "task_id": "civil_comments",
    }


def _event(job: dict[str, object], status: str = "ok") -> dict[str, object]:
    return {**job, "status": status}


def test_resume_skips_successes_and_schedules_one_linked_retry() -> None:
    job = _job()
    assert b3.pending_work([job], []) == [job]
    assert b3.pending_work([job], [_event(job)]) == []
    retry = b3.pending_work([job], [_event(job, "truncated")])
    assert len(retry) == 1
    assert retry[0]["attempt"] == 2
    assert retry[0]["parent_attempt_id"] == job["attempt_id"]
    assert b3.pending_work([job], [_event(job, "truncated"), _event(retry[0])]) == []


def test_terminal_retry_truncation_is_an_executed_unresolved_output() -> None:
    job = _job()
    retry = b3._retry(job)
    report = b3.completeness([job], [_event(job, "truncated"), _event(retry, "truncated")])
    assert report["all_calls_attempted"] is True
    assert report["complete"] is True
    assert report["execution_complete"] is True
    assert report["parsed_complete"] is False
    assert report["terminal_failures"] == 0
    assert report["terminal_malformed_outputs"] == 1
    assert report["unresolved_outputs"] == 1


def test_terminal_worker_failure_still_leaves_the_cell_incomplete() -> None:
    job = _job()
    retry = b3._retry(job)
    report = b3.completeness([job], [_event(job, "worker_failed"), _event(retry, "worker_failed")])
    assert report["complete"] is False
    assert report["terminal_failures"] == 1


def test_resume_rejects_conflicts_and_foreign_events() -> None:
    job = _job()
    duplicate = _event(job)
    with pytest.raises(ValueError, match="conflicting"):
        b3.resume_state([job], [duplicate, {**duplicate, "status": "truncated"}])
    with pytest.raises(ValueError, match="foreign"):
        b3.resume_state([job], [{**duplicate, "task_id": "imdb"}])


@pytest.mark.parametrize(
    ("task_id", "good", "bad"),
    [
        ("civil_comments", '{"labels":{"d":1}}', '{"labels":{"d":2}}'),
        ("multi_nli", '{"labels":{"d":2}}', '{"labels":{"d":3}}'),
        ("imdb", '{"labels":{"d":0}}', '{"labels":{"d":-1}}'),
        ("ag_news", '{"labels":{"d":3}}', '{"labels":{"d":4}}'),
    ],
)
def test_parser_behavior_is_frozen(task_id: str, good: str, bad: str) -> None:
    from freezeb.e2 import parse_result

    parsed = parse_result(good, ["d"], "stop", task_id=task_id)
    assert parsed == {"status": "ok", "parse_status": "ok", "predictions": {"d": int(good[-3])}}
    malformed = parse_result(bad, ["d"], "stop", task_id=task_id)
    assert malformed["status"] == "ok"
    assert malformed["parse_status"] == "malformed"
    assert malformed["predictions"] is None
    truncated = parse_result(good, ["d"], "length", task_id=task_id)
    assert truncated == {"status": "truncated", "parse_status": "truncated", "predictions": None}
