"""The active A8 release must run the full grid without a cost checkpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "seal_extension_full_grid_execution.py"
SPEC = importlib.util.spec_from_file_location("seal_extension_full_grid_execution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_complete_grid_is_released_without_cost_gate() -> None:
    payload = release.build()
    policy = payload["execution_policy"]
    assert payload["status"] == "complete_grid_released_model_by_model"
    assert len(payload["ordered_cells"]) == 16
    assert payload["items_per_cell"] == 8192
    assert policy["released_calls"] == 548864
    assert policy["first_1000_attempt_pause_required"] is False
    assert policy["runtime_reforecast_required"] is False
    assert policy["a100_hour_stop"] is None
    assert policy["cost_based_block_admission"] is False


def test_execution_order_is_model_major() -> None:
    payload = release.build()
    assert len(payload["model_runs"]) == 4
    assert all(run["calls"] == 137216 for run in payload["model_runs"])
    assert [run["family"] for run in payload["model_runs"]] == [
        "qwen",
        "granite",
        "llama",
        "mistral",
    ]
    assert [cell.rsplit("::", 1)[1] for cell in payload["model_runs"][0]["cells"]] == [
        "civil_comments",
        "multi_nli",
        "imdb",
        "ag_news",
    ]


def test_release_does_not_claim_gpu_execution() -> None:
    state = release.build()["run_state"]
    assert state["budget_policy_allows_complete_grid"] is True
    assert state["gpu_work_launched_by_this_release"] is False
    assert state["scientific_model_outputs_generated_by_this_release"] is False
