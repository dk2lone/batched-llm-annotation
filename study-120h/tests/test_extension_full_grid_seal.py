"""The A8 CPU lock must bind the agreed fixed 4 x 4 design."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "seal_extension_full_grid.py"
SPEC = importlib.util.spec_from_file_location("seal_extension_full_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
a8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a8)


def test_full_grid_binds_all_cells_items_and_calls() -> None:
    lock = a8.build()
    assert len(lock["models"]) == 4
    assert len(lock["datasets"]) == 4
    assert len(lock["ordered_cells"]) == 16
    assert len(lock["selected_schedules"]["cells"]) == 16
    assert all(row["n_items"] == 8192 for row in lock["datasets"].values())
    assert all(row["n_items"] == 8192 for row in lock["selected_schedules"]["cells"].values())
    assert lock["call_totals"]["per_cell"] == {
        "joint": 1536,
        "exact_context": 24576,
        "solo": 8192,
        "total": 34304,
    }
    assert lock["call_totals"]["full_grid"]["total"] == 548864


def test_first_round_covers_each_model_and_task_once() -> None:
    lock = a8.build()
    cells = lock["first_runtime_checkpoint"]
    assert len(cells) == 4
    assert len({key.split("::", 1)[0] for key in cells}) == 4
    assert len({key.rsplit("::", 1)[1] for key in cells}) == 4


def test_analysis_and_budget_cannot_drop_cells_on_results() -> None:
    lock = a8.build()
    analysis = lock["analysis_lock"]
    assert analysis["primary_malformed_treatment"] == "unresolved_as_correct"
    assert len(analysis["malformed_treatments"]) == 4
    assert "complete 16-cell grid" in analysis["recurrence_gate"]
    assert lock["design"]["accuracy_and_parse_role"].startswith("reported moderators")
    assert lock["budget_lock"]["gpu_start_authorized_by_this_lock"] is False
    assert "predictions" in lock["budget_lock"]["outcome_blinding"]


def test_missing_frozen_plan_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = a8.load

    def without_one(path: Path):
        value = original(path)
        if path.name == "B3_PLANS.json":
            missing = "Qwen/Qwen3-8B::civil_comments"
            value["plans"] = [
                row
                for row in value["plans"]
                if not (row["cell_key"] == missing and row["n_items"] == 8192)
            ]
        return value

    monkeypatch.setattr(a8, "load", without_one)
    with pytest.raises(ValueError, match="lacks one or more"):
        a8.selected_plans()
