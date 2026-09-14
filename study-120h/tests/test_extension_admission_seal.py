"""A7 must bind the completed solo screen to frozen B3/B4 work."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "seal_extension_admission.py"
SPEC = importlib.util.spec_from_file_location("seal_extension_admission", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
a7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a7)


def test_a7_binds_all_selected_calls_and_budget() -> None:
    lock = a7.build()
    assert len(lock["solo_metrics"]) == 20
    assert len(lock["ordered_cells"]["primary"]) == 7
    assert len(lock["ordered_cells"]["family_core"]) == 4
    assert len(lock["ordered_cells"]["e5_cells"]) == 4
    assert set(lock["selected_schedules"]["b3"]) == set(lock["ordered_cells"]["primary"])
    assert set(lock["selected_schedules"]["b4"]) == set(lock["ordered_cells"]["e5_cells"])
    assert lock["selected_schedules"]["b3_total_calls"] == 94336
    assert lock["selected_schedules"]["b4_total_calls"] == 352000
    assert all(row["power_target_met"] for row in lock["item_counts"].values())
    assert lock["budget_lock"]["guarded_headroom_after_reserve_a100_hours"] > 0
    assert len(set(lock["worker_ownership"]["family_to_logical_slot"].values())) == 4


def test_a7_rejects_missing_selected_schedule() -> None:
    lock = a7.build()
    b3 = a7.load(a7.SEAL / "B3_PLANS.json")
    b4 = a7.load(a7.SEAL / "B4_SCHEDULES.json")
    key = lock["ordered_cells"]["primary"][0]
    n = lock["item_counts"][key]["n_items"]
    b3["plans"] = [r for r in b3["plans"] if (r["cell_key"], r["n_items"]) != (key, n)]
    b3["n_plans"] -= 1
    with pytest.raises(ValueError, match="no frozen B3 schedule"):
        a7.selected_schedule(lock, b3, b4)


def test_a7_rejects_changed_m1_completion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = a7.M1
    for name in ("admission-inputs.json", "SUMMARY.json", "plan.json", "receipts.jsonl"):
        (tmp_path / name).symlink_to(source / name)
    summary = a7.load(source / "SUMMARY.json")
    summary["screen_complete"] = False
    (tmp_path / "SUMMARY.json").unlink()
    a7.write_json(tmp_path / "SUMMARY.json", summary)
    monkeypatch.setattr(a7, "M1", tmp_path)
    with pytest.raises(ValueError, match="completion record"):
        a7.build()
