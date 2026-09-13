"""Focused tests for the cross-family extension (X1R4) planning and analysis code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from freezeb import assign, frames, meter, repeated, screen
from freezeb import extension as ext
from freezeb.claims import CellData
from freezeb.power import choose_n, simulate
from freezeb.selection import Cell
from step0 import stats
from step0.hashing import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
SEAL = ROOT / ext.FREEZE_B1
WORK = ROOT / "study-120h/ext-work/20260913-cpu-draft"
HAS_FRAMES = all((WORK / task.key / "manifest.json").is_file() for task in ext.EXT_TASKS)
HAS_SEAL = (SEAL / "frame-identifiers").is_dir()


# --- Frozen identity -------------------------------------------------------


def test_panel_is_five_families_and_four_tasks() -> None:
    assert len(ext.EXT_MODELS) == 5
    assert len({m.family for m in ext.EXT_MODELS}) == 5
    assert [m.order for m in ext.EXT_MODELS] == [1, 2, 3, 4, 5]
    assert ext.EXT_TASK_KEYS == ("civil_comments", "multi_nli", "imdb", "ag_news")
    assert all(len(m.revision) == 40 for m in ext.EXT_MODELS)


def test_extension_imdb_uses_the_train_split_and_scans_every_row() -> None:
    imdb = next(t for t in ext.EXT_TASKS if t.key == "imdb")
    freeze_b_imdb = next(t for t in frames.TASKS if t.key == "imdb")
    assert imdb.split == "train"
    assert freeze_b_imdb.split == "test"
    assert imdb.revision == freeze_b_imdb.revision
    # The train split is label sorted; a smaller scan yields negative reviews only.
    assert imdb.scan_limit == 25000


def test_extension_namespaces_do_not_collide_with_freeze_b() -> None:
    assert ext.ASSIGN_NAMESPACE != assign.NAMESPACE
    assert ext.E5_NAMESPACE != repeated.NAMESPACE
    assert ext.EXT_SEED != frames.SEED
    frame = assign.Frame(tuple(f"d{i:03d}" for i in range(128)), tuple(range(128)))
    freeze_b = assign.partition(frame, "Qwen/Qwen3-8B::imdb", 16, 1)
    extension = assign.partition(
        frame, "Qwen/Qwen3-8B::imdb", 16, 1, namespace=ext.ASSIGN_NAMESPACE
    )
    assert not np.array_equal(freeze_b, extension)
    assert set(freeze_b.ravel()) == set(extension.ravel())


def test_freeze_b_defaults_are_unchanged() -> None:
    """Every sealed Freeze B default must still produce its original value."""
    assert frames.SEED == 20260912
    assert assign.NAMESPACE == "bigdata2026-freezeb-v1"
    assert repeated.NAMESPACE == "bigdata2026-freezeb-e5-v1"
    assert (meter.TARGET_HOURS, meter.STOP_HOURS, meter.RESERVE_HOURS) == (120.0, 150.0, 10.0)
    assert meter.may_start(100, 20).start is True
    assert meter.may_start(110, 20).reason == "guarded_complete_block_exceeds_stop"
    assert meter.must_stop(150) is True
    assert len(frames.TASKS) == 6
    assert screen.TASK_KEYS == tuple(t.key for t in frames.TASKS)


def test_runtime_policy_attempts_every_candidate_and_holds_no_access_flag() -> None:
    """Access must never be baked into the immutable policy."""
    for model in ext.EXT_MODELS:
        policy = ext.ext_policy_for(model.repo)
        assert policy.runnable is True
        assert policy.exclusion_reason is None
    # No family gets a private context allowance.
    assert len({p.max_model_len for p in ext.EXT_MODEL_POLICIES}) == 1
    assert len(ext.EXT_MODEL_POLICIES) == 5


def _access(**overrides: str) -> list[ext.AccessRecord]:
    out = []
    for model_id in ext.EXT_MODEL_IDS:
        status = overrides.get(model_id, "verified")
        out.append(ext.AccessRecord(model_id, status == "verified", status, "test record"))
    return out


def test_access_lock_decides_every_candidate_and_can_enable_llama() -> None:
    llama = "meta-llama/Llama-3.1-8B-Instruct"
    blocked = ext.runtime_access_lock(
        _access(**{llama: "gated_pending_access"}),
        signed_by="daniel",
        runtime_lock_sha256="a" * 64,
        batch_invariance_report_sha256="c" * 64,
    )
    assert blocked["unavailable_models"] == [llama]
    assert blocked["n_attempted"] == 4
    # Granting access needs no code change, only a new signed lock.
    granted = ext.runtime_access_lock(
        _access(),
        signed_by="daniel",
        runtime_lock_sha256="b" * 64,
        batch_invariance_report_sha256="d" * 64,
    )
    assert granted["unavailable_models"] == []
    assert granted["n_attempted"] == 5
    assert [c["model_id"] for c in granted["candidates"]] == list(ext.EXT_MODEL_IDS)
    assert all(c["candidate_status"] in ext.CANDIDATE_STATUSES for c in granted["candidates"])


def test_access_lock_refuses_partial_or_unsigned_decisions() -> None:
    with pytest.raises(ValueError):
        ext.runtime_access_lock(
            _access()[:4],
            signed_by="daniel",
            runtime_lock_sha256="a" * 64,
            batch_invariance_report_sha256="b" * 64,
        )
    with pytest.raises(ValueError):
        ext.runtime_access_lock(
            _access(),
            signed_by="",
            runtime_lock_sha256="a" * 64,
            batch_invariance_report_sha256="b" * 64,
        )
    with pytest.raises(ValueError):
        ext.runtime_access_lock(
            _access(),
            signed_by="daniel",
            runtime_lock_sha256="short",
            batch_invariance_report_sha256="b" * 64,
        )
    with pytest.raises(ValueError):
        ext.runtime_access_lock(
            _access(),
            signed_by="daniel",
            runtime_lock_sha256="a" * 64,
            batch_invariance_report_sha256="short",
        )
    with pytest.raises(ValueError):
        ext.AccessRecord("not/a-candidate", True, "verified", "x")
    with pytest.raises(ValueError):
        ext.AccessRecord(ext.EXT_MODEL_IDS[0], True, "gated_pending_access", "x")


def test_unknown_model_and_task_are_refused() -> None:
    with pytest.raises(ValueError):
        ext.ext_policy_for("meta-llama/Llama-3.1-70B-Instruct")
    with pytest.raises(ValueError):
        ext.build_ext_task(ROOT, frames.TASKS[4], WORK)


# --- Budget ----------------------------------------------------------------


def test_extension_meter_uses_its_own_eighty_hour_ceiling() -> None:
    assert ext.STOP_HOURS == 80.0
    assert ext.RESERVE_HOURS == 10.0
    assert ext.METER_ORIGIN_HOURS == 0.0
    fits = ext.ext_may_start(0.0, 1.62)
    assert fits["start"] is True
    assert fits["projected_total_hours"] == pytest.approx(12.592)
    # 60 + 1.6 x 7 + 10 = 81.2, above the stop.
    blocked = ext.ext_may_start(60.0, 7.0)
    assert blocked["start"] is False
    assert blocked["reason"] == "guarded_complete_block_exceeds_extension_stop"
    assert ext.ext_may_start(80.0, 0.0)["reason"] == "extension_stop_reached"
    assert ext.ext_must_stop(79.999) is False
    assert ext.ext_must_stop(80.0) is True
    with pytest.raises(ValueError):
        ext.ext_may_start(-1.0, 1.0)


def test_extension_budget_does_not_move_the_freeze_b_meter() -> None:
    # A 60-hour extension meter must not read as inside the 150-hour Freeze B stop.
    assert ext.ext_may_start(75.0, 1.0)["start"] is False
    assert meter.may_start(75.0, 1.0, target_hours=80.0, stop_hours=80.0).start is False
    assert meter.may_start(75.0, 1.0).start is True


# --- Run order -------------------------------------------------------------


def _fake_rows(prefix: str, n: int, *, start: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": f"{prefix}{i + start:06d}",
            "text": f"synthetic document number {i + start} for planning only",
            "n_tokens": 8,
            "n_chars": 40 + (i * 7) % 400,
        }
        for i in range(n)
    ]


def test_m1_schedule_is_the_exact_40960_call_order() -> None:
    screens = {task: _fake_rows(task[:2], 2048) for task in ext.EXT_TASK_KEYS}
    order = ext.m1_call_order(screens)
    assert order["n_cells"] == 20
    assert order["n_calls"] == 40_960
    assert order["candidates_listed"] == list(ext.EXT_MODEL_IDS)
    assert len(order["cells"]) == 20
    assert sum(c["n_calls"] for c in order["cells"]) == 40_960
    # Candidate by candidate, then task in the frozen order.
    assert [c["model_id"] for c in order["cells"][:4]] == [ext.EXT_MODEL_IDS[0]] * 4
    assert [c["task_id"] for c in order["cells"][:4]] == list(ext.EXT_TASK_KEYS)
    # Llama is frozen into the schedule, never omitted.
    assert any(c["model_id"].startswith("meta-llama/") for c in order["cells"])
    assert ext.m1_call_order(screens) == order


def test_m1_call_ids_are_unique_extension_scoped_and_carry_prompt_hashes() -> None:
    rows = _fake_rows("cc", 2048)
    calls = list(ext.m1_cell_calls(ext.EXT_MODEL_IDS[0], "civil_comments", rows))
    assert len(calls) == 2048
    assert all(c["call_id"].startswith(f"{ext.CALL_PREFIX}::m1::") for c in calls)
    assert len({c["call_id"] for c in calls}) == 2048
    assert all(len(c["prompt_sha256"]) == 64 for c in calls)
    assert [c["position"] for c in calls] == list(range(1, 2049))
    # The frozen order follows the sealed frame row order exactly.
    assert [c["doc_id"] for c in calls] == [r["doc_id"] for r in rows]
    # No dataset text ever enters a frozen record.
    assert all("text" not in c for c in calls)
    digest = ext.m1_cell_digest(ext.EXT_MODEL_IDS[0], "civil_comments", rows)
    assert digest["n_calls"] == 2048
    assert len(digest["call_id_order_sha256"]) == 64
    assert len(digest["prompt_sha256_order_sha256"]) == 64
    assert digest["record_stream"]["count"] == 2048
    assert len(digest["record_stream"]["first_records"]) == 3
    assert len(digest["record_stream"]["last_records"]) == 3


def test_m1_prompt_hashes_differ_by_task_but_not_by_candidate() -> None:
    """One grammar for every family: the prompt cannot depend on the model."""
    rows = _fake_rows("cc", 2048)
    a = ext.m1_cell_digest(ext.EXT_MODEL_IDS[0], "civil_comments", rows)
    b = ext.m1_cell_digest(ext.EXT_MODEL_IDS[2], "civil_comments", rows)
    assert a["prompt_sha256_order_sha256"] == b["prompt_sha256_order_sha256"]
    assert a["call_id_order_sha256"] != b["call_id_order_sha256"]
    other_task = ext.m1_cell_digest(ext.EXT_MODEL_IDS[0], "imdb", _fake_rows("im", 2048))
    assert other_task["prompt_sha256_order_sha256"] != a["prompt_sha256_order_sha256"]


def test_m1_refuses_a_wrong_sized_or_duplicated_frame() -> None:
    with pytest.raises(ValueError):
        list(ext.m1_cell_calls(ext.EXT_MODEL_IDS[0], "imdb", _fake_rows("im", 2047)))
    duplicated = _fake_rows("im", 2048)
    duplicated[7] = dict(duplicated[6])
    with pytest.raises(ValueError):
        list(ext.m1_cell_calls(ext.EXT_MODEL_IDS[0], "imdb", duplicated))
    with pytest.raises(ValueError):
        ext.m1_call_order({"imdb": _fake_rows("im", 2048)})


# --- B3 and B4 exact schedules ---------------------------------------------


@pytest.mark.parametrize("n_items", [2048, 4096, 8192])
def test_b3_plan_has_the_exact_counts_and_interleaving(n_items: int) -> None:
    rows = _fake_rows("cc", n_items)
    model_id = ext.EXT_MODEL_IDS[0]
    calls = list(ext.b3_calls(model_id, "civil_comments", rows, n_items))
    expected = ext.b3_expected_counts(n_items)
    assert len(calls) == expected["total"]
    counts: dict[str, int] = {}
    for call in calls:
        counts[str(call["condition"])] = counts.get(str(call["condition"]), 0) + 1
    assert counts["joint"] == expected["joint"] == n_items // 16 * 3
    assert counts["exact_context"] == expected["exact_context"] == n_items * 3
    assert counts["solo"] == expected["solo"] == n_items
    assert len({c["call_id"] for c in calls}) == expected["total"]
    assert all(str(c["call_id"]).startswith(f"{ext.CALL_PREFIX}::b3::") for c in calls)
    assert all(len(str(c["doc_order_sha256"])) == 64 for c in calls)
    assert all(len(str(c["target_ids_sha256"])) == 64 for c in calls)
    assert all(len(str(c["prompt_sha256"])) == 64 for c in calls)
    assert [c["position"] for c in calls] == list(range(1, expected["total"] + 1))
    # Every block serves all three conditions, and the lead condition rotates.
    leads: dict[int, str] = {}
    for call in calls:
        leads.setdefault(int(cast(int, call["execution_block"])), str(call["condition"]))
    assert len(set(leads.values())) == 3
    assert sorted(leads) == list(range(8))


def test_b3_plans_are_deterministic_and_cannot_collide_with_freeze_b() -> None:
    rows = _fake_rows("cc", 2048)
    first = ext.b3_plan_digest(ext.EXT_MODEL_IDS[0], "civil_comments", rows, 2048)
    assert first == ext.b3_plan_digest(ext.EXT_MODEL_IDS[0], "civil_comments", rows, 2048)
    assert first["assignment_namespace"] == ext.ASSIGN_NAMESPACE
    assert first["counts"]["total"] == 8576
    assert len(first["first_call_ids"]) == 3
    assert len(first["last_call_ids"]) == 3
    # A different candidate on the same frame gets a different plan.
    other = ext.b3_plan_digest(ext.EXT_MODEL_IDS[3], "civil_comments", rows, 2048)
    assert other["call_id_order_sha256"] != first["call_id_order_sha256"]
    # Freeze B call IDs never carry the extension prefix.
    freeze_b_shape = "e2::Qwen--Qwen3-8B::civil_comments::k16::a1::b0000::joint"
    assert not freeze_b_shape.startswith(ext.CALL_PREFIX)
    assert all(cid.startswith(ext.CALL_PREFIX) for cid in first["first_call_ids"])


def test_b3_refuses_unfrozen_sizes_and_short_frames() -> None:
    rows = _fake_rows("cc", 2048)
    with pytest.raises(ValueError):
        list(ext.b3_calls(ext.EXT_MODEL_IDS[0], "civil_comments", rows, 1024))
    with pytest.raises(ValueError):
        list(ext.b3_calls(ext.EXT_MODEL_IDS[0], "civil_comments", rows, 4096))
    with pytest.raises(ValueError):
        ext.b3_expected_counts(3000)


def test_b4_schedule_prefixes_are_nested_and_streams_are_disjoint() -> None:
    doc_ids = tuple(f"cc{i:06d}" for i in range(8192))
    schedule = ext.b4_schedule_digest(ext.EXT_MODEL_IDS[0], "civil_comments", doc_ids)
    assert schedule["reference"]["replicates"] == 250
    assert schedule["reference"]["n_calls"] == 250 * 32
    assert schedule["evaluation"]["n_calls"] == 2500 * 32
    assert schedule["evaluation"]["prefixes_are_nested"] is True
    prefixes = schedule["evaluation"]["prefixes"]
    assert sorted(int(k) for k in prefixes) == [400, 1000, 2500]
    assert prefixes["400"]["n_calls"] == 12_800
    assert prefixes["2500"]["n_calls"] == 80_000
    # Expanding a prefix never changes a call that already ran.
    assert prefixes["400"]["first_call_id"] == prefixes["2500"]["first_call_id"]
    assert all(
        str(row["call_id"]).startswith(f"{ext.CALL_PREFIX}::b4::")
        for row in schedule["reference"]["first_records"]
    )
    assert all(
        len(str(row["doc_order_sha256"])) == 64 for row in schedule["reference"]["first_records"]
    )
    assert schedule["repeated_namespace"] == ext.E5_NAMESPACE


def test_b4_streams_and_candidates_never_share_a_call_id() -> None:
    doc_ids = tuple(f"cc{i:06d}" for i in range(8192))
    reference = {
        str(c["call_id"])
        for c in ext.b4_calls(ext.EXT_MODEL_IDS[0], "civil_comments", doc_ids, "reference", 3)
    }
    evaluation = {
        str(c["call_id"])
        for c in ext.b4_calls(ext.EXT_MODEL_IDS[0], "civil_comments", doc_ids, "evaluation", 3)
    }
    other = {
        str(c["call_id"])
        for c in ext.b4_calls(ext.EXT_MODEL_IDS[1], "civil_comments", doc_ids, "reference", 3)
    }
    assert len(reference) == len(evaluation) == len(other) == 96
    assert not reference & evaluation
    assert not reference & other
    with pytest.raises(ValueError):
        list(ext.b4_calls(ext.EXT_MODEL_IDS[0], "imdb", doc_ids, "capacity", 1))


def test_extension_e5_draws_differ_from_freeze_b_for_the_same_cell() -> None:
    doc_ids = tuple(f"cc{i:06d}" for i in range(8192))
    freeze_b = repeated.replicate_calls(doc_ids, "Qwen/Qwen3-8B::civil_comments", "reference", 0)
    extension = repeated.replicate_calls(
        doc_ids,
        "Qwen/Qwen3-8B::civil_comments",
        "reference",
        0,
        namespace=ext.E5_NAMESPACE,
        call_id_prefix=f"{ext.CALL_PREFIX}::b4::",
    )
    assert [c["doc_order"] for c in freeze_b] != [c["doc_order"] for c in extension]
    assert not {c["call_id"] for c in freeze_b} & {c["call_id"] for c in extension}


def test_panels_put_one_cell_per_family_first_and_pick_one_e5_cell_each() -> None:
    cells = [
        Cell("Qwen/Qwen3-8B", "qwen", "civil_comments", 1, 1),
        Cell("Qwen/Qwen3-8B", "qwen", "imdb", 1, 3),
        Cell("ibm-granite/granite-3.3-8b-instruct", "granite", "imdb", 2, 3),
        Cell("mistralai/Mistral-7B-Instruct-v0.3", "mistral", "multi_nli", 4, 2),
    ]
    panel = ext.ext_panels(cells)
    assert len(panel["primary"]) == 4
    assert panel["families"] == ["granite", "mistral", "qwen"]
    assert len(panel["family_core"]) == 3
    assert len({key.split("::")[0] for key in panel["family_core"]}) == 3
    assert len(panel["e5_cells"]) == 3
    # Qwen's E5 cell is Civil Comments, its first admitted task in the frozen order.
    assert "Qwen/Qwen3-8B::civil_comments" in panel["e5_cells"]
    assert set(panel["e5_cells"]) <= set(panel["primary"])


def test_panel_refuses_more_than_twenty_cells() -> None:
    cells = [Cell(f"m{i}", f"f{i % 5}", ext.EXT_TASK_KEYS[i % 4], i, i % 4 + 1) for i in range(21)]
    with pytest.raises(ValueError):
        ext.ext_panels(cells)


def test_b3_b4_order_is_conditional_and_breadth_first() -> None:
    panel = {
        "primary": ["a::imdb", "b::imdb", "a::ag_news"],
        "family_core": ["a::imdb", "b::imdb"],
        "e5_cells": ["a::imdb", "b::imdb"],
    }
    stages = ext.b3_b4_order(panel)
    names = [s["stage"] for s in stages]
    assert names == [
        "M2_b3_family_core",
        "M3_b3_remaining",
        "M4_b4_reference",
        "M4_b4_evaluation",
        "M5_b4_prefixes",
    ]
    assert stages[0]["cells"] == ["a::imdb", "b::imdb"]
    assert stages[1]["cells"] == ["a::ag_news"]
    assert stages[2]["replicates"] == 250
    assert stages[3]["replicates"] == 400
    assert stages[4]["prefixes"] == [1000, 2500]
    assert all(s["condition"] for s in stages)


# --- Claim gates -----------------------------------------------------------


def _cell_data(cell: Cell, effect: float, *, seed: int, n_items: int = 256) -> CellData:
    """Build one synthetic cell whose joint errors carry a known excess covariance."""
    rng = np.random.default_rng(seed)
    frame = assign.Frame(
        tuple(f"{cell.key}:{i:04d}" for i in range(n_items)), tuple(range(n_items))
    )
    matrices = assign.observed_stat_matrices(frame, cell.key, 16, namespace=ext.ASSIGN_NAMESPACE)
    base = 0.30
    joint = np.zeros(n_items, dtype=float)
    if effect > 0:
        rho = effect / (base * (1 - base))
        concentration = 1 / rho - 1
        for call in matrices[0].T:
            rate = rng.beta(base * concentration, (1 - base) * concentration)
            joint[call] = (rng.random(call.size) < rate).astype(float)
    else:
        joint = (rng.random(n_items) < base).astype(float)
    exact = (rng.random(n_items) < base).astype(float)

    def replay(draw: int) -> list[np.ndarray]:
        return assign.replay_stat_matrices(
            frame, cell.key, 16, draw, namespace=ext.ASSIGN_NAMESPACE
        )

    return CellData(
        cell=cell, joint=[joint] * 3, exact=[exact] * 3, observed=matrices, replay=replay
    )


def test_c1_gate_runs_on_a_single_cell_family() -> None:
    cell = Cell("Qwen/Qwen3-8B", "qwen", "imdb", 1, 3)
    result = ext.c1_panel_gate([_cell_data(cell, 0.0, seed=1)], [cell], n_draws=99)
    assert result["n_cells"] == 1
    assert result["n_families"] == 1
    # One family can never reach the three-family recurrence gate.
    assert result["passed"] is False
    assert result["leave_one_family_out"] == {}


def test_c1_gate_weights_families_equally_not_cells() -> None:
    """Three Qwen cells at zero must not be outvoted by cell counting."""
    cells = [
        Cell("Qwen/Qwen3-8B", "qwen", task, 1, i + 1)
        for i, task in enumerate(ext.EXT_TASK_KEYS[:3])
    ] + [Cell("mistralai/Mistral-7B-Instruct-v0.3", "mistral", "imdb", 4, 3)]
    data = [
        _cell_data(c, 0.0 if c.family == "qwen" else 0.20, seed=10 + i) for i, c in enumerate(cells)
    ]
    result = ext.c1_panel_gate(data, cells, n_draws=99)
    cell_equal = result["global_cell_equal_delta_sensitivity"]
    family_equal = result["global_family_equal_delta"]
    # Mistral is one of four cells but one of two families, so it weighs more.
    assert family_equal > cell_equal
    assert result["n_families"] == 2
    assert set(result["leave_one_family_out"]) == {"qwen", "mistral"}


def test_c1_gate_needs_three_positive_families() -> None:
    families = ["qwen", "granite", "mistral"]
    cells = [Cell(f"model-{fam}", fam, "imdb", i + 1, 3) for i, fam in enumerate(families)]
    data = [_cell_data(c, 0.20, seed=30 + i) for i, c in enumerate(cells)]
    result = ext.c1_panel_gate(data, cells, n_draws=499)
    assert result["global_family_equal_delta"] > ext.PRACTICAL_FLOOR
    assert result["passed"] is True
    assert len(result["positive_families"]) == 3
    assert min(result["leave_one_family_out"].values()) > 0

    two = cells[:2]
    result_two = ext.c1_panel_gate([d for d in data[:2]], two, n_draws=499)
    assert result_two["passed"] is False


def test_c1_gate_refuses_an_incomplete_or_mismatched_panel() -> None:
    cells = [Cell(f"model-{i}", f"fam{i}", "imdb", i + 1, 3) for i in range(3)]
    data = [_cell_data(cells[0], 0.0, seed=5)]
    partial = ext.c1_panel_gate(data, cells, n_draws=99)
    assert partial["passed"] is False
    assert partial["reason"] == "incomplete_primary_panel"
    assert (partial["completed"], partial["required"]) == (1, 3)
    other = Cell("stranger", "stranger", "imdb", 9, 3)
    with pytest.raises(ValueError):
        ext.c1_panel_gate([_cell_data(other, 0.0, seed=6)], cells, n_draws=99)
    with pytest.raises(ValueError):
        ext.c1_panel_gate(data, [], n_draws=99)
    with pytest.raises(ValueError):
        ext.c1_panel_gate(data, cells, n_draws=10)


def _coverage(key: str, **kwargs: Any) -> dict[str, Any]:
    row = {
        "cell_key": key,
        "n_evaluation": 400,
        "n_covered": 340,
        "coverage": 0.85,
        "worst_reference_upper_coverage": 0.91,
        "reference_precise": True,
    }
    row.update(kwargs)
    return row


def test_c2_gate_needs_two_supported_cells() -> None:
    keys = ["a::imdb", "b::imdb", "c::imdb"]
    good = ext.c2_coverage_gate([_coverage(k) for k in keys], keys)
    assert good["passed"] is True
    assert len(good["supported_cells"]) == 3
    one = ext.c2_coverage_gate(
        [
            _coverage(keys[0]),
            _coverage(keys[1], coverage=0.94, n_covered=376),
            _coverage(keys[2], coverage=0.95, n_covered=380),
        ],
        keys,
    )
    assert one["passed"] is False
    assert one["supported_cells"] == [keys[0]]


def test_c2_gate_discards_imprecise_reference_and_short_streams() -> None:
    keys = ["a::imdb", "b::imdb"]
    result = ext.c2_coverage_gate(
        [
            _coverage(keys[0], reference_precise=False),
            _coverage(keys[1], n_evaluation=250, n_covered=210),
        ],
        keys,
    )
    assert result["passed"] is False
    assert result["supported_cells"] == []
    assert all(p == 1.0 for p in result["holm_adjusted_p"].values())


def test_c2_gate_refuses_unlocked_cells() -> None:
    with pytest.raises(ValueError):
        ext.c2_coverage_gate([_coverage("x::imdb")], ["a::imdb"])
    with pytest.raises(ValueError):
        ext.c2_coverage_gate([], [])
    with pytest.raises(ValueError):
        ext.c2_coverage_gate([_coverage("a::imdb")], ["a::imdb", "a::imdb"])


# --- Power and item counts -------------------------------------------------


def test_planning_alpha_matches_the_twenty_cell_family() -> None:
    assert ext.PLANNING_ALPHA == pytest.approx(0.05 / 20)
    row = simulate(0.35, 2048, sims=200, alpha=ext.PLANNING_ALPHA)
    assert row["alpha"] == ext.PLANNING_ALPHA
    assert row["n"] == 2048.0
    # A stricter alpha must not raise power at a fixed size.
    loose = simulate(0.35, 2048, sims=200, alpha=0.05 / 16)
    assert row["critical"] >= loose["critical"]


@pytest.mark.skipif(
    not (WORK / "power-x1.json").is_file(), reason="extension power table not built"
)
def test_item_counts_come_from_the_frozen_pooled_extension_table() -> None:
    payload = json.loads((WORK / "power-x1.json").read_text())
    rows = payload["rows"]
    assert len(rows) == 15
    assert payload["namespaces"] == list(ext.POWER_REPLICATE_NAMESPACES)
    # Every scenario pools three independent batches.
    assert all(row["sims"] == 6000.0 for row in rows)
    assert all(len(row["power_batches"]) == 3 for row in rows)
    low = choose_n(0.09, rows)
    high = choose_n(0.45, rows)
    assert low["n_items"] == 2048
    assert high["n_items"] == 8192
    assert high["power_target_met"] is True
    assert all(choose_n(p, rows)["n_items"] in (2048, 4096, 8192) for p in (0.06, 0.2, 0.3))


def test_pooled_power_table_removes_the_single_batch_flip() -> None:
    """One batch left the 0.35 grid rate on the target; pooling must settle it."""
    grid, sizes = (0.35,), (2048, 4096, 8192)
    rows = ext.ext_power_table(simulate, grid, sizes)
    assert len(rows) == 3
    for row in rows:
        assert row["sims"] == 6000.0
        assert row["power"] == pytest.approx(float(np.mean(row["power_batches"])))
        assert row["power_batch_min"] <= row["power"] <= row["power_batch_max"]
    assert ext.ext_power_table(simulate, grid, sizes) == rows


# --- Frames ----------------------------------------------------------------


@pytest.mark.skipif(not HAS_SEAL, reason="Freeze B part-1 seal not present")
def test_freeze_b_exclusion_set_covers_every_role() -> None:
    ids = ext.freeze_b_exclusions(ROOT, "imdb")
    assert len(ids) == 2048 + 8192 + 4096
    with pytest.raises(FileNotFoundError):
        ext.freeze_b_exclusions(ROOT, "not_a_task")


@pytest.mark.skipif(not (HAS_FRAMES and HAS_SEAL), reason="extension frames not built")
def test_extension_frames_are_disjoint_from_every_prior_frame() -> None:
    seen: dict[str, set[str]] = {}
    for task in ext.EXT_TASKS:
        prior = ext.ext_exclusions(ROOT, task)
        blocked = prior["historical"] | prior["freeze_b1"]
        roles = {
            role: {str(r["doc_id"]) for r in read_jsonl(WORK / task.key / f"{role}.jsonl")}
            for role in ext.FRAME_ROLES
        }
        assert [len(roles[r]) for r in ext.FRAME_ROLES] == [2048, 8192, 4096]
        combined = set().union(*roles.values())
        assert len(combined) == 14_336
        assert combined & blocked == set()
        seen[task.key] = combined
    for a in seen:
        for b in seen:
            if a != b:
                assert seen[a] & seen[b] == set()


@pytest.mark.skipif(not HAS_FRAMES, reason="extension frames not built")
def test_extension_frame_manifests_record_the_frozen_identity() -> None:
    for task in ext.EXT_TASKS:
        manifest = json.loads((WORK / task.key / "manifest.json").read_text())
        assert manifest["selection_seed"] == ext.EXT_SEED
        assert manifest["task"]["split"] == task.split
        assert manifest["task"]["revision"] == task.revision
        assert set(manifest["files"]) == {"screen.jsonl", "confirm.jsonl", "filler.jsonl"}


@pytest.mark.skipif(not HAS_FRAMES, reason="extension frames not built")
def test_confirmation_frame_carries_sixteen_equal_length_strata() -> None:
    rows = list(read_jsonl(WORK / "imdb/confirm.jsonl"))
    counts: dict[int, int] = {}
    for row in rows:
        counts[int(row["length_stratum"])] = counts.get(int(row["length_stratum"]), 0) + 1
    assert sorted(counts) == list(range(16))
    assert set(counts.values()) == {512}


# --- Admission lock --------------------------------------------------------


def _summary(model: str, task: str, *, admitted: bool, error_rate: float = 0.25) -> dict[str, Any]:
    return {
        "cell_key": f"{model}::{task}",
        "model_id": model,
        "task_id": task,
        "technical_complete": True,
        "admitted": admitted,
        "error_rate": error_rate,
        "status_counts": {"ok": 2048},
    }


def _all_summaries(admitted: set[str]) -> list[dict[str, Any]]:
    return [
        _summary(m, t, admitted=f"{m}::{t}" in admitted)
        for m in ext.EXT_MODEL_IDS
        for t in ext.EXT_TASK_KEYS
    ]


def _lock(**overrides: str) -> dict[str, Any]:
    return ext.runtime_access_lock(
        _access(**overrides),
        signed_by="daniel",
        runtime_lock_sha256="c" * 64,
        batch_invariance_report_sha256="d" * 64,
    )


@pytest.mark.skipif(
    not (WORK / "power-x1.json").is_file(), reason="extension power table not built"
)
def test_admission_lock_is_solo_only_and_discloses_opened_families() -> None:
    rows = json.loads((WORK / "power-x1.json").read_text())["rows"]
    admitted = {
        "Qwen/Qwen3-8B::civil_comments",
        "ibm-granite/granite-3.3-8b-instruct::imdb",
        "mistralai/Mistral-7B-Instruct-v0.3::multi_nli",
    }
    lock = ext.build_x1_admission_payload(
        _all_summaries(admitted), rows, _lock(), choose_n=choose_n
    )
    assert lock["extension_joint_outputs_seen"] is False
    assert lock["prior_opened_joint_families"] == ["granite", "qwen"]
    assert lock["selection_evidence"] == "solo_capability_and_technical_timing_only"
    assert set(lock["item_counts"]) == admitted
    assert lock["claim_scope"]["distinct_primary_model_families"] == 3
    assert lock["claim_scope"]["recurrence_scope"].startswith("narrowed_3_family")
    assert len(lock["ordered_cells"]["e5_cells"]) == 3
    assert [s["stage"] for s in lock["complete_block_queue"]][0] == "M2_b3_family_core"


@pytest.mark.skipif(
    not (WORK / "power-x1.json").is_file(), reason="extension power table not built"
)
def test_admission_lock_refuses_an_incomplete_screen() -> None:
    rows = json.loads((WORK / "power-x1.json").read_text())["rows"]
    with pytest.raises(ValueError):
        ext.build_x1_admission_payload(_all_summaries(set())[:-1], rows, _lock(), choose_n=choose_n)
    unfinished = _all_summaries(set())
    unfinished[0]["technical_complete"] = False
    with pytest.raises(ValueError):
        ext.build_x1_admission_payload(unfinished, rows, _lock(), choose_n=choose_n)
    # Dropping Llama's four cells must be refused, not silently accepted.
    without_llama = [
        row for row in _all_summaries(set()) if not str(row["model_id"]).startswith("meta-llama/")
    ]
    assert len(without_llama) == 16
    with pytest.raises(ValueError):
        ext.build_x1_admission_payload(without_llama, rows, _lock(), choose_n=choose_n)


@pytest.mark.skipif(
    not (WORK / "power-x1.json").is_file(), reason="extension power table not built"
)
def test_unavailable_candidate_still_carries_a_terminal_status() -> None:
    rows = json.loads((WORK / "power-x1.json").read_text())["rows"]
    llama = "meta-llama/Llama-3.1-8B-Instruct"
    summaries = _all_summaries({"Qwen/Qwen3-8B::imdb"})
    for row in summaries:
        if str(row["model_id"]) == llama:
            row["technical_complete"] = False
            row["admitted"] = False
    lock = ext.build_x1_admission_payload(
        summaries, rows, _lock(**{llama: "gated_pending_access"}), choose_n=choose_n
    )
    assert lock["unavailable_models"] == [llama]
    assert set(lock["candidate_status"]) == set(ext.EXT_MODEL_IDS)
    assert lock["candidate_status"][llama]["candidate_status"] == "technically_unavailable"
    assert len(lock["candidate_status"][llama]["cells"]) == 4
    assert len(lock["attempted_candidates"]) == 20
    assert len(lock["runtime_access_lock_sha256"]) == 64
    # An unavailable candidate can never have an admitted cell.
    bad = _all_summaries({f"{llama}::imdb"})
    with pytest.raises(ValueError):
        ext.build_x1_admission_payload(
            bad, rows, _lock(**{llama: "gated_pending_access"}), choose_n=choose_n
        )


def test_two_admitted_families_cannot_claim_recurrence() -> None:
    rows = [simulate(0.35, n, sims=200, alpha=ext.PLANNING_ALPHA) for n in (2048, 4096, 8192)]
    admitted = {"Qwen/Qwen3-8B::imdb", "ibm-granite/granite-3.3-8b-instruct::imdb"}
    lock = ext.build_x1_admission_payload(
        _all_summaries(admitted), rows, _lock(), choose_n=choose_n
    )
    assert lock["claim_scope"]["recurrence_scope"] == "cross_family_recurrence_claim_not_supported"


# --- E5 end-to-end analysis ------------------------------------------------


def _e5_errors(n_replicates: int, rate: float, intra_call: float, seed: int) -> np.ndarray:
    """(replicates, 32 calls, 16 appearances) of 0/1 errors with call clustering."""
    rng = np.random.default_rng(seed)
    shape = (n_replicates, repeated.CALLS_PER_REPLICATE)
    if intra_call <= 0:
        rates = np.full((*shape, 1), rate)
    else:
        concentration = 1 / intra_call - 1
        rates = rng.beta(rate * concentration, (1 - rate) * concentration, size=shape)[..., None]
    return (rng.random((*shape, repeated.K)) < rates).astype(float)


def test_e5_analysis_composes_every_frozen_calculation() -> None:
    reference = _e5_errors(250, 0.30, 0.0, seed=1)
    evaluation = _e5_errors(400, 0.30, 0.0, seed=2)
    result = ext.ext_e5_analyze("m::imdb", reference, evaluation, n_frame=8192)
    # The top level is exactly the row the C2 gate reads.
    for key in (
        "cell_key",
        "n_evaluation",
        "n_covered",
        "coverage",
        "worst_reference_upper_coverage",
        "reference_precise",
    ):
        assert key in result
    assert result["n_evaluation"] == 400
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["n_covered"] == pytest.approx(result["coverage"] * 400, abs=1.0)
    intervals = result["intervals"]
    assert set(intervals) >= {
        "iid_wald_coverage",
        "call_aware_t_coverage",
        "whole_call_bootstrap_coverage",
        "mean_width_iid_wald",
        "mean_width_call_aware_t",
        "mean_width_whole_call_bootstrap",
    }
    # Clopper-Pearson bounds bracket the point estimate.
    cp = intervals["iid_wald_coverage"]
    assert cp["low"] <= cp["point"] <= cp["high"]
    assert set(result["coverage_with_reference_uncertainty"]["variants"]) == {
        "at_target",
        "target_minus_z_se",
        "target_plus_z_se",
    }
    assert result["variance_ratio"]["n_replicates"] == 400
    fpc = result["finite_population_correction"]
    assert 0.0 < fpc["factor"] <= 1.0
    assert fpc["applied_to_iid_wald"] is False
    assert fpc["applied_to_call_aware_t"] is False
    assert result["reference"]["n_replicates"] == 250
    assert result == ext.ext_e5_analyze("m::imdb", reference, evaluation, n_frame=8192)


def test_e5_analysis_detects_call_clustering_as_undercoverage() -> None:
    """Independent errors must cover near nominal; clustered errors must not."""
    reference_iid = _e5_errors(250, 0.30, 0.0, seed=11)
    independent = ext.ext_e5_analyze(
        "m::imdb", reference_iid, _e5_errors(400, 0.30, 0.0, seed=12), n_frame=8192
    )
    reference_dep = _e5_errors(250, 0.30, 0.20, seed=13)
    clustered = ext.ext_e5_analyze(
        "m::imdb", reference_dep, _e5_errors(400, 0.30, 0.20, seed=14), n_frame=8192
    )
    assert independent["coverage"] > clustered["coverage"]
    assert clustered["variance_ratio"]["point"] > independent["variance_ratio"]["point"]
    # The call-aware interval is wider than the i.i.d. one under clustering.
    assert (
        clustered["intervals"]["mean_width_call_aware_t"]
        > clustered["intervals"]["mean_width_iid_wald"]
    )


def test_e5_analysis_feeds_the_c2_gate_directly() -> None:
    rows = [
        ext.ext_e5_analyze(
            f"m{i}::imdb",
            _e5_errors(250, 0.30, 0.20, seed=20 + i),
            _e5_errors(400, 0.30, 0.20, seed=40 + i),
            n_frame=8192,
        )
        for i in range(3)
    ]
    keys = [str(row["cell_key"]) for row in rows]
    gate = ext.c2_coverage_gate([dict(row) for row in rows], keys)
    assert set(gate["holm_adjusted_p"]) == set(keys)
    assert isinstance(gate["passed"], bool)


def test_e5_analysis_refuses_malformed_streams() -> None:
    good = _e5_errors(250, 0.3, 0.0, seed=5)
    with pytest.raises(ValueError):
        ext.ext_e5_analyze("m::imdb", good, good[:, :31], n_frame=8192)
    with pytest.raises(ValueError):
        ext.ext_e5_analyze("m::imdb", good, good[:, :, :15], n_frame=8192)
    with pytest.raises(ValueError):
        ext.ext_e5_analyze("m::imdb", good[:1], good, n_frame=8192)
    with pytest.raises(ValueError):
        ext.ext_e5_analyze("m::imdb", good, good, n_frame=8)


def test_every_frozen_e5_primitive_exists_in_step0_stats() -> None:
    for name in (
        "wald_interval",
        "call_t_interval",
        "whole_call_bootstrap",
        "clopper_pearson",
        "coverage_with_reference_uncertainty",
        "variance_ratio",
        "reference_precision_check",
        "fpc",
    ):
        assert callable(getattr(stats, name)), name


# --- C1 gate against the written claim text ---------------------------------


def test_c1_gate_matches_every_clause_of_the_written_claim() -> None:
    """Each clause of the C1 text must be able to fail the gate on its own."""
    families = ["qwen", "granite", "mistral"]
    cells = [Cell(f"model-{f}", f, "imdb", i + 1, 3) for i, f in enumerate(families)]
    data = [_cell_data(c, 0.20, seed=70 + i) for i, c in enumerate(cells)]
    passing = ext.c1_panel_gate(data, cells, n_draws=499)
    assert passing["passed"] is True

    # Clause 1: the family-equal global contrast must be above zero.
    assert passing["global_family_equal_delta"] > 0
    # Clause 2: the global replay p must be at most 0.05.
    assert passing["global_p"] <= ext.ALPHA
    # Clause 3: three families each need a Holm-positive cell at the 0.002 floor.
    assert len(passing["positive_families"]) >= ext.MIN_POSITIVE_FAMILIES
    assert all(
        passing["cell_delta"][key] >= ext.PRACTICAL_FLOOR for key in passing["positive_cells"]
    )
    assert all(passing["holm_adjusted_p"][key] <= ext.ALPHA for key in passing["positive_cells"])
    # Clause 4: every leave-one-family-out family-equal mean must be above zero.
    assert min(passing["leave_one_family_out"].values()) > 0
    assert set(passing["leave_one_family_out"]) == set(families)
    # The cell-equal mean is reported but never gates.
    assert "global_cell_equal_delta_sensitivity" in passing
    # Two positive families cannot pass, whatever the global contrast says.
    two = ext.c1_panel_gate(data[:2], cells[:2], n_draws=499)
    assert two["passed"] is False
    assert len(two["positive_families"]) < ext.MIN_POSITIVE_FAMILIES


def test_c1_null_cells_do_not_pass_the_gate() -> None:
    families = ["qwen", "granite", "mistral"]
    cells = [Cell(f"model-{f}", f, "imdb", i + 1, 3) for i, f in enumerate(families)]
    data = [_cell_data(c, 0.0, seed=80 + i) for i, c in enumerate(cells)]
    result = ext.c1_panel_gate(data, cells, n_draws=499)
    assert result["passed"] is False
    assert result["positive_families"] == []


def test_family_equal_mean_refuses_an_empty_panel() -> None:
    with pytest.raises(ValueError):
        ext._family_equal_mean({}, {})


def test_conditional_order_matches_the_written_milestone_rules() -> None:
    cells = [
        Cell("Qwen/Qwen3-8B", "qwen", "civil_comments", 1, 1),
        Cell("Qwen/Qwen3-8B", "qwen", "imdb", 1, 3),
        Cell("ibm-granite/granite-3.3-8b-instruct", "granite", "imdb", 2, 3),
        Cell("mistralai/Mistral-7B-Instruct-v0.3", "mistral", "multi_nli", 4, 2),
    ]
    panel = ext.ext_panels(cells)
    stages = ext.b3_b4_order(panel)
    core = set(panel["family_core"])
    rest = set(stages[1]["cells"])
    # The core and the remainder partition the primary panel exactly once.
    assert core | rest == set(panel["primary"])
    assert not core & rest
    # Every family appears exactly once in the core.
    assert len(core) == len(panel["families"])
    # B4 runs reference before evaluation, and prefixes come last.
    assert stages[2]["stage"].endswith("reference")
    assert stages[3]["stage"].endswith("evaluation")
    assert stages[4]["prefixes"] == list(ext.E5_EVALUATION_PREFIXES[1:])
    # A B4 cell is always an admitted primary cell of its own family.
    for key in panel["e5_cells"]:
        assert key in panel["primary"]


# --- Disclosure ------------------------------------------------------------


def test_disclosure_is_stable_and_carries_the_budget_and_seeds() -> None:
    record = ext.disclosure()
    assert record == ext.disclosure()
    assert record["separate_from_freeze_b"] is True
    assert record["budget"]["stop_a100_hours"] == 80.0
    assert record["budget"]["reserve_a100_hours"] == 10.0
    assert record["budget"]["shares_freeze_b_allowance"] is False
    assert record["seeds"]["assignment_namespace"] == "bigdata2026-ext-v1"
    assert record["statistics"]["primary_contrast"].startswith("family_equal")
    assert len(record["identity_sha256"]) == 64


def test_prompts_and_parser_have_no_model_branch() -> None:
    """One rendered prompt per task and target set, whatever the model."""
    from freezeb.prompts import build_messages

    docs = [(f"d{i}", f"document number {i}") for i in range(16)]
    for task in ext.EXT_TASK_KEYS:
        joint = build_messages(task, docs, [d for d, _ in docs])
        exact = build_messages(task, docs, ["d7"])
        assert joint[0]["content"] == exact[0]["content"]
        # The exact control shows identical documents in identical order.
        for doc_id, text in docs:
            assert f'<doc id="{doc_id}">\n{text}\n</doc>' in exact[1]["content"]
        assert 'Return only the label for document ID "d7"' in exact[1]["content"]
        assert build_messages(task, docs, ["d7"]) == exact


def test_screen_reforecast_scales_to_the_four_task_extension() -> None:
    events = [
        {
            "status": "ok",
            "attempt_id": f"a{i}",
            "metered_a100_seconds": 1.0,
            "prompt_tokens": 10,
            "output_tokens": 5,
            "worker_session_id": "w1",
            "worker_load_seconds": 100.0,
            "timing_probe_method": "probe",
            "probe_prompt_plus_first_token_seconds": 0.1,
            "probe_remaining_generation_seconds": 0.2,
        }
        for i in range(1000)
    ]
    four = screen.technical_reforecast("Qwen/Qwen3-8B", events, 1.0, n_tasks=4)
    six = screen.technical_reforecast("Qwen/Qwen3-8B", events, 1.0)
    assert four is not None and six is not None
    assert (
        four["projected_model_screen_base_a100_hours"]
        < six["projected_model_screen_base_a100_hours"]
    )
    assert four["projected_model_screen_guarded_a100_hours"] == pytest.approx(
        four["projected_model_screen_base_a100_hours"] * 1.6
    )
