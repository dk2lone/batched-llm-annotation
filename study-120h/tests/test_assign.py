"""Invariant and property tests for the deterministic assignment generator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from step0 import assign
from step0.lock import PANEL


def make_frame(n_items: int = 512, n_strata: int = 16) -> assign.Frame:
    per = n_items // n_strata
    return assign.Frame(
        doc_ids=tuple(f"t{i:05d}" for i in range(n_items)),
        strata=tuple(i // per for i in range(n_items)),
    )


def make_filler(n: int = 256) -> list[tuple[str, int]]:
    """A stand-in filler pool: stable ids and a wide spread of lengths."""
    return [(f"f{i:05d}", 40 + (i * 37) % 900) for i in range(n)]


def chars_by_doc(frame: assign.Frame) -> dict[str, int]:
    return {doc: 60 + (i * 13) % 400 for i, doc in enumerate(frame.doc_ids)}



def test_frame_rejects_unequal_strata() -> None:
    with pytest.raises(ValueError, match="same number of items"):
        assign.Frame(doc_ids=("a", "b", "c"), strata=(0, 0, 1))


def test_every_item_appears_once_per_assignment() -> None:
    frame = make_frame()
    calls = assign.build_cell(frame, "cell", 3)
    for assignment in (1, 2, 3):
        docs = [d for c in calls if c["assignment"] == assignment for d in c["doc_order"]]
        assert len(docs) == frame.n_items
        assert len(set(docs)) == frame.n_items


def test_each_call_holds_one_item_per_length_stratum() -> None:
    calls = assign.build_cell(make_frame(), "cell", 3)
    for call in calls:
        assert sorted(call["length_strata"]) == list(range(16))
        assert len(call["doc_order"]) == 16


def test_each_item_uses_distinct_slots_across_assignments() -> None:
    frame = make_frame()
    calls = assign.build_cell(frame, "cell", 3)
    slots: dict[str, list[int]] = {d: [] for d in frame.doc_ids}
    for call in calls:
        for slot, doc in enumerate(call["doc_order"]):
            slots[doc].append(slot)
    assert all(len(v) == 3 for v in slots.values())
    assert all(len(set(v)) == 3 for v in slots.values())


def test_slot_usage_is_uniform_within_each_assignment() -> None:
    frame = make_frame()
    calls = assign.build_cell(frame, "cell", 3)
    for assignment in (1, 2, 3):
        counts = np.zeros(16, dtype=int)
        for call in calls:
            if call["assignment"] == assignment:
                counts[: len(call["doc_order"])] += 1
        assert counts.tolist() == [frame.n_calls] * 16


def test_generator_is_deterministic() -> None:
    frame = make_frame()
    first = assign.build_cell(frame, "cell", 3)
    second = assign.build_cell(frame, "cell", 3)
    assert first == second


def test_different_cells_get_different_partitions() -> None:
    frame = make_frame()
    a = assign.build_cell(frame, "cell_a", 1)
    b = assign.build_cell(frame, "cell_b", 1)
    assert a != b


def test_assignments_differ_from_one_another() -> None:
    frame = make_frame()
    calls = assign.build_cell(frame, "cell", 3)
    first = {tuple(sorted(c["doc_order"])) for c in calls if c["assignment"] == 1}
    second = {tuple(sorted(c["doc_order"])) for c in calls if c["assignment"] == 2}
    assert first != second


def test_replay_is_reproducible_and_differs_from_the_observed_draw() -> None:
    frame = make_frame()
    observed = assign.observed_partition(frame, "cell", 1)
    draw_a = assign.replay_partition(frame, "cell", 1, 7)
    draw_b = assign.replay_partition(frame, "cell", 1, 7)
    other = assign.replay_partition(frame, "cell", 1, 8)
    assert np.array_equal(draw_a, draw_b)
    assert not np.array_equal(draw_a, observed)
    assert not np.array_equal(draw_a, other)


def test_replay_keeps_every_item_exactly_once_per_stratum() -> None:
    frame = make_frame()
    observed = assign.observed_partition(frame, "cell", 2)
    draw = assign.replay_partition(frame, "cell", 2, 3)
    assert draw.shape == (frame.n_calls, 16)
    assert sorted(draw.ravel().tolist()) == list(range(frame.n_items))
    for stratum in range(16):
        column = draw[:, stratum]
        assert {frame.strata[i] for i in column.tolist()} == {stratum}
        for execution_block in range(assign.N_BLOCKS):
            calls = np.arange(execution_block, frame.n_calls, assign.N_BLOCKS)
            assert set(draw[calls, stratum]) == set(observed[calls, stratum])


def test_replay_rejects_a_negative_draw_index() -> None:
    with pytest.raises(ValueError, match="0 or greater"):
        assign.replay_partition(make_frame(), "cell", 1, -1)


def test_balance_report_confirms_the_invariants() -> None:
    frame = make_frame()
    calls = assign.build_cell(frame, "cell", 3)
    references = [i % 2 for i in range(frame.n_items)]
    report = assign.balance_report(frame, calls, references)
    assert report["appearances_per_item"]["all_equal"]
    assert report["distinct_slots_per_item"]["all_distinct"]
    assert report["slot_usage_per_assignment"]["uniform"]
    assert report["one_item_per_stratum_per_call"]
    assert report["length_stratum_sum_per_call"]["constant_by_construction"]


def test_job_skeletons_have_unique_ids_and_the_planned_counts() -> None:
    frame = make_frame(n_items=2048)
    cell = next(c for c in PANEL["cells"] if c["cell_id"] == "qwen_civil")
    calls = assign.build_cell(frame, str(cell["cell_id"]), 3)
    jobs = assign.job_skeletons(cell, calls, chars_by_doc(frame), make_filler(256))
    counts: dict[str, int] = {}
    for job in jobs:
        counts[str(job["condition"])] = counts.get(str(job["condition"]), 0) + 1
    assert counts["joint"] == 384
    assert counts["exact_context"] == 6144
    assert counts["solo"] == 2048
    assert counts["filler"] == 6144
    assert len({j["call_id"] for j in jobs}) == len(jobs)
    assert len({j["execution_order_key"] for j in jobs}) == len(jobs)
    assert all(len(j["execution_order_key"]) == 64 for j in jobs)
    assert all(j["parent_call_id"] is None and j["retry_index"] == 0 for j in jobs)


def test_double_new_cell_gets_the_small_filler_and_serial_subsets() -> None:
    frame = make_frame(n_items=2048)
    cell = next(c for c in PANEL["cells"] if c["cell_id"] == "mistral_secondtask")
    jobs = assign.job_skeletons(
        cell,
        assign.build_cell(frame, str(cell["cell_id"]), 3),
        chars_by_doc(frame),
        make_filler(256),
    )
    counts: dict[str, int] = {}
    for job in jobs:
        counts[str(job["condition"])] = counts.get(str(job["condition"]), 0) + 1
    assert counts["filler"] == 512
    assert counts["serial_joint"] == 32
    assert counts["serial_exact"] == 512
    assert all(j["serving_concurrency"] == 1 for j in jobs if j["condition"].startswith("serial"))


def test_cells_without_a_filler_spec_emit_no_filler() -> None:
    frame = make_frame(n_items=2048)
    cell = next(c for c in PANEL["cells"] if c["cell_id"] == "mistral_civil")
    jobs = assign.job_skeletons(
        cell,
        assign.build_cell(frame, str(cell["cell_id"]), 3),
        chars_by_doc(frame),
        make_filler(256),
    )
    assert not [j for j in jobs if j["condition"] in ("filler", "serial_joint", "serial_exact")]


def test_execution_blocks_cover_every_arm() -> None:
    frame = make_frame(n_items=2048)
    cell = next(c for c in PANEL["cells"] if c["cell_id"] == "qwen_civil")
    jobs = assign.job_skeletons(
        cell,
        assign.build_cell(frame, str(cell["cell_id"]), 3),
        chars_by_doc(frame),
        make_filler(256),
    )
    for condition in ("joint", "exact_context", "filler"):
        blocks = {j["execution_block"] for j in jobs if j["condition"] == condition}
        assert blocks == set(range(8))






def test_replay_stratum_matrices_keep_the_call_and_block_structure() -> None:
    frame = make_frame()
    observed = assign.observed_stratum_matrices(frame, "cell", 3)
    replayed = assign.replay_stratum_matrices(frame, "cell", 3, 5)
    assert len(observed) == len(replayed) == 3
    for a in range(3):
        assert observed[a].shape == replayed[a].shape == (16, frame.n_calls)
        assert sorted(replayed[a].ravel().tolist()) == list(range(frame.n_items))
        for stratum in range(16):
            assert {frame.strata[i] for i in replayed[a][stratum].tolist()} == {stratum}
            for execution_block in range(assign.N_BLOCKS):
                calls = np.arange(execution_block, frame.n_calls, assign.N_BLOCKS)
                assert set(replayed[a][stratum, calls]) == set(observed[a][stratum, calls])
    assert not np.array_equal(observed[0], replayed[0])
    again = assign.replay_stratum_matrices(frame, "cell", 3, 5)
    assert all(np.array_equal(x, y) for x, y in zip(replayed, again, strict=True))


def test_choose_filler_lands_near_the_requested_total_length() -> None:
    pool = [n for _, n in make_filler(400)]
    rng = assign.rng_for("test", "filler")
    picks = assign.choose_filler(pool, 4500, 15, rng, 64)
    assert len(picks) == len(set(picks)) == 15
    total = sum(pool[i] for i in picks)
    assert abs(total - 4500) / 4500 < 0.05


def test_choose_filler_refuses_a_pool_smaller_than_the_request() -> None:
    with pytest.raises(ValueError, match="filler pool holds"):
        assign.choose_filler([100, 200], 500, 15, assign.rng_for("x"), 64)


def _jobs_for(cell_id: str) -> tuple[assign.Frame, list[dict[str, Any]]]:
    frame = make_frame()
    calls = assign.build_cell(frame, cell_id, 3)
    cell = next(c for c in PANEL["cells"] if c["cell_id"] == cell_id)
    return frame, assign.job_skeletons(cell, calls, chars_by_doc(frame), make_filler(256))




def test_a_filler_call_shows_the_target_plus_fifteen_length_matched_peers() -> None:
    _, jobs = _jobs_for("qwen_civil")
    pool_ids = {doc for doc, _ in make_filler(256)}
    for job in [j for j in jobs if j["condition"] == "filler"][:50]:
        docs = job["doc_order"]
        target = job["target_ids"][0]
        assert len(docs) == 16
        assert len(set(docs)) == 16
        assert docs[job["target_slots"][0]] == target
        assert set(docs) - {target} <= pool_ids
        gap = abs(job["match_chars_actual"] - job["match_chars_target"])
        assert gap / job["match_chars_target"] < 0.2
