"""Freeze B parsers and selection rules, checked without experimental labels."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from freezeb.admission import evaluate
from freezeb.assign import (
    Frame,
    build_calls,
    observed_stat_matrices,
    partition,
    replay,
    replay_stat_matrices,
)
from freezeb.claims import CellData, cell_delta, e2_panel_gate, e5_coverage_gate
from freezeb.frames import N_CONFIRM, N_FILLER, N_SCREEN, TASKS, select_frames, source_record
from freezeb.labels import ParseError, disagreement, entity_spans, parse_first_response
from freezeb.mechanism import build_arms, reference_group, select_items
from freezeb.meter import may_start, must_stop
from freezeb.power import ERROR_GRID, SIZES, _covariance, choose_n
from freezeb.prompts import build_messages
from freezeb.repeated import prefix_calls, replicate_calls
from freezeb.selection import Cell, balanced_order, panels
from step0.stats import covariance_from_stratified, randomization_test


@pytest.mark.parametrize(
    ("task", "label", "expected"),
    [
        ("civil_comments", 1, 1),
        ("ag_news", 3, 3),
        ("multi_nli", 2, 2),
        ("imdb", 0, 0),
        ("go_emotions", [3, 1], (1, 3)),
        ("multinerd", [0, 1, 2, 0], ((1, 3, "PER"),)),
    ],
)
def test_parse_first_response(task: str, label: object, expected: object) -> None:
    raw = json.dumps({"labels": {"x": label}})
    assert parse_first_response(raw, ["x"], task, token_counts={"x": 4}) == {"x": expected}


@pytest.mark.parametrize(
    ("task", "label"),
    [
        ("civil_comments", True),
        ("ag_news", 4),
        ("go_emotions", [2, 2]),
        ("multinerd", [0, 1]),
        ("multinerd", [0, 31, 0, 0]),
    ],
)
def test_bad_label_is_unresolved(task: str, label: object) -> None:
    with pytest.raises(ParseError):
        parse_first_response(
            json.dumps({"labels": {"x": label}}), ["x"], task, token_counts={"x": 4}
        )


def test_exact_ids_and_single_json_object() -> None:
    for raw in ('{"labels":{"x":0,"y":1}}', '{"labels":{"x":0}} trailing',
                '{"labels":{"x":0},"note":"ok"}',
                '{"labels":{"x":0,"x":1}}'):
        with pytest.raises(ParseError):
            parse_first_response(raw, ["x", "z"], "civil_comments")


def test_entity_spans_and_reference_disagreement() -> None:
    assert entity_spans([2, 2, 0, 4, 6, 0]) == ((0, 2, "PER"), (3, 4, "ORG"), (4, 5, "LOC"))
    assert disagreement([0, 1, 2, 0], [0, 1, 2, 0], "multinerd", n_tokens=4) == 0
    assert disagreement([0, 1, 0, 0], [0, 1, 2, 0], "multinerd", n_tokens=4) == 1
    assert disagreement([1, 3], [3, 1], "go_emotions") == 0
    assert disagreement([27, 1], [1, 27], "go_emotions") == 0


def test_balanced_selection_is_deterministic_and_distinct_per_round() -> None:
    cells = [
        Cell(f"m{m}", f"f{m}", f"t{t}", m, t)
        for m in range(4)
        for t in range(6)
    ]
    ordered = balanced_order(cells)
    assert ordered == balanced_order(list(reversed(cells)))
    assert len(ordered) == 24
    for start in range(0, 16, 4):
        round_cells = ordered[start:start + 4]
        assert len({c.family for c in round_cells}) == 4
        assert len({c.task for c in round_cells}) == 4
    assert len(panels(cells)["primary"]) == 16
    assert len(panels(cells)["held_out"]) == 8


def test_balanced_selection_maximizes_round_cardinality() -> None:
    cells = [
        Cell("a", "a", "x", 0, 0),
        Cell("a", "a", "y", 0, 1),
        Cell("b", "b", "x", 1, 0),
    ]
    assert {c.key for c in balanced_order(cells)[:2]} == {"a::y", "b::x"}
    with pytest.raises(ValueError, match="duplicate"):
        balanced_order([cells[0], cells[0]])


def test_frame_selection_is_deterministic_and_disjoint() -> None:
    rows = [
        {"doc_id": f"x{i:05d}", "n_chars": i % 271, "source_index": i}
        for i in range(N_SCREEN + N_CONFIRM + N_FILLER + 20)
    ]
    excluded = {"x00000", "x00001"}
    first = select_frames(rows, excluded)
    second = select_frames(list(reversed(rows)), excluded)
    assert first == second
    ids = [row["doc_id"] for frame in first.values() for row in frame]
    assert len(ids) == len(set(ids)) == N_SCREEN + N_CONFIRM + N_FILLER
    assert not excluded.intersection(ids)
    assert {s: sum(r["length_stratum"] == s for r in first["confirm"]) for s in range(16)} == {
        s: N_CONFIRM // 16 for s in range(16)
    }


def test_multinerd_english_filter_and_reference_spans() -> None:
    task = next(task for task in TASKS if task.key == "multinerd")
    row = {"tokens": ["Alice", "from", "Berlin", "today"],
           "ner_tags": [1, 0, 5, 0], "lang": "de"}
    assert source_record(task, row, 0) is None
    row["lang"] = "en"
    normalized = source_record(task, row, 1)
    assert normalized is not None
    assert normalized["reference_normalized"] == ((0, 1, "PER"), (2, 3, "LOC"))


def test_joint_and_exact_prompts_share_document_context() -> None:
    docs = [("x", "Alice in Berlin"), ("y", "Bob in Paris")]
    joint = build_messages("multinerd", docs, ["x", "y"])
    exact = build_messages("multinerd", docs, ["x"])
    for doc_id, text in docs:
        block = f'<doc id="{doc_id}">\n{text}\n</doc>'
        assert block in joint[1]["content"]
        assert block in exact[1]["content"]
    raw_example = joint[1]["content"].split("Answer with this JSON object and nothing else:\n")[1]
    assert parse_first_response(raw_example, ["x", "y"], "multinerd",
                                token_counts={"x": 3, "y": 3}) == {"x": (), "y": ()}


def _order(call: dict[str, object]) -> list[str]:
    doc_order = call["doc_order"]
    assert isinstance(doc_order, list)
    return [str(doc_id) for doc_id in doc_order]


@pytest.mark.parametrize("k", [2, 4, 8, 16, 32])
def test_assignment_and_replay_preserve_items_strata_and_blocks(k: int) -> None:
    frame = Frame(tuple(f"x{i:03d}" for i in range(128)), tuple(i % 73 for i in range(128)))
    calls = build_calls(frame, "m::task", k)
    assert len(calls) == 3 * 128 // k
    for assignment in (1, 2, 3):
        observed = partition(frame, "m::task", k, assignment)
        regrouped = replay(frame, "m::task", k, assignment, 0)
        assert set(observed.ravel()) == set(regrouped.ravel()) == set(range(128))
        for stratum in range(k):
            for block in range(8):
                assert set(observed[block::8, stratum]) == set(regrouped[block::8, stratum])
        one_assignment = [c for c in calls if c["assignment"] == assignment]
        assert len({doc for c in one_assignment for doc in _order(c)}) == 128
    slots_by_id: dict[str, list[int]] = {}
    for call in calls:
        for slot, doc_id in enumerate(_order(call)):
            slots_by_id.setdefault(doc_id, []).append(slot)
    assert all(len(slots) == 3 for slots in slots_by_id.values())
    assert all(max(slots.count(slot) for slot in set(slots)) <= (2 if k == 2 else 1)
               for slots in slots_by_id.values())


def test_replay_statistic_accepts_freezeb_assignments() -> None:
    frame = Frame(tuple(f"x{i:03d}" for i in range(128)), tuple(i % 97 for i in range(128)))
    observed = observed_stat_matrices(frame, "m::task", 16)
    assert all(m.shape == (16, 8) for m in observed)
    rng = np.random.default_rng(26)
    joint = [rng.integers(0, 2, 128).astype(float) for _ in observed]
    exact = [rng.integers(0, 2, 128).astype(float) for _ in observed]
    direct = covariance_from_stratified(joint[0][partition(frame, "m::task", 16, 1)])
    adapted = covariance_from_stratified(joint[0][observed[0]].T)
    assert direct == pytest.approx(adapted)
    result = randomization_test(
        joint, exact, observed, 99,
        lambda draw: replay_stat_matrices(frame, "m::task", 16, draw),
    )
    assert 0 <= result["p_value_one_sided_upper"] <= 1
    assert result["n_draws"] == 99


def test_power_size_rule_and_covariance_closed_form() -> None:
    calls = np.zeros((1, 3, 2, 16), dtype=np.float32)
    calls[:, :, :, :8] = 1
    assert _covariance(calls)[0] < 0
    rows = [{"p": p, "n": float(n), "sims": 800.0,
             "power": 0.9 if n == 4096 else 0.5,
             "power_mc_se": 0.01}
            for p in ERROR_GRID for n in SIZES]
    assert choose_n(0.12, rows)["n_items"] == 4096
    assert choose_n(0.01, rows)["information_poor"] is True


def test_repeated_streams_have_distinct_call_ids_and_unique_items_per_replicate() -> None:
    ids = tuple(f"x{i:04d}" for i in range(8192))
    reference = replicate_calls(ids, "m::task", "reference", 0)
    evaluation = replicate_calls(ids, "m::task", "evaluation", 0)
    assert reference == replicate_calls(ids, "m::task", "reference", 0)
    assert len(reference) == len(evaluation) == 32
    assert len({d for c in reference for d in _order(c)}) == 512
    assert not {c["call_id"] for c in reference} & {c["call_id"] for c in evaluation}
    assert prefix_calls(6, 5000) == 1_008_000


def test_meter_uses_cumulative_guarded_hours() -> None:
    assert may_start(100, 20).start is True
    assert may_start(110, 20).reason == "guarded_complete_block_exceeds_stop"
    assert may_start(120, 1).reason == "target_reached_after_complete_block"
    assert must_stop(149.9) is False
    assert must_stop(150) is True


def test_civil_admission_rejects_majority_class_shortcut() -> None:
    refs = [0] * 93 + [1] * 7
    result = evaluate("civil_comments", [0] * 100, refs)
    assert result["accuracy"] == 0.93
    assert result["admitted"] is False
    assert "balanced_accuracy_below_0.60" in result["reasons"]
    assert "toxic_recall_below_0.25" in result["reasons"]


def test_multilabel_and_span_admission_use_task_metrics() -> None:
    emotions: list[Any] = [(27,), (0,), (1,), (2,)] * 25
    good_emotions = evaluate("go_emotions", emotions, emotions)
    assert good_emotions["admitted"] is False  # no error information
    poor_emotions = evaluate("go_emotions", [(27,)] * 100, emotions)
    assert poor_emotions["admitted"] is False
    spans = [((0, 1, "PER"),), ((1, 2, "LOC"),)] * 50
    poor_spans = evaluate("multinerd", [()] * 100, spans)
    assert "span_micro_f1_below_0.30" in poor_spans["reasons"]


def test_e2_gate_requires_locked_complete_panel_and_correct_matrix_orientation() -> None:
    frame = Frame(tuple(f"x{i:03d}" for i in range(128)), tuple(range(128)))
    cell = Cell("m", "f", "t", 0, 0)
    arrays = [np.zeros(128) for _ in range(3)]
    data = CellData(cell, arrays, arrays, observed_stat_matrices(frame, cell.key, 16),
                    lambda draw: replay_stat_matrices(frame, cell.key, 16, draw))
    assert cell_delta(data, data.observed) == 0
    primary = [cell] + [Cell(f"m{i}", f"f{i % 4}", f"t{i % 6}", i, i % 6)
                        for i in range(1, 16)]
    assert e2_panel_gate([data], primary, n_draws=319)["reason"] == "incomplete_primary_panel"
    with pytest.raises(ValueError, match="strata"):
        cell_delta(data, [m.T for m in data.observed])


def test_e5_gate_requires_two_precise_prespecified_cells() -> None:
    keys = [f"c{i}" for i in range(6)]
    rows = [{"cell_key": key, "n_evaluation": 400, "n_covered": 350,
             "coverage": 0.875, "reference_precise": True,
             "worst_reference_upper_coverage": 0.91} for key in keys[:2]]
    assert e5_coverage_gate(rows, keys)["passed"] is True
    rows[1]["reference_precise"] = False
    assert e5_coverage_gate(rows, keys)["passed"] is False


def test_mechanism_arms_share_items_and_rotate_fixed_batches() -> None:
    rows = [{"doc_id": f"x{i:04d}", "reference_normalized": i // 300}
            for i in range(600)]
    selected = select_items("civil_comments", rows)
    assert len(selected) == 512
    assert sum(reference_group("civil_comments", r["reference_normalized"])
               for r in selected) == 256
    calls = build_arms("civil_comments", rows, "m::civil_comments")
    assert len(calls) == 3 * (32 * 4 + 32)
    for assignment in (1, 2, 3):
        for arm in ("mixed", "homogeneous"):
            zero = [c for c in calls if c["assignment"] == assignment and
                    c["arm"] == arm and c["rotation"] == 0]
            assert {item for call in zero for item in call["doc_order"]} == {
                r["doc_id"] for r in selected
            }
        mixed = [c for c in calls if c["assignment"] == assignment and
                 c["arm"] == "mixed" and c["call_index"] == 0]
        assert all(set(c["doc_order"]) == set(mixed[0]["doc_order"]) for c in mixed)


def test_multinerd_mechanism_group_uses_person_presence() -> None:
    assert reference_group("multinerd", [[0, 1, "PER"]]) == 1
    assert reference_group("multinerd", [[0, 1, "ORG"]]) == 0
