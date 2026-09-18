"""CPU checks for the frozen B5 batch-size dose-response schedule and output contract."""

from __future__ import annotations

import json
from typing import Any

import pytest

from freezeb import b5
from freezeb.prompts import build_messages
from step0.hashing import sha256_json

CELL = "Qwen/Qwen3-8B::civil_comments"
IMDB_CELL = "ibm-granite/granite-3.3-8b-instruct::imdb"


def frame_ids(n: int = b5.N_ITEMS) -> list[str]:
    return [f"cc{index:014x}" for index in range(n)]


def frame_rows(n: int = b5.N_ITEMS) -> list[dict[str, Any]]:
    return [
        {"doc_id": doc_id, "text": f"synthetic comment {index}", "n_chars": 20 + index % 7}
        for index, doc_id in enumerate(frame_ids(n))
    ]


def token_evidence(ids: list[str], per_document: int) -> dict[str, Any]:
    return {
        "solo_prompt_tokens": dict.fromkeys(ids, per_document),
        "exact_offset_max": -100,
    }


# --- schedule totals ---------------------------------------------------------


def test_expected_counts_match_the_frozen_totals() -> None:
    counts = b5.expected_counts()
    assert counts["joint"] == b5.EXPECTED_JOINT == 95_232
    assert counts["exact_context"] == b5.EXPECTED_EXACT == 491_520
    assert counts["total"] == b5.EXPECTED_TOTAL == 586_752
    assert counts["joint"] == 3 * 8192 * sum(1 / k for k in b5.SIZES) * 4
    assert counts["exact_context"] == 3 * 8192 * 5 * 4
    b5.verify_counts({"joint": 95_232, "exact_context": 491_520, "total": 586_752})


def test_verify_counts_fails_closed_on_any_difference() -> None:
    for key in ("joint", "exact_context", "total"):
        counts = {"joint": 95_232, "exact_context": 491_520, "total": 586_752}
        counts[key] += 1
        with pytest.raises(ValueError, match="expected"):
            b5.verify_counts(counts)


def test_cell_call_counts_by_size_and_condition() -> None:
    ids = frame_ids()
    by_size: dict[int, dict[str, int]] = {k: {"joint": 0, "exact_context": 0} for k in b5.SIZES}
    by_assignment: dict[int, int] = dict.fromkeys(range(1, 4), 0)
    call_ids = set()
    for call in b5.cell_calls(CELL, ids):
        by_size[int(call["batch_size"])][str(call["condition"])] += 1
        by_assignment[int(call["assignment"])] += 1
        call_ids.add(str(call["call_id"]))
    for k in b5.SIZES:
        assert by_size[k]["joint"] == 3 * (8192 // k)
        assert by_size[k]["exact_context"] == 3 * 8192
    assert set(by_assignment.values()) == {b5.expected_counts()["total_per_cell"] // 3}
    assert len(call_ids) == b5.expected_counts()["total_per_cell"] == 146_688


def test_three_assignments_exist_for_every_size() -> None:
    ids = frame_ids()
    seen = {
        (int(task["batch_size"]), int(task["assignment"]))
        for task in b5.group_tasks(CELL, ids)
    }
    assert seen == {(k, a) for k in b5.SIZES for a in (1, 2, 3)}


# --- nesting and coverage ----------------------------------------------------


def test_groups_are_nested_across_powers_of_two() -> None:
    report = b5.nesting_report(CELL, n_items=256)
    assert report["nested_pairs_checked"] == 3 * (64 + 32 + 16 + 8)
    assert report["sizes"] == [2, 4, 8, 16, 32]


def test_every_item_appears_exactly_once_per_assignment_and_size() -> None:
    ids = frame_ids()
    for k in (2, 32):
        for assignment in (1, 2, 3):
            permutation = b5.canonical_permutation(CELL, assignment, b5.N_ITEMS)
            flat = [value for group in b5.group_positions(permutation, k) for value in group]
            assert sorted(flat) == list(range(b5.N_ITEMS))
            assert len(set(flat)) == b5.N_ITEMS
    assert len(ids) == b5.N_ITEMS


def test_assignments_use_different_permutations() -> None:
    first = b5.canonical_permutation(CELL, 1, 512)
    second = b5.canonical_permutation(CELL, 2, 512)
    assert not (first == second).all()


def test_canonical_permutation_is_stable_across_calls() -> None:
    again = b5.canonical_permutation(CELL, 1, 512)
    assert (b5.canonical_permutation(CELL, 1, 512) == again).all()


def test_permutation_rejects_a_foreign_cell_or_assignment() -> None:
    with pytest.raises(ValueError, match="not a B5 cell"):
        b5.canonical_permutation("nobody/model::imdb", 1, 32)
    with pytest.raises(ValueError, match="assignment must be"):
        b5.canonical_permutation(CELL, 4, 32)


# --- prompt equivalence and targets -----------------------------------------


def test_joint_and_exact_prompts_show_identical_documents_in_identical_order() -> None:
    rows = frame_rows()
    texts = {str(row["doc_id"]): str(row["text"]) for row in rows}
    task = next(b5.group_tasks(CELL, [str(row["doc_id"]) for row in rows]))
    calls = list(b5.group_calls(task))
    joint, exact = calls[0], calls[1:]
    assert joint["condition"] == "joint"
    assert [call["condition"] for call in exact] == ["exact_context"] * int(task["batch_size"])
    shown = [
        _documents_block(b5.render_messages("civil_comments", texts, call)) for call in calls
    ]
    assert all(block == shown[0] for block in shown)
    assert shown[0].count("<doc id=") == int(task["batch_size"])
    assert [call["target_ids"] for call in exact] == [[doc_id] for doc_id in task["doc_order"]]
    assert joint["target_ids"] == task["doc_order"]


def test_exact_context_requests_only_its_own_target() -> None:
    rows = frame_rows()
    texts = {str(row["doc_id"]): str(row["text"]) for row in rows}
    task = next(b5.group_tasks(CELL, [str(row["doc_id"]) for row in rows]))
    for call in list(b5.group_calls(task))[1:]:
        target = str(call["target_ids"][0])
        content = b5.render_messages("civil_comments", texts, call)[1]["content"]
        assert f'Return only the label for document ID "{target}"' in content
        assert content.count('"labels"') == 1


def test_render_messages_matches_the_shared_builder() -> None:
    rows = frame_rows()
    texts = {str(row["doc_id"]): str(row["text"]) for row in rows}
    task = next(b5.group_tasks(CELL, [str(row["doc_id"]) for row in rows]))
    call = next(b5.group_calls(task))
    order = [str(value) for value in task["doc_order"]]
    assert b5.render_messages("civil_comments", texts, call) == build_messages(
        "civil_comments", [(doc_id, texts[doc_id]) for doc_id in order], order
    )


# --- output contract ---------------------------------------------------------


def test_label_schema_requires_exactly_the_requested_ids() -> None:
    schema = b5.label_schema("multi_nli", ["a", "b"])
    labels = schema["properties"]["labels"]
    assert labels["required"] == ["a", "b"]
    assert set(labels["properties"]) == {"a", "b"}
    assert labels["additionalProperties"] is False
    assert schema["additionalProperties"] is False
    assert labels["properties"]["a"]["enum"] == [0, 1, 2]
    assert schema["$schema_contract"] == b5.CONTRACT_VERSION


def test_label_schema_uses_only_that_task_legal_labels() -> None:
    for task, values in (("civil_comments", [0, 1]), ("imdb", [0, 1]), ("ag_news", [0, 1, 2, 3])):
        schema = b5.label_schema(task, ["x"])
        assert schema["properties"]["labels"]["properties"]["x"]["enum"] == values
    with pytest.raises(ValueError, match="unsupported B5 task"):
        b5.label_schema("go_emotions", ["x"])


def test_label_schema_rejects_empty_or_duplicate_targets() -> None:
    with pytest.raises(ValueError, match="nonempty and unique"):
        b5.label_schema("imdb", [])
    with pytest.raises(ValueError, match="nonempty and unique"):
        b5.label_schema("imdb", ["a", "a"])


def test_output_limit_bounds_the_longest_valid_response() -> None:
    ids = [f"cc{index:014x}" for index in range(32)]
    for task, widest in (("civil_comments", 1), ("multi_nli", 2), ("ag_news", 3)):
        longest = json.dumps(
            {"labels": dict.fromkeys(ids, widest)}, separators=(",", ":")
        )
        cap = b5.max_output_tokens(task, ids)
        assert cap == len(longest.encode("utf-8")) + b5.OUTPUT_TOKEN_ALLOWANCE
        assert cap > len(longest)


def test_output_limit_grows_with_batch_size_and_is_derived_before_output() -> None:
    ids = [f"cc{index:014x}" for index in range(32)]
    caps = [b5.max_output_tokens("imdb", ids[:k]) for k in b5.SIZES]
    assert caps == sorted(caps)
    assert caps[0] < caps[-1] < 1024


# --- context fit -------------------------------------------------------------


def test_size_32_fits_the_sealed_context_limit_on_measured_evidence() -> None:
    ids = frame_ids()
    audit = b5.context_audit(CELL, ids, token_evidence(ids, 400))
    assert audit["fits"] is True
    largest = audit["by_size"][-1]
    assert largest["batch_size"] == 32
    assert largest["max_required_tokens"] <= b5.MAX_MODEL_LEN
    assert largest["max_required_tokens"] > audit["by_size"][0]["max_required_tokens"]


def test_context_audit_fails_closed_on_an_oversized_frame() -> None:
    ids = frame_ids()
    audit = b5.context_audit(CELL, ids, token_evidence(ids, 2_000))
    assert audit["fits"] is False
    assert audit["by_size"][-1]["fits"] is False


def test_context_audit_requires_evidence_for_every_item() -> None:
    ids = frame_ids()
    evidence = token_evidence(ids, 400)
    evidence["solo_prompt_tokens"].pop(ids[0])
    with pytest.raises(ValueError, match="does not cover the frame"):
        b5.context_audit(CELL, ids, evidence)


# --- deterministic identities ------------------------------------------------


def test_call_ids_schemas_and_hashes_are_deterministic() -> None:
    ids = frame_ids()
    first = [
        (c["call_id"], c["json_schema_sha256"], c["doc_order_sha256"], c["max_tokens"])
        for c in b5.cell_calls(CELL, ids)
    ]
    second = [
        (c["call_id"], c["json_schema_sha256"], c["doc_order_sha256"], c["max_tokens"])
        for c in b5.cell_calls(CELL, ids)
    ]
    assert first == second
    assert len({row[0] for row in first}) == len(first)


def test_cell_digest_covers_every_prompt_hash() -> None:
    rows = frame_rows()
    digest = b5.cell_digest(CELL, rows)
    assert digest["counts"]["total"] == 146_688
    assert digest["counts_by_size"]["32"] == {"joint": 768, "exact_context": 24_576}
    assert digest["prompt_sha256_order_sha256"] != digest["call_id_order_sha256"]
    assert b5.cell_digest(CELL, rows)["record_stream_sha256"] == digest["record_stream_sha256"]


def test_cell_digest_changes_when_a_document_changes() -> None:
    rows = frame_rows()
    before = b5.cell_digest(CELL, rows)["prompt_sha256_order_sha256"]
    rows[0]["text"] = "a different synthetic comment"
    assert b5.cell_digest(CELL, rows)["prompt_sha256_order_sha256"] != before


# --- chunking ----------------------------------------------------------------


def test_chunks_cover_the_schedule_once_and_never_split_a_group() -> None:
    ids = frame_ids()
    chunks = list(b5.chunks(CELL, ids, 512))
    assert sum(int(chunk["calls"]) for chunk in chunks) == 146_688
    assert len({call_id for chunk in chunks for call_id in _chunk_call_ids(chunk)}) == 146_688
    for chunk in chunks:
        sizes = {int(group["batch_size"]) for group in chunk["groups"]}
        assignments = {int(group["assignment"]) for group in chunk["groups"]}
        assert len(sizes) == 1
        assert len(assignments) == 1
        assert int(chunk["calls"]) == len(chunk["groups"]) * (int(chunk["batch_size"]) + 1)
        assert int(chunk["calls"]) <= 512


def test_smoke_and_full_schedules_use_the_same_generation_path() -> None:
    ids = frame_ids()
    small = [
        call_id
        for chunk in b5.chunks(CELL, ids, 64)
        for call_id in _chunk_call_ids(chunk)
    ]
    large = [
        call_id
        for chunk in b5.chunks(CELL, ids, 512)
        for call_id in _chunk_call_ids(chunk)
    ]
    assert small == large
    assert len(small) == 146_688


def test_chunk_identity_is_the_ordered_call_id_digest() -> None:
    chunk = next(iter(b5.chunks(CELL, frame_ids(), 512)))
    assert chunk["chunk_id"] == sha256_json(_chunk_call_ids(chunk))


def test_chunk_budget_must_hold_one_largest_group() -> None:
    with pytest.raises(ValueError, match="largest group"):
        list(b5.chunks(CELL, frame_ids(64), 8))


def _documents_block(messages: list[dict[str, str]]) -> str:
    """The shown documents, in order, exactly as the prompt renders them."""
    body = messages[1]["content"].split("Documents:\n", 1)[1]
    return body.split("\n\nReturn ", 1)[0]


def _chunk_call_ids(chunk: dict[str, Any]) -> list[str]:
    return [
        str(call["call_id"])
        for group in chunk["groups"]
        for call in b5.group_calls(group)
    ]


# --- forecast ----------------------------------------------------------------


def anchors() -> list[dict[str, Any]]:
    return [
        {"cell_key": key, "calls": 20_800, "metered_hours": hours,
         "mean_prompt_tokens": prompt, "mean_output_tokens": output, "chunk_calls": 32}
        for key, prompt, output, hours in (
            (b5.CELLS[0], 1815.0, 296.0, 3.46),
            (b5.CELLS[1], 5415.8, 296.2, 8.28),
            (b5.CELLS[2], 1722.8, 228.4, 3.25),
            (b5.CELLS[3], 1602.7, 302.2, 3.18),
        )
    ]


def evidence() -> dict[str, dict[str, Any]]:
    return {
        cell_key: {
            "joint_template_tokens": 110.0,
            "exact_template_tokens": 136.0,
            "target_tokens": 18.0,
            "frame_document_tokens": 740_000.0,
            "joint_response_frame_tokens": 4.0,
            "exact_response_tokens": 22.0,
        }
        for cell_key in b5.CELLS
    }


def test_runtime_fit_reproduces_the_measured_lanes() -> None:
    coefficients = b5.runtime_coefficients(anchors())
    assert coefficients["max_absolute_relative_error"] < 0.05
    assert coefficients["seconds_per_prompt_token"] > 0
    assert coefficients["seconds_per_output_token"] > 0


def test_runtime_fit_needs_enough_lanes() -> None:
    with pytest.raises(ValueError, match="at least three"):
        b5.runtime_coefficients(anchors()[:2])


def test_prefix_reuse_scenario_costs_less_than_the_no_reuse_bound() -> None:
    record = b5.forecast(
        evidence(),
        {"b4_anchors": anchors()},
        usd_per_hour=1.59,
        boot_hours_per_model=0.25,
        retry_fraction=0.02,
    )
    assert record["point_metered_a100_hours"] < record["upper_metered_a100_hours"]
    assert record["point_estimated_usd"] == record["point_metered_a100_hours"] * 1.59
    blocks = record["scenarios"]["prefix_reuse"]["blocks"]
    assert len(blocks) == len(b5.CELLS) * len(b5.SIZES)
    assert sum(int(row["calls"]) for row in blocks) == b5.EXPECTED_TOTAL


def test_required_balance_is_capped_by_the_ceiling_and_adds_the_reserve() -> None:
    record = b5.forecast(
        evidence(),
        {"b4_anchors": anchors()},
        usd_per_hour=2.0,
        boot_hours_per_model=0.0,
        retry_fraction=0.0,
    )
    assert b5.required_balance_usd(record, 0.5, 1.0) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="invalid balance"):
        b5.required_balance_usd(record, 0.0, 1.0)


def test_block_workload_counts_every_scheduled_call() -> None:
    total = sum(
        b5.block_workload(evidence()[b5.CELLS[0]], k, prefix_reuse=True)["calls"]
        for k in b5.SIZES
    )
    assert int(total) == b5.expected_counts()["total_per_cell"]
