"""Prompt construction and the strict parser."""

from __future__ import annotations

import json

import pytest

from batchgate.prompts import (
    build_joint_prompt,
    build_single_prompt,
    parse_labels,
    prompt_text,
    target_status,
)

DOCS = [(f"cc{i:014d}", f"document number {i}") for i in range(16)]
IDS = [doc_id for doc_id, _ in DOCS]


def test_joint_prompt_lists_every_id_once() -> None:
    text = prompt_text(build_joint_prompt(DOCS))
    for doc_id, body in DOCS:
        assert text.count(f'<doc id="{doc_id}">') == 1
        assert body in text
    assert "Return one label for every document id" in text


def test_single_prompt_shows_context_but_asks_for_one_label() -> None:
    target = IDS[7]
    text = prompt_text(build_single_prompt(DOCS, target))
    assert text.count("<doc id=") == 16
    assert f'Label only the document with id "{target}"' in text
    assert '{"labels": {"' + target + '": 0}}' in text


def test_solo_and_filler_share_one_builder() -> None:
    solo = prompt_text(build_single_prompt(DOCS[:1], IDS[0]))
    filler = prompt_text(build_single_prompt(DOCS, IDS[0]))
    assert "Label one document for toxicity." in solo
    assert "Label one document for toxicity." in filler
    assert len(filler) > len(solo)


def test_duplicate_document_in_prompt_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_joint_prompt([DOCS[0], DOCS[0]])
    with pytest.raises(ValueError):
        build_single_prompt([DOCS[0], DOCS[0]], IDS[0])


def test_target_must_be_in_the_prompt() -> None:
    with pytest.raises(ValueError):
        build_single_prompt(DOCS[:3], "ccffffffffffffff")


def test_parse_accepts_exact_json() -> None:
    raw = json.dumps({"labels": {i: 1 for i in IDS[:3]}})
    result = parse_labels(raw, IDS[:3])
    assert result.ok
    assert result.labels == dict.fromkeys(IDS[:3], 1)
    assert result.missing_ids == ()


def test_parse_strips_one_code_fence_and_records_it() -> None:
    raw = "```json\n" + json.dumps({"labels": {IDS[0]: 0}}) + "\n```"
    result = parse_labels(raw, IDS[:1])
    assert result.ok
    assert "stripped_code_fence" in result.notes


def test_parse_reports_missing_ids_without_repairing_them() -> None:
    raw = json.dumps({"labels": {IDS[0]: 1}})
    result = parse_labels(raw, IDS[:3])
    assert result.ok
    assert result.missing_ids == (IDS[1], IDS[2])
    assert target_status(result, IDS[0]) == "ok"
    assert target_status(result, IDS[1]) == "missing_id"


def test_parse_reports_unknown_ids() -> None:
    raw = json.dumps({"labels": {IDS[0]: 1, "ccdeadbeefdeadbe": 0}})
    result = parse_labels(raw, IDS[:1])
    assert result.unknown_ids == ("ccdeadbeefdeadbe",)
    assert "unknown_ids_present" in result.notes


def test_parse_fails_on_duplicate_ids() -> None:
    raw = f'{{"labels": {{"{IDS[0]}": 0, "{IDS[0]}": 1}}}}'
    result = parse_labels(raw, IDS[:1])
    assert result.status == "duplicate_ids"
    assert result.labels == {}


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ("the first one is toxic", "no_json"),
        ('{"labels": {', "json_error"),
        ('{"answer": 1}', "missing_labels"),
        ('{"labels": {"x": 2}}', "bad_value"),
        ('{"labels": {"x": true}}', "bad_value"),
        ('{"labels": {"x": "toxic"}}', "bad_value"),
        ('{"labels": []}', "missing_labels"),
    ],
)
def test_malformed_output_is_a_parse_failure(raw: str, status: str) -> None:
    result = parse_labels(raw, ["x"])
    assert result.status == status
    assert result.labels == {}


def test_string_digits_are_accepted_but_flagged() -> None:
    result = parse_labels('{"labels": {"x": "1"}}', ["x"])
    assert result.ok
    assert result.labels == {"x": 1}
    assert "string_label" in result.notes


def test_every_requested_id_missing_is_a_job_failure() -> None:
    result = parse_labels('{"labels": {"other": 1}}', ["x"])
    assert result.status == "missing_labels"
    assert target_status(result, "x") == "missing_labels"
