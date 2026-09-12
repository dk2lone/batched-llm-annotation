"""Data freeze tests. Selection, exclusion, licensing, and pin evidence.

Every test runs offline. Rows and API payloads are injected, so a network failure
cannot make a freeze look complete.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from step0 import datafreeze as D
from step0 import lock
from step0.hashing import RefusedOverwriteError, write_jsonl


def spec_for(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "dataset": "example/demo",
        "revision": "abc123",
        "license": "cc0-1.0",
        "redistribution_permitted": True,
        "split": "validation",
        "text_field": "text",
        "label_field": "label",
        "n_classes": 2,
        "reference_label_rule": "label 1 is the reference class",
        "labelling_limits": "a demo frame",
        "sampling": "natural prevalence, no class balancing",
        "sampling_rationale": "demo",
        "min_chars": 10,
        "max_chars": 200,
        "scan_limit": 10_000,
        "n_items": 32,
        "filler_pool_size": 32,
        "selection_seed": 7,
        "prefix": "dm",
        "excluded_id_sources": [],
    }
    spec.update(overrides)
    return spec


def rows(n: int, classes: int = 2, pad: int = 20) -> list[dict[str, Any]]:
    return [
        {"text": f"record {i} " + "x" * (pad + i % 30), "label": i % classes} for i in range(n)
    ]


def info_for(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sha": "abc123",
        "cardData": {"license": "cc0-1.0"},
        "gated": False,
        "disabled": False,
        "lastModified": "2026-01-01T00:00:00.000Z",
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------------------
# Document text and candidate filtering
# --------------------------------------------------------------------------------------


def test_item_text_uses_one_field_or_the_frozen_template() -> None:
    assert D.item_text(spec_for(), {"text": "  a  b  "}) == "a b"
    paired = spec_for(
        text_fields=["premise", "hypothesis"],
        text_template="premise: {premise}\nhypothesis: {hypothesis}",
    )
    joined = D.item_text(paired, {"premise": "p  p", "hypothesis": "h"})
    assert joined == "premise: p p\nhypothesis: h"


def test_candidates_apply_length_bounds_deduplicate_and_exclude() -> None:
    spec = spec_for(min_chars=15, max_chars=25)
    source = [
        {"text": "short", "label": 0},
        {"text": "a" * 20, "label": 1},
        {"text": "a" * 20, "label": 1},
        {"text": "b" * 300, "label": 0},
        {"text": "c" * 18, "label": 0},
    ]
    kept = D.candidates(source, spec, set())
    assert [item.n_chars for item in kept] == [20, 18]
    excluded = {kept[0].doc_id}
    assert [i.n_chars for i in D.candidates(source, spec, excluded)] == [18]


def test_candidates_stop_at_the_frozen_scan_limit() -> None:
    assert len(D.candidates(rows(500), spec_for(scan_limit=100), set())) == 100


def test_civil_comments_reference_rule_thresholds_the_score() -> None:
    spec = spec_for(dataset="google/civil_comments", label_field="toxicity")
    source = [{"text": "x" * 30, "toxicity": 0.5}, {"text": "y" * 30, "toxicity": 0.49}]
    kept = D.candidates(source, spec, set())
    assert [i.reference for i in kept] == [1, 0]
    assert [i.raw_label for i in kept] == [0.5, 0.49]


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


def test_select_returns_disjoint_frames_with_equal_length_strata() -> None:
    spec = spec_for()
    items, filler = D.select(D.candidates(rows(400), spec, set()), spec)
    assert len(items) == 32
    assert len(filler) == 32
    assert not ({i.doc_id for i in items} & {f.doc_id for f in filler})
    counts = [sum(1 for i in items if i.length_stratum == s) for s in range(16)]
    assert counts == [2] * 16


def test_select_is_deterministic_and_seed_dependent() -> None:
    pool = D.candidates(rows(400), spec_for(), set())
    first, _ = D.select(pool, spec_for())
    same, _ = D.select(pool, spec_for())
    other, _ = D.select(pool, spec_for(selection_seed=99))
    assert [i.doc_id for i in first] == [i.doc_id for i in same]
    assert [i.doc_id for i in first] != [i.doc_id for i in other]


def test_stratified_selection_takes_equal_counts_per_class() -> None:
    spec = spec_for(sampling="stratified, 16 records per class", n_items=32, n_classes=2)
    items, _ = D.select(D.candidates(rows(400), spec, set()), spec)
    assert sum(1 for i in items if i.reference == 0) == 16
    assert sum(1 for i in items if i.reference == 1) == 16


def test_select_refuses_a_pool_that_is_too_small() -> None:
    spec = spec_for(n_items=32, filler_pool_size=32)
    with pytest.raises(ValueError, match="filler pool too small"):
        D.select(D.candidates(rows(40), spec, set()), spec)


def test_stratified_selection_refuses_an_indivisible_frame() -> None:
    spec = spec_for(sampling="stratified", n_items=32, n_classes=3)
    with pytest.raises(ValueError, match="not divisible"):
        D.select(D.candidates(rows(400, classes=3), spec, set()), spec)


def test_length_strata_need_an_exact_division() -> None:
    pool = D.candidates(rows(100), spec_for(), set())
    with pytest.raises(ValueError, match="do not divide"):
        D.assign_strata(pool[:33])


# --------------------------------------------------------------------------------------
# Licences and pins
# --------------------------------------------------------------------------------------


def test_audit_license_compares_the_live_card_with_the_protocol() -> None:
    audit = D.audit_license(spec_for(), info_for())
    assert audit["revision_pinned_matches_live_main"]
    assert audit["license_is_in_permissive_allowlist"]
    stale = D.audit_license(spec_for(), info_for(sha="different"))
    assert not stale["revision_pinned_matches_live_main"]


def test_a_gated_or_disabled_or_stale_repo_is_a_blocker() -> None:
    assert D.license_blockers(D.audit_license(spec_for(), info_for())) == []
    assert D.license_blockers(D.audit_license(spec_for(), info_for(gated="auto")))
    assert D.license_blockers(D.audit_license(spec_for(), info_for(disabled=True)))
    assert D.license_blockers(D.audit_license(spec_for(), info_for(sha="other")))


def test_an_unknown_license_is_a_recorded_restriction_not_a_blocker() -> None:
    spec = spec_for(license="unknown", redistribution_permitted=False)
    audit = D.audit_license(spec, info_for(cardData={"license": ["unknown"]}))
    assert D.license_blockers(audit) == []
    restrictions = " ".join(D.license_restrictions(spec, audit))
    assert "never text" in restrictions
    assert "no redistribution" in restrictions.lower()


def test_model_pins_record_the_live_access_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    models = {"org/model": {"revision": "sha1", "license": "apache-2.0"}}
    monkeypatch.setattr(
        D,
        "fetch_json",
        lambda url, timeout=60.0: {
            "sha": "sha1",
            "gated": False,
            "private": False,
            "disabled": False,
            "cardData": {"license": "apache-2.0"},
            "lastModified": "2026-01-01",
        },
    )
    audit = D.audit_model_pins(models)
    assert audit["ok"]
    assert audit["models"]["org/model"]["gated_is_false"]
    assert audit["models"]["org/model"]["evidence_url"].endswith("org/model")


def test_a_gated_model_repo_blocks_the_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        D,
        "fetch_json",
        lambda url, timeout=60.0: {"sha": "sha1", "gated": "manual", "cardData": {}},
    )
    audit = D.audit_model_pins({"org/model": {"revision": "sha1", "license": "x"}})
    assert not audit["ok"]
    assert "gated" in audit["blockers"][0]
    assert not audit["models"]["org/model"]["gated_is_false"]


def test_the_engine_pin_is_checked_against_the_release_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serving = {
        "engine": "vllm",
        "engine_version_pin": "vllm==0.28.0",
        "engine_wheel_pin": "vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl",
    }
    payload = {
        "info": {"version": "0.28.0", "requires_python": "<3.15,>=3.10"},
        "urls": [
            {
                "filename": "vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl",
                "packagetype": "bdist_wheel",
                "upload_time": "2026-08-26T10:07:25",
                "digests": {"sha256": "deadbeef"},
            }
        ],
    }
    monkeypatch.setattr(D, "fetch_json", lambda url, timeout=60.0: payload)
    record = D.audit_engine_pin(serving)
    assert record["ok"]
    assert record["pinned_wheel_sha256"] == "deadbeef"
    monkeypatch.setattr(D, "fetch_json", lambda url, timeout=60.0: {"info": {}, "urls": []})
    assert not D.audit_engine_pin(serving)["ok"]


def test_a_failed_retrieval_raises_instead_of_returning_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(url: str, timeout: float = 60.0) -> dict[str, Any]:
        raise OSError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(D.RetrievalBlockedError, match="failed"):
        D.fetch_json("https://example.invalid/x")


# --------------------------------------------------------------------------------------
# Whole-task freeze
# --------------------------------------------------------------------------------------


def _pilot_files(root: Path, n_targets: int, n_filler: int) -> Path:
    """Build a stand-in Gate A tree whose exclusion files hold known ids."""
    data = root / "gate-experiments" / "a" / "artifacts" / "full" / "data"
    write_jsonl(data / "targets.jsonl", [{"doc_id": f"old{i}"} for i in range(n_targets)])
    write_jsonl(data / "filler_pool.jsonl", [{"doc_id": f"oldf{i}"} for i in range(n_filler)])
    return root / "gate-experiments" / "a"


def test_freeze_task_writes_frames_provenance_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_a = _pilot_files(tmp_path, 5, 7)
    spec = spec_for(
        excluded_id_sources=[
            "gate-experiments/a/artifacts/full/data/targets.jsonl",
            "gate-experiments/a/artifacts/full/data/filler_pool.jsonl",
        ],
        excluded_id_counts_expected={"targets": 5, "filler_pool": 7},
        task_id="demo",
        exclusion_rationale="demo",
        license_finding="demo finding, never text",
    )
    monkeypatch.setitem(lock.TASKS, "demo_task", spec)
    out = tmp_path / "data"
    provenance = D.freeze_task("demo_task", out, gate_a, rows=rows(400), info=info_for())

    assert provenance["items"]["n"] == 32
    assert provenance["filler_pool"]["n"] == 32
    assert provenance["disjoint_items_and_filler"]
    assert provenance["no_excluded_id_selected"]
    assert provenance["n_excluded_pilot_ids"] == 12
    assert provenance["text_is_published"] is False
    assert "No new experimental model labels" in provenance["no_new_experimental_model_labels"]
    assert (out / "demo" / "items.jsonl").exists()
    assert len(provenance["files"]["items.jsonl"]["sha256"]) == 64

    loaded = D.load_items(out / "demo")
    assert len(loaded) == 32
    assert D.load_items(out / "demo", "filler_pool.jsonl")[0].n_chars > 0
    recorded = json.loads((out / "demo" / "provenance.json").read_text())
    assert recorded["selection_seed"] == spec["selection_seed"]


def test_freeze_task_never_overwrites_an_existing_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_a = _pilot_files(tmp_path, 1, 1)
    monkeypatch.setitem(lock.TASKS, "demo_task", spec_for(task_id="demo"))
    out = tmp_path / "data"
    D.freeze_task("demo_task", out, gate_a, rows=rows(400), info=info_for())
    with pytest.raises(RefusedOverwriteError):
        D.freeze_task("demo_task", out, gate_a, rows=rows(400), info=info_for())


def test_freeze_task_stops_when_the_exclusion_count_is_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_a = _pilot_files(tmp_path, 4, 4)
    monkeypatch.setitem(
        lock.TASKS,
        "demo_task",
        spec_for(
            task_id="demo",
            excluded_id_sources=["gate-experiments/a/artifacts/full/data/targets.jsonl"],
            excluded_id_counts_expected={"targets": 2000},
        ),
    )
    with pytest.raises(D.RetrievalBlockedError, match="protocol expects 2000"):
        D.freeze_task("demo_task", tmp_path / "data", gate_a, rows=rows(400), info=info_for())


def test_freeze_task_stops_on_a_gated_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_a = _pilot_files(tmp_path, 1, 1)
    monkeypatch.setitem(lock.TASKS, "demo_task", spec_for(task_id="demo"))
    with pytest.raises(D.RetrievalBlockedError, match="gated"):
        D.freeze_task(
            "demo_task", tmp_path / "data", gate_a, rows=rows(400), info=info_for(gated="auto")
        )
    assert not (tmp_path / "data" / "demo" / "items.jsonl").exists()


def test_a_missing_pilot_exclusion_file_is_a_blocker(tmp_path: Path) -> None:
    spec = spec_for(excluded_id_sources=["gate-experiments/a/artifacts/full/data/targets.jsonl"])
    with pytest.raises(D.RetrievalBlockedError, match="missing"):
        D.pilot_excluded_ids(tmp_path / "gate-experiments" / "a", spec)
