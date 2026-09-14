"""CPU tests for the X1R4 M1 solo-screen runner rules.

No test here loads a model, starts a container, or reads a credential. Seals,
locks, frames, and worker payloads are all synthetic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from freezeb import extension as ext
from freezeb import extension_screen as m1
from freezeb.frames import N_SCREEN
from freezeb.prompts import build_messages
from freezeb.screen import MAX_TECHNICAL_ATTEMPTS
from step0.hashing import sha256_file, sha256_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
QWEN = ext.EXT_MODEL_IDS[0]
GRANITE = ext.EXT_MODEL_IDS[1]
LLAMA = ext.EXT_MODEL_IDS[2]
TASK = "civil_comments"
CELL = f"{QWEN}::{TASK}"


# --- Synthetic evidence ----------------------------------------------------


def _rows(task: str = TASK, count: int = N_SCREEN) -> list[dict[str, Any]]:
    classes = {"civil_comments": 2, "imdb": 2, "multi_nli": 3, "ag_news": 4}[task]
    return [
        {
            "doc_id": f"{task[:2]}{index:014x}",
            "text": f"Synthetic {task} record {index} for a CPU test.",
            "reference": index % classes,
            "reference_normalized": index % classes,
            "n_tokens": 8,
        }
        for index in range(count)
    ]


def _backend(model_id: str = QWEN, **overrides: Any) -> dict[str, Any]:
    policy = ext.ext_policy_for(model_id)
    return {
        "backend": "vllm",
        "model_id": model_id,
        "model_revision": policy.revision,
        "tokenizer_revision": policy.revision,
        "dtype": "bfloat16",
        "max_model_len": policy.max_model_len,
        "max_num_seqs": 32,
        "tensor_parallel_size": 1,
        "enable_thinking": False,
        "seed": m1.EXPECTED_SEED,
        "batch_invariant": True,
        "batch_invariance_required": True,
        **overrides,
    }


def _host(**overrides: Any) -> dict[str, Any]:
    return {
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "installed_engine_version_string": m1.EXPECTED_VLLM_VERSION,
        **overrides,
    }


def _response(doc_id: str, label: int) -> str:
    return json.dumps({"labels": {doc_id: label}})


def _event(
    spec: dict[str, Any], *, status: str = "ok", label: int = 0, **extra: Any
) -> dict[str, Any]:
    event = {key: spec[key] for key in m1.EVENT_KEYS}
    event["status"] = status
    if status == "ok":
        event["first_response"] = _response(str(spec["doc_id"]), label)
    event.update(extra)
    return event


def _ledger_event(spec: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """One accepted, metered ledger row of the shape the runner appends."""
    accepted, quarantined = m1.classify_events(
        [spec], [_event(spec, **kwargs)], host=_host(), backend=_backend(str(spec["model_id"]))
    )
    assert not quarantined
    row = accepted[0]
    row.update(
        {
            "worker_session_id": "session-1",
            "worker_load_seconds": 60.0,
            "worker_host": _host(),
            "worker_backend": _backend(str(spec["model_id"])),
            "metered_a100_seconds": 0.5,
            "cumulative_a100_hours": 0.1,
        }
    )
    return row


def _write_seal(root: Path) -> dict[str, Any]:
    """A minimal X1R4-shaped seal that verify_seal must accept."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "M1_CALL_ORDER.json").write_text(json.dumps({"stage": "test"}), encoding="utf-8")
    (root / "EXTENSION_PROTOCOL.md").write_text("synthetic\n", encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": ext.SCHEMA_VERSION,
        "identity_sha256": ext.disclosure()["identity_sha256"],
        "batch_invariance_plan_sha256": "b" * 64,
        "contains_model_outputs": False,
        "contains_dataset_text": False,
        "frames": {},
        "files_sha256": {
            name: sha256_file(root / name)
            for name in ("EXTENSION_PROTOCOL.md", "M1_CALL_ORDER.json")
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _write_lock(root: Path, seal_manifest: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """A minimal signed runtime/access lock bound to the given seal."""
    root.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = overrides.get("candidate_status", {})
    runtime = {
        "status": "passed",
        "scientific_model_outputs": False,
        "sealed_batch_plan_sha256": overrides.get(
            "batch_plan", seal_manifest["batch_invariance_plan_sha256"]
        ),
        "batch_report_sha256": "r" * 64,
        "candidates": [
            {
                "model_id": model.repo,
                "revision": overrides.get("revisions", {}).get(model.repo, model.revision),
                "status": statuses.get(model.repo, "attempted"),
                "synthetic_runtime_status": overrides.get("runtime_status", {}).get(
                    model.repo, "passed"
                ),
                "batch_invariance_status": overrides.get("batch_status", {}).get(
                    model.repo, "passed"
                ),
                "context_window_tokens": 32768,
            }
            for model in ext.EXT_MODELS
        ],
    }
    (root / "RUNTIME_LOCK.json").write_text(json.dumps(runtime), encoding="utf-8")
    access = {
        "runtime_lock_sha256": overrides.get(
            "bound_runtime_sha256", sha256_file(root / "RUNTIME_LOCK.json")
        ),
        "batch_invariance_report_sha256": "r" * 64,
        "signed_by": "test",
        "candidates": [
            {
                "model_id": model.repo,
                "candidate_status": statuses.get(model.repo, "attempted"),
                "access_status": "verified",
            }
            for model in ext.EXT_MODELS
        ],
        "attempted_models": [
            model.repo
            for model in ext.EXT_MODELS
            if statuses.get(model.repo, "attempted") == "attempted"
        ],
        "unavailable_models": [
            model.repo
            for model in ext.EXT_MODELS
            if statuses.get(model.repo, "attempted") != "attempted"
        ],
    }
    (root / "ACCESS_LOCK.json").write_text(json.dumps(access), encoding="utf-8")
    (root / "evidence").mkdir(exist_ok=True)
    (root / "evidence" / "staging").mkdir(exist_ok=True)
    (root / "evidence" / "staging" / "report.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": model.repo,
                        "revision": model.revision,
                        "file_count": 14,
                        "bytes": 16_000_000_000,
                    }
                    for model in ext.EXT_MODELS[2:]
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "status": "sealed",
        "x1r4_manifest_identifier": overrides.get(
            "seal_identifier", seal_manifest["manifest_sha256"]
        ),
        "files_sha256": {
            name: sha256_file(root / name)
            for name in ("ACCESS_LOCK.json", "RUNTIME_LOCK.json", "evidence/staging/report.json")
        },
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _runtime_evidence(model_id: str = QWEN, latency: float = 0.5) -> dict[str, Any]:
    events: list[dict[str, Any]] = [
        {
            "case_id": f"{task}{m1.RUNTIME_SOLO_PROXY_SUFFIX}",
            "phase": phase,
            "status": "ok",
            "latency_seconds": latency,
        }
        for task in ext.EXT_TASK_KEYS
        for phase in ("warmup", "measure", "measure")
    ]
    return {
        "model_id": model_id,
        "container_gpu_seconds": sum(float(row["latency_seconds"]) for row in events) + 120.0,
        "events": events,
    }


# --- Seal and lock binding -------------------------------------------------


def test_verify_seal_accepts_an_intact_seal(tmp_path: Path) -> None:
    manifest = _write_seal(tmp_path / "seal")
    verified = m1.verify_seal(tmp_path / "seal")
    assert verified["manifest_sha256"] == manifest["manifest_sha256"]
    assert verified["files_verified"] == 2
    assert verified["identity_sha256"] == ext.disclosure()["identity_sha256"]


def test_verify_seal_rejects_a_changed_sealed_file(tmp_path: Path) -> None:
    _write_seal(tmp_path / "seal")
    (tmp_path / "seal" / "EXTENSION_PROTOCOL.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed files changed"):
        m1.verify_seal(tmp_path / "seal")


def test_verify_seal_rejects_a_rewritten_manifest(tmp_path: Path) -> None:
    _write_seal(tmp_path / "seal")
    path = tmp_path / "seal" / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["batch_invariance_plan_sha256"] = "c" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not hash to its own recorded digest"):
        m1.verify_seal(tmp_path / "seal")


def test_verify_runtime_lock_returns_attempted_candidates(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest)
    seal = m1.verify_seal(tmp_path / "seal")
    lock = m1.verify_runtime_lock(tmp_path / "lock", seal)
    assert lock["attempted_models"] == list(ext.EXT_MODEL_IDS)
    assert lock["unavailable_models"] == []
    assert lock["candidate_status"][LLAMA] == "attempted"


def test_verify_runtime_lock_keeps_an_unavailable_candidate_out_of_the_screen(
    tmp_path: Path,
) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(
        tmp_path / "lock", seal_manifest, candidate_status={LLAMA: "technically_unavailable"}
    )
    lock = m1.verify_runtime_lock(tmp_path / "lock", m1.verify_seal(tmp_path / "seal"))
    assert LLAMA not in lock["attempted_models"]
    assert lock["unavailable_models"] == [LLAMA]


def test_verify_runtime_lock_rejects_a_lock_from_another_seal(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest, seal_identifier="d" * 64)
    with pytest.raises(ValueError, match="different seal"):
        m1.verify_runtime_lock(tmp_path / "lock", m1.verify_seal(tmp_path / "seal"))


def test_verify_runtime_lock_rejects_a_failed_batch_invariance_gate(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest, batch_status={QWEN: "failed"})
    with pytest.raises(ValueError, match="batch-invariance gate"):
        m1.verify_runtime_lock(tmp_path / "lock", m1.verify_seal(tmp_path / "seal"))


def test_verify_runtime_lock_rejects_an_unpinned_revision(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest, revisions={QWEN: "0" * 40})
    with pytest.raises(ValueError, match="unpinned revision"):
        m1.verify_runtime_lock(tmp_path / "lock", m1.verify_seal(tmp_path / "seal"))


def test_verify_runtime_lock_rejects_an_access_lock_bound_elsewhere(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest, bound_runtime_sha256="e" * 64)
    with pytest.raises(ValueError, match="does not bind this runtime lock"):
        m1.verify_runtime_lock(tmp_path / "lock", m1.verify_seal(tmp_path / "seal"))


def test_verify_runtime_lock_rejects_disagreeing_candidate_lists(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest)
    path = tmp_path / "lock" / "ACCESS_LOCK.json"
    access = json.loads(path.read_text(encoding="utf-8"))
    access["attempted_models"] = access["attempted_models"][:3]
    path.write_text(json.dumps(access), encoding="utf-8")
    manifest = json.loads((tmp_path / "lock" / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["files_sha256"]["ACCESS_LOCK.json"] = sha256_file(path)
    (tmp_path / "lock" / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate lists disagree"):
        m1.verify_runtime_lock(tmp_path / "lock", m1.verify_seal(tmp_path / "seal"))


def test_verify_runtime_lock_rejects_a_tampered_lock_file(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest)
    path = tmp_path / "lock" / "RUNTIME_LOCK.json"
    runtime = json.loads(path.read_text(encoding="utf-8"))
    runtime["status"] = "failed"
    path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="lock files changed"):
        m1.verify_runtime_lock(tmp_path / "lock", m1.verify_seal(tmp_path / "seal"))


# --- Frames, calls, and the sealed order -----------------------------------


def test_verify_screen_frame_rejects_a_changed_source(tmp_path: Path) -> None:
    rows = _rows()
    path = tmp_path / TASK / "screen.jsonl"
    write_jsonl(path, rows)
    seal = {
        "frames": {
            TASK: {
                "screen": {
                    "source_file_sha256": sha256_file(path),
                    "row_ids_sha256": sha256_json([row["doc_id"] for row in rows]),
                    "rows": N_SCREEN,
                }
            }
        }
    }
    assert len(m1.verify_screen_frame(tmp_path, seal, TASK)) == N_SCREEN
    write_jsonl(path, [*rows[1:], rows[0]], allow_existing=True)
    with pytest.raises(ValueError, match="differs from the X1R4 seal"):
        m1.verify_screen_frame(tmp_path, seal, TASK)


def test_build_cell_calls_reproduces_the_sealed_call_and_prompt_order() -> None:
    rows = _rows()
    order = {"cells": [{**ext.m1_cell_digest(QWEN, TASK, rows), "position": 1}]}
    calls = m1.build_cell_calls(QWEN, TASK, rows)
    verified = m1.verify_cell_against_seal(order, QWEN, TASK, calls)
    assert verified["n_calls"] == N_SCREEN
    assert calls[0]["condition"] == "solo"
    assert calls[0]["attempt_id"].endswith("::attempt::1")
    assert calls[0]["messages"] == build_messages(
        TASK, [(rows[0]["doc_id"], rows[0]["text"])], [rows[0]["doc_id"]]
    )
    assert len({call["call_id"] for call in calls}) == N_SCREEN


def test_verify_cell_against_seal_rejects_a_reordered_frame() -> None:
    rows = _rows()
    order = {"cells": [{**ext.m1_cell_digest(QWEN, TASK, rows), "position": 1}]}
    reordered = m1.build_cell_calls(QWEN, TASK, [*rows[1:], rows[0]])
    with pytest.raises(ValueError, match="call-ID order differs"):
        m1.verify_cell_against_seal(order, QWEN, TASK, reordered)


def test_verify_cell_against_seal_rejects_a_changed_prompt() -> None:
    rows = _rows()
    order = {"cells": [{**ext.m1_cell_digest(QWEN, TASK, rows), "position": 1}]}
    calls = m1.build_cell_calls(QWEN, TASK, rows)
    calls[7]["prompt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="prompt-hash order differs"):
        m1.verify_cell_against_seal(order, QWEN, TASK, calls)


def test_execution_order_follows_the_seal_and_drops_unavailable_models() -> None:
    order = {
        "cells": [
            {
                "cell_key": f"{model}::{task}",
                "model_id": model,
                "task_id": task,
                "position": position,
            }
            for position, (model, task) in enumerate(
                [(m, t) for m in ext.EXT_MODEL_IDS for t in ext.EXT_TASK_KEYS], start=1
            )
        ]
    }
    cells = m1.execution_order(order, [QWEN, GRANITE])
    assert [cell["position"] for cell in cells] == list(range(1, 9))
    assert {cell["model_id"] for cell in cells} == {QWEN, GRANITE}
    with pytest.raises(ValueError, match="unknown candidates"):
        m1.execution_order(order, ["openai/not-a-candidate"])


def _plan_lock(attempted: list[str] | None = None) -> dict[str, Any]:
    models = attempted if attempted is not None else list(ext.EXT_MODEL_IDS)
    return {
        "runtime_lock_sha256": "b" * 64,
        "access_lock_sha256": "c" * 64,
        "attempted_models": models,
        "unavailable_models": [m for m in ext.EXT_MODEL_IDS if m not in models],
    }


def _plan_cells(attempted: list[str] | None = None) -> list[dict[str, Any]]:
    models = attempted if attempted is not None else list(ext.EXT_MODEL_IDS)
    return [
        {
            "cell_key": f"{model_id}::{task_id}",
            "position": position,
            "n_calls": N_SCREEN,
            "call_id_order_sha256": "1" * 64,
            "prompt_sha256_order_sha256": "2" * 64,
        }
        for position, (model_id, task_id) in enumerate(
            [(m, t) for m in models for t in ext.EXT_TASK_KEYS], start=1
        )
    ]


def _plan_arguments(attempted: list[str] | None = None) -> dict[str, Any]:
    return {
        "seal": {"manifest_sha256": "a" * 64, "files_verified": 55},
        "lock": _plan_lock(attempted),
        "m1_order": {
            "stage": "M1",
            "condition": "solo only",
            "n_calls": m1.M1_TOTAL_CALLS,
            "global_call_id_order_sha256": "d" * 64,
            "global_prompt_sha256_order_sha256": "e" * 64,
        },
        "verified_cells": _plan_cells(attempted),
        "forecast": m1.m1_base_forecast({QWEN: _runtime_evidence()}),
        "chunk_size": 32,
    }


def test_build_plan_is_stable_and_requires_batch_invariance() -> None:
    arguments = _plan_arguments()
    first = m1.build_plan(**arguments, batch_invariant_env="1")
    second = m1.build_plan(**arguments, batch_invariant_env="1")
    assert first == second
    assert first["budget"]["stop_a100_hours"] == ext.STOP_HOURS
    assert first["first_gate"] == m1.GATE_NAME
    assert first["first_gate_attempts"] == m1.FIRST_REFORECAST_ATTEMPTS
    assert first["planned_calls"] == 5 * 4 * N_SCREEN
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT"):
        m1.build_plan(**arguments, batch_invariant_env="0")


def test_the_plan_binds_the_exact_schedule_that_will_run() -> None:
    """A different panel or a different call count must change the plan hash."""
    full = m1.build_plan(**_plan_arguments(), batch_invariant_env="1")
    narrowed = m1.build_plan(**_plan_arguments([QWEN, GRANITE, LLAMA]), batch_invariant_env="1")
    assert full["plan_sha256"] != narrowed["plan_sha256"]
    assert full["schedule"]["cell_keys"] != narrowed["schedule"]["cell_keys"]
    assert narrowed["planned_calls"] == 3 * 4 * N_SCREEN
    assert full["schedule"]["cell_order_sha256"] != narrowed["schedule"]["cell_order_sha256"]


def test_a_partial_panel_or_task_subset_is_refused() -> None:
    """M1 has no subset mode; a resumed run cannot silently screen fewer cells."""
    lock = _plan_lock()
    with pytest.raises(ValueError, match="complete attempted panel"):
        m1.verify_full_schedule(_plan_cells()[:-1], lock)
    with pytest.raises(ValueError, match="complete attempted panel"):
        m1.verify_full_schedule(_plan_cells([QWEN]), lock)
    one_task = [cell for cell in _plan_cells() if cell["cell_key"].endswith(f"::{TASK}")]
    with pytest.raises(ValueError, match="complete attempted panel"):
        m1.verify_full_schedule(one_task, lock)
    shuffled = _plan_cells()
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(ValueError, match="complete attempted panel"):
        m1.verify_full_schedule(shuffled, lock)
    short_cell = _plan_cells()
    short_cell[3] = {**short_cell[3], "n_calls": N_SCREEN - 1}
    with pytest.raises(ValueError, match="sealed call count"):
        m1.verify_full_schedule(short_cell, lock)
    schedule = m1.verify_full_schedule(_plan_cells(), lock)
    assert schedule["planned_calls"] == m1.M1_TOTAL_CALLS
    assert schedule["tasks"] == list(ext.EXT_TASK_KEYS)


# --- Pinned checkpoints ----------------------------------------------------


def test_expected_checkpoint_inventory_merges_both_evidence_sources(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest)
    freeze_b = tmp_path / "checkpoint-manifests.json"
    freeze_b.write_text(
        json.dumps(
            [
                {
                    "model_id": QWEN,
                    "revision": ext.EXT_MODELS[0].revision,
                    "files": [{"path": "config.json", "bytes": 10, "sha256": "a" * 64}],
                },
                {"model_id": "other/model", "revision": "0" * 40, "files": []},
            ]
        ),
        encoding="utf-8",
    )
    inventory = m1.expected_checkpoint_inventory(tmp_path / "lock", freeze_b)
    assert inventory[QWEN]["source"] == "freeze_b_benchmark_checkpoint_lock"
    assert "files_sha256" in inventory[QWEN]
    assert inventory[LLAMA]["source"] == "x1r4_staging_record"
    assert "files_sha256" not in inventory[LLAMA]
    assert "other/model" not in inventory


def test_verify_checkpoint_inventory_rejects_drift() -> None:
    expected = {
        "model_id": QWEN,
        "revision": ext.EXT_MODELS[0].revision,
        "file_count": 15,
        "total_bytes": 100,
        "files_sha256": "a" * 64,
        "source": "freeze_b_benchmark_checkpoint_lock",
    }
    actual = {
        "model_id": QWEN,
        "revision": ext.EXT_MODELS[0].revision,
        "file_count": 15,
        "total_bytes": 100,
        "files_sha256": "a" * 64,
    }
    assert m1.verify_checkpoint_inventory(expected, actual)["per_file_hashes_compared"] is True
    with pytest.raises(ValueError, match="total_bytes differs"):
        m1.verify_checkpoint_inventory(expected, {**actual, "total_bytes": 99})
    with pytest.raises(ValueError, match="per-file hashes differ"):
        m1.verify_checkpoint_inventory(expected, {**actual, "files_sha256": "b" * 64})
    with pytest.raises(ValueError, match="different checkpoint identity"):
        m1.verify_checkpoint_inventory(expected, {**actual, "revision": "0" * 40})


# --- Resume, retries, and immutable first responses ------------------------


def test_pending_calls_skips_completed_calls_on_resume() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())
    events = [_ledger_event(call) for call in calls[:10]]
    pending = m1.pending_calls(calls, events)
    assert len(pending) == N_SCREEN - 10
    assert all(spec["attempt"] == 1 for spec in pending)
    assert {spec["call_id"] for spec in pending}.isdisjoint({event["call_id"] for event in events})
    assert m1.pending_calls(calls, [_ledger_event(call) for call in calls]) == []


def test_pending_calls_links_a_retry_to_its_preserved_first_attempt() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows(count=N_SCREEN))
    failed = _ledger_event(calls[0], status="generation_failed")
    pending = m1.pending_calls(calls, [failed])
    retry = next(spec for spec in pending if spec["call_id"] == calls[0]["call_id"])
    assert retry["attempt"] == 2
    assert retry["attempt_id"].endswith("::attempt::2")
    assert retry["previous_attempt_id"] == failed["attempt_id"]
    assert retry["previous_status"] == "generation_failed"
    assert retry["attempt_id"] != failed["attempt_id"]


def test_assert_first_attempts_immutable_rejects_a_changed_response() -> None:
    call = m1.build_cell_calls(QWEN, TASK, _rows())[0]
    recorded = {call["attempt_id"]: _ledger_event(call, label=0)}
    m1.assert_first_attempts_immutable(recorded, [dict(recorded[call["attempt_id"]])])
    changed = {**recorded[call["attempt_id"]], "first_response": _response(call["doc_id"], 1)}
    with pytest.raises(ValueError, match="would change recorded attempt"):
        m1.assert_first_attempts_immutable(recorded, [changed])


# --- Provenance and quarantine ---------------------------------------------


def test_validate_worker_provenance_requires_the_frozen_serving_policy() -> None:
    m1.validate_worker_provenance(QWEN, _host(), _backend())
    for broken in (
        {"batch_invariant": False},
        {"batch_invariance_required": False},
        {"seed": 1},
        {"max_num_seqs": 16},
        {"max_model_len": 8192},
        {"model_revision": "0" * 40},
        {"tensor_parallel_size": 2},
        {"dtype": "float16"},
    ):
        with pytest.raises(ValueError, match="differs from the freeze"):
            m1.validate_worker_provenance(QWEN, _host(), _backend(**broken))
    with pytest.raises(ValueError, match="vLLM version"):
        m1.validate_worker_provenance(
            QWEN, _host(installed_engine_version_string="0.27.0"), _backend()
        )
    with pytest.raises(ValueError, match="not an A100"):
        m1.validate_worker_provenance(QWEN, _host(gpu_name="NVIDIA H100"), _backend())
    with pytest.raises(ValueError, match="lacks worker provenance"):
        m1.validate_worker_provenance(QWEN, None, _backend())


def test_classify_events_accepts_a_valid_chunk() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())[:4]
    accepted, quarantined = m1.classify_events(
        calls, [_event(call) for call in calls], host=_host(), backend=_backend()
    )
    assert not quarantined
    assert len(accepted) == 4
    assert all(row["terminal"] and row["final_for_call"] for row in accepted)


def test_classify_events_quarantines_mismatched_and_duplicate_payloads() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())[:4]
    events = [_event(call) for call in calls]
    events[0]["prompt_sha256"] = "f" * 64
    events[1]["call_id"] = "extx1::m1::unknown::0000::zz"
    events[2]["attempt"] = 2
    events.append(_event(calls[3]))
    accepted, quarantined = m1.classify_events(calls, events, host=_host(), backend=_backend())
    reasons = [row["quarantine_reason"] for row in quarantined]
    assert reasons[0].startswith("identity_mismatch:prompt_sha256")
    assert reasons[1] == "unknown_call_id"
    assert reasons[2] == "attempt_id_mismatch"
    assert reasons[3] == "duplicate_attempt_id"
    assert [row["call_id"] for row in accepted] == [calls[3]["call_id"]]
    assert all(row["quarantined"] for row in quarantined)


def test_classify_events_quarantines_a_worker_outside_the_freeze() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    accepted, quarantined = m1.classify_events(
        calls,
        [_event(call) for call in calls],
        host=_host(),
        backend=_backend(batch_invariant=False),
    )
    assert accepted == []
    assert all(row["quarantine_reason"].startswith("worker_provenance") for row in quarantined)


def test_classify_events_quarantines_unknown_status_and_missing_response() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    broken = _event(calls[0])
    del broken["first_response"]
    accepted, quarantined = m1.classify_events(
        calls,
        [broken, _event(calls[1], status="invented_status")],
        host=_host(),
        backend=_backend(),
    )
    assert accepted == []
    assert [row["quarantine_reason"] for row in quarantined] == [
        "missing_first_response",
        "unknown_status",
    ]


def test_classify_events_skips_already_recorded_attempts() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    events = [_event(call) for call in calls]
    accepted, quarantined = m1.classify_events(
        calls,
        events,
        host=_host(),
        backend=_backend(),
        recorded_attempt_ids={str(calls[0]["attempt_id"])},
    )
    assert [row["call_id"] for row in accepted] == [calls[1]["call_id"]]
    assert quarantined[0]["quarantine_reason"] == "duplicate_attempt_id"


def test_classify_events_marks_the_last_technical_attempt_exhausted() -> None:
    call = m1.build_cell_calls(QWEN, TASK, _rows())[0]
    last = {
        **call,
        "attempt": MAX_TECHNICAL_ATTEMPTS,
        "attempt_id": f"{call['call_id']}::attempt::{MAX_TECHNICAL_ATTEMPTS}",
    }
    accepted, _ = m1.classify_events(
        [call], [_event(last, status="generation_failed")], host=_host(), backend=_backend()
    )
    assert accepted[0]["status"] == "technical_attempts_exhausted"
    assert accepted[0]["final_for_call"] is True
    over_limit = {
        **call,
        "attempt": MAX_TECHNICAL_ATTEMPTS + 1,
        "attempt_id": f"{call['call_id']}::attempt::{MAX_TECHNICAL_ATTEMPTS + 1}",
    }
    _, quarantined = m1.classify_events(
        [call], [_event(over_limit)], host=_host(), backend=_backend()
    )
    assert quarantined[0]["quarantine_reason"] == "attempt_outside_technical_limit"


# --- Metering, forecast, and the first-1,000-attempt gate ------------------


def test_session_delta_meters_each_worker_once_and_sums_across_workers() -> None:
    sessions: dict[str, float] = {}
    first = m1.session_delta_seconds(sessions, "worker-a", 100.0)
    second = m1.session_delta_seconds(sessions, "worker-a", 250.0)
    replay = m1.session_delta_seconds(sessions, "worker-a", 250.0)
    other = m1.session_delta_seconds(sessions, "worker-b", 40.0)
    assert (first, second, replay, other) == (100.0, 150.0, 0.0, 40.0)
    assert first + second + replay + other == 290.0
    assert m1.session_delta_seconds(sessions, None, 12.0) == 12.0
    assert m1.session_delta_seconds(sessions, "worker-a", 10.0) == 0.0


def test_m1_base_forecast_uses_the_single_document_proxy_case() -> None:
    forecast = m1.m1_base_forecast({QWEN: _runtime_evidence(latency=0.5)})
    assert forecast["cell_base_a100_hours"][CELL] == pytest.approx(0.5 * N_SCREEN / 3600)
    assert forecast["model_load_a100_hours"][QWEN] == pytest.approx(120.0 / 3600)
    assert m1.cell_base_hours(forecast, CELL, include_load=True) == pytest.approx(
        forecast["cell_base_a100_hours"][CELL] + 120.0 / 3600
    )
    stripped = _runtime_evidence()
    stripped["events"] = [row for row in stripped["events"] if not row["case_id"].startswith(TASK)]
    with pytest.raises(ValueError, match="proxy case"):
        m1.m1_base_forecast({QWEN: stripped})


def _payload(
    session_id: str | None, elapsed: float, attempt_ids: list[str], load: float = 60.0
) -> dict[str, Any]:
    return {
        "worker_session_id": session_id,
        "session_elapsed_seconds": elapsed,
        "load_seconds": load,
        "events": [{"attempt_id": attempt_id} for attempt_id in attempt_ids],
    }


def _absorb(state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    return m1.meter_receipt(
        state,
        payload,
        cell_key=CELL,
        attempt_ids=[str(row["attempt_id"]) for row in payload["events"]],
        wall_seconds=0.0,
    )


def test_receipt_identity_is_stable_for_the_same_committed_payload() -> None:
    payload = _payload("worker-a", 120.0, ["c::attempt::1", "d::attempt::1"])
    assert m1.receipt_identity(payload, ["c::attempt::1", "d::attempt::1"]) == m1.receipt_identity(
        payload, ["d::attempt::1", "c::attempt::1"]
    )
    later = _payload("worker-a", 121.0, ["c::attempt::1", "d::attempt::1"])
    assert m1.receipt_identity(payload, ["c::attempt::1"]) != m1.receipt_identity(
        later, ["c::attempt::1"]
    )
    with pytest.raises(ValueError, match="at least one attempt"):
        m1.receipt_identity(payload, [])


def test_meter_receipts_sum_across_workers_and_never_meter_a_replay_twice() -> None:
    state = m1.empty_meter_state()
    first = _absorb(state, _payload("worker-a", 100.0, ["a::attempt::1"]))
    second = _absorb(state, _payload("worker-a", 250.0, ["b::attempt::1"]))
    replay = _absorb(state, _payload("worker-a", 250.0, ["b::attempt::1"]))
    other = _absorb(state, _payload("worker-b", 40.0, ["c::attempt::1"]))
    assert replay is None
    assert first is not None and second is not None and other is not None
    assert first["metered_a100_seconds"] == 100.0
    assert second["metered_a100_seconds"] == 150.0
    assert other["metered_a100_seconds"] == 40.0
    assert state["metered_a100_seconds"] == 290.0
    assert state["cumulative_a100_hours"] == pytest.approx(290.0 / 3600)
    assert state["receipts"] == 3


def test_the_meter_is_rebuilt_from_receipts_at_every_crash_boundary() -> None:
    """A crash between any two writes must not change the metered total."""
    state = m1.empty_meter_state()
    receipts = [
        _absorb(state, _payload("worker-a", 100.0, ["a::attempt::1"])),
        _absorb(state, _payload("worker-a", 250.0, ["b::attempt::1"])),
        _absorb(state, _payload("worker-b", 40.0, ["c::attempt::1"])),
    ]
    rows = [row for row in receipts if row is not None]
    assert len(rows) == 3
    for boundary in range(len(rows) + 1):
        # The crash happens after `boundary` receipts reached the file.
        resumed = m1.fold_receipts(rows[:boundary])
        assert resumed["metered_a100_seconds"] == sum(
            float(row["metered_a100_seconds"]) for row in rows[:boundary]
        )
        # Replaying the payloads that were already receipted adds nothing.
        for row in rows[:boundary]:
            assert (
                m1.meter_receipt(
                    resumed,
                    _payload(
                        row["worker_session_id"],
                        float(row["session_elapsed_seconds"]),
                        list(row["attempt_ids"]),
                    ),
                    cell_key=CELL,
                    attempt_ids=list(row["attempt_ids"]),
                    wall_seconds=0.0,
                )
                is None
            )
        assert resumed["metered_a100_seconds"] == sum(
            float(row["metered_a100_seconds"]) for row in rows[:boundary]
        )
    complete = m1.fold_receipts(rows)
    assert complete["cumulative_a100_hours"] == pytest.approx(290.0 / 3600)
    # A duplicated append, and a repeated fold, both stay at the same total.
    assert m1.fold_receipts([*rows, *rows])["metered_a100_seconds"] == 290.0


def test_a_resumed_session_meters_only_its_new_seconds() -> None:
    state = m1.empty_meter_state()
    rows = [_absorb(state, _payload("worker-a", 100.0, ["a::attempt::1"]))]
    resumed = m1.fold_receipts([row for row in rows if row is not None])
    later = m1.meter_receipt(
        resumed,
        _payload("worker-a", 260.0, ["b::attempt::1"]),
        cell_key=CELL,
        attempt_ids=["b::attempt::1"],
        wall_seconds=0.0,
    )
    assert later is not None
    assert later["metered_a100_seconds"] == 160.0
    assert resumed["metered_a100_seconds"] == 260.0


def test_a_local_failure_without_a_session_still_meters_its_wall_time() -> None:
    state = m1.empty_meter_state()
    receipt = m1.meter_receipt(
        state,
        {"worker_session_id": None, "load_seconds": None, "events": [{"attempt_id": "a::1"}]},
        cell_key=CELL,
        attempt_ids=["a::1"],
        wall_seconds=12.5,
    )
    assert receipt is not None
    assert receipt["metered_a100_seconds"] == 12.5
    assert receipt["worker_session_id"] is None
    assert state["cumulative_a100_hours"] == pytest.approx(12.5 / 3600)


def test_m1_reforecast_waits_for_the_gate_then_projects_the_remaining_work() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())
    events = [_ledger_event(call) for call in calls[: m1.FIRST_REFORECAST_ATTEMPTS]]
    assert m1.m1_reforecast(events[:-1], planned_calls=m1.M1_TOTAL_CALLS, planned_models=5) is None
    reforecast = m1.m1_reforecast(events, planned_calls=m1.M1_TOTAL_CALLS, planned_models=5)
    assert reforecast is not None
    assert reforecast["gate"] == m1.GATE_NAME
    assert reforecast["attempts"] == m1.FIRST_REFORECAST_ATTEMPTS
    assert reforecast["failure_rate"] == 0.0
    variable = 0.5 * m1.FIRST_REFORECAST_ATTEMPTS - 60.0
    per_call = variable / m1.FIRST_REFORECAST_ATTEMPTS
    assert reforecast["per_call_a100_seconds"] == pytest.approx(per_call)
    assert reforecast["remaining_calls"] == m1.M1_TOTAL_CALLS - len(events)
    assert reforecast["remaining_base_a100_hours"] == pytest.approx(
        (per_call * (m1.M1_TOTAL_CALLS - len(events)) + 60.0 * 4) / 3600
    )
    assert reforecast["remaining_guarded_a100_hours"] == pytest.approx(
        reforecast["remaining_base_a100_hours"] * ext.FORECAST_MULTIPLIER
    )


def test_the_first_gate_counts_one_global_thousand_across_cells() -> None:
    """The frozen gate is one reforecast for the whole screen, not one per model."""
    rows = _rows()
    civil = m1.build_cell_calls(QWEN, TASK, rows)[:600]
    nli_rows = _rows("multi_nli")
    nli = m1.build_cell_calls(QWEN, "multi_nli", nli_rows)[:400]
    events = [_ledger_event(call) for call in civil] + [_ledger_event(call) for call in nli]
    reforecast = m1.m1_reforecast(events, planned_calls=m1.M1_TOTAL_CALLS, planned_models=5)
    assert reforecast is not None
    assert reforecast["attempts"] == m1.FIRST_REFORECAST_ATTEMPTS
    assert reforecast["models_with_load_evidence"] == [QWEN]
    assert len({str(event["task_id"]) for event in events}) == 2


def test_the_reforecast_never_claims_a_model_load_without_evidence() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())
    events = [_ledger_event(call) for call in calls[: m1.FIRST_REFORECAST_ATTEMPTS]]
    blind = [{**event, "worker_load_seconds": None} for event in events]
    with pytest.raises(ValueError, match="no worker load evidence"):
        m1.m1_reforecast(blind, planned_calls=m1.M1_TOTAL_CALLS, planned_models=5)
    one_model = m1.m1_reforecast(events, planned_calls=m1.M1_TOTAL_CALLS, planned_models=5)
    assert one_model is not None
    # Four of the five models have not loaded, so four loads stay in the forecast.
    assert one_model["models_with_load_evidence"] == [QWEN]
    assert one_model["remaining_base_a100_hours"] > one_model["remaining_calls"] * (
        one_model["per_call_a100_seconds"] / 3600
    )


def _reforecast(
    remaining: float = 1.0, attempts: int = m1.FIRST_REFORECAST_ATTEMPTS
) -> dict[str, Any]:
    return {
        "gate": m1.GATE_NAME,
        "attempts": attempts,
        "remaining_base_a100_hours": remaining,
    }


def test_release_decision_holds_the_rest_of_the_screen_when_the_guard_fails() -> None:
    allowed = m1.release_decision(0.5, _reforecast(), plan_sha256="p" * 64)
    assert allowed["release_remaining_work"] is True
    assert allowed["plan_sha256"] == "p" * 64
    assert allowed["reforecast_sha256"] == sha256_json(_reforecast())
    blocked = m1.release_decision(60.0, _reforecast(20.0), plan_sha256="p" * 64)
    assert blocked["release_remaining_work"] is False
    assert blocked["reason"] == "guarded_complete_block_exceeds_extension_stop"
    stopped = m1.release_decision(ext.STOP_HOURS, _reforecast(), plan_sha256="p" * 64)
    assert stopped["reason"] == "extension_stop_reached"
    with pytest.raises(ValueError, match="first-gate reforecast"):
        m1.release_decision(0.5, {"attempts": 1000}, plan_sha256="p" * 64)


def test_a_resumed_run_re_proves_the_saved_gate_decision() -> None:
    reforecast = _reforecast()
    decision = m1.release_decision(0.5, reforecast, plan_sha256="p" * 64)
    verified = m1.verify_release_decision(
        decision, reforecast, plan_sha256="p" * 64, recorded_attempts=1200
    )
    assert verified["verified"] is True
    assert verified["attempts"] == m1.FIRST_REFORECAST_ATTEMPTS

    with pytest.raises(ValueError, match="different plan"):
        m1.verify_release_decision(
            decision, reforecast, plan_sha256="q" * 64, recorded_attempts=1200
        )
    with pytest.raises(ValueError, match="does not bind the saved reforecast"):
        m1.verify_release_decision(
            decision, _reforecast(2.0), plan_sha256="p" * 64, recorded_attempts=1200
        )
    with pytest.raises(ValueError, match="below the 1000-attempt gate"):
        m1.verify_release_decision(
            decision, reforecast, plan_sha256="p" * 64, recorded_attempts=999
        )
    with pytest.raises(ValueError, match="refused the remaining work"):
        refused = m1.release_decision(ext.STOP_HOURS, reforecast, plan_sha256="p" * 64)
        m1.verify_release_decision(
            refused, reforecast, plan_sha256="p" * 64, recorded_attempts=1200
        )
    with pytest.raises(ValueError, match="not this gate's record"):
        m1.verify_release_decision(
            {**decision, "gate": "invented"},
            reforecast,
            plan_sha256="p" * 64,
            recorded_attempts=1200,
        )
    short = _reforecast(attempts=900)
    with pytest.raises(ValueError, match="full first gate"):
        m1.verify_release_decision(
            m1.release_decision(0.5, short, plan_sha256="p" * 64),
            short,
            plan_sha256="p" * 64,
            recorded_attempts=1200,
        )
    with pytest.raises(ValueError, match="disagree on the gate size"):
        m1.verify_release_decision(
            {**decision, "attempts": 1200},
            reforecast,
            plan_sha256="p" * 64,
            recorded_attempts=1200,
        )


# --- Admission-analysis inputs ---------------------------------------------


def _complete_cell_events(
    model_id: str, task_id: str, rows: list[dict[str, Any]], *, wrong: int = 0
) -> list[dict[str, Any]]:
    calls = m1.build_cell_calls(model_id, task_id, rows)
    classes = {"civil_comments": 2, "imdb": 2, "multi_nli": 3, "ag_news": 4}[task_id]
    events = []
    for index, (call, row) in enumerate(zip(calls, rows, strict=True)):
        reference = int(row["reference"])
        label = (reference + 1) % classes if index < wrong else reference
        events.append(_ledger_event(call, label=label))
    return events


def test_summarize_cell_admits_a_capable_solo_cell() -> None:
    rows = _rows()
    summary = m1.summarize_cell(
        QWEN, TASK, rows, _complete_cell_events(QWEN, TASK, rows, wrong=300)
    )
    assert summary["technical_complete"] is True
    assert summary["terminal_calls"] == N_SCREEN
    assert summary["parse_rate"] == 1.0
    assert summary["error_rate"] == pytest.approx(300 / N_SCREEN)
    assert summary["admitted"] is True
    assert summary["candidate_status"] == "attempted"
    assert summary["selection_evidence"] == "solo_capability_and_technical_timing_only"


def test_summarize_cell_reports_an_incomplete_screen() -> None:
    rows = _rows()
    events = _complete_cell_events(QWEN, TASK, rows)[:-1]
    summary = m1.summarize_cell(QWEN, TASK, rows, events)
    assert summary["technical_complete"] is False
    assert summary["admitted"] is False
    assert summary["reasons"] == ["technical_screen_incomplete"]


def test_summarize_cell_refuses_quarantined_or_duplicate_payloads() -> None:
    rows = _rows()
    events = _complete_cell_events(QWEN, TASK, rows)
    with pytest.raises(ValueError, match="quarantined payload"):
        m1.summarize_cell(QWEN, TASK, rows, [*events[:-1], {**events[-1], "quarantined": True}])
    with pytest.raises(ValueError, match="duplicate or missing attempt"):
        m1.summarize_cell(QWEN, TASK, rows, [*events, events[0]])


def test_summarize_cell_records_an_unavailable_candidate_without_execution() -> None:
    summary = m1.summarize_cell(
        LLAMA, TASK, _rows(), [], candidate_status="technically_unavailable"
    )
    assert summary["executed"] is False
    assert summary["admitted"] is False
    assert summary["reasons"] == ["candidate_technically_unavailable"]
    assert summary["family"] == "llama"


def test_admission_inputs_cover_every_candidate_cell(tmp_path: Path) -> None:
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(
        tmp_path / "lock", seal_manifest, candidate_status={LLAMA: "technically_unavailable"}
    )
    seal = m1.verify_seal(tmp_path / "seal")
    lock = m1.verify_runtime_lock(tmp_path / "lock", seal)
    rows = _rows()
    summaries = []
    for model_id in ext.EXT_MODEL_IDS:
        status = lock["candidate_status"][model_id]
        for task_id in ext.EXT_TASK_KEYS:
            events = (
                _complete_cell_events(model_id, TASK, rows, wrong=300)
                if status == "attempted" and task_id == TASK
                else []
            )
            summaries.append(
                m1.summarize_cell(
                    model_id,
                    task_id,
                    rows if task_id == TASK else _rows(task_id),
                    events,
                    candidate_status=status,
                )
            )
    inputs = m1.admission_inputs(summaries, lock, seal=seal, meter={"cumulative_a100_hours": 1.25})
    assert len(inputs["summaries"]) == 20
    assert inputs["unavailable_models"] == [LLAMA]
    assert inputs["metered_a100_hours"] == 1.25
    assert inputs["extension_joint_outputs_seen"] is False
    assert inputs["admitted_cells"] == sorted(
        f"{model}::{TASK}" for model in inputs["attempted_models"]
    )
    assert inputs["screen_complete"] is False
    with pytest.raises(ValueError, match="all 20 candidate cells"):
        m1.admission_inputs(summaries[:-1], lock, seal=seal, meter={"cumulative_a100_hours": 1.25})


def test_admission_inputs_feed_the_sealed_admission_lock(tmp_path: Path) -> None:
    """The summaries must satisfy the sealed X1R4 admission-lock builder."""
    seal_manifest = _write_seal(tmp_path / "seal")
    _write_lock(tmp_path / "lock", seal_manifest)
    lock = m1.verify_runtime_lock(tmp_path / "lock", m1.verify_seal(tmp_path / "seal"))
    rows_by_task = {task_id: _rows(task_id) for task_id in ext.EXT_TASK_KEYS}
    summaries = [
        m1.summarize_cell(
            model_id,
            task_id,
            rows_by_task[task_id],
            _complete_cell_events(model_id, task_id, rows_by_task[task_id], wrong=300),
            candidate_status="attempted",
        )
        for model_id in ext.EXT_MODEL_IDS
        for task_id in ext.EXT_TASK_KEYS
    ]
    payload = ext.build_x1_admission_payload(
        summaries,
        [],
        lock["access_lock"],
        choose_n=lambda error_rate, rows_: {"n_items": 2048, "solo_error_rate": error_rate},
    )
    assert payload["claim_scope"]["distinct_primary_model_families"] == 5
    assert payload["extension_joint_outputs_seen"] is False


# --- Runner defaults --------------------------------------------------------


def test_runner_is_dry_run_by_default_and_binds_the_active_seal_and_lock() -> None:
    path = ROOT / "modal_extension_m1.py"
    spec = importlib.util.spec_from_file_location("modal_extension_m1", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.RUNTIME_ENV["VLLM_BATCH_INVARIANT"] == "1"
    assert runner.VLLM_VERSION == m1.EXPECTED_VLLM_VERSION
    assert runner.SEAL_ROOT == ROOT / m1.X1R4_SEAL.split("study-120h/", 1)[1]
    assert runner.LOCK_ROOT == ROOT / m1.X1R4_RUNTIME_LOCK.split("study-120h/", 1)[1]
    source = path.read_text(encoding="utf-8")
    assert "execute: bool = False" in source
    # The screen has no partial-panel mode, so no model or task selector exists.
    assert "models: str" not in source
    assert "tasks: str" not in source
    # The run lock is released on every path, including an exception.
    assert "    finally:\n        run_lock.close()\n" in source


def test_a_torn_last_line_is_tolerated_only_where_it_is_allowed(tmp_path: Path) -> None:
    """A crash can leave half a line. Receipts tolerate it; a ledger never does."""
    target = tmp_path / "receipts.jsonl"
    target.write_text('{"receipt_id": "a"}\n{"receipt_id": "b"', encoding="utf-8")
    assert m1.read_jsonl_rows(target, tolerate_partial=True) == [{"receipt_id": "a"}]
    with pytest.raises(json.JSONDecodeError):
        m1.read_jsonl_rows(target)
    assert m1.read_jsonl_rows(tmp_path / "missing.jsonl") == []


# --- Crash boundaries inside absorb_payload --------------------------------


def _absorb_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "ledger_path": tmp_path / "cells" / "qwen" / f"{TASK}.jsonl",
        "receipts_path": tmp_path / "receipts.jsonl",
        "quarantine_path": tmp_path / "quarantine.jsonl",
    }


def _chunk(
    specs: list[dict[str, Any]], *, session: str = "worker-a", elapsed: float = 100.0
) -> dict[str, Any]:
    return {
        "worker_session_id": session,
        "session_elapsed_seconds": elapsed,
        "load_seconds": 60.0,
        "host": _host(),
        "backend": _backend(QWEN),
        "events": [_event(spec) for spec in specs],
    }


def _run_absorb(
    state: dict[str, Any],
    specs: list[dict[str, Any]],
    payload: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    return m1.absorb_payload(
        state=state,
        payload=payload,
        specs=specs,
        cell_key=CELL,
        wall_seconds=0.0,
        ledger_path=paths["ledger_path"],
        receipts_path=paths["receipts_path"],
        quarantine_path=paths["quarantine_path"],
    )


def test_absorb_payload_writes_the_receipt_before_any_scientific_row(tmp_path: Path) -> None:
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:3]
    paths = _absorb_paths(tmp_path)
    state = m1.empty_meter_state()
    result = _run_absorb(state, specs, _chunk(specs), paths)
    assert len(result["accepted"]) == 3
    receipts = m1.read_jsonl_rows(paths["receipts_path"])
    assert len(receipts) == 1
    assert receipts[0]["schema_version"] == m1.RECEIPT_SCHEMA
    assert receipts[0]["metered_a100_seconds"] == pytest.approx(100.0)
    assert m1.fold_receipts(receipts)["metered_a100_seconds"] == pytest.approx(100.0)


def test_absorb_payload_replay_after_a_crash_before_the_ledger_write(tmp_path: Path) -> None:
    """The receipt survived, the rows did not. The replay adds rows and no seconds."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:3]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    paths["ledger_path"].unlink()  # the crash point: the receipt landed, the rows did not

    resumed = m1.fold_receipts(m1.read_jsonl_rows(paths["receipts_path"], tolerate_partial=True))
    before = resumed["metered_a100_seconds"]
    result = _run_absorb(resumed, specs, payload, paths)
    assert resumed["metered_a100_seconds"] == before
    assert result["metered_a100_seconds"] == 0.0
    assert len(m1.read_jsonl_rows(paths["receipts_path"])) == 1
    assert len(m1.read_jsonl_rows(paths["ledger_path"])) == 3


def test_absorb_payload_replay_after_a_torn_receipt_line(tmp_path: Path) -> None:
    """The receipt append was cut in half. The payload is not recorded, so it meters once."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    paths["receipts_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["receipts_path"].write_text('{"receipt_id": "tor', encoding="utf-8")

    resumed = m1.fold_receipts(m1.read_jsonl_rows(paths["receipts_path"], tolerate_partial=True))
    assert resumed["metered_a100_seconds"] == 0.0
    _run_absorb(resumed, specs, payload, paths)
    assert resumed["metered_a100_seconds"] == pytest.approx(100.0)


def test_absorb_payload_replay_of_a_complete_absorb_changes_nothing(tmp_path: Path) -> None:
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:4]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    state = m1.empty_meter_state()
    _run_absorb(state, specs, payload, paths)
    ledger = paths["ledger_path"].read_bytes()
    receipts = paths["receipts_path"].read_bytes()

    for _ in range(3):
        result = _run_absorb(state, specs, payload, paths)
        assert result["already_recorded"] == 4
        assert result["accepted"] == []
        assert result["metered_a100_seconds"] == 0.0
    assert paths["ledger_path"].read_bytes() == ledger
    assert paths["receipts_path"].read_bytes() == receipts
    assert state["metered_a100_seconds"] == pytest.approx(100.0)


def test_absorb_payload_meters_a_quarantined_payload_and_keeps_the_ledger_clean(
    tmp_path: Path,
) -> None:
    """Paid time is paid even when the payload is invalid. The rows go to quarantine."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    payload["backend"] = _backend(QWEN, seed=1)
    result = _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    assert result["accepted"] == []
    assert len(result["quarantined"]) == 2
    assert result["metered_a100_seconds"] == pytest.approx(100.0)
    assert m1.read_jsonl_rows(paths["ledger_path"]) == []
    assert len(m1.read_jsonl_rows(paths["quarantine_path"])) == 2
    # A quarantined row never reaches the ledger, so both calls stay pending.
    assert result["retries"] == []
    assert len(m1.pending_calls(specs, m1.read_jsonl_rows(paths["ledger_path"]))) == 2


def test_absorb_payload_counts_dispatched_specs_not_surplus_worker_rows(tmp_path: Path) -> None:
    """A corrupt payload cannot push the global gate past the dispatched calls."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:3]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    payload["events"].append(_event(specs[-1]))

    result = _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    attempts = _attempts(paths)
    quarantined = m1.read_jsonl_rows(paths["quarantine_path"])

    assert len(result["accepted"]) == 3
    assert len(quarantined) == 1
    assert quarantined[0]["counts_as_attempt"] is False
    assert len(attempts) == len(specs)
    receipt = m1.read_jsonl_rows(paths["receipts_path"])[0]
    assert receipt["attempt_ids"] == sorted(str(spec["attempt_id"]) for spec in specs)
    assert receipt["counts"]["payload_events"] == 4
    assert receipt["counts"]["dispatched_attempts"] == 3


def test_absorb_payload_counts_a_dispatched_call_when_its_result_is_missing(
    tmp_path: Path,
) -> None:
    """Every bought call reaches the gate even when the worker omits its result row."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:3]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    payload["events"] = payload["events"][:2]

    result = _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    attempts = _attempts(paths)

    assert len(attempts) == len(specs)
    assert len(result["accepted"]) == 3
    missing = next(row for row in result["accepted"] if row["call_id"] == specs[-1]["call_id"])
    assert missing["status"] == "worker_failed"
    assert result["retries"][0]["call_id"] == specs[-1]["call_id"]


def test_committed_chunk_recovers_only_its_exact_dispatched_specs() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())
    first = calls[20]
    retry = {
        **calls[21],
        "attempt": 2,
        "attempt_id": f"{calls[21]['call_id']}::attempt::2",
        "previous_attempt_id": f"{calls[21]['call_id']}::attempt::1",
        "previous_status": "worker_failed",
    }
    manifest = m1.dispatch_manifest([first, retry])
    recovered = m1.recover_dispatched_specs(calls, manifest)

    assert [row["attempt_id"] for row in recovered] == [first["attempt_id"], retry["attempt_id"]]
    assert recovered[1]["previous_attempt_id"] == retry["previous_attempt_id"]
    assert recovered[1]["messages"] == calls[21]["messages"]
    assert all(row["task_id"] == TASK for row in recovered)


def test_committed_chunk_recovery_fails_closed_without_a_valid_dispatch_manifest() -> None:
    calls = m1.build_cell_calls(QWEN, TASK, _rows())
    with pytest.raises(ValueError, match="lacks its dispatched-attempt manifest"):
        m1.recover_dispatched_specs(calls, None)
    with pytest.raises(ValueError, match="unknown call"):
        m1.recover_dispatched_specs(
            calls,
            [{"call_id": "unknown", "attempt_id": "unknown::attempt::1", "attempt": 1}],
        )
    duplicate = m1.dispatch_manifest([calls[0], calls[0]])
    with pytest.raises(ValueError, match="repeats a dispatched attempt"):
        m1.recover_dispatched_specs(calls, duplicate)
    bad_retry = [
        {
            "call_id": calls[0]["call_id"],
            "attempt_id": f"{calls[0]['call_id']}::attempt::2",
            "attempt": 2,
            "previous_attempt_id": "wrong",
            "previous_status": "worker_failed",
        }
    ]
    with pytest.raises(ValueError, match="does not link"):
        m1.recover_dispatched_specs(calls, bad_retry)


def test_absorb_payload_refuses_to_change_a_recorded_first_attempt(tmp_path: Path) -> None:
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    paths = _absorb_paths(tmp_path)
    state = m1.empty_meter_state()
    _run_absorb(state, specs, _chunk(specs), paths)
    tampered = _chunk(specs)
    tampered["events"][0]["first_response"] = _response(str(specs[0]["doc_id"]), 1)
    with pytest.raises(ValueError, match="would change recorded attempt"):
        _run_absorb(state, specs, tampered, paths)


# --- Recovered rows must carry the receipt's own seconds -------------------


def test_recovered_ledger_rows_carry_the_original_metered_seconds(tmp_path: Path) -> None:
    """A replay writes the rows the crash lost, timed by the receipt of that payload."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:4]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    first = _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    original = [row["metered_a100_seconds"] for row in first["accepted"]]
    assert original == [pytest.approx(25.0)] * 4
    paths["ledger_path"].unlink()  # the crash point: the receipt landed, the rows did not

    resumed = m1.fold_receipts(m1.read_jsonl_rows(paths["receipts_path"], tolerate_partial=True))
    result = _run_absorb(resumed, specs, payload, paths)
    assert result["metered_a100_seconds"] == 0.0
    assert result["per_call_a100_seconds"] == pytest.approx(25.0)
    recovered = m1.read_jsonl_rows(paths["ledger_path"])
    assert [row["metered_a100_seconds"] for row in recovered] == original
    assert [row["cumulative_a100_hours"] for row in recovered] == [
        first["accepted"][0]["cumulative_a100_hours"]
    ] * 4
    # The receipt file, which is the only authority, still holds 100 seconds once.
    assert resumed["metered_a100_seconds"] == pytest.approx(100.0)
    assert len(m1.read_jsonl_rows(paths["receipts_path"])) == 1


def test_the_reforecast_reads_the_same_hours_after_a_recovery(tmp_path: Path) -> None:
    """The ledger the reforecast reads must agree with the receipt file after a replay."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[: m1.FIRST_REFORECAST_ATTEMPTS]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs, elapsed=4000.0)
    clean = _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    assert len(clean["accepted"]) == m1.FIRST_REFORECAST_ATTEMPTS
    intact = m1.m1_reforecast(
        m1.read_jsonl_rows(paths["ledger_path"]), planned_calls=40960, planned_models=5
    )
    assert intact is not None

    paths["ledger_path"].unlink()
    resumed = m1.fold_receipts(m1.read_jsonl_rows(paths["receipts_path"], tolerate_partial=True))
    _run_absorb(resumed, specs, payload, paths)
    recovered = m1.m1_reforecast(
        m1.read_jsonl_rows(paths["ledger_path"]), planned_calls=40960, planned_models=5
    )
    assert recovered is not None
    assert recovered["per_call_a100_seconds"] == pytest.approx(intact["per_call_a100_seconds"])
    assert recovered["remaining_base_a100_hours"] == pytest.approx(
        intact["remaining_base_a100_hours"]
    )
    # Without the receipt lookup these rows would read zero seconds.
    assert recovered["per_call_a100_seconds"] > 0.0


def test_partial_recovery_times_the_missing_rows_like_the_rows_that_survived(
    tmp_path: Path,
) -> None:
    """Only some rows of a receipted payload are lost. The replay writes the rest alike."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:4]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    first = _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    kept = m1.read_jsonl_rows(paths["ledger_path"])[:2]
    paths["ledger_path"].unlink()
    m1.append_jsonl(paths["ledger_path"], kept)  # two rows survived, two were lost

    resumed = m1.fold_receipts(m1.read_jsonl_rows(paths["receipts_path"], tolerate_partial=True))
    result = _run_absorb(resumed, specs, payload, paths)
    assert result["already_recorded"] == 2
    assert len(result["accepted"]) == 2
    rows = m1.read_jsonl_rows(paths["ledger_path"])
    assert len(rows) == 4
    assert [row["metered_a100_seconds"] for row in rows] == [pytest.approx(25.0)] * 4
    assert [row["attempt_id"] for row in rows] == [str(spec["attempt_id"]) for spec in specs]
    # The surviving first responses are untouched.
    assert rows[0]["first_response"] == first["accepted"][0]["first_response"]
    assert resumed["metered_a100_seconds"] == pytest.approx(100.0)


def test_a_replay_refuses_a_receipt_that_covers_another_cell(tmp_path: Path) -> None:
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    tampered = m1.read_jsonl_rows(paths["receipts_path"])
    tampered[0]["cell_key"] = f"{GRANITE}::{TASK}"
    paths["receipts_path"].write_text("", encoding="utf-8")
    m1.append_jsonl(paths["receipts_path"], tampered)
    resumed = m1.fold_receipts(m1.read_jsonl_rows(paths["receipts_path"]))
    paths["ledger_path"].unlink()
    with pytest.raises(ValueError, match="belongs to cell"):
        _run_absorb(resumed, specs, payload, paths)


def test_a_replay_refuses_a_receipt_that_covers_other_attempts(tmp_path: Path) -> None:
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    paths = _absorb_paths(tmp_path)
    payload = _chunk(specs)
    _run_absorb(m1.empty_meter_state(), specs, payload, paths)
    tampered = m1.read_jsonl_rows(paths["receipts_path"])
    tampered[0]["attempt_ids"] = ["extx1::m1::other::attempt::1"]
    paths["receipts_path"].write_text("", encoding="utf-8")
    m1.append_jsonl(paths["receipts_path"], tampered)
    resumed = m1.fold_receipts(m1.read_jsonl_rows(paths["receipts_path"]))
    paths["ledger_path"].unlink()
    with pytest.raises(ValueError, match="covers different attempts"):
        _run_absorb(resumed, specs, payload, paths)


# --- Restart safety of the receipt file ------------------------------------


def test_repair_torn_receipt_tail_drops_only_unterminated_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipts.jsonl"
    assert m1.repair_torn_receipt_tail(path) == 0  # a missing file is already clean
    path.write_text('{"receipt_id": "a"}\n', encoding="utf-8")
    assert m1.repair_torn_receipt_tail(path) == 0  # a clean file is left alone
    path.write_text('{"receipt_id": "a"}\n{"receipt_id": "b"', encoding="utf-8")
    assert m1.repair_torn_receipt_tail(path) == 18
    assert m1.read_jsonl_rows(path) == [{"receipt_id": "a"}]


def test_repair_torn_receipt_tail_fails_closed_on_a_malformed_kept_line(tmp_path: Path) -> None:
    path = tmp_path / "receipts.jsonl"
    path.write_text('{"receipt_id": "a"\n{"receipt_id": "b"', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        m1.repair_torn_receipt_tail(path)
    assert path.read_text(encoding="utf-8").endswith('{"receipt_id": "b"')


def test_read_jsonl_rows_fails_closed_on_a_complete_malformed_last_line(tmp_path: Path) -> None:
    """A terminated line that does not parse is corruption, not a torn write."""
    path = tmp_path / "receipts.jsonl"
    path.write_text('{"receipt_id": "a"}\n{"receipt_id"\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        m1.read_jsonl_rows(path, tolerate_partial=True)


def test_absorb_after_a_torn_receipt_line_leaves_a_file_that_rereads(tmp_path: Path) -> None:
    """The next append must not join itself to the torn bytes of the last crash."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:2]
    paths = _absorb_paths(tmp_path)
    good = _chunk(specs, session="worker-a", elapsed=100.0)
    _run_absorb(m1.empty_meter_state(), specs, good, paths)
    with paths["receipts_path"].open("a", encoding="utf-8") as handle:
        handle.write('{"receipt_id": "torn", "metered_a100_s')  # the crash point

    resumed = m1.fold_receipts(m1.read_jsonl_rows(paths["receipts_path"], tolerate_partial=True))
    assert resumed["receipts"] == 1
    later_specs = m1.build_cell_calls(QWEN, TASK, _rows())[2:4]
    later = _chunk(later_specs, session="worker-a", elapsed=180.0)
    _run_absorb(resumed, later_specs, later, paths)

    # The file rereads without tolerance, so a restart of the whole run is safe.
    rows = m1.read_jsonl_rows(paths["receipts_path"])
    assert len(rows) == 2
    assert m1.fold_receipts(rows)["metered_a100_seconds"] == pytest.approx(180.0)
    assert resumed["metered_a100_seconds"] == pytest.approx(180.0)
    assert len(m1.read_jsonl_rows(paths["ledger_path"])) == 4


# --- The global gate counts every actual attempt ---------------------------


def _receipt_order(paths: dict[str, Path]) -> list[str]:
    return [
        str(row["receipt_id"])
        for row in m1.read_jsonl_rows(paths["receipts_path"], tolerate_partial=True)
    ]


def _attempts(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return m1.actual_attempts(
        m1.read_jsonl_rows(paths["ledger_path"]),
        m1.read_jsonl_rows(paths["quarantine_path"]),
        receipt_order=_receipt_order(paths),
    )


def _mixed_chunk(
    specs: list[dict[str, Any]], *, bad: int, session: str, elapsed: float
) -> dict[str, Any]:
    """One chunk whose first `bad` events are invalid and must be quarantined."""
    payload = _chunk(specs, session=session, elapsed=elapsed)
    for event in payload["events"][:bad]:
        event["status"] = "not_a_status"
    return payload


def test_quarantined_attempts_count_toward_the_global_first_gate(tmp_path: Path) -> None:
    """Accepted and quarantined attempts jointly stop the gate at exactly 1,000."""
    gate = m1.FIRST_REFORECAST_ATTEMPTS
    specs = m1.build_cell_calls(QWEN, TASK, _rows())
    paths = _absorb_paths(tmp_path)
    state = m1.empty_meter_state()

    # Every chunk loses a quarter of its attempts to quarantine.
    chunk, elapsed, taken = 100, 0.0, 0
    while taken < gate:
        take = min(chunk, gate - len(_attempts(paths)))
        assert take > 0
        group = specs[taken : taken + take]
        elapsed += 40.0
        _run_absorb(
            state, group, _mixed_chunk(group, bad=take // 4, session="w", elapsed=elapsed), paths
        )
        taken += take

    rows = _attempts(paths)
    assert len(rows) == gate
    ledger = m1.read_jsonl_rows(paths["ledger_path"])
    quarantined = m1.read_jsonl_rows(paths["quarantine_path"])
    assert len(ledger) == 750
    assert len(quarantined) == 250
    # The ledger alone would have let the runner buy 250 more real attempts.
    assert len(ledger) < gate
    assert len(rows) == len(ledger) + len(quarantined)
    assert len({m1.attempt_identity(row) for row in rows}) == gate


def test_the_gate_sample_is_in_execution_order_and_meters_once(tmp_path: Path) -> None:
    """The first 1,000 actual attempts arrive in receipt order and sum to the receipts."""
    gate = m1.FIRST_REFORECAST_ATTEMPTS
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:gate]
    paths = _absorb_paths(tmp_path)
    state = m1.empty_meter_state()
    for index in range(0, gate, 250):
        group = specs[index : index + 250]
        _run_absorb(
            state,
            group,
            _mixed_chunk(group, bad=10, session="w", elapsed=float(index) + 250.0),
            paths,
        )

    rows = _attempts(paths)
    assert len(rows) == gate
    assert [row["position"] for row in rows] == list(range(1, gate + 1))
    assert [row["receipt_id"] for row in rows] == sorted(
        (row["receipt_id"] for row in rows), key=_receipt_order(paths).index
    )
    reforecast = m1.m1_reforecast(rows, planned_calls=40960, planned_models=5)
    assert reforecast is not None
    assert reforecast["attempts"] == gate
    assert reforecast["quarantined_attempts"] == 40
    # The sampled seconds equal the receipts exactly: no attempt is counted twice.
    assert reforecast["all_in_a100_seconds"] == pytest.approx(state["metered_a100_seconds"])
    assert reforecast["all_in_a100_seconds"] == pytest.approx(1000.0)


def test_a_ledger_only_sample_under_times_the_gate(tmp_path: Path) -> None:
    """Proof the defect is real: ledger rows alone lose the quarantined seconds."""
    gate = m1.FIRST_REFORECAST_ATTEMPTS
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:gate]
    paths = _absorb_paths(tmp_path)
    state = m1.empty_meter_state()
    _run_absorb(state, specs, _mixed_chunk(specs, bad=200, session="w", elapsed=1000.0), paths)

    ledger_only = m1.m1_reforecast(
        m1.read_jsonl_rows(paths["ledger_path"]),
        planned_calls=40960,
        planned_models=5,
        attempts=800,
    )
    whole = m1.m1_reforecast(_attempts(paths), planned_calls=40960, planned_models=5)
    assert ledger_only is not None and whole is not None
    assert ledger_only["all_in_a100_seconds"] == pytest.approx(800.0)
    assert whole["all_in_a100_seconds"] == pytest.approx(1000.0)
    assert whole["all_in_a100_seconds"] > ledger_only["all_in_a100_seconds"]


def test_a_replayed_quarantined_attempt_is_not_counted_twice(tmp_path: Path) -> None:
    """A resumed run re-absorbs a committed chunk. Its rejected rows stay one attempt."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:8]
    paths = _absorb_paths(tmp_path)
    payload = _mixed_chunk(specs, bad=3, session="w", elapsed=80.0)
    state = m1.empty_meter_state()
    _run_absorb(state, specs, payload, paths)
    assert len(_attempts(paths)) == 8

    for _ in range(3):
        resumed = m1.fold_receipts(
            m1.read_jsonl_rows(paths["receipts_path"], tolerate_partial=True)
        )
        _run_absorb(resumed, specs, payload, paths)
    # The quarantine file grew, because it is append-only and never deduplicated.
    assert len(m1.read_jsonl_rows(paths["quarantine_path"])) == 12
    # The gate still sees eight attempts and the original 80 seconds.
    rows = _attempts(paths)
    assert len(rows) == 8
    assert sum(float(row["metered_a100_seconds"]) for row in rows) == pytest.approx(80.0)


def test_a_call_rerun_after_a_quarantine_is_a_second_actual_attempt(tmp_path: Path) -> None:
    """A quarantined call has no result, so it runs again. That run is its own attempt."""
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:4]
    paths = _absorb_paths(tmp_path)
    state = m1.empty_meter_state()
    _run_absorb(state, specs, _mixed_chunk(specs, bad=4, session="w", elapsed=40.0), paths)
    assert m1.read_jsonl_rows(paths["ledger_path"]) == []
    assert len(_attempts(paths)) == 4

    # The same four calls are still pending, so the worker runs them again.
    pending = m1.pending_calls(specs, m1.read_jsonl_rows(paths["ledger_path"]))
    assert len(pending) == 4
    _run_absorb(state, specs, _chunk(specs, session="w", elapsed=90.0), paths)
    rows = _attempts(paths)
    assert len(rows) == 8  # four rejected attempts, then four that were recorded
    assert len(m1.read_jsonl_rows(paths["ledger_path"])) == 4
    assert sum(float(row["metered_a100_seconds"]) for row in rows) == pytest.approx(90.0)


def test_quarantined_rows_never_reach_an_admission_statistic(tmp_path: Path) -> None:
    """The gate counts a quarantined attempt. The science still refuses it."""
    rows = _rows()
    specs = m1.build_cell_calls(QWEN, TASK, rows)
    paths = _absorb_paths(tmp_path)
    state = m1.empty_meter_state()
    for index in range(0, N_SCREEN, 512):
        group = specs[index : index + 512]
        _run_absorb(
            state,
            group,
            _mixed_chunk(group, bad=0 if index else 12, session="w", elapsed=float(index) + 512.0),
            paths,
        )
    assert len(m1.read_jsonl_rows(paths["quarantine_path"])) == 12
    summary = m1.summarize_cell(QWEN, TASK, rows, m1.read_jsonl_rows(paths["ledger_path"]))
    assert summary["terminal_calls"] == N_SCREEN - 12
    assert summary["technical_complete"] is False
    assert summary["admitted"] is False
    # Those twelve attempts were still bought, and the gate sees them.
    assert len(_attempts(paths)) == N_SCREEN


def test_verify_release_decision_counts_actual_attempts(tmp_path: Path) -> None:
    """The saved gate is re-proved against ledger rows plus quarantined attempts."""
    gate = m1.FIRST_REFORECAST_ATTEMPTS
    specs = m1.build_cell_calls(QWEN, TASK, _rows())[:gate]
    paths = _absorb_paths(tmp_path)
    state = m1.empty_meter_state()
    _run_absorb(state, specs, _mixed_chunk(specs, bad=150, session="w", elapsed=1000.0), paths)
    reforecast = m1.m1_reforecast(_attempts(paths), planned_calls=40960, planned_models=5)
    assert reforecast is not None
    decision = m1.release_decision(
        float(state["cumulative_a100_hours"]), reforecast, plan_sha256="plan-sha"
    )

    ledger_rows = len(m1.read_jsonl_rows(paths["ledger_path"]))
    assert ledger_rows == gate - 150
    with pytest.raises(ValueError, match="below the 1000-attempt gate"):
        m1.verify_release_decision(
            decision, reforecast, plan_sha256="plan-sha", recorded_attempts=ledger_rows
        )
    verified = m1.verify_release_decision(
        decision, reforecast, plan_sha256="plan-sha", recorded_attempts=len(_attempts(paths))
    )
    assert verified["verified"] is True
    assert verified["recorded_attempts"] == gate


def test_the_runner_gate_and_reforecast_read_actual_attempts() -> None:
    """The runner must not hand the gate a ledger-only count."""
    source = (ROOT / "modal_extension_m1.py").read_text(encoding="utf-8")
    assert "return len(attempt_rows())" in source
    assert "reforecast = m1.m1_reforecast(\n            attempt_rows()," in source
    assert "recorded_attempts=recorded_attempts()," in source
    assert "m1.actual_attempts(" in source
    assert '"dispatched_attempts": dispatch_manifest(specs)' in source
    assert "m1.recover_dispatched_specs(" in source
