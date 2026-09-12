"""End-to-end smoke, with the network blocked at the socket layer."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gatek.cli import DEFAULT_SMOKE_CONFIG, main
from gatek.config import load_config
from gatek.planning import load_jobs
from gatek.runner import execute_job, make_backend


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("the smoke path must not open a socket")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def test_smoke_runs_offline_and_writes_every_artifact(tmp_path: Path, no_network: None) -> None:
    code = main(["smoke", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)])
    assert code == 0
    for rel in (
        "synthetic/manifest.json",
        "synthetic/data.npz",
        "synthetic/rows.jsonl",
        "plan/jobs.jsonl",
        "run/results.jsonl",
        "analysis/analysis.json",
        "analysis/REPORT.md",
    ):
        assert (tmp_path / rel).exists(), rel


def test_smoke_is_resumable(tmp_path: Path, no_network: None) -> None:
    assert main(["smoke", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)]) == 0
    first = (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8")
    assert main(["smoke", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8") == first
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["n_written"] == 0
    assert summary["n_already_done"] > 0


def test_smoke_refuses_a_non_mock_backend(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text(
        DEFAULT_SMOKE_CONFIG.read_text(encoding="utf-8").replace(
            'backend = "mock"', 'backend = "transformers"'
        ),
        encoding="utf-8",
    )
    assert main(["smoke", "--config", str(cfg), "--out", str(tmp_path / "out")]) == 2


def test_prepare_openml_refuses_without_the_download_flag(tmp_path: Path, no_network: None) -> None:
    assert main(["prepare-openml", "--out", str(tmp_path)]) == 2


def test_analysis_records_the_terminology(tmp_path: Path, no_network: None) -> None:
    assert main(["smoke", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)]) == 0
    analysis = json.loads((tmp_path / "analysis" / "analysis.json").read_text(encoding="utf-8"))
    assert "distance_concentrated" in analysis["terminology"]
    assert "lower" in analysis["terminology"]["distance_concentrated"]
    assert "higher" in analysis["terminology"]["distance_deconcentrated"]


def test_anchor_phase_runs_only_sparse_2d_jobs(tmp_path: Path, no_network: None) -> None:
    assert (
        main(["prepare-synthetic", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)])
        == 0
    )
    assert main(["plan", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)]) == 0
    assert (
        main(
            [
                "run",
                "--config",
                str(DEFAULT_SMOKE_CONFIG),
                "--out",
                str(tmp_path),
                "--phase",
                "anchor",
            ]
        )
        == 0
    )
    rows = [
        json.loads(line) for line in (tmp_path / "run" / "results.jsonl").read_text().splitlines()
    ]
    assert rows
    assert {(row["arm"], row["dim"], row["job_type"]) for row in rows} == {
        ("sparse_zero_pad", 2, "order_random")
    }
    assert main(["analyze", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)]) == 0
    analysis = json.loads((tmp_path / "analysis" / "analysis.json").read_text())
    assert analysis["anchor"]["n"] == len(rows)
    assert analysis["gate"]["verdict"] == "insufficient_results"


def test_single_job_execution_matches_runner_record(tmp_path: Path, no_network: None) -> None:
    assert (
        main(["prepare-synthetic", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)])
        == 0
    )
    assert main(["plan", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)]) == 0
    cfg = load_config(DEFAULT_SMOKE_CONFIG)
    npz = __import__("numpy").load(tmp_path / "synthetic" / "data.npz")
    record = execute_job(load_jobs(tmp_path)[0], cfg, make_backend(cfg), npz)
    assert record["job_id"] == load_jobs(tmp_path)[0].job_id
    assert record["backend"] == "mock"
    assert record["parse_status"] == "ok"


def test_tokenizer_preflight_counts_frozen_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTokenizer:
        init_kwargs = {"_commit_hash": "fake-commit"}
        chat_template = "fake-template"

        def apply_chat_template(
            self, messages: object, **kwargs: object
        ) -> dict[str, list[list[int]]]:
            del messages, kwargs
            return {"input_ids": [[0] * 32], "attention_mask": [[1] * 32]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            del args, kwargs
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    )
    assert (
        main(["prepare-synthetic", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)])
        == 0
    )
    assert main(["plan", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)]) == 0
    assert (
        main(["preflight-tokenizer", "--config", str(DEFAULT_SMOKE_CONFIG), "--out", str(tmp_path)])
        == 0
    )
    report = json.loads((tmp_path / "plan" / "tokenizer_preflight.json").read_text())
    assert report["passed"] is True
    assert report["max_tokens"] == 32
