"""CLI tests: stage wiring, error exits, and the exit-gate verdict.

The exit gate is the file that decides whether Step 0 is complete, so it is tested
against both an empty directory and a full stand-in freeze.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from step0 import cli, integrity, lock
from step0.hashing import RefusedOverwriteError, write_json, write_jsonl


def test_every_stage_is_reachable_from_the_parser() -> None:
    parser = cli._parser()
    stages = {
        "simulate": cli.cmd_simulate,
        "freeze": cli.cmd_freeze,
        "audit-pins": cli.cmd_pins,
        "freeze-data": cli.cmd_freeze_data,
        "assign": cli.cmd_assign,
        "repro-gatea": cli.cmd_repro,
        "venue": cli.cmd_venue,
        "checks": cli.cmd_checks,
        "seal": cli.cmd_seal,
        "verify": cli.cmd_verify,
        "status": cli.cmd_status,
    }
    for name, func in stages.items():
        args = parser.parse_args([name])
        assert args.func is func, name


class _FixedParser:
    """A parser stub, so an exit path can be tested without touching a real freeze."""

    def __init__(self, args: Any) -> None:
        self._args = args

    def parse_args(self, argv: Any = None) -> Any:
        return self._args


def _fails_with(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def boom(args: Any) -> int:
        raise error

    args = cli._parser().parse_args(["venue"])
    args.func = boom
    monkeypatch.setattr(cli, "_parser", lambda: _FixedParser(args))


def test_an_unsealed_freeze_directory_fails_verification(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["verify", "--stamp", "nope-does-not-exist"]) == 1
    assert "never sealed" in capsys.readouterr().out


def test_a_refused_overwrite_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fails_with(monkeypatch, RefusedOverwriteError("already frozen"))
    assert cli.main(["venue"]) == 2
    assert "already frozen" in capsys.readouterr().err


def test_a_missing_input_file_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fails_with(monkeypatch, FileNotFoundError("LATEST is missing"))
    assert cli.main(["venue"]) == 2
    assert "LATEST is missing" in capsys.readouterr().err


def test_an_invalid_freeze_resolution_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fails_with(monkeypatch, ValueError("protocol and simulation disagree"))
    assert cli.main(["venue"]) == 2
    assert "protocol and simulation disagree" in capsys.readouterr().err


def test_a_blocker_is_recorded_as_machine_readable_incomplete(tmp_path: Path) -> None:
    cli._record_blocker(tmp_path, "data-demo", "the dataset repo is gated")
    stored = json.loads((tmp_path / "BLOCKER-data-demo.json").read_text())
    assert stored["step0_complete"] is False
    assert stored["stage"] == "data-demo"
    assert "gated" in stored["blocker"]
    assert "No new experimental model labels" in stored["no_new_experimental_model_labels"]


# --------------------------------------------------------------------------------------
# Exit gate
# --------------------------------------------------------------------------------------


def test_an_empty_directory_is_incomplete_on_every_item(tmp_path: Path) -> None:
    status = cli.exit_gate(tmp_path)
    assert status["verdict"] == "INCOMPLETE"
    # every item fails except "no recorded blocker": an empty directory records none
    assert status["n_failed"] == status["n_items"] - 1
    assert {i["item"] for i in status["items"] if i["ok"]} == {"no recorded blocker"}
    text = cli.render_status(status)
    assert "**INCOMPLETE**" in text
    assert "No new experimental model labels" in text


def build_complete_freeze(root: Path, *, unresolved: bool = False) -> Path:
    """A stand-in freeze with one artifact per exit-gate item."""
    freeze_dir = root / "20260912T000000Z"
    protocol: dict[str, Any] = {
        "content_hash": "a" * 64,
        "freeze": {"stamp_utc": "20260912T000000Z"},
        "panel": {"serving": {"engine_version_pin": "vllm==0.28.0"}},
        "threshold": "SET_BY_SIMULATION" if unresolved else 0.02,
    }
    write_json(freeze_dir / "protocol.lock.json", protocol)
    (freeze_dir / "PROTOCOL.md").write_text("protocol\n", encoding="utf-8")
    (freeze_dir / "protocol.lock.toml").write_text("x = 1\n", encoding="utf-8")
    write_json(
        freeze_dir / "simulations" / "results.json",
        {
            "frozen_conclusions": {
                "error_information_threshold_error_rate": 0.02,
                "error_information_threshold_min_solo_errors": 41,
            }
        },
    )
    for task in ("civil_comments", "ag_news", "multi_nli"):
        write_jsonl(
            freeze_dir / "data" / task / "items.jsonl",
            ({"doc_id": str(i)} for i in range(2_048)),
        )
    write_json(freeze_dir / "pins" / "model_pins.json", {"ok": True, "blockers": []})
    write_json(freeze_dir / "pins" / "engine_pin.json", {"ok": True, "pinned_wheel_found": True})
    for cell_spec in lock.PANEL["cells"]:
        cell = str(cell_spec["cell_id"])
        expected = {"exact_context": 6_144, "joint": 384, "solo": 2_048}
        filler = lock.CONDITIONS["filler"]["cells"].get(cell)
        if filler:
            expected["filler"] = int(filler["calls"])
        serial = lock.CONDITIONS["serial_serving"]["cells"].get(cell)
        if serial:
            expected["serial_joint"] = int(serial["joint_calls"])
            expected["serial_exact"] = int(serial["joint_calls"]) * 16
        total = sum(expected.values())
        cell_dir = freeze_dir / "assignments" / cell
        write_jsonl(cell_dir / "assignment_matrix.jsonl", ({"call": i} for i in range(384)))
        write_jsonl(cell_dir / "job_skeletons.jsonl", ({"call_id": i} for i in range(total)))
        write_json(
            cell_dir / "meta.json",
            {
                "planned_calls_by_condition": expected,
                "planned_calls_total": total,
                "unique_call_ids": total,
                "every_arm_appears_in_every_block": True,
            },
        )
        write_json(
            cell_dir / "balance.json",
            {
                "appearances_per_item": {"all_equal": True},
                "distinct_slots_per_item": {"all_distinct": True},
                "one_item_per_stratum_per_call": True,
            },
        )
    write_json(
        freeze_dir / "repro" / "repro.json",
        {"ok": True, "historical_artifact_unchanged": True, "n_mismatched": 0},
    )
    write_json(
        freeze_dir / "venue" / "venue.json",
        {"consistency_checks": {"page_limit_matches_protocol": True}},
    )
    write_json(
        freeze_dir / "checks" / "RESULTS.json",
        {"all_passed": True, "checks": {"pytest": {"returncode": 0}}},
    )
    integrity.seal(freeze_dir)
    return freeze_dir


def test_a_full_freeze_passes_every_exit_gate_item(tmp_path: Path) -> None:
    status = cli.exit_gate(build_complete_freeze(tmp_path))
    failed = [item for item in status["items"] if not item["ok"]]
    assert failed == []
    assert status["verdict"] == "COMPLETE"
    assert "**COMPLETE**" in cli.render_status(status)


def test_an_unresolved_design_field_fails_the_gate(tmp_path: Path) -> None:
    status = cli.exit_gate(build_complete_freeze(tmp_path, unresolved=True))
    assert status["verdict"] == "INCOMPLETE"
    failed = {item["item"] for item in status["items"] if not item["ok"]}
    assert failed == {"no unresolved design field"}


def test_a_recorded_blocker_fails_the_gate_even_when_artifacts_exist(tmp_path: Path) -> None:
    freeze_dir = build_complete_freeze(tmp_path)
    write_json(freeze_dir / "BLOCKER-venue.json", {"step0_complete": False}, allow_existing=True)
    status = cli.exit_gate(freeze_dir)
    assert status["verdict"] == "INCOMPLETE"
    failed = {item["item"] for item in status["items"] if not item["ok"]}
    assert "no recorded blocker" in failed
    assert "freeze sealed and integrity verified" in failed


def test_a_missing_frame_fails_the_sample_frame_item(tmp_path: Path) -> None:
    freeze_dir = tmp_path / "20260912T000000Z"
    write_jsonl(freeze_dir / "data" / "civil_comments" / "items.jsonl", [{"doc_id": "x"}])
    status = cli.exit_gate(freeze_dir)
    frames = next(i for i in status["items"] if i["item"] == "sample frames materialised")
    assert not frames["ok"]
    assert "civil_comments=1" in frames["detail"]
