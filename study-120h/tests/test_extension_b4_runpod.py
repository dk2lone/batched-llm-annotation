"""Runpod transport checks use fake Pods/backends and never contact Runpod."""

from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import pytest

from freezeb import extension as ext
from freezeb import extension_b4 as b4
from step0.hashing import sha256_json

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))

import b4_runpod_worker as remote  # noqa: E402
import extension_b4_runpod as transport  # noqa: E402


def configuration() -> dict[str, Any]:
    return {
        "schema_version": "b4-runpod-config-v2",
        "account_id": "user_cpu",
        "cloud_type": "SECURE",
        "gpu_type_id": "NVIDIA A100-SXM4-80GB",
        "image": "example/runtime@sha256:" + "a" * 64,
        "data_center_id": "US-MD-1",
        "network_volumes": {cell: f"volume{i}" for i, cell in enumerate(b4.CELLS)},
        "watchdog": {
            "repository": "dk2lone/batched-llm-annotation",
            "workflow": "b4-pod-watchdog.yml",
            "ref": "main",
            "max_target_seconds": 18000,
        },
    }


def budget(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "runpod_config_sha256": sha256_json(config),
        "runpod_storage_reserve_usd": 5,
        "runpod_preflight": {
            "runtime_smoke_passed": True,
            "four_single_gpu_capacity_verified": True,
            "model_caches_verified": True,
            "balance_verified": True,
            "ssh_verified": True,
            "watchdog_cpu_test_passed": True,
            "auto_pay_disabled_verified": True,
            "evidence_sha256": "b" * 64,
        },
    }


def test_cli_uses_live_pod_api_without_removed_termination_flag(monkeypatch: Any) -> None:
    calls: list[list[str]] = []

    def fake_command(argv: list[str], timeout: float) -> bytes:
        calls.append(argv)
        return b'{"id":"fake-pod"}'

    monkeypatch.setattr(transport, "command", fake_command)
    assert transport.cli(["pod", "create"]) == {"id": "fake-pod"}
    assert calls == [["runpodctl", "pod", "create"]]


@pytest.mark.parametrize(
    "fault",
    ["none", "community", "40gb", "tag", "same-volume", "binding", "smoke", "storage", "account"],
)
def test_configuration_fails_closed(fault: str) -> None:
    config = configuration()
    record = budget(config)
    if fault == "community":
        config["cloud_type"] = "COMMUNITY"
    elif fault == "40gb":
        config["gpu_type_id"] = "NVIDIA A100-SXM4-40GB"
    elif fault == "tag":
        config["image"] = "vllm:latest"
    elif fault == "same-volume":
        config["network_volumes"] = dict.fromkeys(b4.CELLS, "shared")
    elif fault == "binding":
        record["runpod_config_sha256"] = "wrong"
    elif fault == "smoke":
        record["runpod_preflight"]["runtime_smoke_passed"] = False
    elif fault == "storage":
        record["runpod_storage_reserve_usd"] = float("nan")
    elif fault == "account":
        config["account_id"] = "other"
    if fault == "none":
        transport.verify_config(config, record, "user_cpu")
    else:
        with pytest.raises(ValueError):
            transport.verify_config(config, record, "user_cpu")


def test_configuration_accepts_a100_pcie_80gb() -> None:
    config = configuration()
    config["gpu_type_id"] = "NVIDIA A100 80GB PCIe"
    transport.verify_config(config, budget(config), "user_cpu")


def test_imports_and_default_cli_are_cpu_only(tmp_path: Path, monkeypatch: Any) -> None:
    assert "vllm" not in sys.modules
    spec = importlib.util.spec_from_file_location("runpod_b4_cpu", STUDY / "runpod_extension_b4.py")
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    plan = {"plan_sha256": "cpu", "gpu_type_id": "NVIDIA A100-SXM4-80GB", "first_calls": 83200}
    monkeypatch.setattr(b4, "build_plan", lambda *args: plan)
    monkeypatch.setattr(transport, "cli", lambda *args: pytest.fail("remote call on CPU path"))
    monkeypatch.setattr(sys, "argv", ["runner", "--output", str(tmp_path)])
    runner.main()
    assert json.loads((tmp_path / "PLAN-400.json").read_text()) == plan
    monkeypatch.setattr(sys, "argv", ["runner", "--output", str(tmp_path), "--execute"])
    with pytest.raises(ValueError, match="execution requires"):
        runner.main()


@pytest.mark.parametrize("mode", ["success", "cancel", "timeout", "secret-error"])
def test_bounded_commands_and_redacted_errors(mode: str) -> None:
    stop = Event()
    if mode == "cancel":
        stop.set()
    if mode == "success":
        assert transport.command([sys.executable, "-c", "print('ok')"], 2) == b"ok\n"
        return
    if mode == "secret-error":
        argv = [sys.executable, "-c", "import sys;sys.stderr.write('sensitive-value');sys.exit(1)"]
        with pytest.raises(transport.CommandError) as exc:
            transport.command(argv, 2)
        assert "sensitive-value" not in str(exc.value)
        return
    with pytest.raises(RuntimeError, match="deadline"):
        transport.command([sys.executable, "-c", "import time;time.sleep(10)"], 0.05, stop)


def test_ssh_validates_details_and_retains_host_keys(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.touch()
    info = {
        "ip": "127.0.0.1",
        "port": 22,
        "ssh_key": {"path": str(key), "in_account": True},
        "ssh_command": "untrusted shell command",
    }
    argv = transport.ssh_args(info, tmp_path / "known_hosts")
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "untrusted shell command" not in argv
    info["ip"] = "host;malicious"
    with pytest.raises(ValueError):
        transport.ssh_args(info, tmp_path / "known_hosts")


def test_watchdog_is_active_before_pod_creation_and_released_afterward(
    tmp_path: Path, monkeypatch: Any
) -> None:
    owner = transport.RunpodTransport.__new__(transport.RunpodTransport)
    owner.config, owner.out = configuration(), tmp_path
    owner.lease_id = owner.watchdog_run_id = owner.watchdog_deadline = None
    commands: list[list[str]] = []

    def fake_command(argv: list[str], timeout: float) -> bytes:
        commands.append(argv)
        return b""

    def fake_gh(args: list[str], timeout: float = 30) -> Any:
        if args[:2] == ["run", "list"]:
            return [
                {
                    "databaseId": 123,
                    "displayTitle": f"b4 watchdog {owner.lease_id}",
                    "status": "in_progress",
                }
            ]
        if args[:2] == ["run", "view"]:
            return {
                "status": "in_progress",
                "conclusion": "",
                "jobs": [{"steps": [{"name": "watch Runpod lease", "status": "in_progress"}]}],
            }
        pytest.fail(f"unexpected GitHub command {args}")

    monkeypatch.setattr(transport, "command", fake_command)
    monkeypatch.setattr(transport, "gh", fake_gh)
    monkeypatch.setattr(transport, "cli", lambda *args: [])
    owner.arm_watchdog(4, 3600)
    assert owner.watchdog_run_id == 123
    assert commands[0][:3] == ["gh", "workflow", "run"]
    owner.release_watchdog()
    assert any(argv[:3] == ["gh", "run", "cancel"] for argv in commands)
    rows = b4.events(tmp_path / "runpod-watchdog.jsonl")
    assert [row["kind"] for row in rows] == ["armed", "released_after_local_deletion"]


def test_watchdog_refuses_pod_creation_when_workflow_step_is_not_active(
    tmp_path: Path, monkeypatch: Any
) -> None:
    owner = transport.RunpodTransport.__new__(transport.RunpodTransport)
    owner.config, owner.out = configuration(), tmp_path
    owner.lease_id, owner.watchdog_run_id = "0123456789abcdef", 123
    owner.watchdog_deadline = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
    monkeypatch.setattr(
        transport,
        "gh",
        lambda *args: {"status": "completed", "conclusion": "failure", "jobs": []},
    )
    with pytest.raises(RuntimeError, match="not actively guarding"):
        owner.verify_watchdog()


@pytest.mark.parametrize("mode", ["success", "price", "wait-failure", "shutdown", "peer-startup"])
def test_pod_ownership_and_cleanup(tmp_path: Path, monkeypatch: Any, mode: str) -> None:
    config = configuration()
    cell = b4.CELLS[0]
    plan: dict[str, Any] = {"cells": list(b4.CELLS), "schedules": {cell: {"digest": "cpu"}}}
    owner = transport.RunpodTransport.__new__(transport.RunpodTransport)
    owner.config, owner.plan, owner.out = config, plan, tmp_path
    owner.code_sha256, owner.archive, owner.max_rate = "a" * 64, b"tar", 1.59
    owner.lease_id = "0123456789abcdef"
    owner.watchdog_run_id = 123
    owner.watchdog_deadline = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
    owner.creation_lock = Lock()
    monkeypatch.setattr(owner, "verify_watchdog", lambda: None)
    calls: list[list[str]] = []
    stop = Event()

    def fake_cli(args: list[str], timeout: float = 30) -> dict[str, Any]:
        calls.append(args)
        if args[:2] == ["pod", "create"]:
            assert args[args.index("--gpu-count") + 1] == "1"
            assert args[args.index("--cloud-type") + 1] == "SECURE"
            assert "--terminate-after" not in args and "--wait" in args
            assert args[args.index("--name") + 1].startswith("b4-0123456789abcdef-")
            if mode == "wait-failure":
                raise transport.CommandError("network_error", {"id": "created_cpu"})
            if mode == "peer-startup":
                stop.set()
            return {
                "id": "created_cpu",
                "gpuCount": 1,
                "imageName": config["image"],
                "networkVolumeId": config["network_volumes"][cell],
                "costPerHr": 10 if mode == "price" else 1.59,
            }
        if args[:2] == ["pod", "delete"]:
            assert args[-1] == "created_cpu"
            if mode == "shutdown":
                raise transport.CommandError("network_error", {})
            return {}
        if args[:2] == ["pod", "get"]:
            raise transport.CommandError("not_found", {})
        return {}

    monkeypatch.setattr(transport, "cli", fake_cli)
    monkeypatch.setattr(transport, "ssh_args", lambda *args: ["ssh-fake"])
    monkeypatch.setattr(transport, "command", lambda *args: b'{"ready":true}')
    monkeypatch.setattr(b4, "verify_serving", lambda *args: None)
    if mode == "success":
        with owner.phase(cell, "reference", plan["schedules"][cell]) as lane:
            lane.start(300, stop)
            assert lane.ready
    else:
        with pytest.raises(RuntimeError):
            with owner.phase(cell, "reference", plan["schedules"][cell]) as lane:
                lane.start(300, stop)
    assert any(args[:2] == ["pod", "delete"] for args in calls)
    assert not any("other_pod" in args for args in calls)
    records = list((tmp_path / "runpod-lifecycle").glob("*.jsonl"))
    assert len(records) == 1
    audit = b4.events(records[0])
    assert any(row["kind"] == "created" for row in audit)
    assert any(row["kind"] == "termination_verified" for row in audit) == (mode != "shutdown")


def test_remote_receipts_are_durable_idempotent_and_path_safe(tmp_path: Path) -> None:
    namespace = "b4-" + "a" * 24
    path = remote.receipt_path(tmp_path, namespace, "imdb", "b" * 64)
    remote.persist(path, {"events": []})
    assert json.loads(path.read_text()) == {"events": []}
    assert not path.with_suffix(".tmp").exists()
    with pytest.raises(ValueError):
        remote.receipt_path(tmp_path, "../../outside", "imdb", "b" * 64)


def test_remote_generation_preserves_raw_and_cached_calls(tmp_path: Path) -> None:
    cell = b4.CELLS[0]
    rows = [{"doc_id": f"d{i}", "text": "movie"} for i in range(8192)]
    job = next(b4.jobs(cell, rows, "reference", 1))
    generated: list[Any] = []

    def generate(requests: Any) -> list[Any]:
        generated.extend(requests)
        return [
            SimpleNamespace(
                call_id=job["attempt_id"],
                text="malformed but preserved",
                finish_reason="stop",
                prompt_tokens=3,
                output_tokens=4,
                queue_seconds=None,
                prefill_seconds=None,
                decode_seconds=None,
            )
        ]

    worker = remote.Worker.__new__(remote.Worker)
    worker.model, worker.root, worker.host = cell.rsplit("::", 1)[0], tmp_path, {}
    worker.backend = SimpleNamespace(
        count_tokens=lambda messages: 3, generate=generate, describe=lambda: {}
    )
    body: dict[str, Any] = {
        "operation": "run",
        "namespace": "b4-" + "a" * 24,
        "task": "imdb",
        "chunk_id": sha256_json([job["attempt_id"]]),
        "jobs": [job],
    }
    result = worker.request(body)
    assert result["events"][0]["first_response"] == "malformed but preserved"
    assert worker.request(body) == result
    assert len(generated) == 1
    recovered = worker.request({**body, "operation": "recover"})
    assert recovered == [result]
    foreign = copy.deepcopy(body)
    foreign["jobs"][0]["model_id"] = "foreign"
    with pytest.raises(ValueError):
        worker.request(foreign)


def test_runpod_serving_requires_80gb_evidence() -> None:
    cell = b4.CELLS[0]
    model = cell.rsplit("::", 1)[0]
    revision = ext.ext_policy_for(model).revision
    plan = {"provider": "runpod", "serving_policy": {model: revision}}
    payload = {
        "backend": {
            "backend": "vllm",
            "model_id": model,
            "model_revision": revision,
            "tokenizer_revision": revision,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_num_seqs": 32,
            "seed": 20260912,
            "batch_invariant": True,
            "batch_invariance_required": True,
        },
        "host": {
            "installed_engine_version_string": "0.28.0",
            "gpu_name": "A100",
            "provider": "runpod",
            "gpu_memory_bytes": 80 * 1024**3,
        },
    }
    b4.verify_serving(payload, plan, cell)
    payload["host"]["gpu_memory_bytes"] = 40 * 1024**3
    with pytest.raises(ValueError, match="80GB"):
        b4.verify_serving(payload, plan, cell)


@pytest.mark.parametrize("mode", ["success", "balance", "account", "foreign-pod", "price", "stock"])
def test_live_round_gates_do_not_create_pods(tmp_path: Path, monkeypatch: Any, mode: str) -> None:
    owner = transport.RunpodTransport.__new__(transport.RunpodTransport)
    owner.config, owner.out, owner.max_rate = configuration(), tmp_path, 1.59
    owner.budget = {
        "record_sha256": "budget",
        "runpod_storage_reserve_usd": 5,
        "spend_limit_usd": 100,
        "spend_used_usd": 0,
    }
    owner.lease_id = owner.watchdog_run_id = owner.watchdog_deadline = None
    armed: list[tuple[int, float]] = []

    def arm(expected: int, seconds: float) -> None:
        armed.append((expected, seconds))
        owner.lease_id = "0123456789abcdef"
        owner.watchdog_run_id = 123
        owner.watchdog_deadline = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=seconds)

    monkeypatch.setattr(owner, "arm_watchdog", arm)
    monkeypatch.setattr(owner, "release_watchdog", lambda: None)
    b4.append(
        tmp_path / "meter.jsonl",
        [
            {
                "kind": "start",
                "phase_id": "phase",
                "budget_sha256": "budget",
                "guarded_forecast_hours": 1,
            }
        ],
    )
    calls: list[list[str]] = []

    def fake_cli(args: list[str], timeout: float = 30) -> Any:
        calls.append(args)
        if args == ["user"]:
            return {
                "id": "other" if mode == "account" else "user_cpu",
                "clientBalance": 0 if mode == "balance" else 100,
            }
        if args[:2] == ["pod", "list"]:
            return [{"desiredStatus": "RUNNING"}] if mode == "foreign-pod" else []
        if args[:2] == ["gpu", "list"]:
            return [
                {
                    "gpuId": owner.config["gpu_type_id"],
                    "secureCloud": True,
                    "securePricePerHr": 10 if mode == "price" else 1.59,
                    "dataCenterAvailability": [
                        {
                            "dataCenterId": "US-MD-1",
                            "stockStatus": "none" if mode == "stock" else "low",
                        }
                    ],
                }
            ]
        pytest.fail("round precheck made a mutation")

    monkeypatch.setattr(transport, "cli", fake_cli)
    if mode == "success":
        with owner.session():
            pass
        checks = list((tmp_path / "runpod-checks").glob("*.json"))
        assert len(checks) == 1
        assert json.loads(checks[0].read_text())["balance_usd"] == 100
        assert armed == [(1, 3660)]
    else:
        with pytest.raises((RuntimeError, ValueError)):
            with owner.session():
                pytest.fail("failed financial/account gate entered launch context")
    assert not any(args[:2] == ["pod", "create"] for args in calls)


def test_storage_reserve_prevents_dispatch(tmp_path: Path, monkeypatch: Any) -> None:
    import modal_extension_b4 as runner

    monkeypatch.setattr(b4, "frame", lambda *args: [])
    monkeypatch.setattr(b4, "jobs", lambda *args, **kwargs: iter([{"call_id": "cpu"}]))
    monkeypatch.setattr(b4, "validate_events", lambda *args: {})
    monkeypatch.setattr(b4, "pending", lambda *args: ["pending"])
    record = {
        "prior_extension_a100_hours": 0,
        "spend_limit_usd": 100,
        "spend_used_usd": 0,
        "a100_usd_per_hour": 1.59,
        "runpod_storage_reserve_usd": 100,
        "base_forecast_hours": {f"{cell}|reference": 1 for cell in b4.CELLS},
    }
    with pytest.raises(RuntimeError, match="protected budget"):
        runner.dispatch(STUDY, tmp_path, {"evaluation_prefix": 400}, record, None)
    assert not (tmp_path / "meter.jsonl").exists()


@pytest.mark.parametrize("mode", ["profile", "legacy", "environment", "missing"])
def test_flash_credentials_are_resolved_without_rewriting(
    tmp_path: Path,
    monkeypatch: Any,
    mode: str,
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = tmp_path / ".runpod" / "config.toml"
    if mode != "missing":
        path.parent.mkdir()
        path.write_text(
            'apikey = "legacy-test-value"\n[default]\napi_key = "profile-test-value"\n'
            if mode != "legacy"
            else 'apikey = "legacy-test-value"\n'
        )
    original = path.read_bytes() if path.exists() else None
    if mode == "environment":
        monkeypatch.setenv("RUNPOD_API_KEY", "environment-test-value")
    env = transport.credential_env()
    expected = {
        "profile": "profile-test-value",
        "legacy": "legacy-test-value",
        "environment": "environment-test-value",
        "missing": None,
    }[mode]
    assert env.get("RUNPOD_API_KEY") == expected
    assert (path.read_bytes() if path.exists() else None) == original
