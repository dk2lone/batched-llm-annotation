"""Opt-in Runpod Pod/SSH transport. No resources are created at import time."""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import math
import os
import re
import shlex
import subprocess
import time
import tomllib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock
from typing import Any, cast

from freezeb import extension_b4 as b4
from step0.hashing import sha256_json, write_json

STUDY = Path(__file__).resolve().parent


def credential_env() -> dict[str, str]:
    """Read Flash's saved default profile without rewriting or displaying credentials."""
    env = os.environ.copy()
    if env.get("RUNPOD_API_KEY"):
        return env
    path = Path.home() / ".runpod" / "config.toml"
    if not path.is_file():
        return env
    try:
        config = tomllib.loads(path.read_text())
    except (OSError, ValueError):
        raise RuntimeError("Runpod credential configuration cannot be read") from None
    key = config.get("default", {}).get("api_key") or config.get("apikey")
    if isinstance(key, str) and key:
        env["RUNPOD_API_KEY"] = key
    return env


class CommandError(RuntimeError):
    def __init__(self, code: str, payload: dict[str, Any]) -> None:
        super().__init__(f"Runpod command failed ({code}); reconcile if creation was ambiguous")
        self.code, self.payload = code, payload


def command(
    argv: list[str],
    timeout: float,
    stop: Event | None = None,
    data: bytes | None = None,
) -> bytes:
    if timeout <= 0 or (stop is not None and stop.is_set()):
        raise RuntimeError("B4 protected deadline reached or peer lane failed")
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=credential_env() if argv[0] == "runpodctl" else None,
    ) as process:
        first = True
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (stop is not None and stop.is_set()):
                    raise RuntimeError("B4 protected deadline reached or peer lane failed")
                try:
                    output, error = process.communicate(
                        data if first else None, timeout=min(1.0, remaining)
                    )
                    break
                except subprocess.TimeoutExpired:
                    first = False
            if process.returncode:
                # CLI output can contain environment values. Never echo it to chat/logs.
                try:
                    payload = json.loads(error)
                except (ValueError, UnicodeDecodeError):
                    payload = {}
                raise CommandError(
                    str(payload.get("code", "command_error")), {"id": payload.get("id")}
                )
            return output
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


def cli(args: list[str], timeout: float = 30) -> Any:
    output = command(["runpodctl", *args], timeout)
    return json.loads(output) if output.strip() else {}


def gh(args: list[str], timeout: float = 30) -> Any:
    output = command(["gh", *args], timeout)
    return json.loads(output) if output.strip() else {}


def verify_config(config: dict[str, Any], budget: dict[str, Any], profile: str) -> None:
    if set(config) != {
        "schema_version",
        "account_id",
        "cloud_type",
        "gpu_type_id",
        "image",
        "data_center_id",
        "network_volumes",
        "watchdog",
    }:
        raise ValueError("Runpod configuration contains unknown fields; credentials are prohibited")
    if config.get("schema_version") != "b4-runpod-config-v2":
        raise ValueError("unknown Runpod configuration")
    if config.get("account_id") != profile:
        raise ValueError("Runpod account must match the verified profile")
    if config.get("cloud_type") != "SECURE" or config.get("gpu_type_id") not in {
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
    }:
        raise ValueError("B4 Runpod allocation must be Secure Cloud 80GB A100")
    if not re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", str(config.get("image", ""))):
        raise ValueError("Runpod runtime image must be pinned by digest after smoke testing")
    if not re.fullmatch(r"[A-Z0-9-]+", str(config.get("data_center_id", ""))):
        raise ValueError("invalid Runpod data center")
    volumes = config.get("network_volumes", {})
    if set(volumes) != set(b4.CELLS) or len(set(volumes.values())) != 4:
        raise ValueError("four distinct preloaded network volumes are required")
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", str(value)) for value in volumes.values()):
        raise ValueError("invalid Runpod volume ID")
    watchdog = config.get("watchdog", {})
    if watchdog != {
        "repository": "dk2lone/batched-llm-annotation",
        "workflow": "b4-pod-watchdog.yml",
        "ref": "main",
        "max_target_seconds": 18000,
    }:
        raise ValueError("Runpod watchdog configuration differs from the reviewed path")
    if budget.get("runpod_config_sha256") != sha256_json(config):
        raise ValueError("budget must bind the exact Runpod configuration")
    evidence = budget.get("runpod_preflight", {})
    if not all(
        evidence.get(key) is True
        for key in (
            "runtime_smoke_passed",
            "four_single_gpu_capacity_verified",
            "model_caches_verified",
            "balance_verified",
            "ssh_verified",
            "watchdog_cpu_test_passed",
            "auto_pay_disabled_verified",
        )
    ):
        raise ValueError("Runpod smoke, watchdog, funding and infrastructure checks are required")
    if not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("evidence_sha256", ""))):
        raise ValueError("Runpod preflight needs a content-addressed evidence artifact")
    storage = budget.get("runpod_storage_reserve_usd")
    if isinstance(storage, bool) or not isinstance(storage, (int, float)):
        raise ValueError("Runpod storage reserve must be recorded")
    if not math.isfinite(storage) or storage < 0:
        raise ValueError("invalid Runpod storage reserve")


def ssh_args(info: dict[str, Any], known_hosts: Path) -> list[str]:
    ip = str(ipaddress.ip_address(info["ip"]))
    port = info["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("invalid SSH port")
    key = info.get("ssh_key", {})
    path = Path(str(key.get("path", "")))
    if not path.is_absolute() or not path.is_file() or key.get("in_account") is not True:
        raise ValueError("Runpod SSH key is not locally available and registered")
    return [
        "ssh",
        "-i",
        str(path),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        f"root@{ip}",
    ]


class Lane:
    def __init__(self, owner: RunpodTransport, cell: str, stream: str) -> None:
        self.owner, self.cell, self.stream = owner, cell, stream
        self.model, self.task = cell.rsplit("::", 1)
        self.pod_id: str | None = None
        self.ssh: list[str] = []
        self.remote = f"/workspace/b4-runtime/{owner.code_sha256}"
        self.ready = False

    def audit(self, kind: str, **fields: Any) -> None:
        b4.append(
            self.owner.out / "runpod-lifecycle" / f"{self.identity}.jsonl",
            [
                {
                    "kind": kind,
                    "at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "cell_key": self.cell,
                    "stream": self.stream,
                    **fields,
                }
            ],
        )

    def start(self, timeout: float, stop: Event) -> None:
        if self.ready:
            return
        if timeout < 45 or stop.is_set():
            raise RuntimeError("B4 stopped before Pod creation")
        deadline = time.monotonic() + timeout
        config = self.owner.config
        self.identity = uuid.uuid4().hex
        self.owner.verify_watchdog()
        if self.owner.watchdog_run_id is None or self.owner.watchdog_deadline is None:
            raise RuntimeError("GitHub watchdog evidence is incomplete")
        name = f"b4-{self.owner.lease_id}-{self.identity[:12]}"
        self.audit(
            "create_intent",
            name=name,
            watchdog_run_id=self.owner.watchdog_run_id,
            shutdown_target_utc=self.owner.watchdog_deadline.isoformat(),
        )
        wait = min(600, max(1, int(timeout - 30)))
        try:
            # Runpod's low-stock allocator races concurrent create requests. Retain each
            # successful Pod while later lanes allocate, then serve all lanes concurrently.
            with self.owner.creation_lock:
                if stop.is_set() or deadline <= time.monotonic():
                    raise RuntimeError("B4 stopped while waiting to create a Pod")
                # Do not interrupt creation mid-response: recover its ID even if a peer fails.
                pod = cli(
                    [
                        "pod",
                        "create",
                        "--name",
                        name,
                        "--image",
                        config["image"],
                        "--gpu-id",
                        config["gpu_type_id"],
                        "--gpu-count",
                        "1",
                        "--cloud-type",
                        "SECURE",
                        "--data-center-ids",
                        config["data_center_id"],
                        "--network-volume-id",
                        config["network_volumes"][self.cell],
                        "--volume-mount-path",
                        "/workspace",
                        "--container-disk-in-gb",
                        "40",
                        "--ports",
                        "22/tcp",
                        "--ssh",
                        "--wait",
                        "--wait-timeout",
                        f"{wait}s",
                    ],
                    min(timeout, wait + 30),
                )
        except CommandError as exc:
            self.capture_id(exc.payload.get("id"))
            raise
        self.capture_id(pod.get("id"))
        if self.pod_id is None:
            raise RuntimeError("ambiguous Pod creation: reconcile before resuming")
        if stop.is_set() or deadline <= time.monotonic():
            raise RuntimeError("B4 stopped during Pod startup")
        rate = pod.get("costPerHr")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (float, int))
            or not math.isfinite(rate)
            or not 0 < rate <= self.owner.max_rate
        ):
            raise RuntimeError("Pod price exceeds or cannot verify the protected rate")
        if (
            pod.get("gpuCount") != 1
            or pod.get("imageName") != config["image"]
            or pod.get("networkVolumeId") != config["network_volumes"][self.cell]
        ):
            raise RuntimeError("created Pod configuration differs")
        info = cli(["ssh", "info", self.pod_id], min(30, deadline - time.monotonic()))
        self.ssh = ssh_args(info, self.owner.out / "runpod-known-hosts")
        command(
            [
                *self.ssh,
                f"mkdir -p {self.remote} && tar --no-same-owner -xf - -C {self.remote}",
            ],
            deadline - time.monotonic(),
            stop,
            self.owner.archive,
        )
        worker = f"{self.remote}/b4_runpod_worker.py"
        launch = shlex.join(
            [
                "/opt/b4/bin/python",
                worker,
                "serve",
                "--model",
                self.model,
                "--source",
                f"{self.remote}/src",
            ]
        )
        command(
            [
                *self.ssh,
                f"setsid {launch} > {self.remote}/worker-{self.identity}.log 2>&1 < /dev/null &",
            ],
            deadline - time.monotonic(),
            stop,
        )
        while deadline > time.monotonic() and not stop.is_set():
            try:
                payload = self.rpc(
                    {"operation": "ready"}, min(10, deadline - time.monotonic()), stop
                )
                b4.verify_serving(payload, self.owner.plan, self.cell)
                self.ready = True
                return
            except CommandError:
                stop.wait(min(2, max(0, deadline - time.monotonic())))
        raise RuntimeError("B4 worker was not ready within its guarded lane deadline")

    def capture_id(self, value: Any) -> None:
        if value is not None:
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", str(value)):
                raise ValueError("invalid created Pod ID")
            self.pod_id = str(value)
            self.audit("created", pod_id=self.pod_id)

    def rpc(self, payload: dict[str, Any], timeout: float, stop: Event) -> Any:
        argv = [
            "/opt/b4/bin/python",
            f"{self.remote}/b4_runpod_worker.py",
            "request",
            "--timeout",
            str(max(0.01, timeout)),
        ]
        output = command([*self.ssh, shlex.join(argv)], timeout, stop, json.dumps(payload).encode())
        return json.loads(output)

    def recover(self, namespace: str, timeout: float, stop: Event) -> list[dict[str, Any]]:
        started = time.monotonic()
        self.start(timeout, stop)
        return cast(
            list[dict[str, Any]],
            self.rpc(
                {"operation": "recover", "namespace": namespace, "task": self.task},
                timeout - (time.monotonic() - started),
                stop,
            ),
        )

    def run(
        self, namespace: str, chunk_id: str, jobs: list[dict[str, Any]], timeout: float, stop: Event
    ) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("B4 worker has not passed serving verification")
        return cast(
            dict[str, Any],
            self.rpc(
                {
                    "operation": "run",
                    "namespace": namespace,
                    "task": self.task,
                    "chunk_id": chunk_id,
                    "jobs": jobs,
                },
                timeout,
                stop,
            ),
        )

    def close(self) -> None:
        if self.pod_id is None:
            return
        deadline = time.monotonic() + 50
        try:
            cli(["pod", "delete", self.pod_id], 15)
        except CommandError as exc:
            if exc.code != "not_found":
                raise
        while time.monotonic() < deadline:
            try:
                cli(["pod", "get", self.pod_id], min(10, deadline - time.monotonic()))
            except CommandError as exc:
                if exc.code == "not_found":
                    self.audit("termination_verified", pod_id=self.pod_id)
                    return
                raise
            Event().wait(1)
        raise RuntimeError("Pod termination unverified: meter must remain unsettled")


class RunpodTransport:
    def __init__(
        self,
        config: dict[str, Any],
        budget: dict[str, Any],
        plan: dict[str, Any],
        profile: str,
        out: Path,
    ) -> None:
        verify_config(config, budget, profile)
        if config["gpu_type_id"] != plan.get("gpu_type_id"):
            raise ValueError("Runpod configuration GPU differs from the immutable plan")
        self.config, self.plan, self.out = config, plan, out
        self.budget = budget
        self.max_rate = float(budget["a100_usd_per_hour"])
        self.lease_id: str | None = None
        self.watchdog_run_id: int | None = None
        self.watchdog_deadline: dt.datetime | None = None
        self.creation_lock = Lock()
        user = cli(["user"])
        identity = user.get("user", user)
        if identity.get("id") != profile:
            raise ValueError("active Runpod CLI account differs from verified profile")
        for volume in config["network_volumes"].values():
            record = cli(["network-volume", "get", volume])
            if record.get("id") != volume or record.get("dataCenterId") != config["data_center_id"]:
                raise ValueError("preloaded network volume placement differs")
        self.archive = command(
            [
                "tar",
                "--exclude=__pycache__",
                "-cf",
                "-",
                "-C",
                str(STUDY),
                "src",
                "b4_runpod_worker.py",
            ],
            30,
        )
        self.code_sha256 = sha256_json(plan["implementation_sha256"])

    def watchdog_audit(self, kind: str, **fields: Any) -> None:
        b4.append(
            self.out / "runpod-watchdog.jsonl",
            [{"kind": kind, "at_utc": dt.datetime.now(dt.UTC).isoformat(), **fields}],
        )

    def arm_watchdog(self, expected_pods: int, target_seconds: float) -> None:
        maximum = int(self.config["watchdog"]["max_target_seconds"])
        if not 30 <= target_seconds <= maximum:
            raise RuntimeError("Runpod round does not fit the five-hour watchdog target")
        self.lease_id = uuid.uuid4().hex[:16]
        self.watchdog_deadline = dt.datetime.now(dt.UTC) + dt.timedelta(
            seconds=math.ceil(target_seconds)
        )
        title = f"b4 watchdog {self.lease_id}"
        settings = self.config["watchdog"]
        command(
            [
                "gh",
                "workflow",
                "run",
                settings["workflow"],
                "--repo",
                settings["repository"],
                "--ref",
                settings["ref"],
                "--field",
                f"lease_id={self.lease_id}",
                "--field",
                f"deadline_utc={self.watchdog_deadline.isoformat()}",
                "--field",
                f"expected_pods={expected_pods}",
            ],
            30,
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            runs = gh(
                [
                    "run",
                    "list",
                    "--repo",
                    settings["repository"],
                    "--workflow",
                    settings["workflow"],
                    "--event",
                    "workflow_dispatch",
                    "--limit",
                    "20",
                    "--json",
                    "databaseId,displayTitle,status,conclusion,createdAt",
                ]
            )
            matches = [row for row in runs if row.get("displayTitle") == title]
            if len(matches) == 1:
                run_id = matches[0].get("databaseId")
                if isinstance(run_id, bool) or not isinstance(run_id, int):
                    raise RuntimeError("GitHub watchdog returned an invalid run ID")
                self.watchdog_run_id = run_id
                try:
                    self.verify_watchdog()
                except RuntimeError:
                    Event().wait(2)
                    continue
                self.watchdog_audit(
                    "armed",
                    lease_id=self.lease_id,
                    workflow_run_id=run_id,
                    shutdown_target_utc=self.watchdog_deadline.isoformat(),
                    expected_pods=expected_pods,
                )
                return
            Event().wait(2)
        raise RuntimeError("GitHub watchdog did not reach its active deletion step")

    def watchdog_state(self) -> dict[str, Any]:
        if self.watchdog_run_id is None:
            raise RuntimeError("GitHub watchdog is not armed")
        return cast(
            dict[str, Any],
            gh(
                [
                    "run",
                    "view",
                    str(self.watchdog_run_id),
                    "--repo",
                    self.config["watchdog"]["repository"],
                    "--json",
                    "status,conclusion,jobs",
                ]
            ),
        )

    def verify_watchdog(self) -> None:
        state = self.watchdog_state()
        steps = [
            step
            for job in state.get("jobs", [])
            for step in job.get("steps", [])
            if step.get("name") == "watch Runpod lease"
        ]
        if (
            state.get("status") != "in_progress"
            or len(steps) != 1
            or steps[0].get("status") != "in_progress"
        ):
            raise RuntimeError("GitHub watchdog is not actively guarding Pod creation")

    def release_watchdog(self) -> None:
        if self.watchdog_run_id is None or self.lease_id is None:
            return
        pods = cli(["pod", "list", "--all"])
        if not isinstance(pods, list):
            raise RuntimeError("Runpod Pod list response shape changed")
        prefix = f"b4-{self.lease_id}-"
        if any(
            str(pod.get("name", "")).startswith(prefix)
            and str(pod.get("desiredStatus", "")).upper() != "TERMINATED"
            for pod in pods
        ):
            raise RuntimeError("lease Pods remain; keep the watchdog active")
        try:
            command(
                [
                    "gh",
                    "run",
                    "cancel",
                    str(self.watchdog_run_id),
                    "--repo",
                    self.config["watchdog"]["repository"],
                ],
                30,
            )
        except CommandError:
            state = self.watchdog_state()
            if state.get("status") != "completed":
                raise
        self.watchdog_audit(
            "released_after_local_deletion",
            lease_id=self.lease_id,
            workflow_run_id=self.watchdog_run_id,
        )

    @contextmanager
    def session(self) -> Iterator[None]:
        # Repeat live financial/catalog checks at every round barrier, not just startup.
        user = cli(["user"])
        if user.get("id") != self.config["account_id"]:
            raise ValueError("Runpod account changed before round launch")
        records = b4.events(self.out / "meter.jsonl")
        settled = {row["phase_id"] for row in records if row["kind"] == "settled"}
        reserved = [
            row for row in records if row["kind"] == "start" and row["phase_id"] not in settled
        ]
        if not 1 <= len(reserved) <= 4 or any(
            row["budget_sha256"] != self.budget["record_sha256"] for row in reserved
        ):
            raise ValueError("Runpod round lacks valid individual lane reservations")
        needed = sum(
            float(row["guarded_forecast_hours"]) + 60 / 3600 for row in reserved
        ) * self.max_rate + float(self.budget["runpod_storage_reserve_usd"])
        balance = user.get("clientBalance")
        if (
            isinstance(balance, bool)
            or not isinstance(balance, (float, int))
            or not math.isfinite(balance)
            or balance < needed
        ):
            raise RuntimeError("live Runpod balance cannot fund the protected round")
        remaining_budget = float(self.budget["spend_limit_usd"]) - float(
            self.budget["spend_used_usd"]
        )
        if balance > remaining_budget:
            raise RuntimeError("prepaid Runpod balance exceeds the authorized remaining budget")
        pods = cli(["pod", "list", "--all"])
        if not isinstance(pods, list) or any(
            str(pod.get("desiredStatus", "")).upper() == "RUNNING" for pod in pods
        ):
            raise RuntimeError("unrelated active Pods prevent exclusive Runpod metering")
        catalog = cli(["gpu", "list", "--include-unavailable"])
        if not isinstance(catalog, list):
            raise ValueError("Runpod GPU catalog shape differs")
        matches = [gpu for gpu in catalog if gpu.get("gpuId") == self.config["gpu_type_id"]]
        if len(matches) != 1 or matches[0].get("secureCloud") is not True:
            raise RuntimeError("requested GPU is not offered on Secure Cloud")
        gpu = matches[0]
        rate = gpu.get("securePricePerHr")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (float, int))
            or not math.isfinite(rate)
            or not 0 < rate <= self.max_rate
        ):
            raise RuntimeError("live Secure Cloud GPU price exceeds or cannot verify budget rate")
        placements = [
            item
            for item in gpu.get("dataCenterAvailability", [])
            if item.get("dataCenterId") == self.config["data_center_id"]
        ]
        if len(placements) != 1 or str(placements[0].get("stockStatus", "")).lower() not in (
            "low",
            "medium",
            "high",
        ):
            raise RuntimeError(
                "live GPU stock is unavailable or unverified in the volume data center"
            )
        target_seconds = max(float(row["guarded_forecast_hours"]) for row in reserved) * 3600 + 60
        self.arm_watchdog(len(reserved), target_seconds)
        if self.watchdog_run_id is None or self.lease_id is None or self.watchdog_deadline is None:
            raise RuntimeError("GitHub watchdog evidence is incomplete")
        try:
            write_json(
                self.out / "runpod-checks" / f"{uuid.uuid4().hex}.json",
                {
                    "at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "account_id": user["id"],
                    "balance_usd": balance,
                    "authorized_remaining_usd": remaining_budget,
                    "required_round_usd": needed,
                    "secure_price_usd_per_hour": rate,
                    "placement": placements[0],
                    "lane_reservations": [row["phase_id"] for row in reserved],
                    "watchdog_run_id": self.watchdog_run_id,
                    "watchdog_lease_id": self.lease_id,
                    "shutdown_target_utc": self.watchdog_deadline.isoformat(),
                },
            )
            yield
        finally:
            self.release_watchdog()

    @contextmanager
    def phase(self, cell: str, stream: str, schedule: dict[str, Any]) -> Iterator[Lane]:
        if cell not in self.plan["cells"] or schedule != self.plan["schedules"][cell]:
            raise ValueError("foreign B4 Runpod lane")
        lane = Lane(self, cell, stream)
        try:
            yield lane
        finally:
            # Delete only the exact Pod ID returned by this lane's create call.
            lane.close()
