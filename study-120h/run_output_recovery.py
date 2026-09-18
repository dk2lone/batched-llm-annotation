"""Guarded serial Runpod runner for the B3/B4 structured-output recovery."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shlex
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from threading import Event
from typing import Any, cast

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from extension_b4_runpod import (  # noqa: E402
    REGISTRY_AUTH_ID,
    CommandError,
    RunpodTransport,
    cli,
    command,
    ssh_args,
)

from freezeb.extension import ext_policy_for  # noqa: E402
from freezeb.output_recovery import recovery_jobs  # noqa: E402
from step0.hashing import read_jsonl, sha256_file, sha256_json, write_json  # noqa: E402

ACCOUNT_ID = "user_3GCQlNiLDruYedpe87pTOrbYANt"
GPU_ID = "NVIDIA A100 80GB PCIe"
DATA_CENTER = "CA-MTL-3"
IMAGE = (
    "ghcr.io/dk2lone/b4-a100-runtime@sha256:"
    "1cf3be1dfb50dbb800a658b371e8ec58f1ea485b55bae5b92ee281af2feba898"
)
MAX_RATE = 1.59
MODEL_KEYS = {
    "Qwen/Qwen3-8B": "qwen3-8b",
    "ibm-granite/granite-3.3-8b-instruct": "granite-3-3-8b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3-1-8b",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral-7b-v0-3",
}
VOLUMES = {
    "Qwen/Qwen3-8B": "pxk7r2wj35",
    "ibm-granite/granite-3.3-8b-instruct": "nm92sgw9zy",
    "meta-llama/Llama-3.1-8B-Instruct": "871iih1km4",
    "mistralai/Mistral-7B-Instruct-v0.3": "16nchs8xkt",
}
WATCHDOG = {
    "repository": "dk2lone/batched-llm-annotation",
    "workflow": "b4-pod-watchdog.yml",
    "ref": "main",
    "max_target_seconds": 18000,
}


def append(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


class Guard(RunpodTransport):
    """Reuse the already-tested B4 GitHub watchdog without constructing a B4 transport."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.config = {"watchdog": WATCHDOG}
        self.lease_id: str | None = None
        self.watchdog_run_id: int | None = None
        self.watchdog_deadline: dt.datetime | None = None

    def watchdog_audit(self, kind: str, **fields: Any) -> None:
        append(
            self.out / "runpod-watchdog.jsonl",
            [{"kind": kind, "at_utc": dt.datetime.now(dt.UTC).isoformat(), **fields}],
        )


class RecoveryPod:
    def __init__(self, guard: Guard, out: Path, model: str, archive: bytes) -> None:
        self.guard, self.out, self.model, self.archive = guard, out, model, archive
        self.pod_id: str | None = None
        self.ssh: list[str] = []
        self.identity = uuid.uuid4().hex
        self.remote = f"/workspace/output-recovery-runtime/{sha256_json([model, len(archive)])}"

    def audit(self, kind: str, **fields: Any) -> None:
        append(
            self.out / "runpod-lifecycle" / f"{MODEL_KEYS[self.model]}.jsonl",
            [{"kind": kind, "at_utc": dt.datetime.now(dt.UTC).isoformat(), **fields}],
        )

    def start(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        self.guard.verify_watchdog()
        if self.guard.lease_id is None:
            raise RuntimeError("recovery watchdog lacks a lease")
        name = f"b4-{self.guard.lease_id}-{self.identity[:12]}"
        self.audit("create_intent", name=name, model_id=self.model)
        try:
            pod = cli(
                [
                    "pod",
                    "create",
                    "--name",
                    name,
                    "--image",
                    IMAGE,
                    "--registry-auth-id",
                    REGISTRY_AUTH_ID,
                    "--gpu-id",
                    GPU_ID,
                    "--gpu-count",
                    "1",
                    "--cloud-type",
                    "SECURE",
                    "--data-center-ids",
                    DATA_CENTER,
                    "--network-volume-id",
                    VOLUMES[self.model],
                    "--volume-mount-path",
                    "/workspace",
                    "--container-disk-in-gb",
                    "40",
                    "--ports",
                    "22/tcp",
                    "--ssh",
                    "--wait",
                    "--wait-timeout",
                    "10m",
                ],
                min(timeout, 660),
            )
            self.pod_id = str(pod["id"])
        except CommandError as exc:
            value = exc.payload.get("id")
            self.pod_id = str(value) if value else None
            if self.pod_id is None:
                matches = [row for row in cli(["pod", "list", "--all"]) if row.get("name") == name]
                if len(matches) == 1:
                    self.pod_id = str(matches[0]["id"])
            if self.pod_id is None:
                raise
            pod = cli(["pod", "get", self.pod_id])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.pod_id or ""):
            raise RuntimeError("Runpod returned an invalid recovery Pod ID")
        self.audit("created", pod_id=self.pod_id, model_id=self.model)
        rate = pod.get("costPerHr")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not 0 < rate <= MAX_RATE
            or pod.get("gpuCount") != 1
            or pod.get("imageName") != IMAGE
            or pod.get("networkVolumeId") != VOLUMES[self.model]
        ):
            raise RuntimeError("created recovery Pod differs from the guarded configuration")
        ssh_deadline = min(deadline, time.monotonic() + 600)
        while time.monotonic() < ssh_deadline:
            try:
                self.ssh = ssh_args(
                    cli(["ssh", "info", self.pod_id]),
                    self.out / "runpod-known-hosts",
                )
                break
            except (CommandError, KeyError, ValueError):
                Event().wait(5)
        if not self.ssh:
            raise RuntimeError("recovery Pod SSH did not become ready")
        probe = "/opt/b4/bin/python -c 'import torch; assert torch.cuda.device_count() == 1'"
        command([*self.ssh, probe], min(30, deadline - time.monotonic()))
        command(
            [*self.ssh, f"mkdir -p {self.remote} && tar --no-same-owner -xf - -C {self.remote}"],
            deadline - time.monotonic(),
            data=self.archive,
        )
        launch = shlex.join(
            [
                "/opt/b4/bin/python",
                f"{self.remote}/output_recovery_worker.py",
                "serve",
                "--model",
                self.model,
                "--source",
                f"{self.remote}/src",
            ]
        )
        command(
            [*self.ssh, f"setsid {launch} > {self.remote}/worker.log 2>&1 < /dev/null &"],
            deadline - time.monotonic(),
        )
        while time.monotonic() < deadline:
            try:
                ready = self.rpc({"operation": "ready"}, min(10, deadline - time.monotonic()))
                if not isinstance(ready, dict):
                    raise RuntimeError("recovery readiness response is not an object")
                policy = ext_policy_for(self.model)
                backend = ready.get("backend", {})
                if (
                    ready.get("structured") is not True
                    or backend.get("model_id") != self.model
                    or backend.get("model_revision") != policy.revision
                    or backend.get("batch_invariant") is not True
                    or str(ready.get("host", {}).get("installed_engine_version_string")) != "0.28.0"
                ):
                    raise RuntimeError("recovery worker provenance differs")
                self.audit("ready", pod_id=self.pod_id, model_id=self.model)
                return
            except (CommandError, json.JSONDecodeError):
                Event().wait(2)
        raise RuntimeError("recovery worker did not become ready before its deadline")

    def rpc(self, payload: dict[str, Any], timeout: float) -> Any:
        argv = [
            "/opt/b4/bin/python",
            f"{self.remote}/output_recovery_worker.py",
            "request",
            "--timeout",
            str(max(0.01, timeout)),
        ]
        return json.loads(
            command([*self.ssh, shlex.join(argv)], timeout, data=json.dumps(payload).encode())
        )

    def close(self) -> None:
        if self.pod_id is None:
            return
        try:
            cli(["pod", "delete", self.pod_id], 20)
        except CommandError as exc:
            if exc.code != "not_found":
                raise
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                cli(["pod", "get", self.pod_id], 10)
            except CommandError as exc:
                if exc.code == "not_found":
                    self.audit("termination_verified", pod_id=self.pod_id, model_id=self.model)
                    return
                raise
            Event().wait(1)
        raise RuntimeError("recovery Pod deletion could not be verified")


def preflight(mode: str) -> dict[str, Any]:
    user = cli(["user"])
    required_balance = 2.0 if mode == "smoke" else len(MODEL_KEYS) * 1.5 * MAX_RATE + 0.5
    if user.get("id") != ACCOUNT_ID or float(user.get("clientBalance", -1)) < required_balance:
        raise RuntimeError("Runpod identity or required recovery balance differs")
    if any(
        str(row.get("desiredStatus", "")).upper() == "RUNNING"
        for row in cli(["pod", "list", "--all"])
    ):
        raise RuntimeError("an unrelated running Pod prevents exclusive recovery metering")
    volumes = {row.get("id"): row for row in cli(["network-volume", "list"])}
    if any(
        volumes.get(volume, {}).get("dataCenterId") != DATA_CENTER for volume in VOLUMES.values()
    ):
        raise RuntimeError("a recovery model volume is absent from CA-MTL-3")
    catalog = cli(["gpu", "list", "--include-unavailable"])
    gpu = next(row for row in catalog if row.get("gpuId") == GPU_ID)
    placement = next(
        row
        for row in gpu.get("dataCenterAvailability", [])
        if row.get("dataCenterId") == DATA_CENTER
    )
    if (
        gpu.get("secureCloud") is not True
        or float(gpu.get("securePricePerHr", -1)) > MAX_RATE
        or str(placement.get("stockStatus", "none")).lower() not in {"low", "medium", "high"}
    ):
        raise RuntimeError("the approved recovery GPU is unavailable or over rate")
    return {
        "at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "account_id": user["id"],
        "balance_usd": user["clientBalance"],
        "storage_spend_usd_per_hour": user["currentSpendPerHr"],
        "gpu_price_usd_per_hour": gpu["securePricePerHr"],
        "required_balance_usd": required_balance,
        "placement": placement,
    }


def smoke_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for model in MODEL_KEYS:
        rows = [job for job in jobs if job["model_id"] == model]
        for size in (1, 16):
            matches = [job for job in rows if len(job["target_ids"]) == size]
            if not matches:
                raise ValueError(f"smoke lacks a {size}-target recovery for {model}")
            selected.append(sorted(matches, key=lambda row: row["attempt_id"])[0])
    return selected


def verify_event_identity(event: dict[str, Any], job: dict[str, Any]) -> None:
    if (
        event.get("recovery_id") != job["attempt_id"]
        or event.get("source_event_sha256") != job["source_event_sha256"]
        or event.get("json_schema_sha256") != job["json_schema_sha256"]
    ):
        raise RuntimeError(f"structured recovery identity differs: {job['attempt_id']}")
    if (
        event.get("status") == "ok"
        and event.get("parse_status") == "ok"
        and set((event.get("predictions") or {}).keys()) != set(job["target_ids"])
    ):
        raise RuntimeError(f"successful recovery IDs differ: {job['attempt_id']}")


def recovery_succeeded(event: dict[str, Any]) -> bool:
    return event.get("status") == "ok" and event.get("parse_status") == "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    core = {key: value for key, value in inventory.items() if key != "manifest_sha256"}
    if sha256_json(core) != inventory.get("manifest_sha256") or len(inventory["records"]) != 4097:
        raise ValueError("recovery inventory seal or call count differs")
    jobs = recovery_jobs(STUDY, inventory["records"])
    selected = smoke_jobs(jobs) if args.mode == "smoke" else jobs
    jobs_by_id = {str(job["attempt_id"]): job for job in jobs}
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "RUN.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["inventory_sha256"] != sha256_file(args.inventory):
            raise ValueError("recovery output belongs to a different inventory")
    else:
        state = {
            "schema_version": "ext-output-recovery-run-v1",
            "namespace": "repair-" + uuid.uuid4().hex[:24],
            "inventory_sha256": sha256_file(args.inventory),
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        write_json(state_path, state)
    checks = preflight(args.mode)
    write_json(args.output / f"preflight-{uuid.uuid4().hex}.json", checks)
    events_path = args.output / "events.jsonl"
    existing_rows = list(read_jsonl(events_path)) if events_path.exists() else []
    completed = {str(row["recovery_id"]) for row in existing_rows}
    pending = [job for job in selected if job["attempt_id"] not in completed]
    archive = command(
        [
            "tar",
            "--exclude=__pycache__",
            "-cf",
            "-",
            "-C",
            str(STUDY),
            "src",
            "output_recovery_worker.py",
        ],
        30,
    )
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in pending:
        by_model[str(job["model_id"])].append(job)
    for model in MODEL_KEYS:
        model_jobs = by_model.get(model, [])
        if not model_jobs:
            continue
        guard = Guard(args.output)
        guard.arm_watchdog(1, (45 if args.mode == "smoke" else 90) * 60)
        pod = RecoveryPod(guard, args.output, model, archive)
        started = dt.datetime.now(dt.UTC)
        append(
            args.output / "meter.jsonl",
            [
                {
                    "kind": "start",
                    "model_id": model,
                    "started_at_utc": started.isoformat(),
                    "rate_usd_per_hour": checks["gpu_price_usd_per_hour"],
                    "mode": args.mode,
                }
            ],
        )
        try:
            pod.start(40 * 60)
            tasks = sorted({str(job["task_id"]) for job in model_jobs})
            recovered: list[dict[str, Any]] = []
            for task in tasks:
                payload = pod.rpc(
                    {
                        "operation": "recover",
                        "namespace": state["namespace"],
                        "model_key": MODEL_KEYS[model],
                        "task": task,
                    },
                    30,
                )
                if not isinstance(payload, list):
                    raise RuntimeError("recovery receipt listing is not a list")
                for receipt in cast(list[dict[str, Any]], payload):
                    recovered.extend(receipt["events"])
            for event in recovered:
                identity = str(event.get("recovery_id", ""))
                if identity not in jobs_by_id:
                    raise RuntimeError("remote recovery receipt contains a foreign call")
                verify_event_identity(event, jobs_by_id[identity])
            known = completed | {str(row["recovery_id"]) for row in recovered}
            if recovered:
                unseen = [row for row in recovered if row["recovery_id"] not in completed]
                append(events_path, unseen)
            remaining = [job for job in model_jobs if job["attempt_id"] not in known]
            by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for job in remaining:
                by_task[str(job["task_id"])].append(job)
            for task, task_jobs in sorted(by_task.items()):
                for offset in range(0, len(task_jobs), 32):
                    chunk = task_jobs[offset : offset + 32]
                    chunk_id = sha256_json([job["attempt_id"] for job in chunk])
                    payload = pod.rpc(
                        {
                            "operation": "run",
                            "namespace": state["namespace"],
                            "model_key": MODEL_KEYS[model],
                            "task": task,
                            "chunk_id": chunk_id,
                            "jobs": chunk,
                        },
                        30 * 60,
                    )
                    if not isinstance(payload, dict):
                        raise RuntimeError("recovery generation response is not an object")
                    receipt = args.output / "receipts" / f"{chunk_id}.json"
                    write_json(receipt, payload)
                    events = cast(list[dict[str, Any]], payload["events"])
                    expected = {job["attempt_id"]: job for job in chunk}
                    if set(expected) != {row.get("recovery_id") for row in events}:
                        raise RuntimeError("recovery receipt identities differ")
                    for event in events:
                        verify_event_identity(event, expected[str(event["recovery_id"])])
                    append(events_path, events)
                    completed.update(expected)
                    if args.mode == "smoke" and not all(
                        recovery_succeeded(event) for event in events
                    ):
                        raise RuntimeError("a structured-output smoke call failed")
        finally:
            pod.close()
            guard.release_watchdog()
            finished = dt.datetime.now(dt.UTC)
            append(
                args.output / "meter.jsonl",
                [
                    {
                        "kind": "settled",
                        "model_id": model,
                        "finished_at_utc": finished.isoformat(),
                        "charged_hours_upper_bound": (
                            (finished - started).total_seconds() / 3600 + 60 / 3600
                        ),
                        "mode": args.mode,
                    }
                ],
            )
    events = list(read_jsonl(events_path)) if events_path.exists() else []
    selected_ids = {job["attempt_id"] for job in selected}
    selected_events = [row for row in events if row.get("recovery_id") in selected_ids]
    complete = len(selected_events) == len(selected)
    recovered_calls = sum(recovery_succeeded(row) for row in selected_events)
    passed = complete and (args.mode == "full" or recovered_calls == len(selected))
    report = {
        "schema_version": "ext-output-recovery-report-v1",
        "mode": args.mode,
        "inventory_sha256": sha256_file(args.inventory),
        "namespace": state["namespace"],
        "selected_calls": len(selected),
        "completed_calls": len(selected_events),
        "recovered_calls": recovered_calls,
        "unresolved_calls": len(selected_events) - recovered_calls,
        "complete": complete,
        "passed": passed,
        "by_model": dict(
            sorted(
                (model, sum(row.get("model_id") == model for row in selected_events))
                for model in MODEL_KEYS
            )
        ),
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    write_json(args.output / f"{args.mode.upper()}_REPORT.json", report)
    if not passed:
        raise RuntimeError("recovery run did not record every selected call")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
