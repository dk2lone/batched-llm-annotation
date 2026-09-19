"""Guarded single-A100 Runpod launcher for the B5 batch-size dose-response.

Only `--launch` and `--resume` can create a Pod, and both need a valid seal plus the
confirmation token printed by the seal. `--dry-run`, `--preflight`, `--status` and
`--verify-only` never create one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import shlex
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any, cast

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

import seal_b5  # noqa: E402
from extension_b4_runpod import (  # noqa: E402
    REGISTRY_AUTH_ID,
    CommandError,
    cli,
    command,
    ssh_args,
)
from run_output_recovery import (  # noqa: E402
    ACCOUNT_ID,
    DATA_CENTER,
    GPU_ID,
    IMAGE,
    MAX_RATE,
    MODEL_KEYS,
    VOLUMES,
    Guard,
    append,
)

from freezeb import b5  # noqa: E402
from freezeb import extension_b4 as b4  # noqa: E402
from freezeb.extension import ext_policy_for  # noqa: E402
from step0.hashing import read_jsonl, sha256_json, write_json  # noqa: E402

RUN_SCHEMA = "b5-run-v1"
CEILING_HOURS = 35.0
RESERVE_USD = 2.00
SMOKE_CEILING_HOURS = 2.0
SMOKE_CHUNK_CALLS = 64
CHUNK_TIMEOUT_SECONDS = 45 * 60
POD_START_TIMEOUT_SECONDS = 40 * 60
WATCHDOG_TARGET_SECONDS = 5 * 3600
MODEL_PHASES = {
    "1": tuple(b5.CELLS[:2]),
    "2": tuple(b5.CELLS[2:]),
}


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


# --- plan --------------------------------------------------------------------


class Plan:
    """The sealed schedule, its frames and its forecast. Building one touches no network."""

    def __init__(self, seal_directory: Path, *, recompute: bool) -> None:
        self.seal = seal_b5.verify(seal_directory)
        self.directory = seal_directory
        preparation = STUDY / str(self.seal["preparation_directory"])
        self.preparation = preparation
        self.schedule = json.loads((preparation / "SCHEDULE_DIGEST.json").read_text())
        self.forecast = json.loads((preparation / "FORECAST.json").read_text())
        self.context = json.loads((preparation / "CONTEXT_AUDIT.json").read_text())
        self.frames: dict[str, list[dict[str, Any]]] = {}
        for cell_key in b5.CELLS:
            task_id = b5.split_cell(cell_key)[1]
            if task_id not in self.frames:
                self.frames[task_id] = b4.frame(STUDY, task_id)
        b5.verify_counts(cast(dict[str, int], self.schedule["observed_counts"]))
        if any(not bool(row["fits"]) for row in self.context["cells"]):
            raise ValueError("the sealed context audit does not fit every scheduled prompt")
        self.digests = {str(row["cell_key"]): row for row in self.schedule["cell_digests"]}
        if recompute:
            self.recompute()

    def doc_ids(self, cell_key: str) -> list[str]:
        task_id = b5.split_cell(cell_key)[1]
        return [str(row["doc_id"]) for row in self.frames[task_id]]

    def texts(self, cell_key: str) -> dict[str, str]:
        task_id = b5.split_cell(cell_key)[1]
        return {str(row["doc_id"]): str(row["text"]) for row in self.frames[task_id]}

    def recompute(self) -> None:
        """Rebuild every call and prompt hash and compare against the sealed digest."""
        for cell_key in b5.CELLS:
            task_id = b5.split_cell(cell_key)[1]
            fresh = b5.cell_digest(cell_key, self.frames[task_id])
            sealed = self.digests[cell_key]
            for key in (
                "counts",
                "counts_by_size",
                "call_id_order_sha256",
                "prompt_sha256_order_sha256",
                "record_stream_sha256",
            ):
                if fresh[key] != sealed[key]:
                    raise ValueError(f"{cell_key} rebuilt {key} differs from the seal")

    def blocks(self) -> list[tuple[str, int]]:
        return [(cell_key, k) for cell_key in b5.CELLS for k in b5.SIZES]

    def chunks(self, cell_key: str, k: int, chunk_calls: int) -> list[dict[str, Any]]:
        return [
            chunk
            for chunk in b5.chunks(cell_key, self.doc_ids(cell_key), chunk_calls)
            if int(chunk["batch_size"]) == k
        ]

    def block_hours(self) -> dict[tuple[str, int], float]:
        return {
            (str(row["cell_key"]), int(row["batch_size"])): float(row["generation_hours"])
            for row in self.forecast["scenarios"]["prefix_reuse"]["blocks"]
        }


def selected_chunks(plan: Plan, chunk_calls: int, smoke: bool) -> list[dict[str, Any]]:
    """The full schedule, or the first chunk of every cell and size.

    Both paths read the same generator, so a smoke run differs only in which chunks it
    selects and in the chunk call budget.
    """
    rows: list[dict[str, Any]] = []
    for cell_key, k in plan.blocks():
        block = plan.chunks(cell_key, k, chunk_calls)
        rows.extend(block[:1] if smoke else block)
    return rows


# --- durable local state -----------------------------------------------------


def expected_calls(chunk: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(call["call_id"]): call
        for group in cast(list[dict[str, Any]], chunk["groups"])
        for call in b5.group_calls(group)
    }


def verify_event(
    texts: dict[str, str], event: dict[str, Any], call: dict[str, Any], attempt: int
) -> None:
    """Refuse any event whose prompt, schema, target set or limit differs from the plan."""
    expected_prompt = sha256_json(b5.render_messages(str(call["task_id"]), texts, call))
    if (
        event.get("attempt_id") != f"{call['call_id']}::attempt::{attempt}"
        or event.get("attempt") != attempt
        or event.get("prompt_sha256") != expected_prompt
        or event.get("doc_order_sha256") != call["doc_order_sha256"]
        or event.get("target_ids_sha256") != call["target_ids_sha256"]
        or event.get("json_schema_sha256") != call["json_schema_sha256"]
        or event.get("max_tokens") != call["max_tokens"]
        or event.get("condition") != call["condition"]
        or event.get("batch_size") != call["batch_size"]
        or event.get("assignment") != call["assignment"]
        or event.get("structured_output") is not True
    ):
        raise RuntimeError(f"B5 event identity differs: {call['call_id']}")
    if attempt == 2 and event.get("parent_attempt_id") != f"{call['call_id']}::attempt::1":
        raise RuntimeError(f"B5 retry linkage differs: {call['call_id']}")
    predictions = event.get("predictions")
    if event.get("status") == "ok" and event.get("parse_status") == "ok":
        if not isinstance(predictions, dict) or set(predictions) != set(call["target_ids"]):
            raise RuntimeError(f"B5 resolved response IDs differ: {call['call_id']}")


class Ledger:
    """Append-only events plus durable chunk receipts. The Pod stores nothing durable."""

    def __init__(self, output: Path, texts_of: Callable[[str], dict[str, str]]) -> None:
        self.output = output
        self.texts_of = texts_of
        self.path = output / "events.jsonl"
        self.receipts = output / "receipts"
        self.by_attempt: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.path) if self.path.exists() else []:
            self.by_attempt[str(row["attempt_id"])] = row

    def has(self, attempt_id: str) -> bool:
        return attempt_id in self.by_attempt

    def add(self, events: list[dict[str, Any]]) -> int:
        fresh = []
        for event in events:
            attempt_id = str(event["attempt_id"])
            stored = self.by_attempt.get(attempt_id)
            if stored is not None:
                if sha256_json(stored) != sha256_json(event):
                    raise RuntimeError(f"conflicting duplicate scientific call: {attempt_id}")
                continue
            self.by_attempt[attempt_id] = event
            fresh.append(event)
        append(self.path, fresh)
        return len(fresh)

    def reconcile(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Rebuild the ledger from durable receipts before anything is scheduled."""
        known = {str(chunk["chunk_id"]): chunk for chunk in chunks}
        imported = 0
        checked = 0
        for path in sorted(self.receipts.glob("*.json")):
            payload = json.loads(path.read_text())
            chunk_id = str(payload["chunk_id"])
            attempt = int(payload.get("attempt", 1))
            source = known.get(str(payload.get("source_chunk_id", chunk_id)))
            if source is None:
                raise RuntimeError(f"receipt {path.name} does not belong to this schedule")
            calls = expected_calls(source)
            events = cast(list[dict[str, Any]], payload["events"])
            for event in events:
                call = calls.get(str(event["call_id"]))
                if call is None:
                    raise RuntimeError(f"receipt {path.name} carries a foreign call")
                verify_event(self.texts_of(str(call["cell_key"])), event, call, attempt)
                checked += 1
            imported += self.add(events)
        return {
            "receipts": len(list(self.receipts.glob("*.json"))),
            "events_checked": checked,
            "events_imported": imported,
        }

    def pending(self, chunk: dict[str, Any]) -> list[str]:
        return [
            call_id for call_id in expected_calls(chunk) if not self.has(f"{call_id}::attempt::1")
        ]

    def retryable(self, chunk: dict[str, Any]) -> list[str]:
        out = []
        for call_id in expected_calls(chunk):
            first = self.by_attempt.get(f"{call_id}::attempt::1")
            if (
                first is not None
                and str(first.get("status")) in b5.RETRYABLE
                and not self.has(f"{call_id}::attempt::2")
            ):
                out.append(call_id)
        return out

    def progress(
        self, chunks: list[dict[str, Any]], block_hours: dict[tuple[str, int], float]
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for cell_key, k in sorted(block_hours, key=lambda key: (b5.CELLS.index(key[0]), key[1])):
            block = [
                chunk
                for chunk in chunks
                if chunk["cell_key"] == cell_key and int(chunk["batch_size"]) == k
            ]
            planned = sum(int(chunk["calls"]) for chunk in block)
            done = sum(
                1
                for chunk in block
                for call_id in expected_calls(chunk)
                if self.has(f"{call_id}::attempt::1")
            )
            rows.append(
                {
                    "cell_key": cell_key,
                    "batch_size": k,
                    "planned_calls": planned,
                    "completed_calls": done,
                    "complete": planned == done,
                    "forecast_generation_hours": block_hours[(cell_key, k)],
                }
            )
        terminal = sum(
            1
            for event in self.by_attempt.values()
            if event.get("status") != "ok" or event.get("parse_status") != "ok"
        )
        return {
            "blocks": rows,
            "planned_calls": sum(int(row["planned_calls"]) for row in rows),
            "completed_calls": sum(int(row["completed_calls"]) for row in rows),
            "stored_attempts": len(self.by_attempt),
            "terminal_unresolved_attempts": terminal,
        }


def metered_hours(output: Path) -> dict[str, float]:
    path = output / "meter.jsonl"
    if not path.is_file():
        return {"settled_hours": 0.0, "open_lanes": 0.0}
    rows = list(read_jsonl(path))
    settled = {str(row["phase_id"]) for row in rows if row["kind"] == "settled"}
    charged = sum(float(row["charged_hours"]) for row in rows if row["kind"] == "settled")
    open_lanes = [row for row in rows if row["kind"] == "start" and row["phase_id"] not in settled]
    return {"settled_hours": charged, "open_lanes": float(len(open_lanes))}


def projection(progress: dict[str, Any], measured: float) -> dict[str, Any]:
    """Scale the forecast by what this run actually spent on the work it finished."""
    finished = sum(
        float(row["forecast_generation_hours"]) for row in progress["blocks"] if row["complete"]
    )
    total = sum(float(row["forecast_generation_hours"]) for row in progress["blocks"])
    ratio = measured / finished if finished > 0 else math.nan
    projected = total * ratio if finished > 0 else math.nan
    return {
        "measured_metered_hours": measured,
        "forecast_hours_of_completed_blocks": finished,
        "forecast_hours_of_all_blocks": total,
        "measured_over_forecast_ratio": b5.safe_float(ratio),
        "projected_total_metered_hours": b5.safe_float(projected),
    }


def calibrate_smoke(
    plan: Plan, output: Path, chunks: list[dict[str, Any]], ceiling_hours: float
) -> dict[str, Any]:
    """Turn a completed 20-block smoke run into the full-launch runtime gate."""
    ledger = Ledger(output, plan.texts)
    ledger.reconcile(chunks)
    progress = ledger.progress(chunks, plan.block_hours())
    timings = list(read_jsonl(output / "block-meter.jsonl"))
    timing_by_block: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in timings:
        key = (str(row["cell_key"]), int(row["batch_size"]))
        timing_by_block.setdefault(key, []).append(row)
    expected_blocks = set(plan.blocks())
    complete = (
        all(bool(row["complete"]) for row in progress["blocks"])
        and set(timing_by_block) == expected_blocks
    )
    coefficients = cast(dict[str, float], plan.forecast["coefficients"])
    prompt_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    missing_metrics = 0
    cache_by_block: list[dict[str, Any]] = []
    for cell_key, k in plan.blocks():
        events = [
            event
            for event in ledger.by_attempt.values()
            if int(event.get("attempt", 0)) == 1
            and event.get("cell_key") == cell_key
            and int(event.get("batch_size", 0)) == k
        ]
        block_prompt = block_cached = 0
        block_missing = 0
        for event in events:
            prompt = event.get("prompt_tokens")
            cached = event.get("num_cached_tokens")
            generated = event.get("output_tokens")
            if not isinstance(prompt, int) or not isinstance(cached, int):
                block_missing += 1
                continue
            block_prompt += prompt
            block_cached += cached
            output_tokens += int(generated or 0)
        prompt_tokens += block_prompt
        cached_tokens += block_cached
        missing_metrics += block_missing
        cache_by_block.append(
            {
                "cell_key": cell_key,
                "batch_size": k,
                "calls": len(events),
                "prompt_tokens": block_prompt,
                "cached_tokens": block_cached,
                "cache_hit_fraction": (block_cached / block_prompt if block_prompt else None),
                "metrics_complete": block_missing == 0 and bool(events),
                "prefix_reuse_observed": block_cached > 0,
            }
        )
    active_prompt_tokens = prompt_tokens - cached_tokens
    predicted_smoke_seconds = (
        coefficients["seconds_per_prompt_token"] * active_prompt_tokens
        + coefficients["seconds_per_output_token"] * output_tokens
        + coefficients["seconds_per_chunk"] * len(chunks)
    )
    measured_generation_seconds = sum(
        float(row["elapsed_seconds"]) for rows in timing_by_block.values() for row in rows
    )
    ratio = (
        measured_generation_seconds / predicted_smoke_seconds
        if predicted_smoke_seconds > 0
        else math.nan
    )
    full_generation = sum(plan.block_hours().values())
    retry_fraction = float(plan.forecast["retry_fraction"])
    settled = metered_hours(output)["settled_hours"]
    observed_overhead = max(0.0, settled - measured_generation_seconds / 3600.0)
    planned_boot = float(plan.forecast["boot_hours_per_model"]) * len(b5.CELLS)
    projected = (
        full_generation * ratio * (1.0 + retry_fraction) + max(observed_overhead, planned_boot)
        if math.isfinite(ratio)
        else math.nan
    )
    metrics_complete = missing_metrics == 0 and all(
        bool(row["metrics_complete"]) for row in cache_by_block
    )
    prefix_reuse_observed = all(bool(row["prefix_reuse_observed"]) for row in cache_by_block)
    gates = {
        "all_smoke_blocks_complete": complete,
        "cache_metrics_complete": metrics_complete,
        "prefix_reuse_observed_in_every_block": prefix_reuse_observed,
        "projection_is_finite": math.isfinite(projected),
        "projection_fits_ceiling": math.isfinite(projected) and projected <= ceiling_hours,
    }
    record = {
        "schema_version": "b5-smoke-calibration-v1",
        "at_utc": now(),
        "seal_manifest_sha256": plan.seal["manifest_sha256"],
        "schedule_sha256": plan.schedule["schedule_sha256"],
        "gates": gates,
        "passed": all(gates.values()),
        "smoke_calls": int(progress["planned_calls"]),
        "smoke_blocks": len(expected_blocks),
        "measured_smoke_metered_hours": settled,
        "measured_generation_seconds": measured_generation_seconds,
        "predicted_smoke_seconds": b5.safe_float(predicted_smoke_seconds),
        "measured_over_forecast_ratio": b5.safe_float(ratio),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_fraction": cached_tokens / prompt_tokens if prompt_tokens else None,
        "projected_full_metered_a100_hours": b5.safe_float(projected),
        "ceiling_metered_a100_hours": ceiling_hours,
        "blocks": cache_by_block,
    }
    record["record_sha256"] = sha256_json(record)
    return record


def load_calibration(plan: Plan, path: Path | None, ceiling_hours: float) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    record = json.loads(path.read_text())
    recorded_digest = record.pop("record_sha256", None)
    digest_valid = recorded_digest == sha256_json(record)
    record["record_sha256"] = recorded_digest
    projected = record.get("projected_full_metered_a100_hours")
    if (
        not digest_valid
        or record.get("schema_version") != "b5-smoke-calibration-v1"
        or record.get("seal_manifest_sha256") != plan.seal["manifest_sha256"]
        or record.get("schedule_sha256") != plan.schedule["schedule_sha256"]
        or record.get("passed") is not True
        or isinstance(projected, bool)
        or not isinstance(projected, (int, float))
        or not math.isfinite(projected)
        or float(projected) > ceiling_hours
    ):
        return None
    return cast(dict[str, Any], record)


# --- read-only gates ---------------------------------------------------------


def account_checks(required_usd: float) -> dict[str, Any]:
    """Live read-only Runpod checks. Nothing here creates or changes a resource."""
    user = cli(["user"])
    identity = user.get("user", user)
    balance = identity.get("clientBalance")
    if identity.get("id") != ACCOUNT_ID:
        raise RuntimeError("the active Runpod account differs from the verified profile")
    if (
        isinstance(balance, bool)
        or not isinstance(balance, (int, float))
        or not math.isfinite(balance)
        or float(balance) < required_usd
    ):
        raise RuntimeError("the live Runpod balance cannot fund the requested B5 run")
    pods = cli(["pod", "list", "--all"])
    if not isinstance(pods, list):
        raise RuntimeError("the Runpod Pod list response shape changed")
    if any(str(pod.get("desiredStatus", "")).upper() == "RUNNING" for pod in pods):
        raise RuntimeError("an active Pod prevents exclusive B5 metering")
    volumes = {str(row.get("id")): row for row in cli(["network-volume", "list"])}
    for cell_key in b5.CELLS:
        model_id = b5.split_cell(cell_key)[0]
        if volumes.get(VOLUMES[model_id], {}).get("dataCenterId") != DATA_CENTER:
            raise RuntimeError(f"the {model_id} model volume is absent from {DATA_CENTER}")
    catalog = cli(["gpu", "list", "--include-unavailable"])
    gpu = next(row for row in catalog if row.get("gpuId") == GPU_ID)
    placement = next(
        row
        for row in gpu.get("dataCenterAvailability", [])
        if row.get("dataCenterId") == DATA_CENTER
    )
    rate = gpu.get("securePricePerHr")
    if (
        gpu.get("secureCloud") is not True
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not 0 < float(rate) <= MAX_RATE
        or str(placement.get("stockStatus", "none")).lower() not in {"low", "medium", "high"}
    ):
        raise RuntimeError("the approved B5 GPU is unavailable or above the protected rate")
    return {
        "at_utc": now(),
        "account_id": identity["id"],
        "balance_usd": balance,
        "required_balance_usd": required_usd,
        "gpu_price_usd_per_hour": rate,
        "storage_spend_usd_per_hour": identity.get("currentSpendPerHr"),
        "placement": placement,
        "running_pods": 0,
    }


def preflight(
    plan: Plan,
    output: Path,
    chunks: list[dict[str, Any]],
    *,
    ceiling_hours: float,
    usd_per_hour: float,
    reserve_usd: float,
    live: bool,
    confirm: str | None,
    smoke: bool = False,
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    ledger = Ledger(output, plan.texts)
    reconciled = ledger.reconcile(chunks)
    progress = ledger.progress(chunks, plan.block_hours())
    measured = metered_hours(output)
    calibration = None if smoke else load_calibration(plan, calibration_path, ceiling_hours)
    point = (
        SMOKE_CEILING_HOURS
        if smoke
        else float(calibration["projected_full_metered_a100_hours"])
        if calibration is not None
        else ceiling_hours
    )
    remaining = max(0.0, point - float(measured["settled_hours"]))
    required = remaining * usd_per_hour + reserve_usd
    gates: dict[str, Any] = {
        "seal_verifies": True,
        "schedule_counts_verify": True,
        "prompt_and_schema_digests_verify": True,
        "context_fits_every_size": True,
        "cpu_tests_evidence_present": (STUDY / "extension-b5" / "CPU_TESTS.json").is_file(),
        "no_conflicting_active_run": measured["open_lanes"] == 0,
        "ceiling_covers_runtime_budget": point <= ceiling_hours,
        "confirmation_token_supplied": confirm == plan.seal["confirmation_token"],
        "smoke_calibration_passed_or_is_smoke": smoke or calibration is not None,
    }
    checks: dict[str, Any] | None = None
    if live:
        checks = account_checks(required)
        gates["balance_covers_block_plus_reserve"] = True
        gates["no_active_pod"] = True
        gates["gpu_available_at_protected_rate"] = True
    record = {
        "schema_version": "b5-preflight-v1",
        "at_utc": now(),
        "seal": plan.seal,
        "gates": gates,
        "passed": all(bool(value) for value in gates.values()),
        "live_account_checks": checks,
        "required_balance_usd": required,
        "reserve_usd": reserve_usd,
        "ceiling_metered_a100_hours": ceiling_hours,
        "point_metered_a100_hours": point,
        "smoke": smoke,
        "smoke_calibration": calibration,
        "upper_metered_a100_hours": float(plan.forecast["upper_metered_a100_hours"]),
        "receipt_reconciliation": reconciled,
        "progress": progress,
        "metered": measured,
        "pod_created": False,
    }
    return record


# --- execution ---------------------------------------------------------------


class B5Pod:
    """One A100 PCIe 80GB Pod for one model. Creation happens only inside `--launch`."""

    def __init__(self, guard: Guard, output: Path, model: str, archive: bytes) -> None:
        self.guard, self.output, self.model, self.archive = guard, output, model, archive
        self.pod_id: str | None = None
        self.ssh: list[str] = []
        self.identity = uuid.uuid4().hex
        self.remote = f"/workspace/b5-runtime/{sha256_json([model, len(archive)])}"
        self.loaded: set[str] = set()

    def audit(self, kind: str, **fields: Any) -> None:
        append(
            self.output / "runpod-lifecycle" / f"{MODEL_KEYS[self.model]}.jsonl",
            [{"kind": kind, "at_utc": now(), "model_id": self.model, **fields}],
        )

    def start(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        self.guard.verify_watchdog()
        if self.guard.lease_id is None:
            raise RuntimeError("the B5 watchdog lacks a lease")
        name = f"b4-{self.guard.lease_id}-{self.identity[:12]}"
        self.audit("create_intent", name=name)
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
                if len(matches) > 1:
                    raise RuntimeError("ambiguous B5 Pod creation produced duplicates") from exc
                if len(matches) == 1:
                    self.pod_id = str(matches[0]["id"])
            if self.pod_id is None:
                raise
            pod = cli(["pod", "get", self.pod_id])
        self.audit("created", pod_id=self.pod_id)
        rate = pod.get("costPerHr")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not 0 < float(rate) <= MAX_RATE
            or pod.get("gpuCount") != 1
            or pod.get("imageName") != IMAGE
            or pod.get("networkVolumeId") != VOLUMES[self.model]
        ):
            raise RuntimeError("the created B5 Pod differs from the guarded configuration")
        ssh_deadline = min(deadline, time.monotonic() + 600)
        while time.monotonic() < ssh_deadline:
            try:
                self.ssh = ssh_args(
                    cli(["ssh", "info", self.pod_id]), self.output / "runpod-known-hosts"
                )
                break
            except (CommandError, KeyError, ValueError):
                Event().wait(5)
        if not self.ssh:
            raise RuntimeError("the B5 Pod SSH endpoint did not become ready")
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
                f"{self.remote}/b5_worker.py",
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
                policy = ext_policy_for(self.model)
                backend = cast(dict[str, Any], ready.get("backend", {}))
                if (
                    ready.get("structured") is not True
                    or backend.get("model_id") != self.model
                    or backend.get("model_revision") != policy.revision
                    or backend.get("batch_invariant") is not True
                    or backend.get("enable_prefix_caching") is not True
                    or backend.get("prefix_caching_required") is not True
                    or backend.get("max_num_seqs") != 32
                    or backend.get("dtype") != "bfloat16"
                    or str(ready.get("host", {}).get("installed_engine_version_string")) != "0.28.0"
                ):
                    raise RuntimeError("the B5 worker provenance differs from the sealed policy")
                self.audit("ready", pod_id=self.pod_id, backend=backend)
                return
            except (CommandError, json.JSONDecodeError):
                Event().wait(2)
        raise RuntimeError("the B5 worker did not become ready before its deadline")

    def rpc(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        argv = [
            "/opt/b4/bin/python",
            f"{self.remote}/b5_worker.py",
            "request",
            "--timeout",
            str(max(0.01, timeout)),
        ]
        raw = command([*self.ssh, shlex.join(argv)], timeout, data=json.dumps(payload).encode())
        return cast(dict[str, Any], json.loads(raw))

    def load_frame(self, plan: Plan, cell_key: str, timeout: float) -> None:
        task_id = b5.split_cell(cell_key)[1]
        if task_id in self.loaded:
            return
        order = plan.doc_ids(cell_key)
        texts = plan.texts(cell_key)
        response = self.rpc(
            {
                "operation": "frame",
                "task": task_id,
                "doc_order": order,
                "doc_order_sha256": sha256_json(order),
                "documents": texts,
                "documents_sha256": sha256_json([texts[doc_id] for doc_id in order]),
            },
            timeout,
        )
        if response.get("loaded") is not True or int(response.get("items", 0)) != len(order):
            raise RuntimeError(f"the B5 worker did not load the {task_id} frame")
        self.loaded.add(task_id)

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
                    self.audit("termination_verified", pod_id=self.pod_id)
                    return
                raise
            Event().wait(1)
        raise RuntimeError("B5 Pod deletion could not be verified; the meter stays open")


def run_chunk(
    plan: Plan,
    pod: B5Pod,
    ledger: Ledger,
    output: Path,
    namespace: str,
    chunk: dict[str, Any],
    call_ids: list[str],
    attempt: int,
) -> dict[str, int]:
    calls = expected_calls(chunk)
    attempt_ids = [
        f"{call_id}::attempt::{attempt}" for call_id in calls if call_id in set(call_ids)
    ]
    remote_chunk_id = sha256_json(attempt_ids)
    payload = pod.rpc(
        {
            "operation": "run",
            "namespace": namespace,
            "model_key": MODEL_KEYS[pod.model],
            "task": b5.split_cell(str(chunk["cell_key"]))[1],
            "chunk_id": remote_chunk_id,
            "attempt": attempt,
            "only_call_ids": call_ids,
            "groups": chunk["groups"],
        },
        CHUNK_TIMEOUT_SECONDS,
    )
    events = cast(list[dict[str, Any]], payload["events"])
    if {str(event["attempt_id"]) for event in events} != set(attempt_ids):
        raise RuntimeError("the B5 receipt identities differ from the dispatched chunk")
    texts = plan.texts(str(chunk["cell_key"]))
    for event in events:
        verify_event(texts, event, calls[str(event["call_id"])], attempt)
    receipt = {
        **payload,
        "source_chunk_id": chunk["chunk_id"],
        "attempt": attempt,
        "written_at_utc": now(),
    }
    write_json(output / "receipts" / f"{remote_chunk_id}.json", receipt)
    return {"stored": ledger.add(events), "returned": len(events)}


def execute(
    plan: Plan,
    output: Path,
    state: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    ceiling_hours: float,
    usd_per_hour: float,
    reserve_usd: float,
    runtime_budget_hours: float,
    active_cells: tuple[str, ...] = tuple(b5.CELLS),
) -> dict[str, Any]:
    ledger = Ledger(output, plan.texts)
    ledger.reconcile(chunks)
    archive = command(
        ["tar", "--exclude=__pycache__", "-cf", "-", "-C", str(STUDY), "src", "b5_worker.py"],
        60,
    )
    for cell_key in active_cells:
        model_id = b5.split_cell(cell_key)[0]
        outstanding = [
            chunk
            for chunk in chunks
            if chunk["cell_key"] == cell_key and (ledger.pending(chunk) or ledger.retryable(chunk))
        ]
        if not outstanding:
            continue
        measured = metered_hours(output)["settled_hours"]
        if measured >= ceiling_hours:
            raise RuntimeError("the B5 metered ceiling is reached; no further Pod is created")
        remaining = max(0.0, runtime_budget_hours - measured)
        account_checks(remaining * usd_per_hour + reserve_usd)
        guard = Guard(output)
        guard.arm_watchdog(1, WATCHDOG_TARGET_SECONDS)
        pod = B5Pod(guard, output, model_id, archive)
        phase = uuid.uuid4().hex
        started = dt.datetime.now(dt.UTC)
        append(
            output / "meter.jsonl",
            [
                {
                    "kind": "start",
                    "phase_id": phase,
                    "cell_key": cell_key,
                    "model_id": model_id,
                    "started_at_utc": started.isoformat(),
                    "ceiling_hours": ceiling_hours,
                    "seal_manifest_sha256": plan.seal["manifest_sha256"],
                }
            ],
        )
        try:
            pod.start(POD_START_TIMEOUT_SECONDS)
            pod.load_frame(plan, cell_key, 600)
            for k in b5.SIZES:
                block_chunks = [c for c in outstanding if int(c["batch_size"]) == k]
                if not block_chunks:
                    continue
                block_started = dt.datetime.now(dt.UTC)
                for chunk in block_chunks:
                    pending = ledger.pending(chunk)
                    if pending:
                        run_chunk(
                            plan, pod, ledger, output, str(state["namespace"]), chunk, pending, 1
                        )
                    retry = ledger.retryable(chunk)
                    if retry:
                        run_chunk(
                            plan, pod, ledger, output, str(state["namespace"]), chunk, retry, 2
                        )
                    lane = (dt.datetime.now(dt.UTC) - started).total_seconds() / 3600
                    if metered_hours(output)["settled_hours"] + lane > ceiling_hours:
                        raise RuntimeError("the B5 metered ceiling is reached mid-block")
                block_finished = dt.datetime.now(dt.UTC)
                append(
                    output / "block-meter.jsonl",
                    [
                        {
                            "phase_id": phase,
                            "cell_key": cell_key,
                            "batch_size": k,
                            "started_at_utc": block_started.isoformat(),
                            "finished_at_utc": block_finished.isoformat(),
                            "elapsed_seconds": (block_finished - block_started).total_seconds(),
                            "chunks": len(block_chunks),
                            "calls": sum(int(chunk["calls"]) for chunk in block_chunks),
                        }
                    ],
                )
                progress = ledger.progress(chunks, plan.block_hours())
                spent = (
                    metered_hours(output)["settled_hours"]
                    + (dt.datetime.now(dt.UTC) - started).total_seconds() / 3600
                )
                forward = projection(progress, spent)
                append(output / "projection.jsonl", [{"at_utc": now(), **forward}])
                projected = forward["projected_total_metered_hours"]
                if projected is not None and float(projected) > ceiling_hours:
                    raise RuntimeError(
                        "the measured throughput projects past the B5 ceiling; the run stops "
                        "with a complete ledger rather than overspending"
                    )
        finally:
            pod.close()
            guard.release_watchdog()
            finished = dt.datetime.now(dt.UTC)
            append(
                output / "meter.jsonl",
                [
                    {
                        "kind": "settled",
                        "phase_id": phase,
                        "finished_at_utc": finished.isoformat(),
                        "charged_hours": (finished - started).total_seconds() / 3600 + 60 / 3600,
                        "accounting": "lane wall clock plus a sixty second settlement",
                    }
                ],
            )
    return ledger.progress(chunks, plan.block_hours())


# --- state and reporting -----------------------------------------------------


def open_state(output: Path, plan: Plan, chunk_calls: int, smoke: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "RUN.json"
    if path.exists():
        state = json.loads(path.read_text())
        if (
            state["schedule_sha256"] != plan.schedule["schedule_sha256"]
            or state["seal_manifest_sha256"] != plan.seal["manifest_sha256"]
            or int(state["chunk_calls"]) != chunk_calls
            or bool(state["smoke"]) != smoke
        ):
            raise ValueError("this output directory belongs to a different B5 run")
        return cast(dict[str, Any], state)
    state = {
        "schema_version": RUN_SCHEMA,
        "namespace": "b5-" + uuid.uuid4().hex[:24],
        "created_at_utc": now(),
        "seal": plan.seal["seal"],
        "seal_manifest_sha256": plan.seal["manifest_sha256"],
        "schedule_sha256": plan.schedule["schedule_sha256"],
        "chunk_calls": chunk_calls,
        "smoke": smoke,
    }
    write_json(path, state)
    return state


def dry_run(plan: Plan, output: Path, chunks: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    by_size: dict[str, dict[str, int]] = {}
    for chunk in chunks:
        row = by_size.setdefault(str(chunk["batch_size"]), {"chunks": 0, "calls": 0})
        row["chunks"] += 1
        row["calls"] += int(chunk["calls"])
    first = expected_calls(chunks[0])
    last = expected_calls(chunks[-1])
    record = {
        "schema_version": "b5-dry-run-v1",
        "at_utc": now(),
        "smoke": smoke,
        "seal": plan.seal,
        "chunks": len(chunks),
        "calls": sum(int(chunk["calls"]) for chunk in chunks),
        "by_size": by_size,
        "expected_counts": b5.expected_counts(),
        "observed_counts": plan.schedule["observed_counts"],
        "first_call_ids": list(first)[:3],
        "last_call_ids": list(last)[-3:],
        "context_audit": [
            {"cell_key": row["cell_key"], "worst_case": row["worst_case"], "fits": row["fits"]}
            for row in plan.context["cells"]
        ],
        "forecast": {
            "point_metered_a100_hours": plan.forecast["point_metered_a100_hours"],
            "upper_metered_a100_hours": plan.forecast["upper_metered_a100_hours"],
            "planning_point_balance_usd": plan.forecast["planning_point_balance_usd"],
            "full_launch_required_balance_usd": plan.forecast["full_launch_required_balance_usd"],
            "full_launch_requires_smoke_calibration": True,
            "smoke_required_balance_usd": plan.forecast["smoke_required_balance_usd"],
        },
        "pod_created": False,
        "gpu_work_launched": False,
    }
    write_json(output / f"DRY_RUN-{'smoke' if smoke else 'full'}.json", record, allow_existing=True)
    return record


def verify_only(plan: Plan, output: Path, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    ledger = Ledger(output, plan.texts)
    reconciled = ledger.reconcile(chunks) if output.is_dir() else {}
    return {
        "schema_version": "b5-verify-v1",
        "at_utc": now(),
        "seal": plan.seal,
        "schedule_rebuilt_and_matched": True,
        "receipt_reconciliation": reconciled,
        "progress": ledger.progress(chunks, plan.block_hours()) if output.is_dir() else {},
        "pod_created": False,
    }


def status(plan: Plan, output: Path, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    ledger = Ledger(output, plan.texts)
    progress = ledger.progress(chunks, plan.block_hours())
    measured = metered_hours(output)
    forward = projection(progress, float(measured["settled_hours"]))
    done = int(progress["completed_calls"])
    total = int(progress["planned_calls"])
    remaining = forward["projected_total_metered_hours"]
    return {
        "schema_version": "b5-status-v1",
        "at_utc": now(),
        "seal": plan.seal["seal"],
        "percent_complete": round(100.0 * done / total, 3) if total else 0.0,
        "completed_calls": done,
        "planned_calls": total,
        "metered": measured,
        "projection": forward,
        "estimated_remaining_hours": (
            None
            if remaining is None
            else max(0.0, float(remaining) - float(measured["settled_hours"]))
        ),
        "blocks": progress["blocks"],
        "terminal_unresolved_attempts": progress["terminal_unresolved_attempts"],
        "pod_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "preflight", "launch", "resume", "status", "verify-only"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--model-phase",
        choices=tuple(MODEL_PHASES),
        help="run one of two disjoint two-model phases while retaining the full ledger",
    )
    parser.add_argument("--chunk-calls", type=int, default=b5.CHUNK_CALLS)
    parser.add_argument("--ceiling-hours", type=float, default=CEILING_HOURS)
    parser.add_argument("--usd-per-hour", type=float, default=MAX_RATE)
    parser.add_argument("--reserve-usd", type=float, default=RESERVE_USD)
    parser.add_argument("--confirm", help="the confirmation token printed by seal_b5.py")
    parser.add_argument(
        "--calibration",
        type=Path,
        help="SMOKE_CALIBRATION.json from a completed all-cell B5 smoke run",
    )
    args = parser.parse_args()
    if args.smoke and args.model_phase is not None:
        parser.error("--model-phase cannot be combined with --smoke")
    creates_pod = bool(args.launch or args.resume)
    plan = Plan(args.seal or seal_b5.latest(), recompute=not args.status)
    chunk_calls = SMOKE_CHUNK_CALLS if args.smoke else args.chunk_calls
    chunks = selected_chunks(plan, chunk_calls, args.smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "receipts").mkdir(exist_ok=True)
    with (args.output / ".active-run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.dry_run:
            report: dict[str, Any] = dry_run(plan, args.output, chunks, args.smoke)
        elif args.verify_only:
            report = verify_only(plan, args.output, chunks)
        elif args.status:
            report = status(plan, args.output, chunks)
        elif args.preflight:
            report = preflight(
                plan,
                args.output,
                chunks,
                ceiling_hours=args.ceiling_hours,
                usd_per_hour=args.usd_per_hour,
                reserve_usd=args.reserve_usd,
                live=True,
                confirm=args.confirm,
                smoke=args.smoke,
                calibration_path=args.calibration,
            )
            write_json(args.output / f"preflight-{uuid.uuid4().hex}.json", report)
        else:
            state = open_state(args.output, plan, chunk_calls, args.smoke)
            gate = preflight(
                plan,
                args.output,
                chunks,
                ceiling_hours=args.ceiling_hours,
                usd_per_hour=args.usd_per_hour,
                reserve_usd=args.reserve_usd,
                live=True,
                confirm=args.confirm,
                smoke=args.smoke,
                calibration_path=args.calibration,
            )
            write_json(args.output / f"preflight-{uuid.uuid4().hex}.json", gate)
            if not gate["passed"]:
                failed = [key for key, value in gate["gates"].items() if not value]
                raise SystemExit(f"B5 preflight refused the launch: {failed}")
            calibration = gate.get("smoke_calibration")
            if args.smoke:
                runtime_budget = SMOKE_CEILING_HOURS
            elif isinstance(calibration, dict):
                runtime_budget = float(calibration["projected_full_metered_a100_hours"])
            else:
                raise RuntimeError("the passing B5 preflight lacks its smoke calibration")
            run_progress = execute(
                plan,
                args.output,
                state,
                chunks,
                ceiling_hours=args.ceiling_hours,
                usd_per_hour=args.usd_per_hour,
                reserve_usd=args.reserve_usd,
                runtime_budget_hours=runtime_budget,
                active_cells=(
                    MODEL_PHASES[args.model_phase]
                    if args.model_phase is not None
                    else tuple(b5.CELLS)
                ),
            )
            smoke_calibration = None
            if args.smoke:
                calibration_output = args.output / "SMOKE_CALIBRATION.json"
                if calibration_output.is_file():
                    smoke_calibration = json.loads(calibration_output.read_text())
                else:
                    smoke_calibration = calibrate_smoke(
                        plan, args.output, chunks, args.ceiling_hours
                    )
                    write_json(calibration_output, smoke_calibration)
            report = {
                "schema_version": "b5-execution-v1",
                "namespace": state["namespace"],
                "progress": run_progress,
                "metered": metered_hours(args.output),
                "model_phase": args.model_phase,
                "active_cells": (
                    list(MODEL_PHASES[args.model_phase])
                    if args.model_phase is not None
                    else list(b5.CELLS)
                ),
                "smoke_calibration": smoke_calibration,
                "pod_created": True,
            }
            write_json(args.output / f"EXECUTION-{uuid.uuid4().hex}.json", report)
    print(json.dumps(report, indent=2, default=str))
    if not creates_pod and report.get("pod_created") is not False:
        raise SystemExit("a read-only B5 mode reported a Pod; refusing to continue")


if __name__ == "__main__":
    main()
