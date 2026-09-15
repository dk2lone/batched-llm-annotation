"""Prepare B4 locally. Paid execution is opt-in and requires fresh budget evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from threading import Event
from typing import Any

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb import extension as ext  # noqa: E402
from freezeb import extension_b4 as b4  # noqa: E402
from freezeb.prompts import build_messages  # noqa: E402
from step0.hashing import sha256_json, write_json  # noqa: E402


def phases(prefix: int) -> list[tuple[str, str, int]]:
    return [(cell, "reference", 250) for cell in b4.CELLS] + [
        (cell, "evaluation", count)
        for count in ext.E5_EVALUATION_PREFIXES
        if count <= prefix
        for cell in b4.CELLS
    ]


def meter(out: Path, budget: dict[str, Any]) -> float:
    records = b4.events(out / "meter.jsonl")
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("kind") not in ("start", "settled") or not row.get("phase_id"):
            raise ValueError("invalid B4 meter record")
        destination = starts if row["kind"] == "start" else ends
        if row["phase_id"] in destination:
            raise ValueError("duplicate B4 meter phase record")
        destination[row["phase_id"]] = row
    if set(starts) != set(ends):
        raise RuntimeError("unsettled GPU phase: reconcile billing before resuming")
    for identity, end in ends.items():
        if end["budget_sha256"] != starts[identity]["budget_sha256"]:
            raise ValueError("meter phase budget binding differs")
        value = end["charged_hours"]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid metered GPU charge")
    # The budget baseline is prior study use, excluding this output directory.
    return float(budget["prior_extension_a100_hours"]) + sum(
        float(row["charged_hours"]) for row in ends.values()
    )


def dispatch(
    study: Path,
    out: Path,
    plan: dict[str, Any],
    budget: dict[str, Any],
    transport: Any,
    chunk_size: int = 32,
) -> None:
    """Four independent lanes, with a shared budget and barriers between prefixes."""
    if not 1 <= chunk_size <= 64:
        raise ValueError("chunk size must be 1 through 64")
    meter(out, budget)
    for _, group in groupby(phases(int(plan["evaluation_prefix"])), key=lambda phase: phase[1:]):
        prepared = []
        for cell, stream, count in group:
            rows = b4.frame(study, cell.rsplit("::", 1)[1])
            full_count = 250 if stream == "reference" else int(plan["evaluation_prefix"])
            all_jobs = list(b4.jobs(cell, rows, stream, full_count, render=False))
            path = b4.stream_path(out, cell, stream)
            cache = b4.validate_events(all_jobs, b4.events(path))
            planned = all_jobs[: count * 32]
            ids = {job["call_id"] for job in planned}
            if b4.pending(planned, [row for row in cache.values() if row["call_id"] in ids]):
                prepared.append((cell, stream, count, rows, all_jobs, path, cache, planned))
        if not prepared:
            continue
        if len(prepared) > 4 or len({phase[0] for phase in prepared}) != len(prepared):
            raise ValueError("B4 round must have at most four distinct cell lanes")
        used = meter(out, budget)
        available = min(
            80 - used - 10,
            (budget["spend_limit_usd"] - budget["spend_used_usd"]) / budget["a100_usd_per_hour"]
            - (used - budget["prior_extension_a100_hours"]),
        )
        # Allocate separate deadlines. Giving each lane the entire remaining
        # budget would allow four lanes to overspend it simultaneously.
        allowances = [
            1.6 * float(budget["base_forecast_hours"][f"{cell}|{stream}"])
            for cell, stream, *_ in prepared
        ]
        if sum(allowances) + len(prepared) * 60 / 3600 > available:
            raise RuntimeError("complete parallel B4 round does not fit protected budget")
        identities = [uuid.uuid4().hex for _ in prepared]
        b4.append(
            out / "meter.jsonl",
            [
                {
                    "kind": "start",
                    "phase_id": identity,
                    "cell_key": phase[0],
                    "stream": phase[1],
                    "prefix": phase[2],
                    "budget_sha256": budget["record_sha256"],
                    "guarded_forecast_hours": allowance,
                    "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                }
                for identity, phase, allowance in zip(identities, prepared, allowances, strict=True)
            ],
        )
        stop = Event()
        failures: list[BaseException] = []
        charges: dict[str, float] = {}
        # One Modal app context owns every lane in this round. A lane must not
        # stop the shared app while another cell is still generating.
        with transport.session():
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="b4-cell") as pool:
                futures: dict[Future[float], str] = {}
                failure: BaseException | None = None
                try:
                    for identity, phase, allowance in zip(
                        identities, prepared, allowances, strict=True
                    ):
                        futures[
                            pool.submit(
                                run_phase,
                                out,
                                plan,
                                transport,
                                phase,
                                allowance,
                                stop,
                                chunk_size,
                                failures,
                            )
                        ] = identity
                    for future in as_completed(futures):
                        try:
                            charged = future.result()
                        except BaseException as exc:
                            stop.set()
                            if failure is None:
                                failure = exc
                        else:
                            charges[futures[future]] = charged
                except BaseException:
                    stop.set()
                    raise
                if failure is not None:
                    raise failures[0] if failures else failure
        # Settle only after shared app shutdown succeeds. A shutdown failure
        # cannot silently turn uncertain remote billing into a completed round.
        b4.append(
            out / "meter.jsonl",
            [
                {
                    "kind": "settled",
                    "phase_id": identity,
                    "budget_sha256": budget["record_sha256"],
                    "charged_hours": charges[identity],
                    "accounting": "per_lane_wall_plus_60s_settlement",
                }
                for identity in identities
            ],
        )


def run_phase(
    out: Path,
    plan: dict[str, Any],
    transport: Any,
    phase: tuple[Any, ...],
    allowance: float,
    stop: Event,
    chunk_size: int,
    failures: list[BaseException],
) -> float:
    cell, stream, _, rows, all_jobs, path, cache, planned = phase
    started = time.monotonic()
    deadline = started + allowance * 3600
    ids = {job["call_id"] for job in planned}
    try:
        if stop.is_set():
            raise RuntimeError("B4 peer lane failed; round stopped")
        with transport.phase(cell, stream, plan["schedules"][cell]) as worker:
            namespace = (
                "b4-"
                + sha256_json(
                    {
                        "schedule": plan["schedules"][cell],
                        "stream": stream,
                        "output": str(out.resolve()),
                    }
                )[:24]
            )
            recovered = worker.recover(namespace, deadline - time.monotonic(), stop)
            recover_chunks(out, path, all_jobs, recovered, plan, cell, cache)
            by_id = {str(row["doc_id"]): row for row in rows}
            while True:
                stored = [event for event in cache.values() if event["call_id"] in ids]
                work = b4.pending(planned, stored)
                if not work:
                    break
                for offset in range(0, len(work), chunk_size):
                    group = []
                    for job in work[offset : offset + chunk_size]:
                        documents = [
                            (doc_id, str(by_id[doc_id]["text"])) for doc_id in job["doc_order"]
                        ]
                        messages = build_messages(job["task_id"], documents, job["target_ids"])
                        if sha256_json(messages) != job["prompt_sha256"]:
                            raise ValueError("B4 rendered prompt changed")
                        group.append({**job, "messages": messages})
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or stop.is_set():
                        raise RuntimeError("B4 protected deadline reached or peer lane failed")
                    chunk_id = sha256_json([job["attempt_id"] for job in group])
                    payload = worker.run(namespace, chunk_id, group, remaining, stop)
                    ingest(out, path, all_jobs, payload, plan, cell, cache)
                    if payload.get("chunk_id") != chunk_id or {
                        event["attempt_id"] for event in payload.get("events", [])
                    } != {job["attempt_id"] for job in group}:
                        raise ValueError("B4 receipt does not match dispatched attempts")
        charged = (time.monotonic() - started + 60) / 3600
        if charged > allowance + 60 / 3600:
            raise RuntimeError("B4 lane exceeded reserved time; reconcile billing")
        return charged
    except BaseException as exc:
        # Ambiguous lanes stay unsettled. Peers cancel their outstanding calls
        # and no later reference/evaluation round can launch.
        failures.append(exc)
        stop.set()
        raise


def ingest(
    out: Path,
    path: Path,
    planned: list[dict[str, Any]],
    payload: dict[str, Any],
    plan: dict[str, Any],
    cell: str,
    cache: dict[str, dict[str, Any]] | None = None,
    *,
    persist_receipt: bool = True,
) -> None:
    # Persist the receipt before validation, including rejected serving output.
    receipt_path = out / "receipts" / f"{sha256_json(payload)}.json"
    if persist_receipt and not receipt_path.exists():
        write_json(receipt_path, payload)
    b4.verify_serving(payload, plan, cell)
    if cache is None:
        cache = b4.validate_events(planned, b4.events(path))
    incoming = payload["events"]
    calls = {event["call_id"] for event in incoming}
    existing = [event for event in cache.values() if event["call_id"] in calls]
    expected_jobs = [job for job in planned if job["call_id"] in calls]
    seen = b4.validate_events(expected_jobs, existing + incoming)
    b4.append(path, [event for key, event in seen.items() if key not in cache])
    cache.update(seen)


def recover_chunks(
    out: Path,
    path: Path,
    planned: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    plan: dict[str, Any],
    cell: str,
    cache: dict[str, dict[str, Any]],
) -> None:
    # Remote filenames are content hashes, not chronological. Validate serving
    # receipts first, then reconcile first attempts before linked retries.
    for payload in payloads:
        receipt_path = out / "receipts" / f"{sha256_json(payload)}.json"
        if not receipt_path.exists():
            write_json(receipt_path, payload)
        identity = sha256_json([event["attempt_id"] for event in payload["events"]])
        if payload.get("chunk_id") != identity:
            raise ValueError("recovered B4 chunk identity differs")
    for attempt in (1, 2):
        for payload in payloads:
            subset = [event for event in payload["events"] if event["attempt"] == attempt]
            if subset:
                ingest(
                    out,
                    path,
                    planned,
                    {**payload, "events": subset},
                    plan,
                    cell,
                    cache,
                    persist_receipt=False,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=int, choices=(400, 1000, 2500), default=400)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--budget-record", type=Path)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    plan = b4.build_plan(STUDY, args.prefix)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".active-run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / f"PLAN-{args.prefix}.json"
        if path.exists():
            if b4.load(path) != plan:
                raise ValueError("existing immutable B4 plan differs")
        else:
            write_json(path, plan)
        if args.execute:
            if not args.profile or args.budget_record is None:
                raise ValueError("execution requires an explicit profile and fresh budget record")
            budget = b4.load(args.budget_record)
            b4.verify_budget(budget, plan, args.profile)
            meter(args.output, budget)
            budget_path = args.output / "budgets" / f"{budget['record_sha256']}.json"
            if budget_path.exists():
                if b4.load(budget_path) != budget:
                    raise ValueError("existing budget artifact differs")
            else:
                write_json(budget_path, budget)
            # No Modal imports, deployments, or remote calls on the default path.
            from extension_b4_transport import ModalTransport

            dispatch(
                STUDY, args.output, plan, budget, ModalTransport(args.profile), args.chunk_size
            )
        print(
            json.dumps(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "first_calls": plan["first_calls"],
                    "max_concurrent_a100s": plan["max_concurrent_a100s"],
                    "gpu_work_launched": bool(args.execute),
                    "output": str(args.output),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
