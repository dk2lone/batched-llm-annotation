"""Prepare the four-A100 Runpod B4 plan locally. --execute alone is insufficient."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from modal_extension_b4 import dispatch, meter, recover_chunks  # noqa: E402

from freezeb import extension_b4 as b4  # noqa: E402
from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402


def verify_resume_plan(previous: dict[str, Any], current: dict[str, Any]) -> None:
    ignored = {"plan_sha256", "implementation_sha256"}
    old = {key: value for key, value in previous.items() if key not in ignored}
    new = {key: value for key, value in current.items() if key not in ignored}
    if old != new:
        raise ValueError("resume source scientific plan differs")
    old_impl = previous.get("implementation_sha256", {})
    new_impl = current.get("implementation_sha256", {})
    changed = {
        key for key in set(old_impl) | set(new_impl) if old_impl.get(key) != new_impl.get(key)
    }
    reviewed = {"extension_b4_runpod.py", "runpod_extension_b4.py"}
    if not changed <= reviewed:
        raise ValueError("resume requires the reviewed implementation or its direct retry")


def verify_resume_lifecycle(source: Path) -> None:
    lifecycle = [
        row
        for path in sorted((source / "runpod-lifecycle").glob("*.jsonl"))
        for row in b4.events(path)
    ]
    created = {row["pod_id"] for row in lifecycle if row.get("kind") == "created"}
    terminated = {
        row["pod_id"] for row in lifecycle if row.get("kind") == "termination_verified"
    }
    if created != terminated:
        raise ValueError("resume source does not prove termination of every created Pod")


def carry_forward(source: Path, destination: Path, plan: dict[str, Any]) -> None:
    source = source.resolve()
    if source == destination.resolve() or not source.is_dir():
        raise ValueError("resume source must be a distinct existing run directory")
    if (destination / "resume-record.json").exists() or (destination / "receipts").exists():
        raise ValueError("resume destination is not empty")
    with (source / ".active-run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = b4.load(source / f"PLAN-{plan['evaluation_prefix']}.json")
        verify_resume_plan(previous, plan)
        meter(source, b4.load(source / "budget-record.json"))
        verify_resume_lifecycle(source)
        receipt_paths = sorted((source / "receipts").glob("*.json"))
        payloads = [b4.load(path) for path in receipt_paths]
        if not payloads:
            raise ValueError("resume source has no durable receipts")
        contexts: dict[tuple[str, str], list[dict[str, Any]]] = {}
        call_owner: dict[str, tuple[str, str]] = {}
        for cell in b4.CELLS:
            rows = b4.frame(STUDY, cell.rsplit("::", 1)[1])
            for stream, count in (
                ("reference", 250),
                ("evaluation", int(plan["evaluation_prefix"])),
            ):
                jobs = list(b4.jobs(cell, rows, stream, count, render=False))
                key = (cell, stream)
                contexts[key] = jobs
                call_owner.update({str(job["call_id"]): key for job in jobs})
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for payload in payloads:
            owners = {
                call_owner.get(str(event.get("call_id", "")))
                for event in payload.get("events", [])
            }
            if None in owners or len(owners) != 1:
                raise ValueError("resume receipt contains foreign or mixed calls")
            owner = owners.pop()
            assert owner is not None
            grouped.setdefault(owner, []).append(payload)
        imported = 0
        for (cell, stream), group in grouped.items():
            jobs = contexts[(cell, stream)]
            path = b4.stream_path(destination, cell, stream)
            cache: dict[str, dict[str, Any]] = {}
            recover_chunks(destination, path, jobs, group, plan, cell, cache)
            imported += len(cache)
        write_json(
            destination / "resume-record.json",
            {
                "schema_version": "b4-runpod-transport-resume-v1",
                "source": str(source),
                "source_plan_sha256": previous["plan_sha256"],
                "destination_plan_sha256": plan["plan_sha256"],
                "source_meter_sha256": sha256_file(source / "meter.jsonl"),
                "receipt_sha256": {path.name: sha256_file(path) for path in receipt_paths},
                "imported_receipts": len(receipt_paths),
                "imported_events": imported,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=int, choices=(400, 1000, 2500), default=400)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", help="verified Runpod account ID")
    parser.add_argument("--budget-record", type=Path)
    parser.add_argument("--runpod-config", type=Path)
    parser.add_argument("--preflight-evidence", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-concurrent-lanes", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--gpu-type-id",
        choices=("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"),
        default="NVIDIA A100-SXM4-80GB",
    )
    args = parser.parse_args()
    plan = b4.build_plan(
        STUDY, args.prefix, "runpod", args.gpu_type_id, args.max_concurrent_lanes
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".active-run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = args.output / f"PLAN-{args.prefix}.json"
        if plan_path.exists():
            if b4.load(plan_path) != plan:
                raise ValueError("existing immutable B4 plan differs")
        else:
            write_json(plan_path, plan)
        if args.execute:
            if not all(
                (args.profile, args.budget_record, args.runpod_config, args.preflight_evidence)
            ):
                raise ValueError(
                    "execution requires account, budget, configuration and smoke evidence"
                )
            assert args.budget_record is not None and args.runpod_config is not None
            assert args.preflight_evidence is not None and args.profile is not None
            budget, config = b4.load(args.budget_record), b4.load(args.runpod_config)
            b4.verify_budget(budget, plan, args.profile)
            meter(args.output, budget)
            evidence = b4.load(args.preflight_evidence)
            if sha256_file(args.preflight_evidence) != budget.get("runpod_preflight", {}).get(
                "evidence_sha256"
            ):
                raise ValueError("Runpod preflight artifact hash differs")
            if (
                evidence.get("runpod_config_sha256") != sha256_json(config)
                or evidence.get("implementation_sha256") != plan["implementation_sha256"]
                or evidence.get("serving_policy") != plan["serving_policy"]
            ):
                raise ValueError("preflight did not validate this configuration/runtime")
            checks = evidence.get("checks", {})
            if not all(
                checks.get(key) is True
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
                raise ValueError("Runpod preflight artifact does not report all required checks")
            if args.resume_from is not None:
                carry_forward(args.resume_from, args.output, plan)
            from extension_b4_runpod import RunpodTransport

            transport = RunpodTransport(config, budget, plan, args.profile, args.output)
            for relative, value in (
                (f"budgets/{budget['record_sha256']}.json", budget),
                (f"runpod-configs/{sha256_json(config)}.json", config),
                (f"runpod-preflight/{sha256_json(evidence)}.json", evidence),
            ):
                path = args.output / relative
                if path.exists():
                    if b4.load(path) != value:
                        raise ValueError("immutable Runpod execution evidence differs")
                else:
                    write_json(path, value)
            dispatch(
                STUDY,
                args.output,
                plan,
                budget,
                transport,
                args.chunk_size,
                args.max_concurrent_lanes,
            )
        print(
            json.dumps(
                {
                    "provider": "runpod",
                    "cloud_type": "SECURE",
                    "gpu_type_id": plan["gpu_type_id"],
                    "max_concurrent_a100s": args.max_concurrent_lanes,
                    "plan_sha256": plan["plan_sha256"],
                    "first_calls": plan["first_calls"],
                    "gpu_work_launched": bool(args.execute),
                    "output": str(args.output),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
