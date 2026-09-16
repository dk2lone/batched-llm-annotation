"""Prepare the four-A100 Runpod B4 plan locally. --execute alone is insufficient."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from modal_extension_b4 import dispatch, meter  # noqa: E402

from freezeb import extension_b4 as b4  # noqa: E402
from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=int, choices=(400, 1000, 2500), default=400)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", help="verified Runpod account ID")
    parser.add_argument("--budget-record", type=Path)
    parser.add_argument("--runpod-config", type=Path)
    parser.add_argument("--preflight-evidence", type=Path)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    plan = b4.build_plan(STUDY, args.prefix, "runpod")
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
                )
            ):
                raise ValueError("Runpod preflight artifact does not report all required checks")
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
            dispatch(STUDY, args.output, plan, budget, transport, args.chunk_size)
        print(
            json.dumps(
                {
                    "provider": "runpod",
                    "cloud_type": "SECURE",
                    "gpu_type_id": plan["gpu_type_id"],
                    "max_concurrent_a100s": 4,
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
