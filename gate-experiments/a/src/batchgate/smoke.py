"""Stage 5: end-to-end check on synthetic local records. Never touches the network."""

from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path
from typing import Any

from .analyze import analyze, write_outputs
from .config import Config
from .data import prepare_data
from .plan import build_plan, read_plan, write_plan
from .runner import run
from .util import read_jsonl

VERDICTS = {
    "PROCEED",
    "PROCEED_POINT_ONLY",
    "BOUNDED_NEGATIVE_CANDIDATE",
    "FILLER_CONFOUNDED",
    "AMBIGUOUS",
    "UNDETERMINED",
    "NEEDS_FILLER_CONTROL",
}

REQUIRED_ROW_FIELDS = [
    "run_id",
    "job_id",
    "condition",
    "assignment",
    "real_batch_id",
    "doc_id",
    "slot",
    "gold",
    "pred",
    "error",
    "parse_status",
    "prompt_sha256",
    "output_sha256",
    "prompt_chars",
    "prompt_tokens",
    "model_id",
    "backend",
    "temperature",
    "max_model_len",
    "gen_seconds",
    "created_at",
    "raw_text",
]


class NetworkBlocked(RuntimeError):
    pass


def _block_network() -> None:
    def deny(*args: Any, **kwargs: Any) -> None:
        raise NetworkBlocked("the smoke test must not open a socket")

    socket.socket.connect = deny  # type: ignore[method-assign]
    socket.create_connection = deny  # type: ignore[assignment]


def _check(failures: list[str], ok: bool, message: str) -> None:
    if not ok:
        failures.append(message)


def smoke(cfg: Config, keep: bool = False) -> tuple[int, list[str]]:
    _block_network()
    failures: list[str] = []

    for relative in (cfg.data.out_dir, cfg.plan.out_dir, cfg.analyze.out_dir):
        if not keep:
            shutil.rmtree(cfg.path(relative), ignore_errors=True)
    raw_path = cfg.path(cfg.run.out_path)
    if not keep and raw_path.exists():
        raw_path.unlink()

    total = cfg.data.n_targets + cfg.data.filler_pool_size
    provenance = prepare_data(cfg, allow_download=False, synthetic=int(total * 2.0))
    _check(failures, provenance["disjoint"], "targets and filler pool overlap")
    _check(failures, provenance["n_targets"] == cfg.data.n_targets, "wrong target count")
    _check(failures, 0.0 < provenance["target_prevalence"] < 1.0, "degenerate label prevalence")

    jobs, meta = build_plan(cfg)
    write_plan(cfg, jobs, meta, force=True)
    plan_jobs = read_plan(cfg)
    _check(failures, len(plan_jobs) == len(jobs), "plan did not round-trip")
    _check(
        failures,
        meta["token_preflight"]["jobs_over_budget"] == 0,
        "planned prompts exceed the context budget",
    )
    conditions = {j.condition for j in plan_jobs}
    _check(failures, conditions == set(cfg.plan.conditions), f"missing conditions: {conditions}")
    for job in plan_jobs:
        if len(set(job.doc_order)) != len(job.doc_order):
            failures.append(f"job {job.job_id} repeats a document")
            break

    # Half the jobs, then the rest, to prove an interrupted run resumes.
    first = run(cfg, backend_name="mock", limit=len(plan_jobs) // 2, progress=False)
    second = run(cfg, backend_name="mock", progress=False)
    third = run(cfg, backend_name="mock", progress=False)
    _check(failures, third["ran"] == 0, "a completed run recomputed jobs")
    _check(
        failures,
        first["ran"] + second["ran"] == len(plan_jobs),
        "resume did not cover every planned job",
    )

    rows = list(read_jsonl(raw_path))
    done_ids = {str(r["job_id"]) for r in rows}
    _check(failures, done_ids == {j.job_id for j in plan_jobs}, "raw output misses planned jobs")
    expected_rows = sum(len(j.target_ids) for j in plan_jobs)
    _check(failures, len(rows) == expected_rows, f"expected {expected_rows} rows, got {len(rows)}")
    missing_fields = sorted({f for f in REQUIRED_ROW_FIELDS for r in rows if f not in r})
    _check(failures, not missing_fields, f"raw rows miss fields: {missing_fields}")

    result = analyze(cfg)
    write_outputs(cfg, result)
    _check(failures, result["source"]["is_mock"], "mock data was not flagged as mock")
    _check(failures, result["verdict"]["label"] in VERDICTS, "unknown verdict label")
    joint = result["icc"].get("joint")
    _check(failures, bool(joint), "no joint ICC block")
    if joint:
        _check(failures, joint["aggregate"]["point"] is not None, "aggregate ICC is undefined")
        _check(
            failures,
            len(joint["per_assignment"]) == cfg.plan.assignments,
            "missing a per-assignment ICC",
        )
    for condition in cfg.plan.conditions:
        _check(failures, condition in result["accuracy"], f"no accuracy for {condition}")
        _check(failures, condition in result["parse"], f"no parse stats for {condition}")

    print(
        json.dumps(
            {
                "verdict": result["verdict"]["label"],
                "aggregate_icc": result["verdict"]["aggregate_icc"],
                "joint_placebo_filler_icc": result["verdict"]["filler_placebo_icc"],
                "jobs": len(plan_jobs),
                "rows": len(rows),
                "failures": failures,
            },
            indent=2,
        )
    )
    return (1 if failures else 0), failures


def report_path(cfg: Config) -> Path:
    return cfg.path(cfg.analyze.out_dir) / "report.md"
