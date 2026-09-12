"""Step 1 exit-gate status and the Step 0 seal integrity check.

The verdict is never `COMPLETE` while any required artifact is absent. When no solo output
exists at all, the verdict is `IMPLEMENTATION_READY_UNRUN`, which states plainly that the code
is finished and nothing has been executed.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from step0 import integrity
from step0.hashing import read_jsonl

from .frozen import Frozen
from .ledger import meter, validate
from .paths import AMENDMENTS, Paths
from .plan import solo_jobs
from .screen import REQUIRED_HOST_FACTS

#: The manifest hash HANDOFF.md records for the sealed Step 0 freeze.
EXPECTED_STEP0_MANIFEST_SHA256 = "8c9ca2a6a18d3178b91245d3f44c2a2c3776724a6f70beabc5723553b6c38171"
FORBIDDEN_CONDITIONS = ("joint", "exact_context", "filler", "serial_joint", "serial_exact")
VERDICTS = ("COMPLETE", "INCOMPLETE", "IMPLEMENTATION_READY_UNRUN")


def verify_step0_seal(
    frozen: Frozen, *, expected_manifest: str | None = EXPECTED_STEP0_MANIFEST_SHA256
) -> dict[str, Any]:
    """Recompute every frozen hash and compare the manifest against the recorded value."""
    report = integrity.verify(frozen.freeze_dir)
    matches = expected_manifest is None or report.get("manifest_sha256") == expected_manifest
    problems = list(report["problems"])
    if not matches:
        problems.append(
            "the manifest hash does not match the value recorded for the Step 0 seal; "
            "the freeze is not the one Step 0 sealed"
        )
    return {
        "ok": bool(report["ok"] and matches),
        "freeze_dir": str(frozen.freeze_dir),
        "freeze_stamp": frozen.stamp,
        "protocol_content_hash": frozen.content_hash,
        "n_files_checked": report["n_files_checked"],
        "manifest_sha256": report.get("manifest_sha256"),
        "expected_manifest_sha256": expected_manifest,
        "manifest_matches_record": matches,
        "problems": problems,
        "checked_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _screened_ledgers(paths: Paths) -> list[tuple[str, str, str, Path]]:
    if not paths.results.exists():
        return []
    ledgers: list[tuple[str, str, str, Path]] = []
    for directory in sorted(paths.results.iterdir()):
        path = directory / "solo.jsonl"
        if not path.exists():
            continue
        rows = list(read_jsonl(path))
        if not rows:
            continue
        identities = {
            (str(row["cell_id"]), str(row["frame_task_id"]), str(row["model_id"])) for row in rows
        }
        if len(identities) != 1:
            continue
        cell_id, frame, model_id = identities.pop()
        ledgers.append((cell_id, frame, model_id, path))
    return ledgers


def exit_gate(frozen: Frozen, paths: Paths) -> dict[str, Any]:  # noqa: C901 - one linear checklist
    """Check every Step 1 exit-gate item against the artifacts on disk."""
    items: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        items.append({"item": name, "ok": bool(ok), "detail": detail})

    seal = verify_step0_seal(frozen)
    check(
        "Step 0 seal still valid",
        seal["ok"],
        f"{seal['n_files_checked']} files, manifest matches record "
        f"{seal['manifest_matches_record']}",
    )

    amendments = (
        sorted(AMENDMENTS.glob("*-step1-execution-budget.json")) if AMENDMENTS.exists() else []
    )
    check(
        "protocol amendment recorded outside the freeze",
        bool(amendments),
        ", ".join(p.name for p in amendments) or f"no amendment in {AMENDMENTS}",
    )

    preflights = sorted(paths.preflight.glob("*.json")) if paths.preflight.exists() else []
    over_budget = 0
    inexact = 0
    for path in preflights:
        record = _json(path)
        over_budget += int(record.get("n_over_budget", 0))
        inexact += 0 if record.get("exact_tokenizer") else 1
    check(
        "exact pinned-tokenizer preflight for every frozen prompt type",
        bool(preflights) and over_budget == 0 and inexact == 0,
        f"{len(preflights)} reports, {over_budget} prompts over budget, {inexact} inexact"
        if preflights
        else "no preflight report",
    )

    hosts = sorted(paths.host.glob("*.json")) if paths.host.exists() else []
    incomplete_hosts = [
        p.name for p in hosts if not all(_json(p).get(f) for f in REQUIRED_HOST_FACTS)
    ]
    check(
        "host facts recorded for every worker",
        bool(hosts) and not incomplete_hosts,
        f"{len(hosts)} host records, incomplete {incomplete_hosts}" if hosts else "no host record",
    )

    ledgers = _screened_ledgers(paths)
    details: list[str] = []
    ledger_ok = bool(ledgers)
    forbidden_found: list[str] = []
    for cell_id, frame, model_id, path in ledgers:
        report = validate(
            path,
            solo_jobs(frozen, cell_id, frame, model_id=model_id),
            rows_per_call=1,
        )
        ledger_ok = ledger_ok and bool(report["ok"])
        details.append(
            f"{cell_id}/{frame}/{model_id}="
            f"{report['n_first_response_calls']}/{report['n_expected_calls']}"
        )
        conditions = {str(row["condition"]) for row in read_jsonl(path)}
        forbidden_found += sorted(conditions.intersection(FORBIDDEN_CONDITIONS))
    check(
        "solo ledger complete for every screened cell",
        ledger_ok,
        "; ".join(details) or "no solo ledger",
    )
    check(
        "no forbidden condition was generated",
        not forbidden_found,
        f"forbidden conditions found: {sorted(set(forbidden_found))}"
        if forbidden_found
        else "only solo responses exist",
    )

    screen_path = paths.screen / "SCREEN.json"
    if screen_path.exists():
        record = _json(screen_path)
        outcomes = {
            s["cell_id"] + "/" + s["frame_task_id"]: s["outcome"] for s in record["screens"]
        }
        check(
            "screen verdict, task choice, and replacement decision recorded",
            bool(record.get("screens")) and record.get("task_decision") is not None,
            f"outcomes {outcomes}, second task {record['task_decision'].get('chosen_second_task')}",
        )
    else:
        check(
            "screen verdict, task choice, and replacement decision recorded",
            False,
            f"{screen_path} is missing",
        )

    bench_path = paths.benchmark / "THROUGHPUT.json"
    if bench_path.exists():
        record = _json(bench_path)
        check(
            "throughput benchmark and remaining-ledger forecast written",
            bool(record.get("per_model")) and bool(record.get("forecasts")),
            f"{len(record.get('per_model', {}))} models, {len(record.get('forecasts', []))} cells",
        )
    else:
        check(
            "throughput benchmark and remaining-ledger forecast written",
            False,
            f"{bench_path} is missing",
        )

    timing = meter(paths.runs)
    check(
        "metered A100 time recorded",
        bool(timing["n_runs"]),
        f"{timing['n_runs']} runs, {timing['a100_hours_total']:.3f} A100 hours",
    )

    blockers = sorted(paths.blockers.glob("*.json")) if paths.blockers.exists() else []
    resolved: list[str] = []
    unresolved: list[str] = []
    screen_record = _json(screen_path) if screen_path.exists() else {}
    screened_load_failures = {
        str(screen["model_id"])
        for screen in screen_record.get("screens", [])
        if screen.get("solo_summary", {}).get("load_failed")
    }
    replacement = screen_record.get("replacement_decision", {})
    for path in blockers:
        blocker = _json(path)
        is_resolved_model_screen = bool(
            blocker.get("load_failed")
            and blocker.get("model_id") in screened_load_failures
            and replacement.get("claim_narrowed")
            and replacement.get("replacement_passed_load_gate") is False
        )
        (resolved if is_resolved_model_screen else unresolved).append(path.name)
    check(
        "no unresolved blocker",
        not unresolved,
        f"resolved by frozen screen: {resolved}"
        if resolved and not unresolved
        else ", ".join(unresolved) or "none",
    )

    n_failed = sum(1 for item in items if not item["ok"])
    unrun = not ledgers and not timing["n_runs"]
    verdict = (
        "COMPLETE" if n_failed == 0 else ("IMPLEMENTATION_READY_UNRUN" if unrun else "INCOMPLETE")
    )
    return {
        "freeze_dir": str(frozen.freeze_dir),
        "freeze_stamp": frozen.stamp,
        "step0_seal": seal,
        "items": items,
        "n_items": len(items),
        "n_failed": n_failed,
        "verdict": verdict,
        "step1_has_been_run": not unrun,
        "solo_only_statement": (
            "Step 1 generates solo model responses only. No joint, exact-context, filler, "
            "fake-batch, serial, or repeated-sampling response is produced in this step."
        ),
        "checked_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def render_status(status: dict[str, Any]) -> str:
    lines = [
        "# Step 1 exit-gate status",
        "",
        f"Verdict: **{status['verdict']}**",
        "",
        f"Freeze directory: `{status['freeze_dir']}`",
        f"Step 1 has been run: {status['step1_has_been_run']}",
        "",
        status["solo_only_statement"],
        "",
    ]
    if status["verdict"] == "IMPLEMENTATION_READY_UNRUN":
        lines += [
            "The Step 1 code, tests, runbook, and protocol amendment are finished. No model was",
            "loaded, no GPU was started, no Modal job ran, and no model output exists. Step 1 is",
            "**not** complete and no Step 1 result may be cited.",
            "",
        ]
    lines += ["| exit-gate item | pass | detail |", "|---|---|---|"]
    for item in status["items"]:
        detail = str(item["detail"]).replace("|", "/")
        lines.append(f"| {item['item']} | {item['ok']} | {detail} |")
    lines += [
        "",
        f"{status['n_items'] - status['n_failed']} of {status['n_items']} items pass.",
        "",
    ]
    return "\n".join(lines)
