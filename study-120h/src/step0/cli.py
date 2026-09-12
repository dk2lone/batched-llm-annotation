"""Single entry point. Every stage is separate, ordered, and refuses to overwrite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import assign, datafreeze, freeze, integrity, repro_gatea, simulate, venue
from .datafreeze import RetrievalBlockedError
from .hashing import RefusedOverwriteError, write_json, write_text
from .lock import CONDITIONS, MODELS, NO_NEW_LABELS, PANEL, TASKS
from .venue import VenueRetrievalError

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_A = PACKAGE_ROOT.parent / "gate-experiments" / "a"
DEFAULT_SIMULATIONS = PACKAGE_ROOT / "simulations"
TASK_KEYS = ["civil_comments", "second_task_first_candidate", "second_task_conditional_fallback"]


def _record_blocker(freeze_dir: Path, stage: str, message: str) -> None:
    write_json(
        freeze_dir / f"BLOCKER-{stage}.json",
        {
            "stage": stage,
            "blocker": message,
            "step0_complete": False,
            "no_new_experimental_model_labels": NO_NEW_LABELS,
        },
        allow_existing=True,
    )
    print(f"BLOCKER in {stage}: {message}", file=sys.stderr)
    print("Step 0 is incomplete. No evidence was fabricated.", file=sys.stderr)


def cmd_simulate(args: argparse.Namespace) -> int:
    results = simulate.run_all(
        args.out,
        n_sim=args.n_sim,
        n_draws_sim=args.n_draws,
        n_sim_threshold=args.n_sim_threshold,
        n_sim_mc=args.n_sim_mc,
    )
    print(json.dumps(results["frozen_conclusions"], indent=2))
    print(f"total {results['total_seconds']} s")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    out, payload = freeze.create_freeze(PACKAGE_ROOT, args.stamp, args.simulations)
    print(f"froze protocol {payload['protocol_version']} into {out}")
    print(f"content hash {payload['content_hash']}")
    print(f"simulation files copied: {payload['freeze']['simulation_files_sha256']}")
    return 0


def cmd_pins(args: argparse.Namespace) -> int:
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    out_dir = freeze_dir / "pins"
    try:
        models = datafreeze.audit_model_pins(MODELS)
        engine = datafreeze.audit_engine_pin(PANEL["serving"])
    except RetrievalBlockedError as exc:
        _record_blocker(freeze_dir, "pins", str(exc))
        return 2
    models["no_new_experimental_model_labels"] = NO_NEW_LABELS
    engine["no_new_experimental_model_labels"] = NO_NEW_LABELS
    write_json(out_dir / "model_pins.json", models)
    write_json(out_dir / "engine_pin.json", engine)
    for model_id, row in models["models"].items():
        print(
            f"{model_id}: gated={row['gated']!r} "
            f"revision_matches={row['revision_matches_live_main']}"
        )
    print(f"engine {engine['engine_version_pin']}: wheel found {engine['pinned_wheel_found']}")
    if not (models["ok"] and engine["ok"]):
        _record_blocker(
            freeze_dir, "pins", "; ".join(models["blockers"]) or "the engine pin was not confirmed"
        )
        return 2
    return 0


def cmd_freeze_data(args: argparse.Namespace) -> int:
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    summary: dict[str, Any] = {}
    for key in TASK_KEYS:
        try:
            provenance = datafreeze.freeze_task(key, freeze_dir / "data", args.gate_a)
        except RetrievalBlockedError as exc:
            _record_blocker(freeze_dir, f"data-{key}", str(exc))
            return 2
        summary[key] = {
            "task_id": provenance["task_id"],
            "role": TASKS[key]["status"],
            "n_items": provenance["items"]["n"],
            "n_filler": provenance["filler_pool"]["n"],
            "items_sha256": provenance["files"]["items.jsonl"]["sha256"],
            "excluded_pilot_ids": provenance["n_excluded_pilot_ids"],
            "license": provenance["license_audit"]["live_card_license"],
            "redistribution_permitted": provenance["redistribution_permitted"],
            "license_restrictions": provenance["license_restrictions"],
        }
        print(
            f"{provenance['task_id']}: {provenance['items']['n']} items, "
            f"{provenance['filler_pool']['n']} filler, "
            f"{provenance['n_excluded_pilot_ids']} pilot ids excluded"
        )
    write_json(
        freeze_dir / "data" / "SUMMARY.json",
        {
            "materialised": summary,
            "second_task_choice_rule": TASKS["second_task_conditional_fallback"][
                "activation_rule"
            ],
            "second_task_is_decided_at": (
                "Step 1, by the solo-only screen. Step 0 freezes both candidate frames and their "
                "provenance so neither can be designed after a screen outcome is known."
            ),
            "no_new_experimental_model_labels": NO_NEW_LABELS,
        },
    )
    return 0


def cmd_assign(args: argparse.Namespace) -> int:
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    data_dir = freeze_dir / "data"
    out_dir = freeze_dir / "assignments"
    task_dirs = {
        "civil_comments": data_dir / "civil_comments",
        "second_task": data_dir / str(TASKS["second_task_first_candidate"]["task_id"]),
    }
    for name, path in task_dirs.items():
        if not (path / "items.jsonl").exists():
            _record_blocker(
                freeze_dir, "assign", f"{name} frame is missing at {path}; run freeze-data first"
            )
            return 2

    frames: dict[str, tuple[assign.Frame, list[int], list[int], list[tuple[str, int]]]] = {}
    for name, path in task_dirs.items():
        items = datafreeze.load_items(path)
        filler = datafreeze.load_items(path, "filler_pool.jsonl")
        frames[name] = (
            assign.Frame(
                doc_ids=tuple(i.doc_id for i in items),
                strata=tuple(i.length_stratum for i in items),
            ),
            [i.reference for i in items],
            [i.n_chars for i in items],
            [(i.doc_id, i.n_chars) for i in filler],
        )

    metas = []
    for cell in PANEL["cells"]:
        frame, references, chars, filler_pool = frames[str(cell["task_id"])]
        meta = assign.write_cell(
            out_dir,
            cell,
            frame,
            references,
            int(PANEL["assignments_per_cell"]),
            chars,
            filler_pool,
        )
        metas.append(meta)
        print(
            f"{cell['cell_id']}: {meta['planned_calls_total']} planned calls, "
            f"{meta['unique_call_ids']} unique call ids, "
            f"{meta['planned_calls_by_condition']}"
        )
    write_json(
        out_dir / "SUMMARY.json",
        {
            "cells": [m["cell"]["cell_id"] for m in metas],
            "planned_calls_total": sum(int(m["planned_calls_total"]) for m in metas),
            "by_cell": {
                str(m["cell"]["cell_id"]): m["planned_calls_by_condition"] for m in metas
            },
            "second_task_frame_used_for_planning": str(
                TASKS["second_task_first_candidate"]["task_id"]
            ),
            "second_task_note": (
                "assignments are planned on the AG News frame because it is the frozen first "
                "candidate. If the Step 1 solo-only screen selects MultiNLI, the identical "
                "generator is rerun on the frozen MultiNLI frame with no rule change."
            ),
            "no_new_experimental_model_labels": NO_NEW_LABELS,
        },
    )
    return 0


def cmd_repro(args: argparse.Namespace) -> int:
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    out_dir = freeze_dir / "repro"
    result = repro_gatea.run(args.gate_a, out_dir, args.python)
    repro_gatea.write_outputs(out_dir, result)
    print(f"verdict: {'PASS' if result['ok'] else 'FAIL'}")
    print(f"historical artifact unchanged: {result['historical_artifact_unchanged']}")
    if result.get("comparisons"):
        matched = result["n_metrics"] - result["n_mismatched"]
        print(f"metrics matched: {matched}/{result['n_metrics']}")
    return 0 if result["ok"] else 1


def cmd_venue(args: argparse.Namespace) -> int:
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    try:
        record = venue.build(freeze_dir / "venue")
    except VenueRetrievalError as exc:
        _record_blocker(freeze_dir, "venue", str(exc))
        return 2
    print(json.dumps(record["consistency_checks"], indent=2))
    print(f"frozen route: {record['frozen_route']} ({record['frozen_route_page_limit']} pages)")
    print(f"human confirmation still required: {record['human_confirmation_required']}")
    return 0


CHECKS = {
    "pytest": [sys.executable, "-m", "pytest", "-q"],
    "ruff": [sys.executable, "-m", "ruff", "check", "src", "tests"],
    "mypy": [sys.executable, "-m", "mypy"],
}


def cmd_checks(args: argparse.Namespace) -> int:
    """Run the test suite, the linter, and the type checker, and record the results."""
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    results: dict[str, Any] = {"no_new_experimental_model_labels": NO_NEW_LABELS, "checks": {}}
    failed = 0
    for name, command in CHECKS.items():
        process = subprocess.run(  # noqa: S603 - fixed commands, no shell
            command, capture_output=True, text=True, cwd=str(PACKAGE_ROOT), check=False
        )
        ok = process.returncode == 0
        failed += 0 if ok else 1
        results["checks"][name] = {
            "command": command,
            "returncode": process.returncode,
            "ok": ok,
            "tail": (process.stdout + process.stderr)[-3000:],
        }
        print(f"{name}: {'ok' if ok else 'FAILED'} (exit {process.returncode})")
    results["all_passed"] = failed == 0
    write_json(freeze_dir / "checks" / "RESULTS.json", results)
    return 0 if failed == 0 else 1


def cmd_seal(args: argparse.Namespace) -> int:
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    record = integrity.seal(freeze_dir)
    print(json.dumps(record, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    report = integrity.verify(freeze_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


# --------------------------------------------------------------------------------------
# Exit-gate status
# --------------------------------------------------------------------------------------


def _rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def exit_gate(freeze_dir: Path) -> dict[str, Any]:  # noqa: C901 - one linear checklist
    """Check every Step 0 exit-gate item against the artifacts on disk."""
    items: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        items.append({"item": name, "ok": bool(ok), "detail": detail})

    protocol_path = freeze_dir / "protocol.lock.json"
    if protocol_path.exists():
        text = protocol_path.read_text(encoding="utf-8")
        payload = json.loads(text)
        unresolved = text.count("SET_BY_SIMULATION")
        check(
            "timestamped protocol",
            (freeze_dir / "PROTOCOL.md").exists() and (freeze_dir / "protocol.lock.toml").exists(),
            f"content hash {payload['content_hash']}, stamp {payload['freeze']['stamp_utc']}",
        )
        check(
            "no unresolved design field",
            unresolved == 0,
            f"{unresolved} occurrences of SET_BY_SIMULATION in protocol.lock.json",
        )
        check(
            "serving engine and version predeclared",
            str(payload["panel"]["serving"]["engine_version_pin"]).count("==") == 1,
            f"engine pin {payload['panel']['serving']['engine_version_pin']}",
        )
    else:
        check("timestamped protocol", False, f"{protocol_path} is missing")
        check("no unresolved design field", False, "no protocol to check")
        check("serving engine and version predeclared", False, "no protocol to check")

    sim = freeze_dir / "simulations" / "results.json"
    if sim.exists():
        results = _json(sim)
        threshold = results["frozen_conclusions"]["error_information_threshold_error_rate"]
        check(
            "null and power simulation",
            threshold is not None,
            f"threshold error rate {threshold}, "
            f"min solo errors "
            f"{results['frozen_conclusions']['error_information_threshold_min_solo_errors']}",
        )
    else:
        check("null and power simulation", False, f"{sim} is missing")

    frames = sorted((freeze_dir / "data").glob("*/items.jsonl"))
    frame_counts = {p.parent.name: _rows(p) for p in frames}
    detail = ", ".join(f"{name}={count}" for name, count in frame_counts.items())
    check(
        "sample frames materialised",
        len(frames) == 3 and all(count == 2_048 for count in frame_counts.values()),
        detail or "no frame found",
    )

    pins = freeze_dir / "pins" / "model_pins.json"
    engine = freeze_dir / "pins" / "engine_pin.json"
    if pins.exists() and engine.exists():
        model_pins, engine_pin = _json(pins), _json(engine)
        check(
            "model and engine pins evidenced",
            bool(model_pins["ok"] and engine_pin["ok"]),
            f"model blockers {model_pins['blockers']}, engine wheel found "
            f"{engine_pin['pinned_wheel_found']}",
        )
    else:
        check("model and engine pins evidenced", False, "pins/ is missing")

    cells = sorted((freeze_dir / "assignments").glob("*/job_skeletons.jsonl"))
    expected_cell_ids = {str(cell["cell_id"]) for cell in PANEL["cells"]}
    assignment_problems: list[str] = []
    assignment_details: list[str] = []
    for jobs_path in cells:
        cell_id = jobs_path.parent.name
        matrix_path = jobs_path.parent / "assignment_matrix.jsonl"
        meta_path = jobs_path.parent / "meta.json"
        balance_path = jobs_path.parent / "balance.json"
        if not (matrix_path.exists() and meta_path.exists() and balance_path.exists()):
            assignment_problems.append(f"{cell_id}: missing matrix, meta, or balance file")
            continue
        meta = _json(meta_path)
        balance = _json(balance_path)
        expected_counts = {
            "exact_context": 6_144,
            "joint": 384,
            "solo": 2_048,
        }
        filler = CONDITIONS["filler"]["cells"].get(cell_id)
        if filler:
            expected_counts["filler"] = int(filler["calls"])
        serial = CONDITIONS["serial_serving"]["cells"].get(cell_id)
        if serial:
            expected_counts["serial_joint"] = int(serial["joint_calls"])
            expected_counts["serial_exact"] = int(serial["joint_calls"]) * 16
        jobs_count = _rows(jobs_path)
        expected_total = sum(expected_counts.values())
        valid = all(
            (
                _rows(matrix_path) == 384,
                jobs_count == expected_total,
                meta.get("planned_calls_by_condition") == expected_counts,
                meta.get("planned_calls_total") == expected_total,
                meta.get("unique_call_ids") == expected_total,
                meta.get("every_arm_appears_in_every_block") is True,
                balance.get("appearances_per_item", {}).get("all_equal") is True,
                balance.get("distinct_slots_per_item", {}).get("all_distinct") is True,
                balance.get("one_item_per_stratum_per_call") is True,
            )
        )
        if not valid:
            assignment_problems.append(f"{cell_id}: count, uniqueness, block, or balance mismatch")
        assignment_details.append(f"{cell_id}={jobs_count}/{expected_total}")
    check(
        "assignments and job skeletons",
        {p.parent.name for p in cells} == expected_cell_ids and not assignment_problems,
        "; ".join(assignment_details + assignment_problems) or "no cell found",
    )

    repro = freeze_dir / "repro" / "repro.json"
    if repro.exists():
        record = _json(repro)
        check(
            "old Gate A analysis reproduced",
            bool(record["ok"]),
            f"historical artifact unchanged {record['historical_artifact_unchanged']}, "
            f"mismatched metrics {record.get('n_mismatched')}",
        )
    else:
        check("old Gate A analysis reproduced", False, f"{repro} is missing")

    venue_path = freeze_dir / "venue" / "venue.json"
    if venue_path.exists():
        record = _json(venue_path)
        check(
            "live venue record",
            all(record["consistency_checks"].values()),
            f"checks {record['consistency_checks']}, human confirmation still required",
        )
    else:
        check("live venue record", False, f"{venue_path} is missing")

    checks_path = freeze_dir / "checks" / "RESULTS.json"
    if checks_path.exists():
        record = _json(checks_path)
        check(
            "tests, lint, and type check",
            bool(record["all_passed"]),
            ", ".join(f"{k}={v['returncode']}" for k, v in record["checks"].items()),
        )
    else:
        check("tests, lint, and type check", False, f"{checks_path} is missing")

    report = integrity.verify(freeze_dir)
    check(
        "freeze sealed and integrity verified",
        bool(report["ok"]),
        f"{report['n_files_checked']} files checked, problems {report['problems'][:3]}",
    )

    blockers = sorted(p.name for p in freeze_dir.glob("BLOCKER-*.json"))
    check("no recorded blocker", not blockers, ", ".join(blockers) or "none")

    return {
        "freeze_dir": str(freeze_dir),
        "items": items,
        "n_items": len(items),
        "n_failed": sum(1 for i in items if not i["ok"]),
        "verdict": "COMPLETE" if all(i["ok"] for i in items) else "INCOMPLETE",
        "no_new_experimental_model_labels": NO_NEW_LABELS,
    }


def render_status(status: dict[str, Any]) -> str:
    lines = [
        "# Step 0 exit-gate status",
        "",
        f"Verdict: **{status['verdict']}**",
        "",
        f"Freeze directory: `{status['freeze_dir']}`",
        "",
        status["no_new_experimental_model_labels"],
        "",
        "| exit-gate item | pass | detail |",
        "|---|---|---|",
    ]
    for item in status["items"]:
        detail = str(item["detail"]).replace("|", "/")
        lines.append(f"| {item['item']} | {item['ok']} | {detail} |")
    lines += [
        "",
        f"{status['n_items'] - status['n_failed']} of {status['n_items']} items pass.",
        "",
    ]
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    freeze_dir = freeze.resolve_freeze_dir(PACKAGE_ROOT, args.stamp)
    status = exit_gate(freeze_dir)
    write_json(PACKAGE_ROOT / "STATUS.json", status, allow_existing=True)
    write_text(PACKAGE_ROOT / "STEP0_STATUS.md", render_status(status), allow_existing=True)
    print(render_status(status))
    return 0 if status["verdict"] == "COMPLETE" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="step0", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--stamp", default=None, help="freeze directory name; default is LATEST")

    sim_cmd = sub.add_parser("simulate", help="run the null and power simulations, before freezing")
    sim_cmd.add_argument("--out", type=Path, default=DEFAULT_SIMULATIONS)
    sim_cmd.add_argument("--n-sim", type=int, default=300)
    sim_cmd.add_argument("--n-draws", type=int, default=999)
    sim_cmd.add_argument("--n-sim-threshold", type=int, default=200)
    sim_cmd.add_argument("--n-sim-mc", type=int, default=200)
    sim_cmd.set_defaults(func=cmd_simulate)

    freeze_cmd = sub.add_parser("freeze", help="write PROTOCOL.md and both locks")
    common(freeze_cmd)
    freeze_cmd.add_argument("--simulations", type=Path, default=DEFAULT_SIMULATIONS)
    freeze_cmd.set_defaults(func=cmd_freeze)

    pins_cmd = sub.add_parser("audit-pins", help="record live model and engine pin evidence")
    common(pins_cmd)
    pins_cmd.set_defaults(func=cmd_pins)

    data_cmd = sub.add_parser("freeze-data", help="pin revisions and freeze every item frame")
    common(data_cmd)
    data_cmd.add_argument("--gate-a", type=Path, default=DEFAULT_GATE_A)
    data_cmd.set_defaults(func=cmd_freeze_data)

    assign_cmd = sub.add_parser("assign", help="generate assignments and job skeletons")
    common(assign_cmd)
    assign_cmd.set_defaults(func=cmd_assign)

    repro_cmd = sub.add_parser("repro-gatea", help="rerun the Gate A analysis and compare")
    common(repro_cmd)
    repro_cmd.add_argument("--gate-a", type=Path, default=DEFAULT_GATE_A)
    repro_cmd.add_argument("--python", type=Path, default=None)
    repro_cmd.set_defaults(func=cmd_repro)

    venue_cmd = sub.add_parser("venue", help="build the venue and authorship record live")
    common(venue_cmd)
    venue_cmd.set_defaults(func=cmd_venue)

    checks_cmd = sub.add_parser("checks", help="run pytest, ruff, and mypy into the freeze")
    common(checks_cmd)
    checks_cmd.set_defaults(func=cmd_checks)

    seal_cmd = sub.add_parser("seal", help="write MANIFEST.sha256 and make the freeze read-only")
    common(seal_cmd)
    seal_cmd.set_defaults(func=cmd_seal)

    verify_cmd = sub.add_parser("verify", help="validate every frozen hash and refuse mutation")
    common(verify_cmd)
    verify_cmd.set_defaults(func=cmd_verify)

    status_cmd = sub.add_parser("status", help="check every exit-gate item and write STATUS.json")
    common(status_cmd)
    status_cmd.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        code: int = args.func(args)
        return code
    except RefusedOverwriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
