"""The frozen solo-only screen, the task fallback, and the second-model replacement.

Every input here is a solo call, a pinned-tokenizer count, or a parse status. No joint,
exact-context, filler, fake-batch, or repeated-sampling value may reach this module, and none
exists in Step 1. `gate` refuses to evaluate until the required artifacts validate, so a
screen verdict can never be produced from a partial or unmetered run.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from step0.hashing import read_jsonl

from .backends import host_facts
from .frozen import SECOND_TASK_ORDER, Frozen
from .ledger import meter, validate
from .plan import PREFLIGHT_CONDITIONS, job_rows, solo_jobs

#: Facts the freeze allows Step 1 to record. All six must be present before evaluation.
REQUIRED_HOST_FACTS = (
    "cuda_version",
    "driver_version",
    "gpu_name",
    "gpu_uuid",
    "torch_version",
    "installed_engine_version_string",
)


class GateError(RuntimeError):
    """Raised when evaluation is attempted before the required artifacts validate."""


def gate(
    frozen: Frozen,
    ledger_path: Path,
    cell_id: str,
    frame_task_id: str,
    *,
    model_id: str | None = None,
    preflight_report: dict[str, Any] | None = None,
    host: dict[str, Any] | None = None,
    runs_path: Path | None = None,
) -> dict[str, Any]:
    """Check every artifact the screen depends on. `ok` is the permission to evaluate."""
    problems: list[str] = []
    planned = solo_jobs(frozen, cell_id, frame_task_id, model_id=model_id)
    completeness = validate(ledger_path, planned, rows_per_call=1)
    if not completeness["ok"]:
        problems += [f"ledger: {p}" for p in completeness["problems"]]

    if preflight_report is None:
        problems.append("preflight: no exact pinned-tokenizer report was supplied")
        preflight_ok = False
    else:
        preflight_ok = bool(preflight_report.get("ok"))
        if not preflight_report.get("exact_tokenizer"):
            problems.append("preflight: the report does not come from the pinned tokenizer")
        if preflight_report.get("n_over_budget"):
            problems.append(
                f"preflight: {preflight_report['n_over_budget']} frozen prompts exceed the budget"
            )
        required_conditions = {
            str(job["condition"])
            for job in job_rows(
                frozen,
                cell_id,
                frame_task_id,
                model_id=model_id,
                conditions=PREFLIGHT_CONDITIONS,
            )
        }
        missing = sorted(required_conditions - set(preflight_report.get("by_condition", {})))
        if missing:
            problems.append(f"preflight: no token count for conditions {missing}")

    host_record = host or {}
    absent = [name for name in REQUIRED_HOST_FACTS if not host_record.get(name)]
    if absent:
        problems.append(f"host facts: {absent} were not recorded")
    engine_pin = str(frozen.serving["engine_version_pin"]).split("==")[-1]
    installed = str(host_record.get("installed_engine_version_string") or "")
    if installed and installed != engine_pin:
        problems.append(
            f"host facts: the installed engine {installed!r} is not the pinned {engine_pin!r}"
        )

    timing = meter(runs_path) if runs_path else {"a100_hours_total": 0.0, "n_runs": 0}
    if not timing["n_runs"]:
        problems.append("gpu timing: no metered A100 run was recorded")

    engines = (
        {str(row["engine_version"]) for row in read_jsonl(ledger_path)}
        if ledger_path.exists()
        else set()
    )
    foreign = sorted(engine for engine in engines if engine != engine_pin)
    if foreign:
        problems.append(
            f"ledger: rows were produced by {foreign}, not by the pinned engine {engine_pin!r}"
        )

    return {
        "ok": not problems,
        "cell_id": cell_id,
        "frame_task_id": frame_task_id,
        "completeness": completeness,
        "preflight_ok": preflight_ok,
        "host_facts_present": not absent,
        "a100_meter": timing,
        "problems": problems,
        "checked_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def solo_summary(ledger_path: Path, cell_id: str, frame_task_id: str) -> dict[str, Any]:
    """Counts over first-response solo rows only. Retries never enter this summary."""
    rows = [
        row
        for row in read_jsonl(ledger_path)
        if row["condition"] == "solo"
        and int(row["retry_index"]) == 0
        and row["cell_id"] == cell_id
        and row["frame_task_id"] == frame_task_id
    ]
    n_calls = len(rows)
    parsable = [r for r in rows if r["parse_status"] == "ok" and r["pred"] is not None]
    n_errors = sum(1 for r in parsable if int(r["error"]) == 1)
    n_correct = len(parsable) - n_errors
    statuses: dict[str, int] = {}
    for row in rows:
        key = str(row["parse_status"])
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "n_solo_calls": n_calls,
        "n_parsable": len(parsable),
        "n_correct": n_correct,
        "n_errors": n_errors,
        "parsable_first_response_rate": len(parsable) / n_calls if n_calls else 0.0,
        "accuracy_over_all_calls": n_correct / n_calls if n_calls else 0.0,
        "accuracy_over_parsable": n_correct / len(parsable) if parsable else 0.0,
        "error_rate_over_all_calls": n_errors / n_calls if n_calls else 0.0,
        "parse_status_counts": dict(sorted(statuses.items())),
        "first_response_status_counts": _counts(rows, "first_response_status"),
    }


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def screen_cell(
    frozen: Frozen,
    ledger_path: Path,
    cell_id: str,
    frame_task_id: str,
    *,
    model_id: str | None = None,
    preflight_report: dict[str, Any] | None = None,
    host: dict[str, Any] | None = None,
    runs_path: Path | None = None,
    load_failed: bool = False,
) -> dict[str, Any]:
    """Apply the frozen screen criteria in their frozen order.

    The outcome order is context, parse, capability, information, which is the order of the
    frozen `outcome_labels`. The accuracy denominator is every solo call, so an unparsable
    response counts against capability. The parsable-rate gate runs first and caps the gap
    between the two denominators at 2 percent, and both are reported.
    """
    gate_report = gate(
        frozen,
        ledger_path,
        cell_id,
        frame_task_id,
        model_id=model_id,
        preflight_report=preflight_report,
        host=host,
        runs_path=runs_path,
    )
    if not gate_report["ok"] and not load_failed:
        raise GateError(
            "the Step 1 artifacts do not validate, so no screen verdict is produced: "
            + "; ".join(gate_report["problems"][:5])
        )

    criteria = frozen.protocol["solo_screen"]["criteria"]
    floor = frozen.capability_floor(frame_task_id)
    parse_threshold = float(criteria["parsable_first_response_rate"]["threshold"])
    min_errors = int(criteria["error_information_threshold"]["min_solo_errors"])
    ceiling = float(criteria["accuracy_ceiling"]["value"])

    if load_failed:
        summary: dict[str, Any] = {"n_solo_calls": 0, "load_failed": True}
        outcome, reason = "FAIL_CAPABILITY", "the pinned revision could not be loaded on the A100"
        checks: dict[str, Any] = {}
    else:
        summary = solo_summary(ledger_path, cell_id, frame_task_id)
        over_budget = int((preflight_report or {}).get("n_over_budget", 0))
        parsable_rate = float(summary["parsable_first_response_rate"])
        accuracy = float(summary["accuracy_over_all_calls"])
        checks = {
            "context_fit": {
                "n_over_budget": over_budget,
                "threshold": 0,
                "pass": over_budget == 0,
            },
            "parsable_first_response_rate": {
                "value": parsable_rate,
                "threshold": parse_threshold,
                "pass": parsable_rate >= parse_threshold,
            },
            "capability_floor": {
                "value": accuracy,
                "value_over_parsable": float(summary["accuracy_over_parsable"]),
                "threshold": floor,
                "pass": accuracy >= floor,
                "denominators_agree": bool(
                    (accuracy >= floor) == (float(summary["accuracy_over_parsable"]) >= floor)
                ),
            },
            "error_information_threshold": {
                "n_errors": int(summary["n_errors"]),
                "min_solo_errors": min_errors,
                "pass": int(summary["n_errors"]) >= min_errors,
            },
            "accuracy_ceiling": {
                "value": accuracy,
                "ceiling": ceiling,
                "pass": accuracy < ceiling,
            },
        }
        if not checks["context_fit"]["pass"]:
            outcome, reason = "FAIL_CONTEXT", "a frozen prompt exceeds max_model_len"
        elif not checks["parsable_first_response_rate"]["pass"]:
            outcome, reason = "FAIL_PARSE", "the parsable first-response rate is below 0.98"
        elif not checks["capability_floor"]["pass"]:
            outcome, reason = "FAIL_CAPABILITY", "solo accuracy is below the task capability floor"
        elif (
            not checks["error_information_threshold"]["pass"]
            or not checks["accuracy_ceiling"]["pass"]
        ):
            outcome = "INFORMATION_POOR"
            reason = "insufficient events for a precise covariance estimate"
        else:
            outcome, reason = "PASS", "every frozen screen criterion is met"

    return {
        "cell_id": cell_id,
        "model_id": model_id or str(frozen.cell(cell_id)["model_id"]),
        "frame_task_id": frame_task_id,
        "outcome": outcome,
        "reason": reason,
        "checks": checks,
        "solo_summary": summary,
        "gate": gate_report,
        "inputs_used": ["solo first responses", "pinned tokenizer counts", "parse status"],
        "inputs_forbidden_and_absent": list(frozen.protocol["solo_screen"]["inputs_forbidden"]),
        "screened_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


#: Screen outcomes that the frozen policy accepts as a model-replacement trigger.
REPLACEABLE_OUTCOMES = frozenset({"FAIL_CONTEXT", "FAIL_PARSE", "FAIL_CAPABILITY"})


def decide_task(frozen: Frozen, screens: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen task fallback rule to second-task screen outcomes only."""
    rule = frozen.protocol["solo_screen"]["task_fallback_rule"]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for screen in screens:
        if screen["frame_task_id"] in SECOND_TASK_ORDER:
            by_task.setdefault(str(screen["frame_task_id"]), []).append(screen)

    def usable(task: str) -> bool:
        rows = by_task.get(task, [])
        return len(rows) == 2 and all(r["outcome"] == "PASS" for r in rows)

    for task in SECOND_TASK_ORDER:
        if usable(task):
            return {
                "chosen_second_task": task,
                "reason": f"{task} passes the frozen solo-only screen for both panel models",
                "screened_tasks": sorted(by_task),
                "rule": rule["rule"],
                "forbidden_input": rule["forbidden"],
                "claim_narrowed": False,
            }
    if "ag_news" in by_task and "multi_nli" not in by_task:
        return {
            "chosen_second_task": None,
            "reason": "AG News failed the screen, so the frozen MultiNLI fallback must be screened",
            "next_action": "screen multi_nli for both selected panel models",
            "screened_tasks": sorted(by_task),
            "rule": rule["rule"],
            "claim_narrowed": False,
        }
    return {
        "chosen_second_task": "ag_news",
        "reason": (
            "neither frozen task passes for both models, so AG News is kept, its failing cell "
            "is published as uninformative, and the claim narrows to the tested task"
        ),
        "screened_tasks": sorted(by_task),
        "rule": rule["rule"],
        "claim_narrowed": True,
    }


def decide_replacement(
    frozen: Frozen, screens: list[dict[str, Any]], *, replacements_used: int = 0
) -> dict[str, Any]:
    """Apply the frozen second-model replacement policy. At most one replacement, ever.

    INFORMATION_POOR is never a trigger. Only the second model is replaceable; the anchor
    model is the pilot model and the policy does not cover it.
    """
    policy = frozen.protocol["second_model_replacement_policy"]
    second_model = str(policy["primary_second_model"])
    limit = int(policy["replacement_limit"])
    ordered = [str(m) for m in policy["ordered_replacement_list"]]

    triggers = [
        s for s in screens if s["model_id"] == second_model and s["outcome"] in REPLACEABLE_OUTCOMES
    ]
    information_poor = [
        s for s in screens if s["model_id"] == second_model and s["outcome"] == "INFORMATION_POOR"
    ]
    base = {
        "primary_second_model": second_model,
        "replacement_limit": limit,
        "replacements_used": replacements_used,
        "ordered_replacement_list": ordered,
        "information_poor_cells_left_in_panel": [s["cell_id"] for s in information_poor],
        "information_poor_is_not_a_trigger": policy[
            "information_poor_is_not_a_replacement_trigger"
        ],
        "forbidden_triggers": list(policy["forbidden_triggers"]),
    }
    if not triggers:
        return {**base, "replace": False, "reason": "no frozen replacement trigger fired"}
    if replacements_used >= limit:
        return {
            **base,
            "replace": False,
            "reason": policy["if_no_replacement_passes"],
            "claim_narrowed": True,
        }
    return {
        **base,
        "replace": True,
        "replacement_model_id": ordered[replacements_used],
        "triggered_by": [
            {"cell_id": s["cell_id"], "frame_task_id": s["frame_task_id"], "outcome": s["outcome"]}
            for s in triggers
        ],
        "reason": "a frozen solo-only trigger fired on the second model",
        "next_action": "rescreen every second-model cell with the replacement model",
    }


def collect_host(path: Path | None = None) -> dict[str, Any]:
    """Read the host facts. On a non-GPU machine every GPU field stays null."""
    facts = host_facts()
    facts["collected_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    facts["complete"] = all(facts.get(name) for name in REQUIRED_HOST_FACTS)
    if path:
        from step0.hashing import write_json  # noqa: PLC0415

        write_json(path, facts, allow_existing=True)
    return facts


def render_screen(record: dict[str, Any]) -> str:
    """Markdown screen report. Scientific screen facts only, no performance number."""
    lines = [
        "# Step 1 solo-only screen",
        "",
        f"Freeze: `{record['freeze_stamp']}`",
        f"Written: {record['written_at_utc']}",
        "",
        "Only solo first responses, pinned-tokenizer counts, and parse statuses reach this report.",
        "No joint, exact-context, filler, fake-batch, or repeated-sampling value exists in Step 1.",
        "Throughput and cost live in the separate benchmark report and are not screen evidence.",
        "",
        "| cell | model | frame | outcome | parsable rate | accuracy | solo errors |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for screen in record["screens"]:
        summary = screen["solo_summary"]
        lines.append(
            f"| `{screen['cell_id']}` | `{screen['model_id']}` | {screen['frame_task_id']} | "
            f"**{screen['outcome']}** | {summary.get('parsable_first_response_rate', 0.0):.4f} | "
            f"{summary.get('accuracy_over_all_calls', 0.0):.4f} | {summary.get('n_errors', 0)} |"
        )
    task = record["task_decision"]
    replacement = record["replacement_decision"]
    lines += [
        "",
        "## Second-task decision",
        "",
        f"- chosen second task: `{task.get('chosen_second_task')}`",
        f"- reason: {task.get('reason')}",
        f"- claim narrowed: {task.get('claim_narrowed')}",
        "",
        "## Second-model replacement decision",
        "",
        f"- replace: {replacement.get('replace')}",
        f"- reason: {replacement.get('reason')}",
        f"- replacement model: {replacement.get('replacement_model_id')}",
        f"- information-poor cells kept in the panel: "
        f"{replacement.get('information_poor_cells_left_in_panel')}",
        "",
        "An INFORMATION_POOR cell is never replaced. It stays in the published panel with",
        "uncertainty bounds and the label 'insufficient events for a precise covariance estimate'.",
        "",
    ]
    return "\n".join(lines)
