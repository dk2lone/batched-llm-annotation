"""Create the timestamped freeze directory, PROTOCOL.md, and the machine-readable locks."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import tomli_w

from . import lock as L
from .hashing import sha256_file, sha256_json, write_json, write_text

FREEZE_ROOT_NAME = "freeze"
LATEST_POINTER = "LATEST"


def utc_stamp(now: dt.datetime | None = None) -> str:
    moment = now or dt.datetime.now(dt.UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def freeze_root(package_root: Path) -> Path:
    return package_root / FREEZE_ROOT_NAME


def resolve_freeze_dir(package_root: Path, stamp: str | None) -> Path:
    """Return the freeze directory named by `stamp`, or the one LATEST points at."""
    root = freeze_root(package_root)
    if stamp:
        return root / stamp
    pointer = root / LATEST_POINTER
    if not pointer.exists():
        raise FileNotFoundError(
            f"{pointer} is missing. Run `step0 freeze` first, or pass --stamp explicitly."
        )
    return root / pointer.read_text(encoding="utf-8").strip()


def _toml_safe(value: Any) -> Any:
    """TOML has no null. Replace None and coerce tuples, so the lock round-trips."""
    if value is None:
        return "null"
    if isinstance(value, dict):
        return {str(k): _toml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_toml_safe(v) for v in value]
    return value


SIMULATION_FILES = ("results.json", "REPORT.md")
SOURCE_GLOBS = ("src/step0/*.py", "tests/*.py")
SOURCE_ROOT_FILES = ("pyproject.toml", "README.md")
RUNTIME_PACKAGES = ("datasets", "huggingface-hub", "numpy", "scipy", "tomli-w")


def _simulation_results(source: Path) -> dict[str, Any]:
    path = source / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `step0 simulate --out {source}` first")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def validate_simulation_resolution(protocol: dict[str, Any], results: dict[str, Any]) -> None:
    """Refuse a freeze whose resolved fields disagree with its simulation evidence."""
    conclusions = results["frozen_conclusions"]
    threshold = protocol["solo_screen"]["criteria"]["error_information_threshold"]
    expected_high = {
        "highest_grid_error_rate_meeting_target": conclusions[
            "error_information_threshold_upper_error_rate"
        ],
        "error_rates_above_the_band_that_miss_the_target": conclusions[
            "error_rates_above_the_band_that_miss_the_target"
        ],
    }
    checks = {
        "min_solo_errors": (
            threshold["min_solo_errors"],
            conclusions["error_information_threshold_min_solo_errors"],
        ),
        "min_solo_error_rate": (
            threshold["min_solo_error_rate"],
            conclusions["error_information_threshold_error_rate"],
        ),
        "high_error_rate_finding": (threshold["high_error_rate_finding"], expected_high),
        "accuracy_ceiling": (
            protocol["solo_screen"]["criteria"]["accuracy_ceiling"]["value"],
            1.0 - float(conclusions["error_information_threshold_error_rate"]),
        ),
        "monte_carlo_draws": (
            protocol["tests"]["primary_contrast"]["monte_carlo_draws"],
            conclusions["monte_carlo_draws"],
        ),
        "monte_carlo_draws_rationale": (
            protocol["tests"]["primary_contrast"]["monte_carlo_draws_rationale"],
            conclusions["monte_carlo_draws_justification"],
        ),
        "covariance_floor": (
            protocol["tests"]["practical_floors"]["covariance_floor_absolute"],
            results["settings"]["covariance_floor"],
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        detail = ", ".join(
            f"{name}: protocol={checks[name][0]!r}, simulation={checks[name][1]!r}"
            for name in mismatches
        )
        raise ValueError(f"protocol fields do not match simulations/results.json: {detail}")


def _snapshot_source(package_root: Path, out: Path) -> dict[str, str]:
    """Copy the exact generator, analysis code, tests, and package metadata into the freeze."""
    paths = [package_root / name for name in SOURCE_ROOT_FILES]
    for pattern in SOURCE_GLOBS:
        paths.extend(sorted(package_root.glob(pattern)))
    copied: dict[str, str] = {}
    for src in sorted(set(paths)):
        if not src.exists():
            raise FileNotFoundError(f"source snapshot input is missing: {src}")
        rel = src.relative_to(package_root)
        dst = out / "source_snapshot" / rel
        write_text(dst, src.read_text(encoding="utf-8"))
        copied[rel.as_posix()] = sha256_file(dst)
    return copied


def _copy_simulations(source: Path, out: Path) -> dict[str, str]:
    """Copy the completed simulation into the freeze and hash what was copied.

    The simulation runs before the freeze because it resolves five protocol fields.
    Copying it in keeps the freeze self-contained and lets `step0 verify` seal it.
    """
    copied: dict[str, str] = {}
    for name in SIMULATION_FILES:
        src = source / name
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing; run `step0 simulate --out {source}` first")
        write_text(out / "simulations" / name, src.read_text(encoding="utf-8"))
        copied[name] = sha256_file(out / "simulations" / name)
    return copied


def _runtime_versions() -> dict[str, str]:
    """Exact installed versions that can affect data loading, RNG, or statistics."""
    versions: dict[str, str] = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def create_freeze(
    package_root: Path, stamp: str | None = None, simulations: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Write PROTOCOL.md and both locks into a new timestamped directory."""
    stamp = stamp or utc_stamp()
    out = freeze_root(package_root) / stamp
    payload = L.protocol()
    if simulations is None:
        raise FileNotFoundError("completed simulations are required to create a Step 0 freeze")
    simulation_results = _simulation_results(simulations)
    validate_simulation_resolution(payload, simulation_results)
    copied = _copy_simulations(simulations, out)
    source_snapshot = _snapshot_source(package_root, out)
    payload["freeze"] = {
        "stamp_utc": stamp,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "runtime_packages": _runtime_versions(),
        "gpu_used": False,
        "experimental_model_inference_used": False,
        "paid_compute_used": False,
        "simulation_source": str(simulations) if simulations else None,
        "simulation_files_sha256": copied,
        "source_snapshot_sha256": source_snapshot,
    }
    payload["content_hash"] = sha256_json({k: v for k, v in payload.items() if k != "freeze"})

    write_json(out / "protocol.lock.json", payload)
    write_text(out / "protocol.lock.toml", tomli_w.dumps(_toml_safe(payload)))
    write_text(out / "PROTOCOL.md", render_protocol_md(payload))
    write_text(out / "DECISIONS.md", render_decisions_md(payload))
    write_json(out / "decisions.json", payload["decisions"])

    pointer = freeze_root(package_root) / LATEST_POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(stamp + "\n", encoding="utf-8")
    return out, payload


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _kv_table(rows: dict[str, Any], key_head: str = "field", val_head: str = "value") -> list[str]:
    out = [f"| {key_head} | {val_head} |", "|---|---|"]
    for key, value in rows.items():
        if isinstance(value, (dict, list)):
            text = "; ".join(str(v) for v in (value if isinstance(value, list) else value.items()))
        else:
            text = str(value)
        safe = text.replace("|", "/")
        out.append(f"| `{key}` | {safe} |")
    return out


def render_decisions_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Conservative decisions taken where the plan was underspecified",
        "",
        f"Protocol `{payload['protocol_id']}` version {payload['protocol_version']}, "
        f"frozen {payload['freeze']['stamp_utc']}.",
        "",
        "Each row records a point where `plan-120h-a100.md` or `plan.md` left a choice open, the "
        "choice this freeze makes, and why that choice is the conservative one.",
        "",
        "| id | topic | what the plan left open | frozen decision | why this is conservative |",
        "|---|---|---|---|---|",
    ]
    for row in payload["decisions"]:
        lines.append(
            f"| {row['id']} | {row['topic']} | {row['underspecified']} | "
            f"{row['decision']} | {row['why_conservative']} |"
        )
    lines += [
        "",
        "No row above is a result. Every one is a planned rule fixed before any new model output "
        "exists.",
        "",
        payload["no_new_experimental_model_labels"],
        "",
    ]
    return "\n".join(lines)


def render_protocol_md(payload: dict[str, Any]) -> str:  # noqa: C901 - one linear document
    p = payload
    lines: list[str] = []
    add = lines.append

    add("# Frozen protocol: batched LLM annotation dependence, 120-hour single-A100 study")
    add("")
    add(f"- Protocol id: `{p['protocol_id']}`")
    add(f"- Protocol version: `{p['protocol_version']}`")
    add(f"- Freeze stamp (UTC): `{p['freeze']['stamp_utc']}`")
    add(f"- Content hash: `{p['content_hash']}`")
    add("")
    add("## 0. Status")
    add("")
    add(p["status"])
    add("")
    add(
        "This document is Step 0 of `plan-120h-a100.md`. It fixes the design before any new joint "
        "output exists. It reports no finding. Every number below is a planned decision rule, a "
        "pinned revision, a count, or a threshold."
    )
    add("")
    add(
        "The serving engine and its version are pinned in section 1 with retrieval evidence. "
        "The only fields completed after this freeze are the Step 1 host facts listed in "
        "`recorded_from_the_step_1_host`: CUDA version, driver version, GPU name, GPU uuid, "
        "torch version, and the installed engine version string. They are properties of the "
        "machine, not design choices."
    )
    add("")
    add(p["no_new_experimental_model_labels"])
    add("")

    add("## 1. The panel")
    add("")
    add(
        f"A {p['panel']['shape']}. Batch size k = {p['panel']['batch_size_k']}. "
        f"{p['panel']['records_per_task']} distinct records per task, shared by both models. "
        f"{p['panel']['assignments_per_cell']} independently randomized assignments per cell."
    )
    add("")
    add("| cell | model | task | role | tier | in confirmation set | repeated sampling |")
    add("|---|---|---|---|---|---|---|")
    for cell in p["panel"]["cells"]:
        add(
            f"| `{cell['cell_id']}` | `{cell['model_id']}` | `{cell['task_id']}` | "
            f"{cell['role']} | {cell['tier']} | {cell['in_confirmation_set']} | "
            f"{cell['repeated_sampling']} |"
        )
    add("")
    add("### Decoding and serving")
    add("")
    lines.extend(_kv_table(p["panel"]["decoding"]))
    add("")
    lines.extend(_kv_table(p["panel"]["serving"]))
    add("")

    add("## 2. Checkpoints and tokenizers")
    add("")
    add("| model | revision | tokenizer revision | license | new to this study |")
    add("|---|---|---|---|---|")
    for model_id, spec in p["models"].items():
        add(
            f"| `{model_id}` | `{spec['revision']}` | `{spec['tokenizer_revision']}` | "
            f"{spec['license']} | {not spec['same_revision_as_pilot']} |"
        )
    add("")
    add("### Second-model replacement policy, frozen before any joint output")
    add("")
    pol = p["second_model_replacement_policy"]
    add(f"Primary second model: `{pol['primary_second_model']}`.")
    replacements = ", ".join("`" + m + "`" for m in pol["ordered_replacement_list"])
    add(f"Ordered replacement list: {replacements}.")
    add(f"At most {pol['replacement_limit']} replacement is permitted.")
    add("")
    add("A replacement may be triggered only by these solo-only conditions:")
    add("")
    for item in pol["trigger_conditions_all_solo_only"]:
        add(f"- {item}")
    add("")
    add("A replacement must never be triggered by:")
    add("")
    for item in pol["forbidden_triggers"]:
        add(f"- {item}")
    add("")
    add(f"If no replacement passes: {pol['if_no_replacement_passes']}")
    add("")

    add("## 3. Tasks, dataset revisions, licenses, and label limits")
    add("")
    add(
        "AG News is the frozen first candidate for the second task. MultiNLI is the conditional "
        "fallback only. The final task choice stays a Step 1 solo-only decision and must never use "
        "a joint result."
    )
    add("")
    for key, spec in p["tasks"].items():
        add(f"### `{key}`")
        add("")
        lines.extend(_kv_table(spec))
        add("")

    add("## 4. Solo-only screen, frozen before any joint output")
    add("")
    screen = p["solo_screen"]
    add("Allowed inputs: " + ", ".join(screen["inputs_allowed"]) + ".")
    add("")
    add("Forbidden inputs:")
    add("")
    for item in screen["inputs_forbidden"]:
        add(f"- {item}")
    add("")
    for name, crit in screen["criteria"].items():
        add(f"**{name}**")
        add("")
        lines.extend(_kv_table(crit))
        add("")
    add(f"Outcome labels: {', '.join('`' + s + '`' for s in screen['outcome_labels'])}.")
    add("")
    add(screen["information_poor_handling"])
    add("")
    add("### Task fallback rule")
    add("")
    lines.extend(_kv_table(screen["task_fallback_rule"]))
    add("")

    add("## 5. Prompt templates and response grammars")
    add("")
    for key, spec in p["prompts"].items():
        add(f"### `{key}`")
        add("")
        lines.extend(_kv_table(spec))
        add("")

    add("## 6. Conditions and interleaving")
    add("")
    for key, spec in p["conditions"].items():
        add(f"### `{key}`")
        add("")
        lines.extend(_kv_table(spec))
        add("")

    add("## 7. Call lineage schema")
    add("")
    add(
        "One row per requested label. A row is appended once and never edited. Every field below "
        "is mandatory for every call in every condition."
    )
    add("")
    lines.extend(_kv_table(p["call_lineage_schema"]["fields"], "field", "meaning"))
    add("")

    add("## 8. First-response, failure, retry, and exclusion policy")
    add("")
    fp = p["failure_policy"]
    add(f"State machine: {', '.join('`' + s + '`' for s in fp['state_machine'])}.")
    add("")
    add(fp["first_response_policy"])
    add("")
    add("### Retry policy")
    add("")
    lines.extend(_kv_table(fp["retry_policy"]))
    add("")
    add("### Truncation")
    add("")
    lines.extend(_kv_table(fp["truncation"]))
    add("")
    add("### Exclusion rules")
    add("")
    lines.extend(_kv_table(fp["exclusion_rules"]))
    add("")

    add("## 9. Estimands")
    add("")
    for key, spec in p["estimands"].items():
        if isinstance(spec, list):
            add(f"### `{key}`")
            add("")
            for item in spec:
                add(f"- {item}")
            add("")
            continue
        add(f"### `{key}`")
        add("")
        lines.extend(_kv_table(spec))
        add("")

    add("## 10. Tests, floors, multiplicity, and gates")
    add("")
    for key, spec in p["tests"].items():
        add(f"### `{key}`")
        add("")
        if key == "multiplicity":
            for fam, body in spec.items():
                add(f"**{fam}**")
                add("")
                if isinstance(body, dict):
                    lines.extend(_kv_table(body))
                elif isinstance(body, list):
                    for item in body:
                        add(f"- {item}")
                else:
                    add(str(body))
                add("")
            continue
        if key == "recurrence_gate":
            lines.extend(
                _kv_table(
                    {k: v for k, v in spec.items() if k != "pass_conditions_all_required"}
                )
            )
            add("")
            add("Every condition below must hold:")
            add("")
            for item in spec["pass_conditions_all_required"]:
                add(f"- {item}")
            add("")
            continue
        lines.extend(_kv_table(spec))
        add("")

    add("## 11. Repeated-sampling cells, streams, and interval methods")
    add("")
    rs = p["repeated_sampling"]
    add(f"Cells: {', '.join('`' + c + '`' for c in rs['cells'])}, frozen before any output.")
    add("")
    add(f"Target: {rs['target']}")
    add("")
    not_this = p["estimands"]["E5_inferential_consequence"]["is_not"]
    add(f"This is an LLM-policy and reference-label quantity. It is not {not_this}.")
    add("")
    add("### Replicate design")
    add("")
    lines.extend(_kv_table(rs["replicate_design"]))
    add("")
    add("### Reference stream")
    add("")
    lines.extend(_kv_table(rs["reference_stream"]))
    add("")
    add("### Evaluation stream")
    add("")
    lines.extend(_kv_table(rs["evaluation_stream"]))
    add("")
    add("### Interval methods")
    add("")
    lines.extend(_kv_table(rs["interval_methods"]["iid"], "i.i.d. method", "value"))
    add("")
    lines.extend(_kv_table(rs["interval_methods"]["call_aware"], "call-aware method", "value"))
    add("")
    add(rs["interval_methods"]["finite_population_correction_decision"])
    add("")
    lines.extend(_kv_table(rs["interval_methods"]["sensitivity"]["whole_call_bootstrap"]))
    add("")
    add("### Endpoints")
    add("")
    lines.extend(_kv_table(rs["endpoints"]["variance_ratio"], "variance ratio", "value"))
    add("")
    lines.extend(_kv_table(rs["endpoints"]["coverage"], "coverage", "value"))
    add("")
    add("### Call budget")
    add("")
    lines.extend(_kv_table(rs["call_budget"]))
    add("")

    add("## 12. Assignment generator")
    add("")
    ag = p["assignment_generator"]
    add(f"Policy: {ag['policy']}. {ag['policy_definition']}")
    add("")
    add(f"Balanced mechanism designs: {ag['balanced_mechanism_policy']}")
    add("")
    for sub in ("length_strata", "partition", "slot_assignment", "seeds", "replay"):
        add(f"**{sub}**")
        add("")
        lines.extend(_kv_table(ag[sub]))
        add("")
    add("Emitted artifacts:")
    add("")
    for item in ag["emitted_artifacts"]:
        add(f"- {item}")
    add("")

    add("## 13. Execution blocks and the A100 ledger")
    add("")
    el = p["execution_ledger"]
    add("| step | ceiling in A100 hours |")
    add("|---|---:|")
    for step, hours in el["a100_ceilings_hours"].items():
        add(f"| {step} | {hours} |")
    add("")
    add("These are hard stage ceilings, not runtime predictions.")
    add("")
    add("### Planned call counts")
    add("")
    lines.extend(_kv_table(el["planned_call_counts"], "quantity", "calls"))
    add("")
    lines.extend(_kv_table(el["throughput_anchor"]))
    add("")
    add("Trim order if the projected design exceeds the budget:")
    add("")
    for i, item in enumerate(el["trim_order_if_the_budget_is_exceeded"], 1):
        add(f"{i}. {item}")
    add("")
    add("Never trim:")
    add("")
    for item in el["never_trim"]:
        add(f"- {item}")
    add("")

    add("## 14. Discovery and confirmation separation")
    add("")
    dc = p["discovery_confirmation"]
    for key in ("discovery_tier", "disclosed_discovery_inputs_used_to_set_this_design",
                "not_used_to_set_this_design", "confirmation_tier", "anchor_tier"):
        add(f"**{key}**")
        add("")
        for item in dc[key]:
            add(f"- {item}")
        add("")
    add(f"Pooling rule: {dc['pooling_rule']}")
    add("")
    add(f"Post-hoc rule: {dc['post_hoc_rule']}")
    add("")

    add("## 15. Prespecified outcomes and forbidden claims")
    add("")
    co = p["claim_outcomes"]
    for key in ("strong_bounded", "phenomenon_without_demonstrated_damage", "narrow_or_negative"):
        add(f"- **{key}**: {co[key]}")
    add("")
    add("Forbidden claims:")
    add("")
    for item in co["forbidden_claims"]:
        add(f"- {item}")
    add("")

    add("## 16. Simulation specification")
    add("")
    ss = p["simulation_spec"]
    add("Purpose:")
    add("")
    for item in ss["purpose"]:
        add(f"- {item}")
    add("")
    add(ss["design_is_exact"])
    add("")
    lines.extend(_kv_table(ss["error_model"]))
    add("")
    lines.extend(_kv_table(ss["scenario_grid"]))
    add("")
    add(f"Power target {ss['power_target']} at alpha {ss['alpha']}.")
    add("")
    add("Named diagnostics: " + ", ".join("`" + d + "`" for d in ss["diagnostics"]) + ".")
    add("")
    add(ss["tuning_boundary"])
    add("")

    add("## 17. Venue and authorship route")
    add("")
    lines.extend(_kv_table(p["venue"]))
    add("")
    add(
        "Personal eligibility is not established by this record. A human must confirm that Daniel "
        "is an undergraduate degree student at the time of submission before the route is used."
    )
    add("")

    add("## 18. Reproducibility and integrity")
    add("")
    rp = p["reproducibility"]
    for item in rp["immutability"]:
        add(f"- {item}")
    add("")
    add(f"Credentials: {rp['credentials']}")
    add("")
    add(f"Integrity command: `{rp['integrity_command']}`")
    add("")
    lines.extend(_kv_table(rp["gate_a_reproduction"]))
    add("")

    add("## 19. Step 0 exit evidence")
    add("")
    lines.extend(_kv_table(p["step0_exit_evidence"], "required evidence", "artifact"))
    add("")

    add("## 20. Conservative decisions where the plan was underspecified")
    add("")
    add("See `DECISIONS.md` and `decisions.json` in this directory for the full table.")
    add("")
    add("| id | topic | frozen decision |")
    add("|---|---|---|")
    for row in p["decisions"]:
        add(f"| {row['id']} | {row['topic']} | {row['decision']} |")
    add("")
    return "\n".join(lines) + "\n"
