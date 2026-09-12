"""Rerun the existing Gate A analysis from its saved raw rows and check key metrics.

The historical artifact is never modified. The rerun writes into a new directory,
and the SHA-256 of every preserved Gate A file is recorded before and after so an
accidental write is detected rather than assumed absent.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .hashing import sha256_file, write_json, write_text

GATE_A_FILES = (
    "artifacts/full/raw/predictions.jsonl",
    "artifacts/full/analysis/analysis.json",
    "artifacts/full/analysis/report.md",
    "artifacts/full/plan/plan.jsonl",
    "artifacts/full/plan/plan_meta.json",
    "artifacts/full/data/targets.jsonl",
    "artifacts/full/data/filler_pool.jsonl",
    "configs/full.toml",
    "src/batchgate/analyze.py",
    "src/batchgate/stats.py",
)

# Metrics declared in the preserved report and analysis, checked value by value.
KEY_METRICS: dict[str, tuple[str, ...]] = {
    "verdict_label": ("verdict", "label"),
    "aggregate_icc": ("verdict", "aggregate_icc"),
    "decision_ci_low": ("verdict", "decision_ci", "0"),
    "decision_ci_high": ("verdict", "decision_ci", "1"),
    "max_permutation_p": ("verdict", "max_permutation_p"),
    "filler_placebo_icc": ("verdict", "filler_placebo_icc"),
    "joint_minus_solo_error_delta": ("verdict", "joint_minus_solo_error_delta"),
    "filler_minus_solo_error_delta": ("verdict", "filler_minus_solo_error_delta"),
    "joint_parse_failure_rate": ("verdict", "joint_parse_failure_rate"),
    "enough_errors": ("verdict", "enough_errors"),
    "parse_quality_passed": ("verdict", "parse_quality_passed"),
    "config_fingerprint": ("config_fingerprint",),
    "joint_icc_assignment_1": ("icc", "joint", "per_assignment", "1", "point"),
    "joint_icc_assignment_2": ("icc", "joint", "per_assignment", "2", "point"),
    "joint_icc_assignment_3": ("icc", "joint", "per_assignment", "3", "point"),
    "joint_error_rate": ("accuracy", "joint", "error_rate_document_mean"),
    "solo_error_rate": ("accuracy", "solo", "error_rate_document_mean"),
    "filler_error_rate": ("accuracy", "filler", "error_rate_document_mean"),
    "n_rows": ("source", "n_rows"),
    "n_jobs": ("source", "n_jobs"),
}
TOLERANCE = 1e-12


def _dig(obj: Any, path: tuple[str, ...]) -> Any:
    node = obj
    for key in path:
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= TOLERANCE * max(1.0, abs(float(a)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def _report_claims(report_text: str) -> dict[str, float]:
    """Pull the numbers the preserved report.md states in prose."""
    out: dict[str, float] = {}
    match = re.search(
        r"Aggregate ICC (\d+\.\d+), decision interval \[(\d+\.\d+), (\d+\.\d+)\], "
        r"largest fake-batch p (\d+\.\d+)",
        report_text,
    )
    if match:
        out["report_aggregate_icc"] = float(match.group(1))
        out["report_decision_ci_low"] = float(match.group(2))
        out["report_decision_ci_high"] = float(match.group(3))
        out["report_max_permutation_p"] = float(match.group(4))
    return out


def _config_with_new_out_dir(gate_a_root: Path, out_dir: Path) -> str:
    source = (gate_a_root / "configs" / "full.toml").read_text(encoding="utf-8")
    replaced, count = re.subn(
        r'out_dir = "artifacts/full/analysis"',
        f'out_dir = "{out_dir.as_posix()}"',
        source,
    )
    if count != 1:
        raise RuntimeError(
            "configs/full.toml no longer contains exactly one analyze out_dir line; refusing to "
            "guess where the rerun would write"
        )
    return replaced


def run(gate_a_root: Path, work_dir: Path, python: Path | None = None) -> dict[str, Any]:
    """Rerun `batchgate analyze` into `work_dir` and compare against the preserved result."""
    before = {rel: sha256_file(gate_a_root / rel) for rel in GATE_A_FILES}
    preserved = json.loads(
        (gate_a_root / "artifacts/full/analysis/analysis.json").read_text(encoding="utf-8")
    )
    preserved_report = (gate_a_root / "artifacts/full/analysis/report.md").read_text(
        encoding="utf-8"
    )

    rerun_dir = work_dir / "rerun"
    config_path = work_dir / "gate_a_repro.toml"
    write_text(config_path, _config_with_new_out_dir(gate_a_root, rerun_dir))

    interpreter = python or gate_a_root / ".venv" / "bin" / "python"
    command = [
        str(interpreter),
        "-m",
        "batchgate.cli",
        "--config",
        str(config_path),
        "--root",
        str(gate_a_root),
        "analyze",
    ]
    started = dt.datetime.now(dt.UTC)
    process = subprocess.run(  # noqa: S603 - fixed command, no shell
        command, capture_output=True, text=True, cwd=str(gate_a_root), check=False
    )
    after = {rel: sha256_file(gate_a_root / rel) for rel in GATE_A_FILES}
    touched = sorted(rel for rel in GATE_A_FILES if before[rel] != after[rel])

    result: dict[str, Any] = {
        "started_at_utc": started.isoformat(),
        "gate_a_root": str(gate_a_root),
        "command": command,
        "returncode": process.returncode,
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-2000:],
        "historical_files_hashed": len(GATE_A_FILES),
        "historical_files_modified": touched,
        "historical_artifact_unchanged": not touched,
        "rerun_dir": str(rerun_dir),
    }

    rerun_json = rerun_dir / "analysis.json"
    if process.returncode != 0 or not rerun_json.exists():
        result["ok"] = False
        result["blocker"] = (
            f"the Gate A analyze stage did not produce {rerun_json}. "
            f"returncode {process.returncode}."
        )
        result["comparisons"] = {}
        return result

    fresh = json.loads(rerun_json.read_text(encoding="utf-8"))
    comparisons: dict[str, Any] = {}
    for name, path in KEY_METRICS.items():
        try:
            want = _dig(preserved, path)
            got = _dig(fresh, path)
        except (KeyError, IndexError, TypeError) as exc:
            comparisons[name] = {"match": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        comparisons[name] = {"preserved": want, "recomputed": got, "match": _same(want, got)}

    for name, claimed in _report_claims(preserved_report).items():
        key = name.replace("report_", "")
        recomputed = _dig(fresh, KEY_METRICS[key]) if key in KEY_METRICS else None
        rounded = None if recomputed is None else round(float(recomputed), 4)
        comparisons[name] = {
            "report_states": claimed,
            "recomputed_rounded": rounded,
            "match": rounded == claimed,
        }

    result["comparisons"] = comparisons
    result["n_metrics"] = len(comparisons)
    result["n_mismatched"] = sum(1 for v in comparisons.values() if not v["match"])
    result["ok"] = bool(result["historical_artifact_unchanged"] and result["n_mismatched"] == 0)
    return result


def write_outputs(out_dir: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    json_path = write_json(out_dir / "repro.json", result)
    lines = [
        "# Gate A reproduction check",
        "",
        f"Verdict: **{'PASS' if result['ok'] else 'FAIL'}**",
        "",
        f"- Gate A root: `{result['gate_a_root']}`",
        f"- Rerun output: `{result['rerun_dir']}`",
        f"- Historical artifact unchanged: **{result['historical_artifact_unchanged']}**",
        f"- Historical files hashed before and after: {result['historical_files_hashed']}",
        f"- Analyze exit code: {result['returncode']}",
        "",
        "The rerun reads `artifacts/full/raw/predictions.jsonl` and writes only into the new "
        "directory above. The preserved analysis and report are inputs, never outputs.",
        "",
    ]
    if not result.get("comparisons"):
        lines += [
            "## Blocker",
            "",
            str(result.get("blocker", "unknown")),
            "",
            "```",
            str(result.get("stderr_tail", ""))[-1200:],
            "```",
            "",
        ]
        return json_path, write_text(out_dir / "REPRO_REPORT.md", "\n".join(lines))

    lines += [
        f"## Key metrics: {result['n_metrics'] - result['n_mismatched']} of "
        f"{result['n_metrics']} match",
        "",
        "| metric | preserved | recomputed | match |",
        "|---|---|---|---|",
    ]
    for name, row in result["comparisons"].items():
        want = row.get("preserved", row.get("report_states"))
        got = row.get("recomputed", row.get("recomputed_rounded"))
        lines.append(f"| `{name}` | {want} | {got} | {row['match']} |")
    lines += [
        "",
        "A mismatch means the preserved Gate A report cannot be regenerated from its own saved "
        "rows, and the historical result must not be cited until the cause is found.",
        "",
    ]
    return json_path, write_text(out_dir / "REPRO_REPORT.md", "\n".join(lines))
