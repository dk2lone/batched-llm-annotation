"""Deterministic CPU analysis of a completed B5 run. It creates no Pod and no model call."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, cast

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

import seal_b5  # noqa: E402

from freezeb import b5, b5_analysis  # noqa: E402
from freezeb import extension_b4 as b4  # noqa: E402
from step0.hashing import read_jsonl, sha256_file, sha256_json, write_json  # noqa: E402

A9_ANALYSIS = "extension-b3/analysis/analysis.json"


def final_events(path: Path) -> dict[str, dict[str, Any]]:
    """One terminal event per call: the retry when the protocol allowed one."""
    by_call: dict[str, list[dict[str, Any]]] = {}
    attempts: dict[str, dict[str, Any]] = {}
    for event in read_jsonl(path):
        attempt_id = str(event["attempt_id"])
        if attempt_id in attempts:
            if sha256_json(attempts[attempt_id]) != sha256_json(event):
                raise ValueError(f"conflicting duplicate attempt: {attempt_id}")
            continue
        attempts[attempt_id] = event
        by_call.setdefault(str(event["call_id"]), []).append(event)
    final = {}
    for call_id, rows in by_call.items():
        rows.sort(key=lambda row: int(row["attempt"]))
        sequence = [int(row["attempt"]) for row in rows]
        if sequence not in ([1], [1, 2]):
            raise ValueError(f"invalid attempt sequence: {call_id}")
        if rows[0]["attempt_id"] != f"{call_id}::attempt::1":
            raise ValueError(f"invalid first-attempt identity: {call_id}")
        if len(rows) == 2:
            if str(rows[0]["status"]) not in b5.RETRYABLE:
                raise ValueError(f"unpermitted retry: {call_id}")
            if rows[1].get("parent_attempt_id") != rows[0]["attempt_id"]:
                raise ValueError(f"invalid retry linkage: {call_id}")
        final[call_id] = rows[-1]
    return final


def historical_b3() -> dict[str, float]:
    path = STUDY / A9_ANALYSIS
    if not path.is_file():
        return {}
    deltas = cast(dict[str, float], json.loads(path.read_text())["primary"]["cell_delta"])
    return {cell_key: float(deltas[cell_key]) for cell_key in b5.CELLS if cell_key in deltas}


def render_report(result: dict[str, Any]) -> str:
    primary = result["primary"]
    interval = primary["cross_family_slope_interval"]
    lines = [
        "# B5 batch-size dose-response analysis",
        "",
        "## Primary result",
        "",
        (
            "The equal-weight cross-family slope of the standardized joint-minus-exact "
            f"contrast on log2(batch size) is {primary['cross_family_slope']:.6f} "
            f"(95% cluster-bootstrap interval [{interval['low']}, {interval['high']}]; "
            f"two-sided bootstrap p = {primary['cross_family_slope_two_sided_p']:.4f})."
        ),
        "",
        f"The prespecified test against zero rejects: **{primary['rejects_zero_at_alpha']}**.",
        "",
        "## Contrast by size",
        "",
        "| size | standardized contrast | 95% interval | raw contrast |",
        "|---:|---:|---|---:|",
    ]
    for k in b5.SIZES:
        standardized = result["cross_family"]["standardized_contrast_by_size"][str(k)]
        bound = result["cross_family"]["standardized_contrast_interval_by_size"][str(k)]
        raw = result["cross_family"]["raw_contrast_by_size"][str(k)]
        lines.append(
            f"| {k} | {standardized:.6f} | [{bound['low']}, {bound['high']}] | {raw:.6f} |"
        )
    lines += [
        "",
        "## Cells",
        "",
        "| cell | slope | 95% interval | two-sided p |",
        "|---|---:|---|---:|",
    ]
    for row in result["cells"]:
        bound = row["slope_interval"]
        lines.append(
            f"| {row['cell_key']} | {row['slope_standardized']:.6f} | "
            f"[{bound['low']}, {bound['high']}] | {row['slope_two_sided_p']:.4f} |"
        )
    lines += [
        "",
        "## Prespecified reading",
        "",
        f"Matched interpretations: {', '.join(result['outcome']['matched']) or 'none'}.",
        "",
        (
            f"Practical contrast {result['outcome']['practical_contrast']:.6f}; practical slope "
            f"{result['outcome']['practical_slope']:.6f}. Success is not defined as a positive "
            "slope, and a wide interval is not evidence of equivalence."
        ),
        "",
        "## Historical comparison",
        "",
        (
            "The sealed A9 size-16 raw covariance contrasts are listed beside this result for "
            "reference only. They used the earlier output contract and are never pooled into "
            "this experiment."
        ),
        "",
    ]
    for cell_key, value in sorted(result["historical_size_16_b3"]["values"].items()):
        lines.append(f"- {cell_key}: {value:.6f}")
    lines.append("")
    return "\n".join(lines)


def analyze(events: Path, output: Path, *, n_bootstrap: int, n_replay: int) -> dict[str, Any]:
    frames: dict[str, list[dict[str, Any]]] = {}
    cells = []
    stored = final_events(events)
    for cell_key in b5.CELLS:
        task_id = b5.split_cell(cell_key)[1]
        if task_id not in frames:
            frames[task_id] = b4.frame(STUDY, task_id)
        rows = frames[task_id]
        data = b5_analysis.CellData(
            cell_key,
            [str(row["doc_id"]) for row in rows],
            {str(row["doc_id"]): row["reference_normalized"] for row in rows},
        )
        data.load({key: row for key, row in stored.items() if row["cell_key"] == cell_key})
        cells.append(data)
    result = b5_analysis.analyze(
        cells, n_bootstrap=n_bootstrap, n_replay=n_replay, historical=historical_b3()
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ANALYSIS.json", result)
    (output / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    manifest = {
        "schema_version": "b5-analysis-manifest-v1",
        "analyzed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "events_sha256": sha256_file(events),
        "seal": seal_b5.verify(seal_b5.latest()),
        "files_sha256": {
            name: sha256_file(output / name) for name in ("ANALYSIS.json", "REPORT.md")
        },
        "result_sha256": result["result_sha256"],
        "gpu_work_launched": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output / "MANIFEST.json", manifest)
    return {"output": str(output), "result_sha256": result["result_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=b5_analysis.N_BOOTSTRAP)
    parser.add_argument("--replay", type=int, default=b5_analysis.N_REPLAY)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.events, args.output, n_bootstrap=args.bootstrap, n_replay=args.replay
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
