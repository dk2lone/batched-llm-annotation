"""Analyze only the sealed, complete B5 endpoint pilot ledger on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

import seal_b5_endpoint as seal  # noqa: E402

from freezeb import b5_endpoint as pilot  # noqa: E402
from freezeb import b5_endpoint_analysis as analysis  # noqa: E402
from freezeb import extension_b4 as b4  # noqa: E402
from step0.hashing import read_jsonl, sha256_file, sha256_json, write_json  # noqa: E402


def terminal_events(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    attempts: dict[str, dict[str, Any]] = {}
    by_call: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        attempt_id = str(row["attempt_id"])
        if attempt_id in attempts:
            raise ValueError(f"duplicate attempt {attempt_id}")
        attempts[attempt_id] = row
        by_call.setdefault(str(row["call_id"]), []).append(row)
    final = {}
    for call_id, rows in by_call.items():
        rows.sort(key=lambda row: int(row["attempt"]))
        if [row["attempt"] for row in rows] not in ([1], [1, 2]):
            raise ValueError(f"invalid retry sequence: {call_id}")
        if len(rows) == 2 and (
            rows[0]["status"] not in pilot.RETRYABLE
            or rows[1].get("parent_attempt_id") != rows[0]["attempt_id"]
        ):
            raise ValueError(f"invalid retry linkage: {call_id}")
        final[call_id] = rows[-1]
    failures = Counter(
        str(row.get("status")) for row in final.values() if row.get("status") != "ok"
    )
    parses = Counter(
        str(row.get("parse_status")) for row in final.values() if row.get("parse_status") != "ok"
    )
    first_failures = Counter(
        str(rows[0].get("status")) for rows in by_call.values() if rows[0].get("status") != "ok"
    )
    first_parse_failures = Counter(
        str(rows[0].get("parse_status"))
        for rows in by_call.values()
        if rows[0].get("parse_status") != "ok"
    )
    return final, {
        "stored_attempts": len(attempts),
        "retried_calls": sum(len(rows) == 2 for rows in by_call.values()),
        "first_attempt_failures_by_status": dict(sorted(first_failures.items())),
        "first_attempt_parse_failures_by_status": dict(sorted(first_parse_failures.items())),
        "terminal_failures_by_status": dict(sorted(failures.items())),
        "terminal_parse_failures_by_status": dict(sorted(parses.items())),
    }


def analyze(events: Path, output: Path, seal_dir: Path) -> dict[str, Any]:
    sealed = seal.verify(seal_dir)
    final, quality = terminal_events(events)
    frames: dict[str, list[dict[str, Any]]] = {}
    cells = []
    expected: set[str] = set()
    for cell_key in pilot.CELLS:
        task = pilot.split_cell(cell_key)[1]
        frames.setdefault(task, b4.frame(STUDY, task))
        rows = frames[task]
        ids = [str(row["doc_id"]) for row in rows]
        expected.update(str(call["call_id"]) for call in pilot.cell_calls(cell_key, ids))
        data = analysis.CellData(
            cell_key, ids, {str(row["doc_id"]): row["reference_normalized"] for row in rows}
        )
        data.load({key: value for key, value in final.items() if value["cell_key"] == cell_key})
        cells.append(data)
    if len(expected) != pilot.EXPECTED_TOTAL or set(final) != expected:
        raise ValueError(
            f"incomplete or foreign pilot ledger: {len(set(final) & expected)}/{len(expected)}"
        )
    result = analysis.analyze_endpoint(cells)
    result["quality"] = quality
    result["completed_calls"] = len(final)
    result["result_sha256"] = sha256_json({k: v for k, v in result.items() if k != "result_sha256"})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ANALYSIS.json", result)
    cross = result["cross_family"]
    lines = [
        "# B5 endpoint pilot",
        "",
        "Separate exploratory spending decision; never pool with confirmatory B5.",
        "",
        f"Completed calls: {len(final):,}. Verdict: **{result['decision']['verdict']}**.",
        "",
        (
            f"Standardized contrasts: size 2 {cross['standardized_contrast_size_2']:.6f}; "
            f"size 32 {cross['standardized_contrast_size_32']:.6f}."
        ),
        (
            f"Raw contrasts: size 2 {cross['raw_contrast_size_2']:.6f}; "
            f"size 32 {cross['raw_contrast_size_32']:.6f}."
        ),
        (
            f"Endpoint difference: {cross['endpoint_difference']:.6f}; "
            f"95% cluster bootstrap interval {cross['intervals']['delta']}."
        ),
        "",
        "## Cells",
        "",
        "| Cell | Size 2 | Size 32 | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in result["cells"]:
        lines.append(
            f"| {row['cell_key']} | {row['by_size'][0]['standardized_contrast']:.6f} | "
            f"{row['by_size'][1]['standardized_contrast']:.6f} | "
            f"{row['endpoint_difference']:.6f} |"
        )
    lines.extend(["", "## Technical outcomes", "", json.dumps(quality, sort_keys=True), ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    meter_path = events.parent / "meter.jsonl"
    meters = list(read_jsonl(meter_path)) if meter_path.is_file() else []
    settled = [row for row in meters if row.get("kind") == "settled"]
    charged_hours = sum(float(row["charged_hours"]) for row in settled)
    cost = {
        "schema_version": "b5-endpoint-pilot-cost-v1",
        "settled_metered_a100_hours_upper_bound": charged_hours,
        "gpu_cost_upper_bound_usd_at_1_59_per_hour": charged_hours * 1.59,
        "meter_source_sha256": sha256_file(meter_path) if meter_path.is_file() else None,
        "billing_note": (
            "settlement includes a one-minute allowance per Pod; actual Runpod billing "
            "requires a separate read-only account check"
        ),
    }
    write_json(output / "EXECUTION_COST.json", cost)
    manifest = {
        "schema_version": "b5-endpoint-pilot-analysis-manifest-v1",
        "seal_manifest_sha256": sealed["manifest_sha256"],
        "events_sha256": sha256_file(events),
        "result_sha256": result["result_sha256"],
        "files_sha256": {
            name: sha256_file(output / name)
            for name in ("ANALYSIS.json", "REPORT.md", "EXECUTION_COST.json")
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output / "MANIFEST.json", manifest)
    return {"completed_calls": len(final), "result_sha256": result["result_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=seal.latest())
    args = parser.parse_args()
    print(json.dumps(analyze(args.events, args.output, args.seal), indent=2))


if __name__ == "__main__":
    main()
