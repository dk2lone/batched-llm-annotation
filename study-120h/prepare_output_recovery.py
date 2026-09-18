"""Build the immutable B3/B4 unresolved-call recovery inventory. CPU only."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb.output_recovery import (  # noqa: E402
    SCHEMA_VERSION,
    final_events,
    inventory_record,
    is_unresolved,
    summarize,
)
from step0.hashing import read_jsonl, sha256_file, sha256_json, write_json  # noqa: E402


def collect(root: Path, pattern: str, study: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    ledgers: dict[str, str] = {}
    for ledger in sorted(root.glob(pattern)):
        ledger_sha256 = sha256_file(ledger)
        events = list(read_jsonl(ledger))
        final = final_events(events)
        selected = [event for event in final.values() if is_unresolved(event)]
        records.extend(
            inventory_record(study, ledger.resolve(), event, ledger_sha256)
            for event in selected
        )
        ledgers[str(ledger.resolve())] = ledger_sha256
    return records, ledgers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b4-run", type=Path, required=True)
    parser.add_argument("--b3-root", type=Path, default=STUDY / "extension-b3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b4_records, b4_ledgers = collect(args.b4_run / "cells", "*/*/events.jsonl", "b4")
    b3_records, b3_ledgers = collect(args.b3_root, "*--*/cells/*/events.jsonl", "b3")
    records = sorted(b4_records + b3_records, key=lambda row: str(row["recovery_id"]))
    summary = summarize(records)
    expected = {
        "calls": 4097,
        "by_study": {"b3": 3002, "b4": 1095},
        "by_failure_class": {"malformed": 3966, "truncated": 131},
    }
    if any(summary[key] != value for key, value in expected.items()):
        raise ValueError(f"recovery inventory differs from audited completion counts: {summary}")
    core = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "b4_run": str(args.b4_run.resolve()),
        "b4_plan_sha256": sha256_file(args.b4_run / "PLAN-400.json"),
        "b3_analysis_manifest_sha256": sha256_file(args.b3_root / "analysis/MANIFEST.json"),
        "source_ledgers": dict(sorted((b4_ledgers | b3_ledgers).items())),
        "summary": summary,
        "policy": {
            "namespace": "repair-v1",
            "attempts_per_call": 1,
            "preserve_original_ledgers": True,
            "structured_output": "exact per-call JSON schema",
            "one_target_max_tokens": 128,
            "sixteen_target_max_tokens": 512,
            "successful_repair": "status=ok and parse_status=ok with exact requested IDs",
        },
        "records": records,
    }
    manifest = {**core, "manifest_sha256": sha256_json(core)}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "RECOVERY_INVENTORY.json", manifest)
    write_json(
        args.output / "RECOVERY_SUMMARY.json",
        {
            "schema_version": SCHEMA_VERSION,
            "inventory_sha256": sha256_file(args.output / "RECOVERY_INVENTORY.json"),
            "manifest_sha256": manifest["manifest_sha256"],
            "summary": summary,
            "gpu_work_launched": False,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
