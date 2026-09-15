"""Settle one interrupted B4 phase using operator-verified billing evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import math
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb.extension_b4 import append, events  # noqa: E402
from step0.hashing import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--billed-a100-hours", type=float, required=True)
    parser.add_argument("--billing-evidence", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.billed_a100_hours) or args.billed_a100_hours < 0:
        raise ValueError("billed hours must be finite and nonnegative")
    if args.billing_evidence.stat().st_size == 0:
        raise ValueError("billing evidence cannot be empty")
    with (args.run / ".active-run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = [
            row for row in events(args.run / "meter.jsonl") if row["phase_id"] == args.phase_id
        ]
        if len(records) != 1 or records[0]["kind"] != "start":
            raise ValueError(
                "exactly one unsettled phase must match; settled entries are immutable"
            )
        append(
            args.run / "meter.jsonl",
            [
                {
                    "kind": "settled",
                    "phase_id": args.phase_id,
                    "budget_sha256": records[0]["budget_sha256"],
                    "charged_hours": args.billed_a100_hours,
                    "accounting": "operator_verified_billing_reconciliation",
                    "billing_evidence_sha256": sha256_file(args.billing_evidence),
                    "verified_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                }
            ],
        )
    print("Meter reconciled locally. No GPU work launched.")


if __name__ == "__main__":
    main()
