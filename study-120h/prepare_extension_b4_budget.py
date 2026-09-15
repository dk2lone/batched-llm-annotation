"""Hash operator-verified budget evidence locally; never contact Modal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb.extension_b4 import load, verify_budget  # noqa: E402
from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load(args.plan)
    if (
        sha256_json({key: value for key, value in plan.items() if key != "plan_sha256"})
        != plan["plan_sha256"]
    ):
        raise ValueError("plan content hash differs")
    record = load(args.input)
    record.update(
        plan_sha256=plan["plan_sha256"],
        profile=args.profile,
        evidence_file_sha256=sha256_file(args.input),
    )
    record.pop("record_sha256", None)
    record["record_sha256"] = sha256_json(record)
    verify_budget(record, plan, args.profile)
    write_json(args.output, record)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "record_sha256": record["record_sha256"],
                "gpu_work_launched": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
