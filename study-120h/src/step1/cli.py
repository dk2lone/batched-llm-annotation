"""Small local control surface for Step 1.

Paid execution lives in ``modal_step1.py`` so a local status command can never
start a GPU by accident.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from step0.hashing import write_json, write_text

from . import amendment
from .frozen import load
from .paths import STATUS_JSON, STATUS_MD, Paths
from .report import exit_gate, render_status, verify_step0_seal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="step1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="verify the sealed Step 0 freeze")
    sub.add_parser("amend", help="record the authorised execution-only amendment")
    status = sub.add_parser("status", help="evaluate the Step 1 exit gate")
    status.add_argument("--out", type=Path, default=Paths().root)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    frozen = load()
    if args.command == "verify":
        record = verify_step0_seal(frozen)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0 if record["ok"] else 1
    if args.command == "amend":
        path, record = amendment.write(frozen)
        print(f"wrote {path}")
        print(f"content hash {record['content_hash']}")
        return 0

    paths = Paths(root=args.out)
    record = exit_gate(frozen, paths)
    write_json(STATUS_JSON, record, allow_existing=True)
    write_text(STATUS_MD, render_status(record), allow_existing=True)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["verdict"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
