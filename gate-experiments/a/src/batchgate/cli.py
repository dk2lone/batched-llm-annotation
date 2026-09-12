"""Single entry point. Every stage is separate and resumable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze, write_outputs
from .config import Config, load_config
from .data import DownloadNotAllowedError, prepare_data
from .plan import PlanExistsError, build_plan, write_plan
from .runner import PlanIntegrityError, PreflightError, run
from .smoke import smoke

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "full.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="batchgate", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config path")
    parser.add_argument("--root", type=Path, default=None, help="base directory for artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data", help="freeze targets and the filler pool")
    prepare.add_argument(
        "--allow-download", action="store_true", help="required before any dataset download"
    )
    prepare.add_argument(
        "--synthetic",
        type=int,
        default=None,
        metavar="N",
        help="build N local synthetic records instead of downloading",
    )

    plan_cmd = sub.add_parser("plan", help="build the immutable job manifest")
    plan_cmd.add_argument("--force", action="store_true", help="overwrite an existing plan")

    run_cmd = sub.add_parser("run", help="execute planned jobs and append raw rows")
    run_cmd.add_argument("--backend", choices=["mock", "vllm"], default=None)
    run_cmd.add_argument("--limit", type=int, default=None, help="run at most N jobs")
    run_cmd.add_argument(
        "--dry-run", action="store_true", help="token preflight only, no generation"
    )
    run_cmd.add_argument(
        "--phase",
        choices=["primary", "controls", "all"],
        default="all",
        help="primary runs solo+joint; controls runs filler; all runs both",
    )

    sub.add_parser("analyze", help="compute the gate result from raw rows")
    smoke_cmd = sub.add_parser("smoke", help="synthetic end-to-end check, no network")
    smoke_cmd.add_argument("--keep", action="store_true", help="do not clear previous artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg: Config = load_config(args.config, args.root)

    try:
        if args.command == "prepare-data":
            provenance = prepare_data(cfg, args.allow_download, args.synthetic)
            print(json.dumps(provenance, indent=2)[:4000])
            return 0

        if args.command == "plan":
            jobs, meta = build_plan(cfg)
            path = write_plan(cfg, jobs, meta, args.force)
            print(f"wrote {len(jobs)} jobs to {path}")
            print(json.dumps(meta, indent=2, default=str))
            return 0

        if args.command == "run":
            summary = run(
                cfg,
                args.backend,
                args.limit,
                args.dry_run,
                phase=args.phase,
            )
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "analyze":
            result = analyze(cfg)
            json_path, md_path = write_outputs(cfg, result)
            print(f"wrote {json_path}\nwrote {md_path}")
            print(json.dumps(result["verdict"], indent=2))
            return 0

        if args.command == "smoke":
            code, failures = smoke(cfg, args.keep)
            for failure in failures:
                print(f"SMOKE FAILURE: {failure}", file=sys.stderr)
            print("smoke ok" if code == 0 else "smoke failed", file=sys.stderr)
            return code
    except (DownloadNotAllowedError, PlanExistsError, PlanIntegrityError, PreflightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
