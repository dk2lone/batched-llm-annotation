"""Single entry point for the gate. One subcommand per stage."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .analyze import analyze
from .config import Config, load_config
from .planning import plan, tokenizer_preflight
from .prepare import prepare_synthetic
from .runner import run

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMOKE_CONFIG = PACKAGE_ROOT / "configs" / "smoke.toml"
DEFAULT_FULL_CONFIG = PACKAGE_ROOT / "configs" / "full.toml"


def _resolve(cfg: Config, args: argparse.Namespace) -> tuple[Config, Path]:
    out_dir = Path(args.out) if args.out else PACKAGE_ROOT / cfg.out_dir
    if getattr(args, "backend", None):
        cfg = replace(cfg, run=replace(cfg.run, backend=args.backend))
    if getattr(args, "model_id", None):
        cfg = replace(cfg, run=replace(cfg.run, model_id=args.model_id))
    if getattr(args, "attn_implementation", None):
        cfg = replace(cfg, run=replace(cfg.run, attn_implementation=args.attn_implementation))
    return cfg, out_dir


def cmd_prepare_synthetic(args: argparse.Namespace) -> int:
    cfg, out_dir = _resolve(load_config(args.config), args)
    manifest = prepare_synthetic(cfg, out_dir)
    print(
        json.dumps(
            {k: manifest[k] for k in ("n_arm_datasets", "construction_ok", "failed_invariants")},
            indent=2,
        )
    )
    print(f"wrote {out_dir / 'synthetic'}")
    if args.strict and not manifest["construction_ok"]:
        print("CONSTRUCTION FAILURE: do not start the GPU run", file=sys.stderr)
        return 1
    return 0


def cmd_prepare_openml(args: argparse.Namespace) -> int:
    cfg, out_dir = _resolve(load_config(args.config), args)
    if not args.yes_download:
        print(
            "prepare-openml downloads data from OpenML. Re-run with --yes-download to confirm.",
            file=sys.stderr,
        )
        return 2
    from .openml_prep import prepare_openml  # noqa: PLC0415  (import only on explicit request)

    manifest = prepare_openml(cfg, out_dir)
    print(f"froze {len(manifest['sources'])} OpenML sources into {out_dir / 'openml'}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cfg, out_dir = _resolve(load_config(args.config), args)
    manifest = plan(cfg, out_dir)
    print(
        json.dumps(
            {"n_jobs": manifest["n_jobs"], "jobs_by_type": manifest["jobs_by_type"]}, indent=2
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg, out_dir = _resolve(load_config(args.config), args)
    summary = run(cfg, out_dir, limit=args.limit, phase=args.phase)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("backend", "n_planned", "n_already_done", "n_written", "n_errors")
            },
            indent=2,
        )
    )
    return 1 if summary["n_errors"] else 0


def cmd_preflight_tokenizer(args: argparse.Namespace) -> int:
    cfg, out_dir = _resolve(load_config(args.config), args)
    result = tokenizer_preflight(cfg, out_dir, allow_download=args.yes_download)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg, out_dir = _resolve(load_config(args.config), args)
    analysis = analyze(cfg, out_dir)
    print(json.dumps(analysis["gate"], indent=2))
    print(f"wrote {out_dir / 'analysis' / 'REPORT.md'}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Tiny end-to-end check. Never touches the network."""
    cfg = load_config(args.config)
    if cfg.run.backend != "mock":
        print("smoke requires the mock backend; it must not access the network", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else PACKAGE_ROOT / cfg.out_dir
    failures: list[str] = []

    manifest = prepare_synthetic(cfg, out_dir)
    if not manifest["construction_ok"]:
        failures.append(f"construction invariants failed: {manifest['failed_invariants']}")
    plan_manifest = plan(cfg, out_dir)
    expected_types = {"order_random", "order_nearest_first", "ablate_nearest", "ablate_random"}
    missing = expected_types - set(plan_manifest["jobs_by_type"])
    if missing:
        failures.append(f"plan is missing job types: {sorted(missing)}")
    run_summary = run(cfg, out_dir)
    if run_summary["n_errors"]:
        failures.append(
            f"run reported {run_summary['n_errors']} errors: {run_summary['errors'][:3]}"
        )
    if run_summary["n_written"] + run_summary["n_already_done"] != plan_manifest["n_jobs"]:
        failures.append("run did not cover every planned job")
    analysis = analyze(cfg, out_dir)
    if analysis["gate"]["verdict"] not in {
        "construction_failure",
        "model_capability_failure",
        "insufficient_results",
        "mechanism_unsupported",
        "mechanism_supported_enough_to_expand",
        "ambiguous",
    }:
        failures.append(f"unknown verdict {analysis['gate']['verdict']}")
    if not analysis["cells"]:
        failures.append("analysis produced no cells")

    print(f"smoke output: {out_dir}")
    print(f"verdict (mock data, not evidence): {analysis['gate']['verdict']}")
    if failures:
        for f in failures:
            print(f"SMOKE FAILURE: {f}", file=sys.stderr)
        return 1
    print("smoke OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gatek", description="Feasibility gate for idea K")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: object, default_config: Path) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(default_config))
        p.add_argument("--out", default=None, help="override the output directory")
        p.set_defaults(func=handler)
        return p

    p = add("prepare-synthetic", cmd_prepare_synthetic, DEFAULT_FULL_CONFIG)
    p.add_argument("--strict", action="store_true", help="exit nonzero when an invariant fails")

    p = add("prepare-openml", cmd_prepare_openml, DEFAULT_FULL_CONFIG)
    p.add_argument("--yes-download", action="store_true", help="confirm the OpenML download")

    add("plan", cmd_plan, DEFAULT_FULL_CONFIG)

    p = add("preflight-tokenizer", cmd_preflight_tokenizer, DEFAULT_FULL_CONFIG)
    p.add_argument(
        "--yes-download",
        action="store_true",
        help="allow downloading tokenizer files when they are not cached",
    )

    p = add("run", cmd_run, DEFAULT_FULL_CONFIG)
    p.add_argument("--backend", choices=["mock", "transformers"], default=None)
    p.add_argument("--model-id", dest="model_id", default=None)
    p.add_argument("--attn-implementation", dest="attn_implementation", default=None)
    p.add_argument("--limit", type=int, default=None, help="run at most this many pending jobs")
    p.add_argument(
        "--phase",
        choices=["anchor", "all"],
        default="all",
        help="run the 2D capability anchor first, or every pending job",
    )

    add("analyze", cmd_analyze, DEFAULT_FULL_CONFIG)
    add("smoke", cmd_smoke, DEFAULT_SMOKE_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
