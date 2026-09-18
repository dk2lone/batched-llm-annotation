"""Run the CPU-only revised B4 analysis from exact linked recovery outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb.b4_revised import analyze, render_report  # noqa: E402
from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original, revised = analyze(STUDY, args.base_run, args.recovery_dir, args.inventory)
    write_json(args.output / "ORIGINAL_ANALYSIS.json", original)
    write_json(args.output / "REVISED_ANALYSIS.json", revised)
    (args.output / "REPORT.md").write_text(render_report(revised), encoding="utf-8")
    files = {
        name: sha256_file(args.output / name)
        for name in ("ORIGINAL_ANALYSIS.json", "REVISED_ANALYSIS.json", "REPORT.md")
    }
    core = {
        "schema_version": "ext-x1r4-a11-b4-repaired-analysis-manifest-v1",
        "base_run": str(args.base_run.resolve()),
        "recovery_dir": str(args.recovery_dir.resolve()),
        "inventory": str(args.inventory.resolve()),
        "files_sha256": files,
        "gpu_work_launched": False,
    }
    manifest = {**core, "manifest_sha256": sha256_json(core)}
    write_json(args.output / "MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "repair_calls_used": revised["repair_calls_used"],
                "coverage_gate": revised["coverage_gate"],
                "gpu_work_launched": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
