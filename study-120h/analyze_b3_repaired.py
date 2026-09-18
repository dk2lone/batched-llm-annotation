"""Run the CPU-only repaired B3 recurrence sensitivity analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb.b3_repaired import analyze, render_report  # noqa: E402
from freezeb.extension_b3_analysis import N_DRAWS  # noqa: E402
from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original, revised = analyze(
        STUDY,
        args.recovery_dir,
        args.inventory,
        n_draws=args.draws,
        workers=args.workers,
    )
    write_json(args.output / "ORIGINAL_ANALYSIS.json", original)
    write_json(args.output / "REPAIRED_SENSITIVITY.json", revised)
    (args.output / "REPORT.md").write_text(render_report(revised), encoding="utf-8")
    files = {
        name: sha256_file(args.output / name)
        for name in ("ORIGINAL_ANALYSIS.json", "REPAIRED_SENSITIVITY.json", "REPORT.md")
    }
    core = {
        "schema_version": "ext-x1r4-a9-b3-repaired-sensitivity-manifest-v1",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "recovery_dir": str(args.recovery_dir.resolve()),
        "inventory": str(args.inventory.resolve()),
        "files_sha256": files,
        "analysis_module_sha256": sha256_file(STUDY / "src/freezeb/b3_repaired.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "n_draws": args.draws,
        "workers": args.workers,
        "gpu_work_launched": False,
        "scientific_model_output_generated": False,
    }
    write_json(args.output / "MANIFEST.json", {**core, "manifest_sha256": sha256_json(core)})
    print(
        json.dumps(
            {
                "output": str(args.output),
                "repair_calls_used": revised["repair_calls_used"],
                "comparison": revised["comparison_to_original"],
                "recurrence_gate": revised["recurrence_gate"],
                "gpu_work_launched": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
