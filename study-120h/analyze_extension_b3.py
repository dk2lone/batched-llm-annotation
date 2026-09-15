"""Run the sealed CPU-only X1R4 A9 full-grid analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from freezeb.extension_b3_analysis import N_DRAWS, analyze_grid, render_report  # noqa: E402
from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402

STUDY = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--output", type=Path, default=STUDY / "extension-b3/analysis")
    args = parser.parse_args()

    result = analyze_grid(STUDY, n_draws=args.draws, workers=args.workers)
    args.output.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output / "analysis.json"
    report_path = args.output / "report.md"
    write_json(analysis_path, result)
    report_path.write_text(render_report(result), encoding="utf-8")
    manifest = {
        "schema_version": "ext-x1r4-a9-analysis-manifest-v1",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "analysis_sha256": sha256_file(analysis_path),
        "report_sha256": sha256_file(report_path),
        "analysis_module_sha256": sha256_file(
            STUDY / "src/freezeb/extension_b3_analysis.py"
        ),
        "runner_sha256": sha256_file(Path(__file__)),
        "n_draws": args.draws,
        "workers": args.workers,
        "gpu_work_launched": False,
        "scientific_model_output_generated": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(args.output / "MANIFEST.json", manifest)
    print(
        json.dumps(
            {"output": str(args.output), "recurrence_gate": result["recurrence_gate"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
