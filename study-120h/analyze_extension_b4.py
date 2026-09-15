"""CPU-only B4 ledger validation and frozen interval/coverage analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb.extension_b4 import analyze, render_report  # noqa: E402
from step0.hashing import sha256_file, sha256_json, write_json, write_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=int, choices=(400, 1000, 2500), default=400)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(
            "analysis output must be a new directory; previous artifacts are immutable"
        )
    result = analyze(STUDY, args.input, args.prefix)
    write_json(args.output / "analysis.json", result)
    write_text(args.output / "report.md", render_report(result))
    manifest = {
        "schema_version": "ext-x1r4-a11-analysis-manifest-v1",
        "files_sha256": {
            name: sha256_file(args.output / name) for name in ("analysis.json", "report.md")
        },
        "code_sha256": {
            name: sha256_file(STUDY / name)
            for name in (
                "analyze_extension_b4.py",
                "src/freezeb/extension_b4.py",
                "src/freezeb/extension.py",
                "src/step0/stats.py",
            )
        },
        "gpu_work_launched": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(args.output / "MANIFEST.json", manifest)
    print(
        json.dumps({"output": str(args.output), "coverage_gate": result["coverage_gate"]}, indent=2)
    )


if __name__ == "__main__":
    main()
