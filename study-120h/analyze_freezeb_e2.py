"""Run the frozen CPU analysis for the first completed Freeze B E2 cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freezeb.e2_analysis import N_DRAWS, analyze, render_report

ROOT = Path(__file__).resolve().parent
DEFAULT_CELL = ROOT / "e2-out" / "qwen3-8b--multi_nli"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--cell", type=Path, default=DEFAULT_CELL)
    args = parser.parse_args()
    out = args.cell / "analysis"
    result = analyze(
        args.cell / "events.jsonl",
        ROOT / "freeze-b-work" / "20260912-cpu-draft",
        ROOT / "freeze" / "20260912T222618Z-B1",
        ROOT / "freeze" / "20260913T022442Z-B2" / "ADMISSION_LOCK.json",
        args.cell / "plan.json",
        args.cell / "SUMMARY.json",
        args.n_draws,
    )
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        out / "analysis.json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        out / "report.md": render_report(result),
    }
    for path, content in artifacts.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite non-identical analysis: {path}")
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
