"""Run the frozen CPU-only analysis for the completed historical bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from step2.analysis import N_DRAWS, analyze, render_report

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "step2-out" / "bridge" / "analysis"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = analyze(
        ROOT / "step2-out" / "bridge" / "calls.jsonl",
        ROOT / "freeze" / "20260912T041633Z" / "data" / "civil_comments" / "items.jsonl",
        ROOT / "step1-out" / "results" / "qwen_civil__civil_comments__Qwen_Qwen3-8B" / "solo.jsonl",
        args.n_draws,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "analysis.json"
    report_path = args.out / "report.md"
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report = render_report(result)
    for path, content in ((json_path, encoded), (report_path, report)):
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite non-identical analysis: {path}")
        path.write_text(content, encoding="utf-8")
    print(verdict := result["verdict"]["label"])
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    if verdict != "REPLICATION_SUPPORTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
