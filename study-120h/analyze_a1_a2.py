"""CPU-only practical-information and cross-stage coherence analyses.

This script reads existing B3, B4, and B5 analyses. It launches no model inference and
does not modify any sealed or prior artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

from step0.hashing import sha256_file  # type: ignore[import-untyped]  # noqa: E402

B3_PATH = HERE / "extension-b3/analysis/analysis.json"
B4_PATH = HERE / "extension-b4/revised-analysis-20260918T034636Z/REVISED_ANALYSIS.json"
B5_PATH = HERE / "extension-b5/run-01/analysis-final-a/ANALYSIS.json"
N_LABELS = 8_192
BATCH_SIZE = 16


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def analyze_a1(b5: dict[str, Any]) -> dict[str, Any]:
    cross_family = b5["cross_family"]
    contrasts = cross_family["standardized_contrast_by_size"]
    intervals = cross_family["standardized_contrast_interval_by_size"]
    cells = b5["cells"]
    rows: list[dict[str, Any]] = []
    for batch_size in sorted(int(value) for value in contrasts):
        key = str(batch_size)
        contrast = float(contrasts[key])
        contrast_low = float(intervals[key]["low"])
        contrast_high = float(intervals[key]["high"])
        excess_de = 1 + (batch_size - 1) * contrast
        excess_de_low = 1 + (batch_size - 1) * contrast_low
        excess_de_high = 1 + (batch_size - 1) * contrast_high
        total_joint_de = float(
            np.mean(
                [
                    next(
                        float(row["approximate_variance_inflation"])
                        for row in cell["by_size"]
                        if int(row["batch_size"]) == batch_size
                    )
                    for cell in cells
                ]
            )
        )
        rows.append(
            {
                "batch_size": batch_size,
                "standardized_excess_dependence": contrast,
                "standardized_excess_interval_95": [contrast_low, contrast_high],
                "excess_design_effect": excess_de,
                "excess_design_effect_interval_95": [excess_de_low, excess_de_high],
                "effective_n_from_8192": N_LABELS / excess_de,
                "effective_n_interval_95": [
                    N_LABELS / excess_de_high,
                    N_LABELS / excess_de_low,
                ],
                "mean_total_joint_design_effect": total_joint_de,
                "mean_total_joint_effective_n_from_8192": N_LABELS / total_joint_de,
            }
        )
    return {
        "analysis": "A1 batch-size dependence translated to statistical information",
        "rows": rows,
        "excess_design_effect_increases_monotonically": all(
            left["excess_design_effect"] < right["excess_design_effect"]
            for left, right in zip(rows[:-1], rows[1:], strict=True)
        ),
        "effective_n_decreases_monotonically": all(
            left["effective_n_from_8192"] > right["effective_n_from_8192"]
            for left, right in zip(rows[:-1], rows[1:], strict=True)
        ),
        "interpretation_boundary": (
            "Post-hoc descriptive transformation of the prespecified B5 standardized "
            "joint-minus-exact contrast. The 95% intervals monotonically transform the stored "
            "B5 parent-group cluster-bootstrap percentile endpoints. The excess design effect "
            "is 1 + (b-1) times the standardized contrast; it is not the total joint-arm design "
            "effect. Effective N is the corresponding variance-equivalent interpretation for "
            "8,192 rows under exchangeable within-call dependence."
        ),
    }


def primary_error_rate(cell: dict[str, Any]) -> float:
    accuracy = cell["accuracy"]["joint"]
    return (
        float(accuracy["error_rate_resolved"])
        * int(accuracy["resolved"])
        / int(accuracy["target_appearances"])
    )


def analyze_a2(b3: dict[str, Any], b4: dict[str, Any]) -> dict[str, Any]:
    b3_by_cell = {cell["cell_key"]: cell for cell in b3["cells"]}
    rows: list[dict[str, Any]] = []
    for b4_cell in b4["cells"]:
        cell_key = b4_cell["cell_key"]
        b3_cell = b3_by_cell[cell_key]
        error_rate = primary_error_rate(b3_cell)
        marginal_variance = error_rate * (1 - error_rate)
        if marginal_variance <= 0:
            raise ValueError(f"degenerate B3 marginal error rate: {cell_key}")
        covariance = float(b3_cell["covariance_unresolved_as_correct"]["joint_mean"])
        implied_de = 1 + (BATCH_SIZE - 1) * covariance / marginal_variance
        ratio = b4_cell["analysis"]["variance_ratio"]
        empirical_ratio = float(ratio["point"])
        rows.append(
            {
                "cell_key": cell_key,
                "b3_joint_covariance": covariance,
                "b3_joint_error_rate": error_rate,
                "b3_implied_design_effect": implied_de,
                "b4_empirical_variance_ratio": empirical_ratio,
                "b4_variance_ratio_interval_95": [
                    float(ratio["low"]),
                    float(ratio["high"]),
                ],
                "empirical_minus_implied": empirical_ratio - implied_de,
            }
        )
    implied = np.asarray([row["b3_implied_design_effect"] for row in rows], dtype=float)
    empirical_values = np.asarray(
        [row["b4_empirical_variance_ratio"] for row in rows], dtype=float
    )
    return {
        "analysis": "A2 B3 covariance-to-B4 variance-inflation coherence",
        "rows": rows,
        "pearson_correlation_descriptive": float(
            sps.pearsonr(implied, empirical_values).statistic
        ),
        "spearman_correlation_descriptive": float(
            sps.spearmanr(implied, empirical_values).statistic
        ),
        "mean_absolute_difference": float(np.mean(np.abs(empirical_values - implied))),
        "root_mean_squared_difference": float(
            math.sqrt(np.mean((empirical_values - implied) ** 2))
        ),
        "interpretation_boundary": (
            "Post-hoc four-cell coherence check only. B3 converts observed joint covariance to "
            "a design effect under exchangeable within-call dependence; B4 measures empirical "
            "variance inflation in fresh repeated samples. Four cells do not support a formal "
            "calibration or generalization claim, and the disclosed B4 technical recovery "
            "boundary remains unchanged."
        ),
    }


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def render(result: dict[str, Any]) -> str:
    a1 = result["a1"]
    a2 = result["a2"]
    lines = [
        "# CPU follow-ups A1-A2",
        "",
        "All calculations use existing B3/B4/B5 outputs. No model inference or GPU work "
        "was launched.",
        "",
        "## A1: batch size and effective information",
        "",
        table(
            ["batch", "excess corr.", "excess DE [95%]", "effective N [95%]", "total joint DE"],
            [
                [
                    str(row["batch_size"]),
                    f"{row['standardized_excess_dependence']:.4f}",
                    f"{row['excess_design_effect']:.3f} "
                    f"[{row['excess_design_effect_interval_95'][0]:.3f}, "
                    f"{row['excess_design_effect_interval_95'][1]:.3f}]",
                    f"{row['effective_n_from_8192']:.0f} "
                    f"[{row['effective_n_interval_95'][0]:.0f}, "
                    f"{row['effective_n_interval_95'][1]:.0f}]",
                    f"{row['mean_total_joint_design_effect']:.3f}",
                ]
                for row in a1["rows"]
            ],
        ),
        "",
        "Pairwise excess dependence decreases after batch size 4, but the excess design effect "
        "increases monotonically because each label shares a call with more labels. The "
        "variance-equivalent effective N falls from 7,589 at batch size 2 to 3,440 at size 32.",
        "",
        a1["interpretation_boundary"],
        "",
        "## A2: covariance-to-variance coherence",
        "",
        table(
            ["cell", "B3 implied DE", "B4 empirical ratio [95%]", "difference"],
            [
                [
                    row["cell_key"],
                    f"{row['b3_implied_design_effect']:.3f}",
                    f"{row['b4_empirical_variance_ratio']:.3f} "
                    f"[{row['b4_variance_ratio_interval_95'][0]:.3f}, "
                    f"{row['b4_variance_ratio_interval_95'][1]:.3f}]",
                    f"{row['empirical_minus_implied']:+.3f}",
                ]
                for row in a2["rows"]
            ],
        ),
        "",
        f"Descriptive Pearson correlation: {a2['pearson_correlation_descriptive']:.4f}. "
        f"Descriptive Spearman correlation: {a2['spearman_correlation_descriptive']:.4f}. "
        f"Mean absolute difference: {a2['mean_absolute_difference']:.3f}.",
        "",
        a2["interpretation_boundary"],
        "",
        "## Input integrity",
        "",
    ]
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in result["input_sha256"].items())
    lines.append("")
    return "\n".join(lines)


def write_csvs(out: Path, result: dict[str, Any]) -> None:
    with (out / "a1_batch_information.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["a1"]["rows"][0]))
        writer.writeheader()
        writer.writerows(result["a1"]["rows"])
    with (out / "a2_cross_stage_coherence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["a2"]["rows"][0]))
        writer.writeheader()
        writer.writerows(result["a2"]["rows"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "cpu-followups/20260919-a1-a2",
        help="new output directory; an existing directory is refused",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {out}")

    b3 = load(B3_PATH)
    b4 = load(B4_PATH)
    b5 = load(B5_PATH)
    result = {
        "schema_version": "cpu-followups-a1-a2-v1",
        "gpu_work_launched": False,
        "a1": analyze_a1(b5),
        "a2": analyze_a2(b3, b4),
        "input_sha256": {
            str(B3_PATH): sha256_file(B3_PATH),
            str(B4_PATH): sha256_file(B4_PATH),
            str(B5_PATH): sha256_file(B5_PATH),
        },
    }
    out.mkdir(parents=True)
    (out / "ANALYSIS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "REPORT.md").write_text(render(result), encoding="utf-8")
    write_csvs(out, result)
    print(out)


if __name__ == "__main__":
    main()
