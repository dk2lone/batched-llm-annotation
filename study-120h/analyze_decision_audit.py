"""Regenerate the CPU-only A1-A7 decision audit from original and recovery data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from freezeb.decision_audit import b3_audit, b4_audit  # noqa: E402
from step0.hashing import sha256_file, sha256_json, write_json  # noqa: E402

BASE = STUDY / "extension-b4/runpod-pcie-auto-20260917T230203Z"
RECOVERY = STUDY / "output-recovery/20260918T015605Z-smoke"
INVENTORY = STUDY / "output-recovery/20260918T013521Z-inventory/RECOVERY_INVENTORY.json"
REVISED = STUDY / "extension-b4/revised-analysis-20260918T034636Z/REVISED_ANALYSIS.json"
ORIGINAL_B3 = STUDY / "extension-b3/analysis/analysis.json"
REPAIRED_B3 = STUDY / "extension-b3/repaired-analysis-20260918T041906Z/REPAIRED_SENSITIVITY.json"


def _short(cell: str) -> str:
    model, task = cell.rsplit("::", 1)
    name = {
        "ibm-granite": "Granite", "meta-llama": "Llama", "mistralai": "Mistral", "Qwen": "Qwen"
    }[model.split("/")[0]]
    task_name = {
        "imdb": "IMDb", "civil_comments": "Civil Comments",
        "multi_nli": "MultiNLI", "ag_news": "AG News",
    }[task]
    return f"{name}/{task_name}"


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _interval(row: dict[str, Any], digits: int = 3) -> str:
    return f"{_f(row['low'], digits)}–{_f(row['high'], digits)}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tables(output: Path, b3: dict[str, Any], b4: dict[str, Any]) -> None:
    a4 = []
    a6 = []
    for cell in b3["cells"]:
        for assignment in cell["assignments"]:
            row = {"cell": cell["cell_key"], "assignment": assignment["assignment"]}
            for condition in ("joint", "exact_context", "solo_fake"):
                for metric in (
                    "marginal_error", "accuracy", "covariance", "standardized",
                    "within_group_error_homogeneity",
                ):
                    row[f"{condition}_{metric}"] = assignment[condition][metric]
                row[f"{condition}_covariance_interval"] = json.dumps(
                    assignment["replay_shift_95_intervals"][condition]
                )
                for metric in ("standardized", "homogeneity"):
                    row[f"{condition}_{metric}_interval"] = json.dumps(
                        assignment["replay_shift_95_intervals"][f"{condition}_{metric}"]
                    )
            for metric in (
                "joint_minus_exact", "standardized_joint_minus_exact",
                "standardized_joint_minus_solo",
            ):
                row[metric] = assignment[metric]
                row[f"{metric}_interval"] = json.dumps(
                    assignment["replay_shift_95_intervals"][metric]
                )
            a6.append(row)
        summary = {"cell": cell["cell_key"], "family": cell["family"], "task": cell["task"]}
        for condition in ("joint", "exact_context", "solo_fake"):
            for metric in ("marginal_error", "accuracy", "covariance", "standardized"):
                summary[f"{condition}_{metric}"] = sum(
                    assignment[condition][metric] for assignment in cell["assignments"]
                ) / 3
        summary.update(cell["cell_mean"])
        for metric, value in summary.items():
            if isinstance(value, list):
                summary[metric] = json.dumps(value)
        a4.append(summary)
    _write_csv(output / "A4_TABLE.csv", a4)
    _write_csv(output / "A6_TABLE.csv", a6)

    a3 = []
    for cell in b4["cells"]:
        for stream in ("reference", "evaluation"):
            result = cell["a3_homogeneity"][stream]
            row = {"cell": cell["cell_key"], "stream": stream,
                   "original_calls": result["original_calls"],
                   "repaired_calls": result["repaired_calls"]}
            for metric in (
                "maximum_label_share", "distinct_labels", "label_entropy",
                "pairwise_label_agreement",
            ):
                row[f"{metric}_repaired_minus_original"] = result[metric]["repaired_minus_original"]
                row[f"{metric}_interval"] = json.dumps(result[metric]["bootstrap_95_interval"])
            a3.append(row)
    _write_csv(output / "A3_TABLE.csv", a3)


def report(b3: dict[str, Any], b4: dict[str, Any]) -> str:
    original = json.loads(ORIGINAL_B3.read_text())
    repaired = json.loads(REPAIRED_B3.read_text())
    original_exact_duplicate = sum(
        cell["covariance_unresolved_as_correct"]["exact_context_mean"]
        == cell["fake_batch_controls_unresolved_as_correct"]["exact_context_mean"]
        for cell in original["cells"]
    )
    repaired_exact_duplicate = sum(
        cell["covariance_unresolved_as_correct"]["exact_context_mean"]
        == cell["fake_batch_controls_unresolved_as_correct"]["exact_context_mean"]
        for cell in repaired["cells"]
    )
    if original_exact_duplicate != 16 or repaired_exact_duplicate != 16:
        raise ValueError("A7 duplicate-column audit differs from the stored analyses")
    gates = b4["a2_sensitivity_c2_checks"]
    if gates["existing_all"]["supported_cells"] != b4["frozen_c2"]["supported_cells"]:
        raise ValueError("existing-target C2 check differs from revised report")
    if set(gates["original_only_all"]["supported_cells"]) != set(
        b4["frozen_c2"]["supported_cells"]
    ):
        raise ValueError("original-only target changes the supported-cell set")
    lines = [
        "# B3/B4 post-hoc decision audit",
        "",
        "The sealed A9 recurrence result remains primary. The original B4 result remains "
        "`coverage_unavailable_original`; the repaired C2 result is a separate post-execution "
        "sensitivity. A1–A6 below are new post-hoc checks and do not change either gate.",
        "",
        "## Provenance and repair classification",
        "",
        "The recovery inventory and linked receipts identify 4,097 repaired calls: 3,002 B3 "
        "and 1,095 B4. The B4 analysis uses the authenticated inventory mapping, then excludes "
        "every evaluation replicate containing at least one mapped repair. Successful original "
        "calls remain original. All 4,097 repairs used the later structured-output contract; "
        "this is a technical amendment, not uniformly generated confirmation data.",
        "",
        "Existing-report reconciliation: sealed A9 has global raw excess 0.009610; the "
        "separately repaired B3 sensitivity has 0.007899, a 17.81% reduction caused by "
        "replacing 3,002 unresolved source calls. Both report 13/16 positive cells. Original "
        "B4 has 1,095 unresolved outputs and no coverage estimate; revised B4 uses exactly "
        "1,095 linked repairs and reports 400 evaluation replicates per cell. This audit "
        "reproduces those revised IID and call-aware coverage counts exactly. Its smaller "
        "repair-free totals follow whole-replicate exclusion, not missing execution.",
        "",
        "## A1 — Repair-free B4 coverage",
        "",
        ("| cell | clean evaluation / 400 | clean reference / 250 | "
         "IID coverage [exact 95%] | call-aware coverage | variance ratio [95%] | "
         "frozen C2 status |"),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in b4["cells"]:
        streams = cell["streams"]
        variant = cell["a1_a2_coverage_2x2"]["existing_repair_free"]
        ratio = variant["variance_ratio"]
        lines.append(
            f"| {_short(cell['cell_key'])} | {streams['evaluation']['retained_replicates']} | "
            f"{streams['reference']['retained_replicates']} | "
            f"{_f(variant['iid_coverage']['point'])} [{_interval(variant['iid_coverage'])}] | "
            f"{_f(variant['call_aware_coverage']['point'])} | "
            f"{_f(ratio['point'])} [{_f(ratio['low'])}–{_f(ratio['high'])}] | "
            "below frozen 400-replicate minimum |"
        )
    lines += [
        "",
        "Granite/IMDb retains only 133 clean evaluation and 82 clean reference replicates; "
        "its original C2 support is **too imprecise under the frozen gate**, rather than "
        "disproved. Qwen/Civil Comments is also below the frozen sample minimum, though "
        "its repair-free IID coverage remains 0.884. Thus no repair-free subset can pass the "
        "unchanged C2 gate. Mean widths and exact intervals are in `B4_AUDIT.json`.",
        "",
        "## A2 — Original-only reference targets",
        "",
        "Each original-only target averages the means of complete, unrepaired reference "
        "replicates. This keeps the frozen equal-replicate weighting and 512 appearances per "
        "replicate; partly repaired reference replicates are excluded whole.",
        "",
        ("| cell | existing target | original-only target (SE) | shift | "
         "inside existing 95% MC band | IID coverage: existing/all, original/all, "
         "existing/clean, original/clean |"),
        "|---|---:|---:|---:|---|---|",
    ]
    for cell in b4["cells"]:
        target = cell["reference_targets"]
        variants = cell["a1_a2_coverage_2x2"]
        coverage = ", ".join(
            _f(variants[key]["iid_coverage"]["point"])
            for key in (
                "existing_all", "original_only_all", "existing_repair_free",
                "original_only_repair_free",
            )
        )
        lines.append(
            f"| {_short(cell['cell_key'])} | {_f(target['existing'], 5)} | "
            f"{_f(target['original_only_complete_replicates'], 5)} "
            f"({_f(target['original_only_se'], 5)}) | "
            f"{_f(target['difference_original_minus_existing'], 5)} | "
            f"{'yes' if target['inside_existing_95_uncertainty_band'] else 'no'} | {coverage} |"
        )
    lines += [
        "",
        "Granite's original-only target shifts by −0.00523, outside the existing target's "
        "Monte Carlo band. Its IID coverage stays about 0.78–0.79 across the four combinations; "
        "Qwen stays about 0.88. With all 400 evaluation replicates, changing the target "
        "does not change the revised supported-cell set. Clean subsets cannot satisfy the "
        "frozen sample minimum. The original-only reference precision check passes in all cells.",
        "",
        "## A3 — Repair/schema homogeneity",
        "",
        "Calls are the bootstrap units, stratified by cell and reference/evaluation stream. "
        "B4 has no assignment arm beyond its single joint condition. The full four-metric, "
        "eight-stratum results are in `A3_TABLE.csv`.",
        "",
        ("| cell | evaluation repaired/original calls | "
         "pairwise label agreement, repaired minus original [95%] |"),
        "|---|---:|---:|",
    ]
    for cell in b4["cells"]:
        row = cell["a3_homogeneity"]["evaluation"]
        metric = row["pairwise_label_agreement"]
        low, high = metric["bootstrap_95_interval"]
        lines.append(
            f"| {_short(cell['cell_key'])} | {row['repaired_calls']}/{row['original_calls']} | "
            f"{_f(metric['repaired_minus_original'])} [{_f(low)}, {_f(high)}] |"
        )
    lines += [
        "",
        "Granite repaired calls are not measurably more homogeneous: their reference and "
        "evaluation agreement intervals both include zero. These observational contrasts "
        "cannot identify the effect of the output schema. `B4_AUDIT.json` also records "
        "all-call, original-call, repaired-call, and clean-replicate error covariance, plus "
        "full versus clean variance ratios.",
        "",
        "## A4 — Standardized B3 dependence",
        "",
        "`A4_TABLE.csv` gives all 16 cells in one table: marginal error/accuracy, raw covariance, "
        "covariance divided by p(1−p), joint-minus-exact and joint-minus-solo contrasts, "
        "and conditional assignment-replay intervals. Original A9 unresolved outputs use its "
        "frozen unresolved-as-correct treatment. Marginal p is fixed under this replay, "
        "so it has a point estimate only. This is post-hoc, not a new gate.",
        "",
        f"The standardized joint-minus-exact contrast is positive in "
        f"{b3['standardized_positive_cells']} "
        f"of 16 cells and all {len(b3['standardized_positive_families'])} model families. "
        "The direction and cross-family recurrence therefore survive standardization; "
        "the A9 practical floor remains on the raw scale.",
        "",
        "## A5 — B4 variance inflation",
        "",
        ("| cell | call-aware / IID variance | empirical / IID ratio [95%] | "
         "IID / call-aware coverage | IID / call-aware mean width | "
         "effective n (approx.) |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in b4["cells"]:
        row = cell["a5_frozen_variance"]
        ratio = row["variance_ratio"]
        lines.append(
            f"| {_short(cell['cell_key'])} | "
            f"{_f(row['call_aware_variance'], 6)} / {_f(row['iid_variance'], 6)} | "
            f"{_f(ratio['point'])} [{_f(ratio['low'])}, {_f(ratio['high'])}] | "
            f"{_f(row['iid_coverage']['point'])} / {_f(row['call_aware_coverage']['point'])} | "
            f"{_f(row['mean_width_iid'])} / {_f(row['mean_width_call_aware'])} | "
            f"{_f(row['effective_n_approx'], 0)} |"
        )
    lines += [
        "",
        "The existing 1.25 ratio floor is exceeded by Granite, Qwen, and Mistral point "
        "estimates; Granite and Qwen have intervals wholly above it. Llama does not. "
        "For equal batch size 16 and common marginal p, a raw pair covariance of 0.002 "
        "implies approximate inflation 1 + 15×0.002/[p(1−p)]. The cell-specific values "
        "are in `B4_AUDIT.json`. This formula assumes exchangeable within-call covariance "
        "and no additional between-call dependence. Effective n = 512/observed ratio is "
        "only the corresponding design-effect interpretation.",
        "",
        "## A6 — Three existing B3 assignments",
        "",
        "The sealed protocol defines three seeded item regroupings, not composition arms. "
        "`A6_TABLE.csv` reports every cell and assignment: joint, exact fake, and solo fake "
        "covariance, marginal errors, standardized contrasts, joint-call error homogeneity, "
        "and assignment-replay intervals. The equal-cell raw excess by assignment is "
        f"{', '.join(_f(value, 5) for value in b3['assignment_global_excess'])}. "
        "All assignments reuse the same items, so this between-assignment range is "
        "descriptive; no independent composition-arm test is available.",
        "",
        "**Composition decision: no evidence motivating a new composition experiment.** "
        "Assignment variation alone does not identify composition effects.",
        "",
        "## A7 — Report integrity",
        "",
        "In both stored B3 analyses, the exact-context fake covariance duplicates the "
        "exact-context covariance in all 16 cells. It is the same regrouping, not a second "
        "control. Solo fake covariance is separate in all 16 cells and is labeled as such "
        "in the new tables. A9 replay p = 0.0002 is the 4,999-draw resolution floor, "
        "not an effect-size measure. No sealed or original artifact was edited.",
        "",
        "## Conclusions across sensitivities",
        "",
        ("| analysis | B3 positive dependence across families | "
         "B4 IID undercoverage in Granite and Qwen | frozen C2 confirmation |"),
        "|---|---|---|---|",
        "| sealed A9 / original B4 | yes | unavailable original | unavailable original |",
        ("| repaired B3 / revised B4 | yes, 13/16 positive | "
         "yes, 0.778 / 0.880 | passes as repaired sensitivity |"),
        ("| repair-free B4 evaluations | unchanged B3 | descriptive signal persists | "
         "cannot pass 400-replicate minimum |"),
        ("| original-only reference target | unchanged B3 | "
         "persists with all evaluations | revised supported set unchanged |"),
        ("| schema homogeneity | unchanged B3 | no Granite homogeneity shift detected | "
         "repair contract remains a threat |"),
        "| standardized B3 dependence | positive in all families | unchanged B4 | no new gate |",
        "",
        "## Decision and paid-work trigger",
        "",
        "The strongest remaining threat is the mixed original/recovery output contract in "
        "Granite/IMDb: only 133 clean evaluation replicates remain, and its original-only "
        "reference target lies outside the revised target's Monte Carlo band. Its coverage "
        "estimate is stable, but that does not turn the repaired C2 result into a clean "
        "confirmation.",
        "",
        "Ranked recommendation: (1) no additional experiment, reporting repaired C2 only "
        "as sensitivity evidence; (2) uniform-contract Granite/IMDb rerun if a clean C2 "
        "claim is required; (3) uniform-contract two-cell Granite+Qwen rerun for a fully "
        "clean C2 confirmation; (4) A11 extension to 1,000 replicates only if a new "
        "substantive precision need is identified; (5) batch sizes 2, 4, 8, 16, 32 only "
        "after the repair concern is resolved and batch-size dependence becomes the question; "
        "(6) batch-composition intervention, which A6 does not motivate.",
        "",
        "No GPU work is triggered by the current robustness results. Paid GPU work would "
        "be triggered by a decision to claim clean C2 confirmation: "
        "the existing Granite cell has 133/400 repair-free evaluations and 82/250 "
        "repair-free references, so it cannot supply that claim. One clean Granite cell "
        "addresses the largest repair objection but does not create a fully clean two-cell "
        "C2 confirmation; that requires at least two uniformly generated cells. Do not "
        "extend A11 merely to narrow intervals. A batch-size experiment requires the repair "
        "concern to be resolved and a genuine question about dependence versus batch size.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=319)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output directory already exists; preserve existing artifacts")
    b4 = b4_audit(STUDY, BASE, RECOVERY, INVENTORY, REVISED)
    b3 = b3_audit(STUDY, draws=args.draws)
    args.output.mkdir(parents=True)
    write_json(args.output / "B4_AUDIT.json", b4)
    write_json(args.output / "B3_AUDIT.json", b3)
    tables(args.output, b3, b4)
    (args.output / "DECISION.md").write_text(report(b3, b4), encoding="utf-8")
    names = (
        "B4_AUDIT.json", "B3_AUDIT.json", "A3_TABLE.csv", "A4_TABLE.csv",
        "A6_TABLE.csv", "DECISION.md",
    )
    manifest_core = {
        "schema_version": "b3-b4-posthoc-decision-v1",
        "files_sha256": {name: sha256_file(args.output / name) for name in names},
        "inputs_sha256": {
            str(path.relative_to(STUDY)): sha256_file(path)
            for path in (
                INVENTORY, RECOVERY / "FULL_REPORT.json", REVISED, ORIGINAL_B3, REPAIRED_B3
            )
        },
        "draws": args.draws,
        "gpu_work_launched": False,
    }
    write_json(args.output / "MANIFEST.json", {
        **manifest_core, "manifest_sha256": sha256_json(manifest_core),
    })
    print(json.dumps({"output": str(args.output), "files": list(names)}, indent=2))


if __name__ == "__main__":
    main()
