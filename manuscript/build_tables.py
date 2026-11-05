"""Format existing study summaries for the camera-ready manuscript."""

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
INPUTS = {
    "grid": "study-120h/extension-b3/analysis/analysis.json",
    "batch": "study-120h/extension-b5/run-01/analysis-final-a/ANALYSIS.json",
    "coverage": "study-120h/extension-b4/revised-analysis-20260918T034636Z/REVISED_ANALYSIS.json",
    "followups": "study-120h/cpu-followups/20260919-e2-e5/ANALYSIS.json",
}
MODELS = ["Qwen", "Granite", "Llama", "Mistral"]
TASKS = ["civil_comments", "multi_nli", "imdb", "ag_news"]
SHORT_TASK = {"civil_comments": "Civil", "multi_nli": "NLI", "imdb": "IMDb", "ag_news": "News"}
FULL_TASK = {"civil_comments": "Civil Comments", "multi_nli": "MultiNLI", "imdb": "IMDb", "ag_news": "AG News"}


def model_name(model):
    return next(name for name in MODELS if name.lower() in model.lower())


def write_csv(name, rows):
    with (HERE / "tables" / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = {name: json.loads((REPO / path).read_text()) for name, path in INPUTS.items()}
    provenance = {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in INPUTS.values()}
    cells = sorted(data["grid"]["cells"], key=lambda c: (MODELS.index(model_name(c["model_id"])), TASKS.index(c["task_id"])))
    assert len(cells) == 16
    grid_rows, tex = [], []
    for index, cell in enumerate(cells):
        accuracy = cell["accuracy"]
        effect = cell["primary_replay"]["effect_interval"]
        row = {"model": model_name(cell["model_id"]), "task": cell["task_id"]}
        row.update({arm + "_error_pct": 100 * accuracy[arm]["error_rate_resolved"] for arm in ["solo", "joint", "exact_context"]})
        row.update({"delta": effect["point"], "replay_shift_low": effect["low"], "replay_shift_high": effect["high"]})
        grid_rows.append(row)
        if index and index % 4 == 0:
            tex.append(r"\midrule")
        name = row["model"] if index % 4 == 0 else ""
        tex.append(f"{name} & {FULL_TASK[row['task']]} & {row['solo_error_pct']:.1f} & {row['joint_error_pct']:.1f} & {row['exact_context_error_pct']:.1f} & {effect['point']:.5f} & [{effect['low']:.5f}, {effect['high']:.5f}] " + r"\\")
    (HERE / "tables" / "grid_table.tex").write_text(
        "\\begin{tabular}{llrrrcl}\n\\toprule\n"
        + r"Model & Dataset & \multicolumn{3}{c}{Error rate (\%)} & \multicolumn{2}{c}{Excess error covariance, $\Delta$} \\" + "\n"
        + r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}" + "\n"
        + r" & & Solo & Joint & Target-only & Estimate & 95\% replay range \\" + "\n\\midrule\n"
        + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_csv("grid.csv", grid_rows)

    batch_rows, tex = [], []
    batch_cells = sorted(data["batch"]["cells"], key=lambda c: MODELS.index(model_name(c["model_id"])))
    for cell in batch_cells:
        values = []
        for size in cell["by_size"]:
            row = {
                "model": model_name(cell["model_id"]), "task": cell["task_id"], "batch_size": size["batch_size"],
                "joint_accuracy_pct": 100 * (1 - size["marginal_error_joint"]),
                "exact_accuracy_pct": 100 * (1 - size["marginal_error_exact"]),
                "joint_rho": size["joint_standardized"], "exact_rho": size["exact_standardized"],
                "joint_design_effect": 1 + (size["batch_size"] - 1) * size["joint_standardized"],
                "exact_design_effect": 1 + (size["batch_size"] - 1) * size["exact_standardized"],
            }
            batch_rows.append(row)
            values.append(f"{row['joint_accuracy_pct']:.1f}")
        tex.append(f"{model_name(cell['model_id'])}/{FULL_TASK[cell['task_id']]} & " + " & ".join(values) + r" \\")
    write_csv("batch_accuracy.csv", batch_rows)
    (HERE / "tables" / "batch_table.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        + r"Cell & 2 & 4 & 8 & 16 & 32 \\" + "\n\\midrule\n"
        + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n"
    )

    coverage_rows = []
    for cell in data["coverage"]["cells"]:
        a = cell["analysis"]
        for method, key in [("item_wald", "iid_wald_coverage"), ("call_t", "call_aware_t_coverage"), ("whole_call_bootstrap", "whole_call_bootstrap_coverage")]:
            estimate = a["intervals"][key]
            coverage_rows.append({"cell": a["cell_key"], "method": method, **{k: estimate[k] for k in ["point", "low", "high", "n_covered", "n_replicates"]}})
    write_csv("coverage.csv", coverage_rows)
    summary = {
        "input_sha256": provenance,
        "grid_equal_cell_delta": sum(r["delta"] for r in grid_rows) / 16,
        "joint_accuracy_minus_solo_pp": [r["solo_error_pct"] - r["joint_error_pct"] for r in grid_rows],
        "batch_equal_cell_joint_design_effect": {str(b): sum(r["joint_design_effect"] for r in batch_rows if r["batch_size"] == b) / 4 for b in [2, 4, 8, 16, 32]},
        "position": data["followups"]["e5"]["position"],
        "pair_distance": data["followups"]["e5"]["pair_distance"],
        "new_model_calls": 0,
    }
    (HERE / "tables" / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Formatted 16 grid rows, 20 batch-size rows, and 12 coverage rows from saved summaries.")


if __name__ == "__main__":
    main()
