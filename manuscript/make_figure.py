from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "figure_source_data.csv"
OUT = ROOT / "figures" / "main_results"

MODELS = ["Qwen", "Granite", "Llama", "Mistral"]
TASKS = ["Civil Comments", "MultiNLI", "IMDb", "AG News"]
METHODS = ["IID Wald", "Call-aware t", "Whole-call bootstrap"]
COLORS = {
    "IID Wald": "#C95E52",
    "Call-aware t": "#3F78A8",
    "Whole-call bootstrap": "#6F98B7",
}
BATCH_COLORS = {
    "Qwen": "#3F78A8",
    "Granite": "#C95E52",
    "Llama": "#7A6FA8",
    "Mistral": "#5B9279",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_figure(rows: list[dict[str, str]]) -> tuple[plt.Figure, dict[str, list]]:
    fig = plt.figure(figsize=(7.16, 2.12), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.92, 1.12])

    ax0 = fig.add_subplot(grid[0, 0])
    covariance = np.zeros((len(MODELS), len(TASKS)))
    for row in rows:
        if row["panel"] != "covariance":
            continue
        covariance[MODELS.index(row["model"]), TASKS.index(row["task"])] = 100 * float(
            row["value"]
        )

    vmax = 3.0
    image = ax0.imshow(covariance, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(MODELS)):
        for j in range(len(TASKS)):
            value = covariance[i, j]
            ax0.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 1.55 else "#111827",
                fontsize=6.5,
                fontweight="bold" if value >= 0.2 else "normal",
            )
    ax0.set_xticks(
        range(len(TASKS)), ["Civil\nComments", "MultiNLI", "IMDb", "AG News"]
    )
    ax0.set_yticks(range(len(MODELS)), MODELS)
    ax0.tick_params(length=0)
    ax0.set_title("Excess covariance ($b=16$; $\\times$100)", loc="left", pad=5)
    colorbar = fig.colorbar(image, ax=ax0, fraction=0.045, pad=0.025)
    colorbar.set_ticks([0, 1, 2, 3])
    colorbar.ax.tick_params(labelsize=6, length=2)
    label_a = ax0.text(
        -0.10, 1.06, "a", transform=ax0.transAxes, fontweight="bold", fontsize=8
    )

    ax1 = fig.add_subplot(grid[0, 1])
    cells = [
        ("Granite", "IMDb"),
        ("Qwen", "Civil Comments"),
        ("Mistral", "MultiNLI"),
        ("Llama", "Civil Comments"),
    ]
    labels = ["Granite\nIMDb", "Qwen\nCivil", "Mistral\nMultiNLI", "Llama\nCivil"]
    x = np.arange(len(cells))
    width = 0.23
    ax1.axhspan(94, 96, color="#E8F1F8", zorder=0)
    ax1.axhline(95, color="#31546D", linestyle=(0, (2, 2)), linewidth=0.8, zorder=1)
    for method_index, method in enumerate(METHODS):
        values = []
        lows = []
        highs = []
        for model, task in cells:
            match = next(
                row
                for row in rows
                if row["panel"] == "coverage"
                and row["model"] == model
                and row["task"] == task
                and row["method"] == method
            )
            values.append(100 * float(match["value"]))
            lows.append(100 * float(match["low"]))
            highs.append(100 * float(match["high"]))
        bars = ax1.bar(
            x + (method_index - 1) * width,
            values,
            width=width,
            color=COLORS[method],
            label=method,
            edgecolor="white",
            linewidth=0.4,
            yerr=np.vstack((np.asarray(values) - lows, np.asarray(highs) - values)),
            error_kw={"ecolor": "#111827", "elinewidth": 0.55, "capsize": 1.4},
            zorder=2,
        )
        for bar, value in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                value * 0.68,
                f"{value:.1f}",
                ha="center",
                va="center",
                rotation=90,
                fontsize=5.4,
                color="white",
                fontweight="bold",
                path_effects=[
                    path_effects.Stroke(linewidth=0.75, foreground="#111827"),
                    path_effects.Normal(),
                ],
                zorder=3,
            )
    ax1.set_ylim(0, 105)
    ax1.set_yticks([0, 25, 50, 75, 95])
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", color="#E5E7EB", linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_ylabel("Empirical coverage (%)")
    ax1.set_title("95% interval coverage ($b=16$)", loc="left", pad=5)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        ncol=3,
        fontsize=5.2,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.3,
    )
    label_b = ax1.text(
        -0.12, 1.06, "b", transform=ax1.transAxes, fontweight="bold", fontsize=8
    )

    ax2 = fig.add_subplot(grid[0, 2])
    batch_sizes = np.array([2, 4, 8, 16, 32])
    x_batch = np.arange(1, 6)
    for model in MODELS:
        values = [
            float(
                next(
                    row["value"]
                    for row in rows
                    if row["panel"] == "batch_size"
                    and row["model"] == model
                    and row["method"] == "Standardized contrast"
                    and int(row["n"]) == size
                )
            )
            for size in batch_sizes
        ]
        ax2.plot(
            x_batch,
            values,
            color=BATCH_COLORS[model],
            marker="o",
            markersize=2.8,
            linewidth=1.0,
            label=model,
            zorder=2,
        )

    def pooled(method: str) -> np.ndarray:
        return np.array(
            [
                float(
                    next(
                        row["value"]
                        for row in rows
                        if row["panel"] == "batch_size"
                        and row["model"] == "Pooled"
                        and row["method"] == method
                        and int(row["n"]) == size
                    )
                )
                for size in batch_sizes
            ]
        )

    pooled_values = pooled("Pooled")
    ax2.fill_between(
        x_batch,
        pooled("Pooled low"),
        pooled("Pooled high"),
        color="#111827",
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )
    ax2.plot(
        x_batch,
        pooled_values,
        color="#111827",
        marker="o",
        markersize=3.2,
        linewidth=1.5,
        label="Pooled",
        zorder=3,
    )
    ax2.axhline(0, color="#6B7280", linestyle=(0, (2, 2)), linewidth=0.7)
    ax2.set_xticks(x_batch, [str(size) for size in batch_sizes])
    ax2.set_ylim(-0.025, 0.19)
    ax2.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax2.grid(axis="y", color="#E5E7EB", linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Standardized excess dependence")
    ax2.set_title("Batch-size robustness (4 cells)", loc="left", pad=5)
    ax2.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=5.2,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    label_c = ax2.text(
        -0.15, 1.06, "c", transform=ax2.transAxes, fontweight="bold", fontsize=8
    )

    return (
        fig,
        {
            "covariance": [ax0, colorbar.ax],
            "coverage": [ax1],
            "batch_size": [ax2],
        },
        {"a": label_a, "b": label_b, "c": label_c},
    )


def save_outputs(fig: plt.Figure, output: Path, bbox_inches="tight") -> None:
    fig.savefig(output.with_suffix(".svg"), bbox_inches=bbox_inches)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches=bbox_inches)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches=bbox_inches)
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches=bbox_inches)


def panel_bbox(fig: plt.Figure, axes: list) -> Bbox:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = Bbox.union([axis.get_tightbbox(renderer) for axis in axes])
    return bounds.transformed(fig.dpi_scale_trans.inverted()).padded(0.04)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, panels, labels = build_figure(read_rows())
    save_outputs(fig, OUT)

    for label in labels.values():
        label.set_visible(False)
    save_outputs(
        fig, OUT.parent / "covariance_results", panel_bbox(fig, panels["covariance"])
    )
    batch_artists = panels["batch_size"][0].get_children()
    batch_visibility = [artist.get_visible() for artist in batch_artists]
    for artist in batch_artists:
        artist.set_visible(False)
    save_outputs(
        fig, OUT.parent / "coverage_results", panel_bbox(fig, panels["coverage"])
    )
    for artist, visible in zip(batch_artists, batch_visibility):
        artist.set_visible(visible)
    save_outputs(
        fig, OUT.parent / "batch_size_results", panel_bbox(fig, panels["batch_size"])
    )

    labels["b"].set_text("a")
    labels["b"].set_visible(True)
    labels["c"].set_text("b")
    labels["c"].set_visible(True)
    save_outputs(
        fig,
        OUT.parent / "coverage_batch_results",
        panel_bbox(fig, panels["coverage"] + panels["batch_size"]),
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
