from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "study_overview"
ROBOT_ASSET = ROOT / "figures" / "assets" / "annotation_robot_flat.png"

INK = "#20262B"
GRAY = "#69757E"
LIGHT = "#D8DEE3"
BLUE = "#3F78A8"
BLUE_PALE = "#E8F1F7"
GREEN_PALE = "#E9F3D9"
CORAL = "#C95E52"
CORAL_PALE = "#F6DDD8"
GOLD = "#D69A2D"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "mathtext.fontset": "stixsans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def arrow(ax, start, end, color=INK, lw=0.7, style="-|>", dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=7.5,
            linewidth=lw,
            linestyle=(0, (2, 2)) if dashed else "solid",
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def draw_context(ax, x, y, w=0.14, h=0.12, target=False):
    for index, offset in enumerate((0.025, 0.012, 0.0)):
        ax.add_patch(
            Polygon(
                [
                    (x + offset, y + offset),
                    (x + w + offset, y + offset),
                    (x + w + offset + 0.018, y + h + offset),
                    (x + 0.018 + offset, y + h + offset),
                ],
                closed=True,
                facecolor="white",
                edgecolor=GRAY if offset else INK,
                linewidth=0.55 if offset else 0.75,
                zorder=index,
            )
        )
    # Keep the 4 x 4 text blocks portrait-shaped so the input reads as a
    # vertically substantial document stack rather than a stretched tray.
    cell_w = 0.018
    cell_h = 0.026
    col_gap = 0.010
    row_gap = 0.006
    grid_w = 4 * cell_w + 3 * col_gap
    grid_h = 4 * cell_h + 3 * row_gap
    y0 = y + (h - grid_h) / 2
    for row in range(4):
        for col in range(4):
            selected = target and row == 2 and col == 1
            cell_y = y0 + (cell_h + row_gap) * row
            page_shear = 0.018 * (cell_y - y) / h
            cell_x = x + (w - grid_w) / 2 + page_shear + (cell_w + col_gap) * col
            cell_shear = 0.018 * cell_h / h
            ax.add_patch(
                Polygon(
                    [
                        (cell_x, cell_y),
                        (cell_x + cell_w, cell_y),
                        (cell_x + cell_w + cell_shear, cell_y + cell_h),
                        (cell_x + cell_shear, cell_y + cell_h),
                    ],
                    closed=True,
                    facecolor=GOLD if selected else BLUE_PALE,
                    edgecolor=GOLD if selected else BLUE,
                    linewidth=0.35,
                    zorder=4,
                )
            )


def draw_model(ax, x, y, w=0.12, h=0.085, small=False):
    """Place the generated annotation-robot cutout at a stable physical size."""
    robot = plt.imread(ROBOT_ASSET)
    scale = min(w / 0.14, h / 0.12)
    image = OffsetImage(robot, zoom=0.023 * scale, interpolation="lanczos")
    ax.add_artist(
        AnnotationBbox(
            image,
            (x + w / 2, y + h / 2),
            frameon=False,
            pad=0,
            zorder=4,
        )
    )


def rounded_card(ax, x, y, w, h, face, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor=face, edgecolor=edge, linewidth=0.55, zorder=0,
        )
    )


def draw_call_box(ax, x, y, stacked=False):
    offsets = ((0.014, 0.014), (0.007, 0.007), (0.0, 0.0)) if stacked else ((0.0, 0.0),)
    for dx, dy in offsets:
        ax.add_patch(
            FancyBboxPatch(
                (x + dx, y + dy), 0.125, 0.17,
                boxstyle="round,pad=0.003,rounding_size=0.012",
                facecolor="white", edgecolor=LIGHT if dx else BLUE,
                linewidth=0.5 if dx else 0.65, zorder=1,
            )
        )
    draw_model(ax, x + 0.012, y + 0.020, w=0.101, h=0.130)


def draw_output_group(ax, x, y, joint=True):
    if joint:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), 0.14, 0.165,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor="white", edgecolor=BLUE, linewidth=0.7,
            )
        )
        colors = (BLUE, CORAL, CORAL, BLUE)
        for row in range(4):
            for col in range(4):
                ax.scatter(
                    [x + 0.022 + 0.032 * col], [y + 0.027 + 0.037 * row],
                    s=8, c=[colors[(row + col) % 4]], edgecolors="none", zorder=3,
                )
    else:
        for idx in range(4):
            yy = y + 0.120 - idx * 0.040
            ax.add_patch(
                FancyBboxPatch(
                    (x, yy), 0.14, 0.032,
                    boxstyle="round,pad=0.002,rounding_size=0.007",
                    facecolor="white", edgecolor=GRAY, linewidth=0.6,
                )
            )
            ax.scatter(
                [x + 0.070], [yy + 0.016], s=8,
                c=[(BLUE, CORAL, CORAL, BLUE)[idx]], edgecolors="none", zorder=3,
            )


def draw_joint_outputs(ax, x, y):
    left, right = x, x + 0.12
    bottom, top = y + 0.006, y + 0.132
    ax.plot([left, left], [bottom, top], color=INK, lw=0.75)
    ax.plot([left, left + 0.014], [top, top], color=INK, lw=0.75)
    ax.plot([left, left + 0.014], [bottom, bottom], color=INK, lw=0.75)
    ax.plot([right, right], [bottom, top], color=INK, lw=0.75)
    ax.plot([right - 0.014, right], [top, top], color=INK, lw=0.75)
    ax.plot([right - 0.014, right], [bottom, bottom], color=INK, lw=0.75)
    labels = [r"$\hat y_1$", r"$\hat y_2$", r"$\hat y_3$", r"$\hat y_{16}$"]
    colors = [BLUE, CORAL, CORAL, BLUE]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        yy = y + 0.111 - 0.030 * idx
        ax.add_patch(Circle((x + 0.027, yy), 0.007, facecolor=color, edgecolor="none"))
        ax.text(x + 0.047, yy, label, ha="left", va="center", fontsize=5.3, color=INK)
    ax.plot([x + 0.015, x + 0.015], [y + 0.067, y + 0.095], color=CORAL, lw=1.1)
    ax.text(x + 0.06, y - 0.014, "shared call $c$", ha="center", va="top",
            fontsize=5.2, color=BLUE, fontstyle="italic")


def draw_separate_outputs(ax, x, y):
    for idx, label in enumerate((r"$c_1$", r"$c_2$", r"$c_{16}$")):
        yy = y + 0.086 - idx * 0.041
        ax.add_patch(Rectangle((x, yy), 0.065, 0.029, facecolor="white",
                               edgecolor=GRAY, linewidth=0.55))
        ax.add_patch(Circle((x + 0.082, yy + 0.0145), 0.0065,
                            facecolor=BLUE, edgecolor="none"))
        ax.text(x + 0.0325, yy + 0.0145, label, ha="center", va="center",
                fontsize=5.0, color=INK)
    ax.text(x + 0.04, y - 0.015, "separate calls", ha="center", va="top",
            fontsize=5.2, color=GRAY, fontstyle="italic")


def draw_covariance(ax, x, y, joint=True):
    size = 0.038
    for row in range(4):
        for col in range(4):
            active = joint and (row, col) in {(0, 1), (1, 0), (1, 2), (2, 1)}
            diagonal = row == col
            face = CORAL_PALE if active else (BLUE_PALE if diagonal else "white")
            edge = CORAL if active else (BLUE if diagonal else LIGHT)
            ax.add_patch(
                Rectangle(
                    (x + col * size, y + (3 - row) * size),
                    size,
                    size,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.45,
                )
            )
    name = "joint" if joint else "exact"
    ax.text(x + 2 * size, y - 0.017, rf"$\mathrm{{Cov}}_{{{name}}}$",
            ha="center", va="top", fontsize=5.3, color=INK)


def lane(ax, y, joint=True):
    label = "a  joint annotation" if joint else "b  exact-context target control"
    ax.text(0.03, y + 0.142, label, ha="left", va="bottom", fontsize=6.4,
            color=INK, fontweight="bold")

    draw_context(ax, 0.045, y, target=not joint)
    ax.text(0.12, y - 0.016, "same 16 texts\nand order", ha="center", va="top",
            fontsize=5.0, color=GRAY, linespacing=0.9, fontstyle="italic")
    arrow(ax, (0.205, y + 0.06), (0.245, y + 0.06))

    if joint:
        draw_model(ax, 0.255, y + 0.018)
        arrow(ax, (0.375, y + 0.06), (0.425, y + 0.06))
        draw_joint_outputs(ax, 0.435, y)
        arrow(ax, (0.57, y + 0.06), (0.665, y + 0.06))
        draw_covariance(ax, 0.68, y + 0.022, joint=True)
        ax.text(0.495, y + 0.151, "coupled errors", ha="center", va="center",
                fontsize=5.0, color=CORAL, fontstyle="italic")
    else:
        for idx in range(3):
            draw_model(ax, 0.255, y + 0.088 - idx * 0.043, w=0.09, h=0.030, small=True)
            arrow(ax, (0.35, y + 0.103 - idx * 0.043),
                  (0.395, y + 0.103 - idx * 0.043), color=GRAY, lw=0.6)
        draw_separate_outputs(ax, 0.405, y + 0.003)
        arrow(ax, (0.505, y + 0.06), (0.665, y + 0.06), color=GRAY)
        draw_covariance(ax, 0.68, y + 0.022, joint=False)
        ax.text(0.12, y + 0.103, "target", ha="center", va="center",
                fontsize=5.0, color=GOLD, fontweight="bold")


def build_figure():
    fig, ax = plt.subplots(figsize=(3.5, 1.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # One matched input branches into two aligned experimental lanes.
    draw_context(ax, 0.005, 0.365, w=0.16, h=0.16)
    ax.text(0.105, 0.325, "same $b$ texts\nand order", ha="center", va="top",
            fontsize=5.0, color=GRAY, linespacing=0.92)
    ax.plot([0.225, 0.225], [0.31, 0.70], color=LIGHT, lw=0.7)
    arrow(ax, (0.195, 0.475), (0.225, 0.475), color=GRAY)

    rounded_card(ax, 0.25, 0.55, 0.70, 0.34, "#F1F6FA", "#C8DAE7")
    rounded_card(ax, 0.25, 0.14, 0.70, 0.34, "#F5F6F7", "#D8DEE3")
    arrow(ax, (0.225, 0.70), (0.295, 0.70), color=BLUE)
    arrow(ax, (0.225, 0.31), (0.295, 0.31), color=GRAY)

    # Joint lane.
    ax.text(0.28, 0.845, "Joint annotation", ha="left", va="center",
            fontsize=6.8, color=INK, fontweight="bold")
    draw_call_box(ax, 0.34, 0.58, stacked=False)
    ax.text(0.91, 0.845, "1 call", ha="right", va="center",
            fontsize=5.2, color=BLUE, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                  "edgecolor": "#C8DAE7", "linewidth": 0.5})
    arrow(ax, (0.48, 0.67), (0.535, 0.67), color=BLUE)
    draw_output_group(ax, 0.55, 0.585, joint=True)
    arrow(ax, (0.70, 0.67), (0.755, 0.67), color=BLUE)
    draw_covariance(ax, 0.77, 0.622, joint=True)

    # Exact-context lane.
    ax.text(0.28, 0.435, "Exact-context control", ha="left", va="center",
            fontsize=6.8, color=INK, fontweight="bold")
    draw_call_box(ax, 0.34, 0.17, stacked=True)
    ax.text(0.91, 0.435, "$b$ calls", ha="right", va="center",
            fontsize=5.0, color=GRAY, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                  "edgecolor": "#D8DEE3", "linewidth": 0.5})
    arrow(ax, (0.48, 0.26), (0.535, 0.26), color=GRAY)
    draw_output_group(ax, 0.55, 0.184, joint=False)
    arrow(ax, (0.70, 0.26), (0.755, 0.26), color=GRAY)
    draw_covariance(ax, 0.77, 0.212, joint=False)

    # Compact estimator footer.
    ax.text(0.50, 0.055,
            r"Estimate: $\Delta\mathrm{Cov}=\mathrm{Cov}_{joint}-\mathrm{Cov}_{exact}$",
            ha="center", va="center", fontsize=6.5, color=INK)
    return fig


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT.with_suffix(".svg"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
