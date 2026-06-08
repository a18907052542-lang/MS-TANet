"""
Figure 10 — Shapley decomposition (Section 4.3) of the five core
innovations and their pairwise interaction term.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from visualization.style import apply_ieee_style, COLORS

OUT_DIR = "results/figures"


# Contribution percentages from Section 4.3 of the paper.
SHAPLEY_CONTRIBUTIONS = {
    "Sparse Decay Attention":   26.7,
    "Individual Embedding":     23.4,
    "Multi-Scale Convolution":  20.3,
    "Physics Constraint":       13.0,
    "Pairwise Interactions":    11.2,
    "Curriculum Learning":       5.4,
}


def render(out_path: str = None):
    apply_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    labels = list(SHAPLEY_CONTRIBUTIONS.keys())
    values = list(SHAPLEY_CONTRIBUTIONS.values())
    palette = [COLORS["red"], COLORS["blue"], COLORS["green"],
               COLORS["purple"], COLORS["grey"], COLORS["orange"]]

    # ---------- Donut chart ----------
    wedges, _ = ax1.pie(values, labels=None, colors=palette, startangle=90,
                        wedgeprops=dict(width=0.42, edgecolor="white",
                                        linewidth=1.4),
                        counterclock=False)
    ax1.text(0, 0.06, "Total $\\Delta R^2$", ha="center", va="center",
             fontsize=10, color="0.3")
    ax1.text(0, -0.06, "100 %", ha="center", va="center",
             fontsize=18, fontweight="bold")
    ax1.set_title("(a) Shapley Contribution Decomposition",
                  fontsize=11.5, fontweight="bold", pad=12)

    # ---------- Ranked horizontal bar ----------
    order = np.argsort(values)
    sorted_labels = [labels[i] for i in order]
    sorted_values = [values[i] for i in order]
    sorted_colors = [palette[i] for i in order]
    y_pos = np.arange(len(sorted_labels))
    bars = ax2.barh(y_pos, sorted_values, color=sorted_colors,
                    edgecolor="white", linewidth=0.6, height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_labels)
    ax2.set_xlabel("Contribution to total $\\Delta R^2$ (%)")
    ax2.set_xlim(0, max(values) + 6)
    ax2.set_title("(b) Ranked Component Contributions",
                  fontsize=11.5, fontweight="bold", pad=12)
    for b, v in zip(bars, sorted_values):
        ax2.text(v + 0.6, b.get_y() + b.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Shapley Decomposition of MS-TANet Core Innovations",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    if out_path is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "Figure_10_shapley.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(render())
