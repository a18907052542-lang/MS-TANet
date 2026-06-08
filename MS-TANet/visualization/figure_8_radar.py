"""
Figure 8 — Multi-dimensional comparative performance radar chart of MS-TANet
vs. nine baselines across five evaluation dimensions: RMSE (inverted),
MAE (inverted), R², calibration (95 % PI coverage), and parameter efficiency.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from visualization.style import apply_ieee_style, COLORS

OUT_DIR = "results/figures"
TABLE_DIR = "results/tables"


def _load_radar_data():
    """Load per-model performance from Table 3/4 outputs (longitudinal dataset)."""
    # Numbers come from the paper's reported main experiment (Tables 3-4)
    return {
        "MS-TANet":        {"RMSE": 4.04, "MAE": 3.21, "R2": 0.853,
                            "Coverage95": 0.937, "Params_M": 1.72},
        "iTransformer":    {"RMSE": 4.41, "MAE": 3.49, "R2": 0.835,
                            "Coverage95": 0.892, "Params_M": 1.95},
        "PatchTST":        {"RMSE": 4.52, "MAE": 3.57, "R2": 0.827,
                            "Coverage95": 0.883, "Params_M": 2.18},
        "TimesNet":        {"RMSE": 4.66, "MAE": 3.69, "R2": 0.820,
                            "Coverage95": 0.875, "Params_M": 2.42},
        "Informer":        {"RMSE": 4.81, "MAE": 3.82, "R2": 0.806,
                            "Coverage95": 0.851, "Params_M": 2.31},
        "TFT":             {"RMSE": 4.74, "MAE": 3.77, "R2": 0.823,
                            "Coverage95": 0.864, "Params_M": 3.62},
        "BiLSTM":          {"RMSE": 5.20, "MAE": 4.15, "R2": 0.781,
                            "Coverage95": 0.812, "Params_M": 1.41},
        "Transformer":     {"RMSE": 5.41, "MAE": 4.30, "R2": 0.759,
                            "Coverage95": 0.798, "Params_M": 1.84},
        "LSTM":            {"RMSE": 5.47, "MAE": 4.36, "R2": 0.768,
                            "Coverage95": 0.793, "Params_M": 1.18},
        "Banister-IR":     {"RMSE": 7.21, "MAE": 5.83, "R2": 0.618,
                            "Coverage95": 0.683, "Params_M": 0.01},
    }


def _normalize(values, invert=False):
    arr = np.array(values, dtype=float)
    if arr.max() == arr.min():
        return np.ones_like(arr)
    norm = (arr - arr.min()) / (arr.max() - arr.min())
    if invert:
        norm = 1.0 - norm
    return norm


def render(out_path: str = None):
    apply_ieee_style()
    data = _load_radar_data()

    models = list(data.keys())
    rmse_n = _normalize([data[m]["RMSE"] for m in models], invert=True)
    mae_n = _normalize([data[m]["MAE"] for m in models], invert=True)
    r2_n = _normalize([data[m]["R2"] for m in models])
    cov_n = _normalize([data[m]["Coverage95"] for m in models])
    eff_n = _normalize([data[m]["RMSE"] * data[m]["Params_M"] for m in models],
                       invert=True)

    axes_labels = ["1 − RMSE", "1 − MAE", "R²",
                   "Coverage₉₅", "Param-Efficiency"]
    n_axes = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")

    palette = [COLORS["red"], COLORS["blue"], COLORS["green"], COLORS["purple"],
               COLORS["orange"], COLORS["cyan"], COLORS["yellow"],
               COLORS["darkred"], COLORS["navy"], COLORS["grey"]]

    for i, m in enumerate(models):
        vals = [rmse_n[i], mae_n[i], r2_n[i], cov_n[i], eff_n[i]]
        vals += vals[:1]
        is_mst = (m == "MS-TANet")
        ax.plot(angles, vals,
                color=palette[i],
                linewidth=2.4 if is_mst else 1.1,
                label=m,
                marker="o" if is_mst else None,
                markersize=4 if is_mst else 0)
        if is_mst:
            ax.fill(angles, vals, alpha=0.18, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Dimensional Performance Comparison (Longitudinal Dataset)",
                 pad=24, fontsize=12, fontweight="bold")

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.10),
                       frameon=True, fontsize=9, title="Model")
    legend.get_title().set_fontweight("bold")

    if out_path is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "Figure_08_radar.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(render())
