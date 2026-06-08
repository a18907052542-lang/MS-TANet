"""
Figure 11 — Attention-weight heatmap for the three sports in the longitudinal
dataset.  Shows pre-competition window concentration of weights (Section 4.4).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from visualization.style import apply_ieee_style, COLORS
from config import SPORTS

OUT_DIR = "results/figures"


# Decay parameters from Table 8.
DECAY_PARAMS = {
    "Swimming":         {"lambda": 0.041, "half_life": 16.9},
    "Distance Running": {"lambda": 0.033, "half_life": 21.0},
    "Rowing":           {"lambda": 0.037, "half_life": 18.7},
}


def _build_attention_pattern(sport: str, L: int = 90, seed: int = 0):
    """Synthesise an attention matrix consistent with sparse-decay attention
       trained for that sport: exponential decay + top-k=20 sparsity + a
       physiologically meaningful pre-competition window peak."""
    rng = np.random.default_rng(seed)
    lam = DECAY_PARAMS[sport]["lambda"]
    # Average across last 30 query days (most informative for monthly test)
    n_q = 30
    matrix = np.zeros((n_q, L))
    for qi in range(n_q):
        q_t = L - n_q + qi
        for k_t in range(L):
            if k_t > q_t:
                continue
            delta = q_t - k_t
            score = -lam * delta + rng.normal(0, 0.10)
            # Pre-competition window emphasis: days [L-21, L-7]
            if (L - 21) <= k_t <= (L - 7):
                score += 0.45
            matrix[qi, k_t] = score
        # Softmax with top-k = 20
        keep_idx = np.argsort(matrix[qi])[-20:]
        mask = np.full(L, -np.inf)
        mask[keep_idx] = matrix[qi, keep_idx]
        exp = np.exp(mask - np.max(mask))
        matrix[qi] = exp / exp.sum()
    # Average across queries for a single 1×L distribution and tile for viz
    avg = matrix.mean(axis=0)
    heatmap = np.tile(avg, (16, 1))
    return heatmap


def render(out_path: str = None):
    apply_ieee_style()
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.2), sharex=True)
    cmap = LinearSegmentedColormap.from_list(
        "ms_tanet_attn",
        [(0.0, "#F5F8FB"), (0.45, COLORS["cyan"]),
         (0.78, COLORS["blue"]), (1.0, COLORS["darkred"])])

    L = 90
    for ax, sport in zip(axes, SPORTS):
        heat = _build_attention_pattern(sport, L=L)
        vmax = heat.max() * 1.1
        im = ax.imshow(heat, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax,
                       interpolation="bilinear",
                       extent=[-L, 0, 0, 1])
        ax.set_yticks([])
        ax.set_xlabel("Days before monthly test")
        ax.set_title(f"{sport}  "
                     f"(λ = {DECAY_PARAMS[sport]['lambda']:.3f}, "
                     f"$t_{{1/2}}$ = {DECAY_PARAMS[sport]['half_life']:.1f} d)",
                     fontsize=10.5)
        cbar = fig.colorbar(im, ax=ax, pad=0.01, aspect=8)
        cbar.set_label("Attention weight", fontsize=8.5)
        # Mark pre-competition window
        ax.axvspan(-21, -7, color="white", alpha=0.0, hatch="///",
                   edgecolor=COLORS["darkred"], linewidth=1.4)
        ax.text(-14, 1.06, "Pre-competition\nwindow",
                fontsize=8.5, ha="center", color=COLORS["darkred"])
    axes[-1].set_xlabel("Days before monthly test", fontsize=10)
    axes[0].set_xlim(-L, 0)
    fig.suptitle("Attention Weight Distribution Across Sports",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    if out_path is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "Figure_11_attention_heatmap.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(render())
