"""
Figure 12 — Three exponential-decay curves a(t-t') = exp(-λ·(t-t')) with
horizontal markers at the half-life t_{1/2} = ln 2 / λ for each sport.
Sport-specific λ matches Table 8 of the paper.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from visualization.style import apply_ieee_style, COLORS
from config import SPORTS

OUT_DIR = "results/figures"


DECAY_PARAMS = {
    "Swimming":         {"lambda": 0.041, "half_life": 16.9,
                         "half_life_ci": 2.5, "color": COLORS["blue"]},
    "Distance Running": {"lambda": 0.033, "half_life": 21.0,
                         "half_life_ci": 3.2, "color": COLORS["red"]},
    "Rowing":           {"lambda": 0.037, "half_life": 18.7,
                         "half_life_ci": 3.5, "color": COLORS["green"]},
}


def render(out_path: str = None):
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    t = np.linspace(0, 90, 600)

    for sport in SPORTS:
        cfg = DECAY_PARAMS[sport]
        lam = cfg["lambda"]
        a = np.exp(-lam * t)
        ax.plot(t, a, color=cfg["color"], linewidth=2.4,
                label=f"{sport}  (λ = {lam:.3f}, $t_{{1/2}}$ = {cfg['half_life']:.1f} ± {cfg['half_life_ci']:.1f} d)")
        # Half-life confidence band
        lam_lo = math.log(2) / (cfg["half_life"] + cfg["half_life_ci"])
        lam_hi = math.log(2) / max(0.1, cfg["half_life"] - cfg["half_life_ci"])
        ax.fill_between(t, np.exp(-lam_hi * t), np.exp(-lam_lo * t),
                        color=cfg["color"], alpha=0.13)
        # Vertical line at half-life
        ax.axvline(cfg["half_life"], color=cfg["color"], linestyle="--",
                   linewidth=1.4, alpha=0.85)
        ax.scatter([cfg["half_life"]], [0.5],
                   color=cfg["color"], s=70, zorder=5, edgecolor="white",
                   linewidth=1.6)

    # Horizontal 0.5 line
    ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(88, 0.51, "$a(t)=0.5$", fontsize=9, color="0.4",
            ha="right", va="bottom")

    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Days before query  $t - t'$", fontsize=11)
    ax.set_ylabel("Normalised attention weight  $\\exp(-\\lambda \\cdot (t - t'))$",
                  fontsize=11)
    ax.set_title("Learned Exponential Decay of Attention Weights",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5)

    # Inset table of half-lives
    txt_lines = ["Sport-specific physiological recovery half-life",
                 "—" * 26]
    for sport in SPORTS:
        cfg = DECAY_PARAMS[sport]
        txt_lines.append(
            f"  {sport:<19s}{cfg['half_life']:.1f} ± {cfg['half_life_ci']:.1f} d")
    ax.text(0.02, 0.02, "\n".join(txt_lines), transform=ax.transAxes,
            fontsize=8.5, va="bottom",
            family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.75", alpha=0.95))

    if out_path is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "Figure_12_decay_curves.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(render())
