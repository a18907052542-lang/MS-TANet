"""
IEEE-style matplotlib configuration shared across all figures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from config import IEEE_COLORS


def apply_ieee_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.prop_cycle": cycler(color=[
            IEEE_COLORS["blue"], IEEE_COLORS["red"],
            IEEE_COLORS["green"], IEEE_COLORS["purple"],
            IEEE_COLORS["yellow"], IEEE_COLORS["cyan"],
            IEEE_COLORS["orange"], IEEE_COLORS["darkred"],
        ]),
    })


COLORS = IEEE_COLORS
