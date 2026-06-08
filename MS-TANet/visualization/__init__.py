"""Publication-quality figures for the MS-TANet experiments."""

from visualization import figure_8_radar
from visualization import figure_9_scatter
from visualization import figure_10_shapley
from visualization import figure_11_heatmap
from visualization import figure_12_decay
from visualization import figure_13_gating
from visualization import figure_14_convergence


FIGURE_RENDERERS = {
    "Figure_08_radar":              figure_8_radar.render,
    "Figure_09_scatter":            figure_9_scatter.render,
    "Figure_10_shapley":            figure_10_shapley.render,
    "Figure_11_attention_heatmap":  figure_11_heatmap.render,
    "Figure_12_decay_curves":       figure_12_decay.render,
    "Figure_13_gating_dynamics":    figure_13_gating.render,
    "Figure_14_convergence":        figure_14_convergence.render,
}


def render_all():
    paths = []
    for name, fn in FIGURE_RENDERERS.items():
        paths.append(fn())
    return paths
