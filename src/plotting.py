"""
Claude Code: Centralized plot styling for project figures.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Color palette — one consistent set across all figures
COLORS = {
    "equal_weight": "#4C78A8",     # blue
    "min_var":      "#F58518",     # orange
    "min_var_lw":   "#54A24B",     # green
    "min_cvar":     "#E45756",     # red
    "neutral":      "#79706E",     # gray
    "highlight":    "#9D755D",     # brown
}


def setup_style():
    """Set matplotlib defaults for consistent figure styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def pct_formatter(decimals=0):
    """Return a percentage formatter for axes (0.15 -> '15%')."""
    return mticker.PercentFormatter(1.0, decimals=decimals)


def save_fig(fig, path):
    """Save a figure with consistent settings."""
    fig.savefig(path, dpi=150, bbox_inches="tight")