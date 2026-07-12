"""Shared figure style: one entity-color mapping across every figure.

The mapping is semantic and fixed so a reader who learns it once can read
the whole gallery: red is always the naive analysis, blue is always the
corrected one, gray dashed is always ground truth.
"""

import matplotlib as mpl

# Categorical slots from the validated reference palette (light surface).
NAIVE = "#e34948"  # the broken analysis
CORRECTED = "#2a78d6"  # the fixed analysis
SECONDARY = "#1baf7a"  # a second comparison method (e.g. Bonferroni)
TERTIARY = "#eda100"  # a third comparison method
TRUTH = "#898781"  # ground truth reference line, always dashed

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Set matplotlib rcParams for the gallery's shared look."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 120,
        }
    )
