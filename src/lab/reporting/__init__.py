"""Figure generation: every visual in the repo is produced here."""

from lab.reporting.figures import (
    fig_contamination,
    fig_interference,
    fig_multiple_comparisons,
    fig_novelty,
    fig_peeking,
    fig_subgroups,
)
from lab.reporting.style import apply_style

__all__ = [
    "apply_style",
    "fig_contamination",
    "fig_interference",
    "fig_multiple_comparisons",
    "fig_novelty",
    "fig_peeking",
    "fig_subgroups",
]
