"""One figure per pathology: naive error and corrected recovery on one canvas.

Every function takes the result object of a pathology simulation, never raw
style choices; the numbers on the canvas are the simulation's, unedited.
Figures return (fig, data) where data is the JSON-serializable content of
the figure, exported for the D3 versions on the site so the animated and
static views can never drift apart.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lab.pathologies import (
    ContaminationResult,
    InterferenceResult,
    MultipleComparisonsResult,
    NoveltyResult,
    PeekingResult,
    SubgroupResult,
)
from lab.reporting.style import (
    CORRECTED,
    INK_SECONDARY,
    NAIVE,
    SECONDARY,
    SURFACE,
    TRUTH,
    apply_style,
)

FigureAndData = tuple[Figure, dict[str, Any]]


def _truth_line(ax: Axes, y: float, label: str, *, side: str = "right") -> None:
    ax.axhline(y, color=TRUTH, linestyle="--", linewidth=1.5, zorder=1)
    x = 1.0 if side == "right" else 0.0
    ax.annotate(
        label,
        xy=(x, y),
        xycoords=("axes fraction", "data"),
        xytext=(2 if side == "left" else -2, 4),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=9,
        ha=side,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
    )


def fig_peeking(result: PeekingResult, *, alpha: float = 0.05) -> FigureAndData:
    """Cumulative false positive rate by day: naive peeking vs sequential."""
    apply_style()
    d = result.decisions
    n_days = int(result.boundaries.size)
    days = np.arange(1, n_days + 1)
    naive_cum = [float((d["naive_stop_day"] <= day).mean()) for day in days]
    seq_cum = [float((d["sequential_stop_day"] <= day).mean()) for day in days]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(days, naive_cum, color=NAIVE, label="peek daily, stop at p < 0.05")
    ax.plot(days, seq_cum, color=CORRECTED, label="peek daily, O'Brien-Fleming boundary")
    _truth_line(ax, alpha, f"nominal alpha = {alpha:.2f}", side="left")
    ax.annotate(
        f"{naive_cum[-1]:.0%}",
        xy=(days[-1], naive_cum[-1]),
        xytext=(-4, 6),
        textcoords="offset points",
        color=NAIVE,
        fontweight="bold",
        ha="right",
    )
    ax.annotate(
        f"{seq_cum[-1]:.1%}",
        xy=(days[-1], seq_cum[-1]),
        xytext=(-4, 6),
        textcoords="offset points",
        color=CORRECTED,
        fontweight="bold",
        ha="right",
    )
    ax.set_xlabel("day of experiment (a look per day, no true effect)")
    ax.set_ylabel("experiments falsely called significant")
    ax.set_title("Checking daily manufactures false positives")
    ax.set_xlim(1, n_days)
    ax.set_ylim(0, max(naive_cum) * 1.25)
    ax.legend(loc="upper left")
    fig.tight_layout()

    data = {
        "days": days.tolist(),
        "naive_cumulative_fpr": naive_cum,
        "sequential_cumulative_fpr": seq_cum,
        "boundaries": result.boundaries.tolist(),
        "z_crit": result.z_crit,
        "alpha": alpha,
        "n_sims": int(len(d)),
    }
    return fig, data


def fig_multiple_comparisons(result: MultipleComparisonsResult) -> FigureAndData:
    """Family-wise error rate per policy against the nominal alpha."""
    apply_style()
    fwer = result.summary["fwer"]
    labels = ["test all 20 at 0.05", "Bonferroni", "Benjamini-Hochberg"]
    colors = [NAIVE, SECONDARY, CORRECTED]
    values = [fwer["naive"], fwer["bonferroni"], fwer["benjamini_hochberg"]]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            color=INK_SECONDARY,
        )
    ax.axhline(0.05, color=TRUTH, linestyle="--", linewidth=1.5, label="nominal alpha = 0.05")
    ax.legend(loc="upper right")
    ax.set_ylabel("experiments with any false significant metric")
    ax.set_title("Twenty metrics, one experiment: the family-wise error")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()

    data = {
        "policies": ["naive", "bonferroni", "benjamini_hochberg"],
        "fwer": values,
        "summary": result.summary.to_dict(),
        "n_metrics": int(result.is_null.size),
    }
    return fig, data


def fig_contamination(result: ContaminationResult) -> FigureAndData:
    """The three estimators against the true effect."""
    apply_style()
    order = ["intent_to_treat", "per_protocol", "iv_wald"]
    labels = ["intent-to-treat", "per-protocol", "IV / Wald"]
    colors = [SECONDARY, NAIVE, CORRECTED]
    values = [float(result.estimates[k]) for k in order]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            color=INK_SECONDARY,
        )
    _truth_line(ax, result.true_effect, f"true effect = {result.true_effect:.2f}", side="left")
    ax.set_ylabel("estimated treatment effect")
    ax.set_title("Contamination: attenuated, biased, and recovered")
    ax.set_ylim(0, max(values + [result.true_effect]) * 1.25)
    fig.tight_layout()

    data = {
        "estimators": order,
        "estimates": values,
        "expected": [float(result.expected[k]) for k in order],
        "true_effect": result.true_effect,
    }
    return fig, data


def fig_interference(result: InterferenceResult) -> FigureAndData:
    """Both randomization designs against the global treatment effect."""
    apply_style()
    order = ["user_randomized", "cluster_randomized"]
    labels = ["randomize users", "randomize clusters"]
    colors = [NAIVE, CORRECTED]
    values = [float(result.estimates[k]) for k in order]
    gte = result.global_treatment_effect

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.45)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            color=INK_SECONDARY,
        )
    _truth_line(ax, gte, f"global treatment effect = {gte:.2f}", side="left")
    ax.set_ylabel("estimated treatment effect")
    ax.set_title("Interference: the design decides what the estimator can see")
    ax.set_ylim(0, gte * 1.3)
    fig.tight_layout()

    data = {
        "designs": order,
        "estimates": values,
        "expected": [float(result.expected[k]) for k in order],
        "exposure_contrast": {k: float(result.exposure_contrast[k]) for k in order},
        "global_treatment_effect": gte,
    }
    return fig, data


def fig_subgroups(result: SubgroupResult) -> FigureAndData:
    """Probability of finding at least one significant null subgroup."""
    apply_style()
    s = result.summary
    labels = ["scan all 20 segments", "pre-registered primary", "Benjamini-Hochberg scan"]
    colors = [NAIVE, CORRECTED, SECONDARY]
    values = [
        float(s["fishing_any_discovery"]),
        float(s["preregistered_primary_fpr"]),
        float(s["bh_corrected_any_discovery"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            color=INK_SECONDARY,
        )
    ax.axhline(0.05, color=TRUTH, linestyle="--", linewidth=1.5, label="nominal alpha = 0.05")
    ax.legend(loc="upper right")
    ax.set_ylabel("experiments with a false subgroup discovery")
    ax.set_title("The subgroup fishing expedition, quantified")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()

    data = {"policies": labels, "rates": values, "summary": s.to_dict()}
    return fig, data


def fig_novelty(result: NoveltyResult, *, path_days: np.ndarray, path: np.ndarray) -> FigureAndData:
    """The decaying effect path and what each readout window concludes."""
    apply_style()
    windows = {
        "first_week": (0, 6, NAIVE, "week-1 readout"),
        "full_run": (0, 27, SECONDARY, "4-week average"),
        "post_burn_in": (21, 27, CORRECTED, "last-week readout"),
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(path_days, path, color=INK_SECONDARY, linewidth=1.5, label="true effect path")
    _truth_line(ax, result.long_run_effect, f"long-run effect = {result.long_run_effect:.2f}")
    for name, (d0, d1, color, label) in windows.items():
        est = float(result.estimates[name])
        ax.hlines(est, d0, d1, color=color, linewidth=3)
        ax.annotate(
            f"{label}: {est:.2f}",
            xy=(d1, est),
            xytext=(4, -3),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("days since launch")
    ax.set_ylabel("treatment effect")
    ax.set_title("Novelty: the first week is not the launch effect")
    ax.set_xlim(0, float(path_days.max()) * 1.28)
    fig.tight_layout()

    data = {
        "days": path_days.tolist(),
        "effect_path": path.tolist(),
        "estimates": {k: float(v) for k, v in result.estimates.items()},
        "expected": {k: float(v) for k, v in result.expected.items()},
        "long_run_effect": result.long_run_effect,
    }
    return fig, data
