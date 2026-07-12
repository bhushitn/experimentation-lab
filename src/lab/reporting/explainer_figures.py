"""Static fallback figures for the non-pathology explainer datasets."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lab.reporting.style import (
    CORRECTED,
    INK_SECONDARY,
    NAIVE,
    SECONDARY,
    TERTIARY,
    TRUTH,
    apply_style,
)


def _bar_labels(ax: Axes, bars: Any, values: list[float], fmt: str) -> None:
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            fmt.format(v),
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            color=INK_SECONDARY,
        )


def fig_randomization(data: dict[str, Any]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    values = [data["observational"], data["randomized"]]
    bars = ax.bar(["compare adopters vs non-adopters", "randomize"], values,
                  color=[NAIVE, CORRECTED], width=0.45)
    _bar_labels(ax, bars, values, "{:.2f}")
    ax.axhline(data["true_effect"], color=TRUTH, linestyle="--", linewidth=1.5)
    ax.annotate(f"true effect = {data['true_effect']:.2f}", xy=(0.0, data["true_effect"]),
                xycoords=("axes fraction", "data"), xytext=(2, 4),
                textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("estimated effect")
    ax.set_title("Self-selection is not an experiment")
    fig.tight_layout()
    return fig


def fig_test_choice(data: dict[str, Any]) -> Figure:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, key, color, title in [
        (axes[0], "continuous", CORRECTED, "engagement-shaped: Welch t"),
        (axes[1], "zero_inflated", TERTIARY, "revenue-shaped: t + Mann-Whitney"),
    ]:
        h = data["distributions"][key]
        edges = np.array(h["edges"])
        ax.bar(edges[:-1], h["density"], width=np.diff(edges), align="edge",
               color=color, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
    fig.suptitle("The metric's shape picks the test", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_cuped(data: dict[str, Any]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    s = data["scatter"]
    colors = [CORRECTED if t else SECONDARY for t in s["treated"]]
    ax.scatter(s["pre"], s["post"], s=8, c=colors, alpha=0.45, linewidths=0)
    ax.set_xlabel("pre-period metric (28-day average)")
    ax.set_ylabel("experiment metric")
    ax.set_title(
        f"CUPED: {data['variance_reduction']:.0%} variance reduction from the pre-period"
    )
    fig.tight_layout()
    return fig


def fig_power(data: dict[str, Any]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"0.005": NAIVE, "0.010": CORRECTED, "0.020": SECONDARY}
    for lift, curve in data["curves"].items():
        ax.plot(data["n_per_arm"], curve, color=colors[lift],
                label=f"+{float(lift) * 100:.1f} point lift")
    ax.axhline(0.8, color=TRUTH, linestyle="--", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("users per arm (log scale)")
    ax.set_ylabel("power")
    ax.set_title("Power is arithmetic, not optimism (10% baseline rate)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_srm(data: dict[str, Any]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for frac, curve, color in [
        ("100k", data["power_100k"], CORRECTED),
        ("1M", data["power_1m"], SECONDARY),
    ]:
        ax.plot([f * 100 for f in data["drop_fractions"]], curve, color=color,
                label=f"{frac} users per arm")
    ax.axhline(0.999, color=TRUTH, linestyle="--", linewidth=1.2)
    ax.set_xlabel("share of one arm silently dropped (%)")
    ax.set_ylabel("probability the SRM alarm fires")
    ax.set_title("The SRM check catches losses that outgrow sampling noise")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_switchback(data: dict[str, Any]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = [r["window_hours"] for r in data["by_window"]]
    ax.plot(w, [r["naive_estimate"] for r in data["by_window"]], color=NAIVE,
            marker="o", label="naive (all hours)")
    ax.plot(w, [r["burn_in_estimate"] for r in data["by_window"]], color=CORRECTED,
            marker="o", label="burn-in (drop first hour after switch)")
    ax.axhline(data["true_effect"], color=TRUTH, linestyle="--", linewidth=1.5)
    ax.set_xlabel("switch window length (hours)")
    ax.set_ylabel("estimated effect")
    ax.set_title("Switchbacks: carryover attenuates short windows")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_quasi(data: dict[str, Any]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    v = data["violated"]
    periods = v["periods"]
    ax.plot(periods, v["treated_mean"], color=CORRECTED, marker="o",
            markersize=3, label="cities that launched")
    ax.plot(periods, v["control_mean"], color=SECONDARY, marker="o",
            markersize=3, label="cities that did not")
    ax.axvline(data["launch_period"] - 0.5, color=TRUTH, linestyle="--", linewidth=1.2)
    ax.set_xlabel("period (launch at the dashed line)")
    ax.set_ylabel("metric, city-group mean")
    ax.set_title("No randomization: the groups differ before the launch")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
