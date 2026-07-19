"""Render the peeking animation (GIF) and the social preview (PNG).

Both are built from the same committed simulation the README quotes:
simulate_daily_peeking(n_sims=4000, n_days=14, effect=0.0, seed=1), which
gives a 24.1% naive false positive rate against a 5.0% corrected rate. The
figures reuse the gallery's shared style (red naive, blue corrected, gray
dashed ground truth), so they cannot drift from the rest of the site.

Run:
    PYTHONPATH=src python scripts/render_peeking_media.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from lab.pathologies.peeking import simulate_daily_peeking
from lab.reporting.style import CORRECTED, INK, INK_SECONDARY, NAIVE, SURFACE, apply_style

FIGURES = Path("docs/assets/figures")
ASSETS = Path("docs/assets")


def cumulative_rates(seed: int = 1, n_days: int = 14):
    """Per-day cumulative false positive rate for each policy, from the sim."""
    result = simulate_daily_peeking(n_sims=4000, n_days=n_days, effect=0.0, seed=seed)
    d = result.decisions
    days = np.arange(1, n_days + 1)
    naive = np.array([float((d["naive_stop_day"] <= day).mean()) for day in days])
    seq = np.array([float((d["sequential_stop_day"] <= day).mean()) for day in days])
    return days, naive, seq, int(len(d))


def render_animation(days, naive, seq, n_sims, alpha=0.05):
    """GIF: the naive false positive rate climbing look by look."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(1, days[-1])
    ax.set_ylim(0, float(naive.max()) * 1.2)
    ax.axhline(alpha, color="#898781", linestyle="--", linewidth=1.5, zorder=1)
    ax.annotate(
        f"nominal alpha = {alpha:.2f}",
        xy=(0.0, alpha),
        xycoords=("axes fraction", "data"),
        xytext=(2, 4),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xlabel("day of experiment (a look per day, no true effect)")
    ax.set_ylabel("experiments falsely called significant")
    ax.set_title("Checking daily manufactures false positives")

    (naive_line,) = ax.plot([], [], color=NAIVE, label="peek daily, stop at p < 0.05")
    (seq_line,) = ax.plot(
        [], [], color=CORRECTED, label="peek daily, O'Brien-Fleming boundary"
    )
    readout = ax.text(
        0.98,
        0.06,
        "",
        transform=ax.transAxes,
        ha="right",
        color=NAIVE,
        fontweight="bold",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    def update(k):
        naive_line.set_data(days[:k], naive[:k])
        seq_line.set_data(days[:k], seq[:k])
        readout.set_text(f"day {days[k - 1]}: {naive[k - 1]:.0%} naive vs {seq[k - 1]:.0%} corrected")
        return naive_line, seq_line, readout

    frames = list(range(1, len(days) + 1)) + [len(days)] * 6
    anim = FuncAnimation(fig, update, frames=frames, interval=450, blit=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    anim.save(FIGURES / "peeking.gif", writer=PillowWriter(fps=2.2))
    plt.close(fig)


def render_social(days, naive, seq, n_sims, alpha=0.05):
    """1280x640 social card: the final curve, sized for a link preview."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    ax.plot(days, naive, color=NAIVE, linewidth=3.0, label="peek daily, stop at p < 0.05")
    ax.plot(days, seq, color=CORRECTED, linewidth=3.0, label="peek daily, O'Brien-Fleming boundary")
    ax.axhline(alpha, color="#898781", linestyle="--", linewidth=2.0, zorder=1)
    ax.annotate(
        f"nominal alpha = {alpha:.2f}",
        xy=(0.0, alpha),
        xycoords=("axes fraction", "data"),
        xytext=(4, 6),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=15,
    )
    for y, color in ((naive[-1], NAIVE), (seq[-1], CORRECTED)):
        ax.annotate(
            f"{y:.1%}",
            xy=(days[-1], y),
            xytext=(-6, 8),
            textcoords="offset points",
            color=color,
            fontweight="bold",
            fontsize=20,
            ha="right",
        )
    ax.set_xlabel("day of experiment (a look per day, no true effect)", fontsize=16)
    ax.set_ylabel("experiments falsely called significant", fontsize=16)
    ax.set_title(
        "Checking an A/B test daily manufactures false positives",
        fontsize=24,
        pad=16,
    )
    ax.set_xlim(1, days[-1])
    ax.set_ylim(0, float(naive.max()) * 1.22)
    ax.tick_params(labelsize=14)
    ax.legend(loc="upper left", fontsize=15)
    fig.text(
        0.01,
        0.02,
        f"experimentation-lab   ·   {n_sims:,} simulated experiments, no true effect, seed 1",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / "social-preview.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    days, naive, seq, n_sims = cumulative_rates()
    render_animation(days, naive, seq, n_sims)
    render_social(days, naive, seq, n_sims)
    print(f"naive final {naive[-1]:.1%}, corrected final {seq[-1]:.1%}")
    print("wrote docs/assets/figures/peeking.gif and docs/assets/social-preview.png")


if __name__ == "__main__":
    main()
