"""Novelty effects: the treatment effect decays, short experiments overclaim.

Effect model: effect(day) = long_run + novelty_amplitude * exp(-day / decay_days),
day counted from launch (day 0 is the first full day). A one-week readout
averages over the novelty spike; the long-run effect, which is what the
launch decision is actually about, can be several times smaller.

The corrected analysis is a post-burn-in window: estimate from days after
the novelty has decayed (here, the last week of a four-week run). Because
the effect path is known, the expected estimate for any window is the
analytic mean of effect(day) over that window, and the simulation is
verified against it exactly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lab.stats import welch_t_test


@dataclass(frozen=True)
class NoveltyResult:
    """Window estimates vs the long-run truth.

    Attributes
    ----------
    estimates : pd.Series
        Difference-in-means over each readout window.
    expected : pd.Series
        Analytic mean of effect(day) over the same windows.
    long_run_effect : float
        The persistent effect, the number the decision needs.
    """

    estimates: pd.Series
    expected: pd.Series
    long_run_effect: float


def effect_path(
    days: np.ndarray, *, long_run: float, novelty_amplitude: float, decay_days: float
) -> np.ndarray:
    """The true effect on each day."""
    return long_run + novelty_amplitude * np.exp(-days / decay_days)


def simulate_novelty(
    *,
    n_users_per_arm: int = 20_000,
    n_days: int = 28,
    long_run: float = 0.10,
    novelty_amplitude: float = 0.40,
    decay_days: float = 5.0,
    mean: float = 10.0,
    within_sd: float = 5.0,
    seed: int = 0,
) -> NoveltyResult:
    """One experiment observed daily; three readout windows compared.

    Windows: first_week (days 0-6, the impatient readout), full_run
    (days 0-27, the naive average), post_burn_in (days 21-27, the
    corrected readout). Users are present every day; the daily metric is
    mean + effect(day) * treated + noise, averaged per user over the
    window before testing so observations are independent across users.
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)
    path = effect_path(
        days, long_run=long_run, novelty_amplitude=novelty_amplitude, decay_days=decay_days
    )

    windows = {
        "first_week": days < 7,
        "full_run": np.ones(n_days, dtype=bool),
        "post_burn_in": days >= n_days - 7,
    }

    # Daily per-user outcomes, reduced to per-user window means.
    yt = mean + path[None, :] + rng.normal(0.0, within_sd, (n_users_per_arm, n_days))
    yc = mean + rng.normal(0.0, within_sd, (n_users_per_arm, n_days))

    estimates, expected = {}, {}
    for name, mask in windows.items():
        res = welch_t_test(yt[:, mask].mean(axis=1), yc[:, mask].mean(axis=1))
        estimates[name] = res.estimate
        expected[name] = float(path[mask].mean())

    return NoveltyResult(
        estimates=pd.Series(estimates),
        expected=pd.Series(expected),
        long_run_effect=long_run,
    )
