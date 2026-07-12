"""Peeking: an experimenter tests daily and stops at first significance.

Simulates many replications of an experiment where users arrive daily and
a Welch z-test is run on the accumulated data each day. Three analysis
policies are compared on identical data:

- fixed_horizon: one test on the final day at |z| > z_crit.
- naive_peeking: stop the first day |z| > z_crit. Inflates the false
  positive rate far above alpha.
- sequential: stop the first day |z| exceeds the Lan-DeMets
  O'Brien-Fleming boundary for that day. Restores the nominal rate.

Vectorized across replications; the only Python loop is over days.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from lab.stats.sequential import obf_boundaries


@dataclass(frozen=True)
class PeekingResult:
    """Per-policy rejection outcomes across replications.

    Attributes
    ----------
    decisions : pd.DataFrame
        One row per replication: fixed_horizon, naive_peeking, sequential
        (bool rejection), naive_stop_day and sequential_stop_day (day of
        first crossing, NaN if never).
    boundaries : np.ndarray
        The daily z boundaries used by the sequential policy.
    z_crit : float
        The fixed-horizon critical value.
    """

    decisions: pd.DataFrame
    boundaries: np.ndarray
    z_crit: float

    def rejection_rates(self) -> pd.Series:
        """Rejection rate per policy; the false positive rate when effect=0."""
        return self.decisions[["fixed_horizon", "naive_peeking", "sequential"]].mean()


def simulate_daily_peeking(
    *,
    n_sims: int = 4000,
    n_days: int = 14,
    users_per_day_per_arm: int = 100,
    effect: float = 0.0,
    mean: float = 10.0,
    sd: float = 5.4,
    alpha: float = 0.05,
    seed: int = 0,
) -> PeekingResult:
    """Simulate daily-peeking experiments and apply all three policies.

    Outcomes are the engine's marginal metric distribution: each user's
    observed metric is N(mean + effect * treated, sd), the marginal of a
    persistent user mean plus period noise (sd**2 = between**2 + within**2).

    Parameters
    ----------
    n_sims : int
        Number of replicated experiments.
    n_days : int
        Days of daily looks; information fractions are day / n_days.
    users_per_day_per_arm : int
        New users entering each arm each day.
    effect : float
        True additive effect. 0 measures the false positive rate;
        nonzero measures power and time-to-decision.
    mean, sd : float
        Metric mean and standard deviation.
    alpha : float
        Two-sided type I error for every policy.
    seed : int
        Seed for the random generator.
    """
    rng = np.random.default_rng(seed)
    z_crit = float(sps.norm.isf(alpha / 2.0))
    boundaries = obf_boundaries(np.arange(1, n_days + 1) / n_days, alpha=alpha)

    # Running sufficient statistics per sim and arm.
    sum_t = np.zeros(n_sims)
    sumsq_t = np.zeros(n_sims)
    sum_c = np.zeros(n_sims)
    sumsq_c = np.zeros(n_sims)
    z_by_day = np.empty((n_sims, n_days))

    for day in range(n_days):
        yt = rng.normal(mean + effect, sd, (n_sims, users_per_day_per_arm))
        yc = rng.normal(mean, sd, (n_sims, users_per_day_per_arm))
        sum_t += yt.sum(axis=1)
        sumsq_t += (yt**2).sum(axis=1)
        sum_c += yc.sum(axis=1)
        sumsq_c += (yc**2).sum(axis=1)
        n = (day + 1) * users_per_day_per_arm
        mean_t, mean_c = sum_t / n, sum_c / n
        var_t = (sumsq_t - n * mean_t**2) / (n - 1)
        var_c = (sumsq_c - n * mean_c**2) / (n - 1)
        z_by_day[:, day] = (mean_t - mean_c) / np.sqrt(var_t / n + var_c / n)

    abs_z = np.abs(z_by_day)
    naive_cross = abs_z > z_crit
    seq_cross = abs_z > boundaries[None, :]

    def first_day(crossed: np.ndarray) -> np.ndarray:
        any_cross = crossed.any(axis=1)
        day = crossed.argmax(axis=1).astype(float) + 1.0
        day[~any_cross] = np.nan
        return day

    decisions = pd.DataFrame(
        {
            "fixed_horizon": abs_z[:, -1] > z_crit,
            "naive_peeking": naive_cross.any(axis=1),
            "sequential": seq_cross.any(axis=1),
            "naive_stop_day": first_day(naive_cross),
            "sequential_stop_day": first_day(seq_cross),
        }
    )
    return PeekingResult(decisions=decisions, boundaries=boundaries, z_crit=z_crit)
