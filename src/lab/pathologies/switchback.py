"""Switchback experiments: when the marketplace is the unit.

One city, one pool of couriers and orders: user-level randomization is
impossible because both arms would share the same supply. The standard
design alternates the whole market between treatment and control in time
windows (a switchback). The catch is carryover: a market does not reset the
instant the toggle flips, so the first hours of each window still carry the
previous window's state.

Model, chosen because it makes every bias exactly computable (the same
linear-exposure trick as the interference module):

    y_h = mu + delta * e_h + noise,  e_h = (1 - gamma) * A_h + gamma * A_{h-1}

where A_h is the assigned arm in hour h and gamma is the carryover
fraction. The naive estimator compares hours by assignment; its expectation
given the realized schedule is

    delta * [ (1 - gamma) + gamma * (P(prev treated | treated) - P(prev treated | control)) ]

so short windows, whose hours sit mostly next to opposite-arm hours,
attenuate the estimate, and the module measures that bias as a function of
window length together with the variance cost of longer windows. The
corrected analysis drops the first hour after each switch (a burn-in),
which removes the contaminated hours and restores the estimate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lab.stats import welch_t_test


@dataclass(frozen=True)
class SwitchbackResult:
    """Bias and precision of switchback analyses across window lengths.

    Attributes
    ----------
    by_window : pd.DataFrame
        One row per window length: naive_estimate, naive_expected (exact,
        given schedules), burn_in_estimate, and the empirical standard
        error of each across replications.
    true_effect : float
        delta, the steady-state effect of running treatment.
    carryover : float
        gamma, the fraction of an hour's outcome carried from the prior hour.
    """

    by_window: pd.DataFrame
    true_effect: float
    carryover: float


def _schedule(n_hours: int, window: int, rng: np.random.Generator) -> np.ndarray:
    """Random arm per window, expanded to hours."""
    n_windows = int(np.ceil(n_hours / window))
    arms = rng.integers(0, 2, n_windows).astype(bool)
    return np.repeat(arms, window)[:n_hours]


def simulate_switchback(
    *,
    n_days: int = 28,
    windows: tuple[int, ...] = (1, 2, 4, 8, 24),
    delta: float = 1.0,
    gamma: float = 0.4,
    mu: float = 20.0,
    noise_sd: float = 2.0,
    day_sd: float = 1.0,
    n_sims: int = 400,
    seed: int = 0,
) -> SwitchbackResult:
    """Replicated switchback experiments at several window lengths.

    Markets have day-level shocks (weather, a concert, a promo) shared by
    every hour of that day; day_sd controls them. They are what make long
    windows expensive: a day-long window rides entirely on one day's shock,
    so the effective number of units is the number of days, while short
    windows average the shocks out within each day. Hourly noise alone
    would hide that cost.

    Vectorized across replications: for each window length, all n_sims
    schedules and outcome series are simulated as one (n_sims, n_hours)
    array; the only Python loop is over the handful of window lengths.
    """
    rng = np.random.default_rng(seed)
    n_hours = 24 * n_days
    rows = []
    for window in windows:
        n_windows = int(np.ceil(n_hours / window))
        arms = rng.integers(0, 2, (n_sims, n_windows)).astype(bool)
        assigned = np.repeat(arms, window, axis=1)[:, :n_hours]
        prev = np.concatenate([assigned[:, :1], assigned[:, :-1]], axis=1)
        exposure = (1.0 - gamma) * assigned + gamma * prev
        day_shocks = np.repeat(rng.normal(0.0, day_sd, (n_sims, n_days)), 24, axis=1)
        y = (
            mu
            + delta * exposure
            + day_shocks
            + rng.normal(0.0, noise_sd, (n_sims, n_hours))
        )

        # Naive: hour-level difference in means by assignment.
        sum_t = (y * assigned).sum(axis=1)
        n_t = assigned.sum(axis=1)
        sum_c = (y * ~assigned).sum(axis=1)
        n_c = (~assigned).sum(axis=1)
        naive = sum_t / n_t - sum_c / n_c

        # Exact expectation given each realized schedule.
        e_t = (exposure * assigned).sum(axis=1) / n_t
        e_c = (exposure * ~assigned).sum(axis=1) / n_c
        expected = delta * (e_t - e_c)

        # Burn-in: drop the first hour after every switch (and hour 0).
        switched = np.concatenate(
            [np.ones((n_sims, 1), dtype=bool), assigned[:, 1:] != assigned[:, :-1]], axis=1
        )
        keep = ~switched
        kt = keep & assigned
        kc = keep & ~assigned
        burn = (y * kt).sum(axis=1) / kt.sum(axis=1) - (y * kc).sum(axis=1) / kc.sum(axis=1)

        rows.append(
            {
                "window_hours": window,
                "naive_estimate": float(naive.mean()),
                "naive_expected": float(expected.mean()),
                "naive_se": float(naive.std(ddof=1)),
                "burn_in_estimate": float(burn.mean()),
                "burn_in_se": float(burn.std(ddof=1)),
            }
        )
    return SwitchbackResult(
        by_window=pd.DataFrame(rows), true_effect=delta, carryover=gamma
    )


def switchback_single_run(
    *,
    n_days: int = 28,
    window: int = 4,
    delta: float = 1.0,
    gamma: float = 0.4,
    mu: float = 20.0,
    noise_sd: float = 2.0,
    day_sd: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, float, float]:
    """One readable run for the explainer.

    Returns the hourly series (hour, assigned, outcome) plus the naive and
    burn-in estimates on this run.
    """
    rng = np.random.default_rng(seed)
    n_hours = 24 * n_days
    assigned = _schedule(n_hours, window, rng)
    prev = np.concatenate([assigned[:1], assigned[:-1]])
    exposure = (1.0 - gamma) * assigned + gamma * prev
    day_shocks = np.repeat(rng.normal(0.0, day_sd, n_days), 24)
    y = mu + delta * exposure + day_shocks + rng.normal(0.0, noise_sd, n_hours)
    naive = welch_t_test(y[assigned], y[~assigned]).estimate
    switched = np.concatenate([[True], assigned[1:] != assigned[:-1]])
    keep = ~switched
    burn = welch_t_test(y[keep & assigned], y[keep & ~assigned]).estimate
    series = pd.DataFrame({"hour": np.arange(n_hours), "assigned": assigned, "outcome": y})
    return series, float(naive), float(burn)
