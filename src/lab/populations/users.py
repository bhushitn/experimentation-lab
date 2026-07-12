"""Synthetic user populations with known ground-truth parameters.

Each factory returns a DataFrame holding one row per user with the latent
parameters that drive outcomes, plus one observed pre-period metric usable
as a CUPED covariate. Because the latent parameters are stored, every
downstream estimate can be checked against exact truth.
"""

import numpy as np
import pandas as pd


def continuous_users(
    n: int,
    *,
    mean: float = 10.0,
    between_sd: float = 2.0,
    within_sd: float = 5.0,
    pre_period_days: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Users whose metric is a persistent user mean plus per-period noise.

    Parameters
    ----------
    n : int
        Number of users.
    mean : float
        Population mean of the per-user latent means.
    between_sd : float
        Standard deviation of latent means across users.
    within_sd : float
        Standard deviation of per-period noise around a user's mean.
    pre_period_days : int
        Days averaged into the pre-period metric. The pre-period noise
        scales as within_sd / sqrt(pre_period_days), so a longer pre
        period gives a covariate closer to the latent user mean and a
        larger CUPED payoff, exactly as in production systems. With the
        default of 1, the pre/post correlation is
        between_sd**2 / (between_sd**2 + within_sd**2).
    seed : int
        Seed for the random generator.

    Returns
    -------
    pd.DataFrame
        Columns: user_id, user_mean (latent), pre_metric (observed).
    """
    rng = np.random.default_rng(seed)
    user_mean = rng.normal(mean, between_sd, n)
    pre_metric = user_mean + rng.normal(0.0, within_sd / np.sqrt(pre_period_days), n)
    return pd.DataFrame(
        {"user_id": np.arange(n), "user_mean": user_mean, "pre_metric": pre_metric}
    )


def binary_users(
    n: int,
    *,
    base_rate: float = 0.10,
    concentration: float = 50.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Users with heterogeneous conversion rates drawn from a Beta distribution.

    Parameters
    ----------
    n : int
        Number of users.
    base_rate : float
        Mean conversion rate across users.
    concentration : float
        Beta concentration (a + b). Larger means less heterogeneity.
    seed : int
        Seed for the random generator.

    Returns
    -------
    pd.DataFrame
        Columns: user_id, user_rate (latent), pre_metric (pre-period
        conversion, a Bernoulli draw from user_rate).
    """
    rng = np.random.default_rng(seed)
    a = base_rate * concentration
    b = (1.0 - base_rate) * concentration
    user_rate = rng.beta(a, b, n)
    pre_metric = rng.binomial(1, user_rate).astype(float)
    return pd.DataFrame(
        {"user_id": np.arange(n), "user_rate": user_rate, "pre_metric": pre_metric}
    )


def zero_inflated_users(
    n: int,
    *,
    active_rate: float = 0.30,
    concentration: float = 20.0,
    log_mean: float = 1.0,
    log_sd: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Users with zero-inflated lognormal spend, the shape of most revenue metrics.

    A user is active in a period with probability p_active (heterogeneous,
    Beta-distributed); an active user's spend is lognormal(log_mean, log_sd).

    Parameters
    ----------
    n : int
        Number of users.
    active_rate : float
        Mean activity probability across users.
    concentration : float
        Beta concentration for the activity probabilities.
    log_mean, log_sd : float
        Parameters of the lognormal spend for active users.
    seed : int
        Seed for the random generator.

    Returns
    -------
    pd.DataFrame
        Columns: user_id, p_active (latent), pre_metric (observed
        pre-period spend, mostly zeros).
    """
    rng = np.random.default_rng(seed)
    a = active_rate * concentration
    b = (1.0 - active_rate) * concentration
    p_active = rng.beta(a, b, n)
    active = rng.binomial(1, p_active)
    pre_metric = active * rng.lognormal(log_mean, log_sd, n)
    return pd.DataFrame(
        {"user_id": np.arange(n), "p_active": p_active, "pre_metric": pre_metric}
    )
