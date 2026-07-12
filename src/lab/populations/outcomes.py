"""Outcome generation under a known true treatment effect.

Each draw_* function takes a population from lab.populations.users, a boolean
treatment vector, and effect parameters, and returns the observed outcome.
Each true_*_ate function returns the exact average treatment effect implied
by the same parameters, so estimates are checkable against truth.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def draw_continuous(
    users: pd.DataFrame,
    treated: BoolArray,
    *,
    effect: float,
    within_sd: float = 5.0,
    seed: int = 0,
) -> FloatArray:
    """Observed outcome: user mean, plus an additive effect if treated, plus noise."""
    rng = np.random.default_rng(seed)
    y = users["user_mean"].to_numpy() + effect * treated + rng.normal(0.0, within_sd, len(users))
    return np.asarray(y, dtype=np.float64)


def true_continuous_ate(effect: float) -> float:
    """The additive effect is the ATE by construction."""
    return effect


def draw_binary(
    users: pd.DataFrame,
    treated: BoolArray,
    *,
    lift: float,
    seed: int = 0,
) -> FloatArray:
    """Observed conversion: Bernoulli with an absolute lift on the rate if treated.

    Rates are clipped to [0, 1]; true_binary_ate accounts for the clipping.
    """
    rng = np.random.default_rng(seed)
    p = np.clip(users["user_rate"].to_numpy() + lift * treated, 0.0, 1.0)
    return np.asarray(rng.binomial(1, p), dtype=np.float64)


def true_binary_ate(users: pd.DataFrame, lift: float) -> float:
    """Exact ATE over this population, accounting for rate clipping."""
    rate = users["user_rate"].to_numpy()
    return float(np.mean(np.clip(rate + lift, 0.0, 1.0) - rate))


def draw_zero_inflated(
    users: pd.DataFrame,
    treated: BoolArray,
    *,
    spend_multiplier: float,
    log_mean: float = 1.0,
    log_sd: float = 1.0,
    seed: int = 0,
) -> FloatArray:
    """Observed spend: activity Bernoulli, then lognormal spend scaled if treated.

    Treatment multiplies an active user's expected spend by spend_multiplier
    and leaves the activity probability unchanged.
    """
    rng = np.random.default_rng(seed)
    active = rng.binomial(1, users["p_active"].to_numpy())
    spend = rng.lognormal(log_mean, log_sd, len(users))
    scale = np.where(treated, spend_multiplier, 1.0)
    return np.asarray(active * spend * scale, dtype=np.float64)


def true_zero_inflated_ate(
    users: pd.DataFrame,
    spend_multiplier: float,
    *,
    log_mean: float = 1.0,
    log_sd: float = 1.0,
) -> float:
    """Exact ATE: mean activity times lognormal mean times (multiplier - 1)."""
    lognormal_mean = float(np.exp(log_mean + log_sd**2 / 2.0))
    return float(users["p_active"].mean()) * lognormal_mean * (spend_multiplier - 1.0)
