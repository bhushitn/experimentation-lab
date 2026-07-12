"""Power, minimum detectable effect, and sample size for two-sample means.

Normal-approximation formulas, two-sided alpha, equal arm sizes. Validated
against statsmodels NormalIndPower in tests/unit/test_stats.py. For binary
metrics pass sd = sqrt(p * (1 - p)).
"""

import math

from scipy import stats as sps


def power_two_sample(n_per_arm: int, *, effect: float, sd: float, alpha: float = 0.05) -> float:
    """Probability of detecting `effect` with n_per_arm users in each arm.

    Includes both rejection tails, matching statsmodels NormalIndPower.
    """
    se = sd * math.sqrt(2.0 / n_per_arm)
    z_crit = float(sps.norm.ppf(1.0 - alpha / 2.0))
    shift = effect / se
    return float(sps.norm.cdf(shift - z_crit) + sps.norm.cdf(-shift - z_crit))


def minimum_detectable_effect(
    n_per_arm: int, *, sd: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest true effect detectable with the given power at this sample size."""
    z_alpha = float(sps.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(sps.norm.ppf(power))
    return (z_alpha + z_power) * sd * math.sqrt(2.0 / n_per_arm)


def sample_size_per_arm(
    *, effect: float, sd: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Users per arm needed to detect `effect` with the given power.

    Standard formula; ignores the far rejection tail, which is negligible
    at conventional alpha and power.
    """
    z_alpha = float(sps.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(sps.norm.ppf(power))
    return math.ceil(2.0 * ((z_alpha + z_power) * sd / effect) ** 2)
