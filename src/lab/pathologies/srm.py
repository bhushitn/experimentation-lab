"""Sample ratio mismatch: the smoke alarm of experimentation.

The assignment says 50/50 but the logged data does not. Here the cause is a
logging bug that silently drops part of one arm, correlated with who the
users are: treatment users on old app versions crash before exposure is
logged, and old-version users skew low-baseline. Two things follow, and the
module quantifies both:

- The split itself betrays the bug. A chi-square goodness-of-fit test on
  the logged counts (the SRM check) flags the mismatch once the missing
  users outnumber the split's sampling noise, roughly a loss of
  3.3 * sqrt(total) users at the conventional alpha of 0.001. At 100,000
  per arm that means a 2% one-arm loss is caught essentially always and a
  1% loss usually is not; at a million per arm the same 1% is caught
  essentially always. Detectability improves with the square root of scale.
- The estimate is no longer trustworthy. Dropping low-baseline users from
  one arm shifts that arm's mean, and the bias is computable exactly from
  the latent baselines.

The lesson the memo carries: an SRM alarm is not a statistics problem to
correct, it is a data bug to find. No reweighting rescues an unknown
missingness mechanism.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from lab.stats import welch_t_test


@dataclass(frozen=True)
class SrmResult:
    """One buggy experiment: the alarm and the damage.

    Attributes
    ----------
    counts : pd.Series
        Logged users per arm (assigned was 50/50).
    srm_p_value : float
        Chi-square goodness-of-fit p-value against the designed split.
    estimates : pd.Series
        naive: difference in means on the logged data.
        expected_naive: exact expectation given who was dropped.
    true_effect : float
        The constructed truth.
    """

    counts: pd.Series
    srm_p_value: float
    estimates: pd.Series
    true_effect: float


def srm_check(n_treatment: int, n_control: int, *, expected_fraction: float = 0.5) -> float:
    """Chi-square goodness-of-fit p-value for the logged assignment counts.

    The standard SRM alarm (see Fabijan et al., "Diagnosing Sample Ratio
    Mismatch in Online Controlled Experiments", KDD 2019). Validated against
    scipy.stats.chisquare in the unit tests.
    """
    total = n_treatment + n_control
    expected = np.array([expected_fraction, 1.0 - expected_fraction]) * total
    stat = float((((np.array([n_treatment, n_control]) - expected) ** 2) / expected).sum())
    return float(sps.chi2.sf(stat, df=1))


def simulate_srm(
    users: pd.DataFrame,
    treated: np.ndarray,
    *,
    effect: float = 0.5,
    drop_fraction: float = 0.02,
    within_sd: float = 5.0,
    seed: int = 0,
) -> SrmResult:
    """One experiment where a logging bug drops treatment users non-randomly.

    The dropped users are drawn from the low-baseline half of the treatment
    arm (old app versions crash before exposure logs; old versions skew
    low-engagement), at a rate that makes the overall treatment-arm loss
    equal drop_fraction.
    """
    rng = np.random.default_rng(seed)
    baseline = users["user_mean"].to_numpy()

    logged = np.ones(len(users), dtype=bool)
    t_idx = np.flatnonzero(treated)
    below_median = t_idx[baseline[t_idx] < np.median(baseline[t_idx])]
    # Concentrate the loss in the low-baseline half: rate doubles there.
    drop = rng.random(below_median.size) < 2.0 * drop_fraction
    logged[below_median[drop]] = False

    y = baseline + effect * treated + rng.normal(0.0, within_sd, len(users))
    t_logged = treated & logged
    c_logged = ~treated & logged

    naive = welch_t_test(y[t_logged], y[c_logged]).estimate
    expected_naive = effect + baseline[t_logged].mean() - baseline[c_logged].mean()

    return SrmResult(
        counts=pd.Series(
            {"treatment_logged": int(t_logged.sum()), "control_logged": int(c_logged.sum())}
        ),
        srm_p_value=srm_check(int(t_logged.sum()), int(c_logged.sum())),
        estimates=pd.Series({"naive": naive, "expected_naive": float(expected_naive)}),
        true_effect=effect,
    )


def srm_detection_power(
    *,
    n_per_arm: int,
    drop_fraction: float,
    n_sims: int = 2000,
    alpha: float = 0.001,
    seed: int = 0,
) -> float:
    """How often the SRM check fires for a given loss rate and sample size.

    Uses the conventional strict alpha of 0.001 (the check runs on every
    experiment, so its own false positive rate must be tiny). Vectorized
    binomial draws; no per-user work needed for the counts alone.
    """
    rng = np.random.default_rng(seed)
    kept = rng.binomial(n_per_arm, 1.0 - drop_fraction, n_sims)
    control = np.full(n_sims, n_per_arm)
    total = kept + control
    expected_t = total / 2.0
    stat = (kept - expected_t) ** 2 / expected_t + (control - expected_t) ** 2 / expected_t
    p = sps.chi2.sf(stat, df=1)
    return float((p < alpha).mean())
