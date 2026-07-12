"""CUPED variance reduction.

Deng, Xu, Kohavi, Walker, "Improving the Sensitivity of Online Controlled
Experiments by Utilizing Pre-Experiment Data", WSDM '13.
DOI: 10.1145/2433396.2433413.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from lab.stats.results import TestResult
from lab.stats.tests import welch_t_test


@dataclass(frozen=True)
class CupedResult:
    """CUPED-adjusted test plus diagnostics.

    Attributes
    ----------
    test : TestResult
        Welch t-test on the CUPED-adjusted outcomes.
    theta : float
        The regression coefficient of outcome on covariate, pooled
        across both arms.
    variance_reduction : float
        1 - var(adjusted) / var(raw), pooled. Approximately the squared
        outcome-covariate correlation.
    """

    test: TestResult
    theta: float
    variance_reduction: float


def cuped_adjust(y: ArrayLike, covariate: ArrayLike, theta: float | None = None) -> np.ndarray:
    """Return y - theta * (covariate - mean(covariate)).

    With theta = cov(y, x) / var(x) this is the minimum-variance linear
    adjustment; centering the covariate leaves the mean of y unchanged.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(covariate, dtype=np.float64)
    if theta is None:
        theta = float(np.cov(y_arr, x_arr)[0, 1] / x_arr.var(ddof=1))
    return y_arr - theta * (x_arr - x_arr.mean())


def cuped_t_test(
    treat_y: ArrayLike,
    treat_covariate: ArrayLike,
    control_y: ArrayLike,
    control_covariate: ArrayLike,
    *,
    alpha: float = 0.05,
) -> CupedResult:
    """Welch t-test on CUPED-adjusted outcomes.

    Theta is estimated on the pooled sample (both arms), the standard
    practice: under randomization the covariate is independent of
    assignment, so pooling does not bias the effect estimate.
    """
    ty = np.asarray(treat_y, dtype=np.float64)
    tx = np.asarray(treat_covariate, dtype=np.float64)
    cy = np.asarray(control_y, dtype=np.float64)
    cx = np.asarray(control_covariate, dtype=np.float64)

    y = np.concatenate([ty, cy])
    x = np.concatenate([tx, cx])
    theta = float(np.cov(y, x)[0, 1] / x.var(ddof=1))

    x_mean = x.mean()
    ty_adj = ty - theta * (tx - x_mean)
    cy_adj = cy - theta * (cx - x_mean)

    raw_var = float(y.var(ddof=1))
    adj_var = float(np.concatenate([ty_adj, cy_adj]).var(ddof=1))
    return CupedResult(
        test=welch_t_test(ty_adj, cy_adj, alpha=alpha),
        theta=theta,
        variance_reduction=1.0 - adj_var / raw_var,
    )
