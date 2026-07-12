"""Percentile bootstrap confidence interval for a difference in means.

Fully vectorized: one (n_boot, n) index matrix per arm, no Python loop.
"""

import numpy as np
from numpy.typing import ArrayLike

from lab.stats.results import TestResult


def bootstrap_mean_diff(
    treat: ArrayLike,
    control: ArrayLike,
    *,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> TestResult:
    """Percentile bootstrap CI for mean(treat) - mean(control).

    Resamples each arm independently with replacement. The p-value field is
    NaN: the percentile bootstrap as implemented yields an interval, not a
    test. Preferred over the t-test CI when the metric is heavy-tailed and
    n is too small to trust the CLT.
    """
    x = np.asarray(treat, dtype=np.float64)
    y = np.asarray(control, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    by = y[rng.integers(0, y.size, size=(n_boot, y.size))].mean(axis=1)
    diffs = bx - by
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return TestResult(
        estimate=float(x.mean() - y.mean()),
        statistic=float("nan"),
        p_value=float("nan"),
        ci_low=float(lo),
        ci_high=float(hi),
    )
