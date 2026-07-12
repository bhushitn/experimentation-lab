"""Two-sample tests, implemented from the formulas and validated against
scipy/statsmodels reference outputs in tests/unit/test_stats.py.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sps

from lab.stats.results import TestResult


def welch_t_test(treat: ArrayLike, control: ArrayLike, *, alpha: float = 0.05) -> TestResult:
    """Welch two-sample t-test (unequal variances).

    Uses the Welch-Satterthwaite degrees of freedom. Reference:
    scipy.stats.ttest_ind(equal_var=False).
    """
    x = np.asarray(treat, dtype=np.float64)
    y = np.asarray(control, dtype=np.float64)
    nx, ny = x.size, y.size
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    diff = float(x.mean() - y.mean())
    se = float(np.sqrt(vx / nx + vy / ny))
    df = float((vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)))
    t = diff / se
    p = float(2.0 * sps.t.sf(abs(t), df))
    t_crit = float(sps.t.ppf(1.0 - alpha / 2.0, df))
    return TestResult(
        estimate=diff,
        statistic=t,
        p_value=p,
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        df=df,
    )


def two_proportion_z_test(
    treat: ArrayLike, control: ArrayLike, *, alpha: float = 0.05
) -> TestResult:
    """Two-proportion z-test on 0/1 outcomes.

    The test statistic uses the pooled standard error (the null assumes equal
    rates); the confidence interval uses the unpooled (Wald) standard error,
    the standard pairing. Reference: statsmodels proportions_ztest.
    """
    x = np.asarray(treat, dtype=np.float64)
    y = np.asarray(control, dtype=np.float64)
    nx, ny = x.size, y.size
    px, py = float(x.mean()), float(y.mean())
    diff = px - py
    p_pool = (x.sum() + y.sum()) / (nx + ny)
    se_pooled = float(np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / nx + 1.0 / ny)))
    z = diff / se_pooled
    p = float(2.0 * sps.norm.sf(abs(z)))
    se_wald = float(np.sqrt(px * (1.0 - px) / nx + py * (1.0 - py) / ny))
    z_crit = float(sps.norm.ppf(1.0 - alpha / 2.0))
    return TestResult(
        estimate=diff,
        statistic=z,
        p_value=p,
        ci_low=diff - z_crit * se_wald,
        ci_high=diff + z_crit * se_wald,
    )


def chi_square_test(treat: ArrayLike, control: ArrayLike) -> TestResult:
    """Pearson chi-square test of independence on the 2x2 arm-by-outcome table.

    No continuity correction, so the statistic equals the square of the pooled
    two-proportion z statistic. Reference:
    scipy.stats.chi2_contingency(correction=False).
    """
    x = np.asarray(treat, dtype=np.float64)
    y = np.asarray(control, dtype=np.float64)
    observed = np.array(
        [
            [x.sum(), x.size - x.sum()],
            [y.sum(), y.size - y.sum()],
        ]
    )
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row * col / observed.sum()
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p = float(sps.chi2.sf(chi2, df=1))
    return TestResult(
        estimate=float(x.mean() - y.mean()),
        statistic=chi2,
        p_value=p,
        df=1.0,
    )


def mann_whitney_u_test(treat: ArrayLike, control: ArrayLike) -> TestResult:
    """Mann-Whitney U test, two-sided, normal approximation.

    Applies the tie correction to the variance and a continuity correction of
    0.5, matching scipy.stats.mannwhitneyu(method="asymptotic"). Suited to
    heavy-tailed metrics where the t-test's mean is dominated by outliers.
    """
    x = np.asarray(treat, dtype=np.float64)
    y = np.asarray(control, dtype=np.float64)
    nx, ny = x.size, y.size
    n = nx + ny
    ranks = sps.rankdata(np.concatenate([x, y]))
    r1 = float(ranks[:nx].sum())
    u1 = r1 - nx * (nx + 1) / 2.0
    mu = nx * ny / 2.0
    _, counts = np.unique(np.concatenate([x, y]), return_counts=True)
    tie_term = float((counts.astype(np.float64) ** 3 - counts).sum())
    sigma = float(np.sqrt(nx * ny / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))))
    z = (abs(u1 - mu) - 0.5) / sigma
    p = float(min(1.0, 2.0 * sps.norm.sf(z)))
    return TestResult(estimate=float("nan"), statistic=u1, p_value=p)
