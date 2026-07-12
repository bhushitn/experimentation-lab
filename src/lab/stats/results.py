"""Shared result container for the statistical tests."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TestResult:
    """Outcome of a two-sample test.

    Attributes
    ----------
    estimate : float
        Effect estimate, treatment minus control. NaN for rank tests,
        which do not estimate a mean difference.
    statistic : float
        The test statistic (t, z, chi-square, or U).
    p_value : float
        Two-sided p-value.
    ci_low, ci_high : float
        Confidence interval for the estimate; NaN where not defined.
    df : float
        Degrees of freedom; NaN where not applicable.
    """

    estimate: float
    statistic: float
    p_value: float
    ci_low: float = math.nan
    ci_high: float = math.nan
    df: float = math.nan

    def significant(self, alpha: float = 0.05) -> bool:
        """Whether the two-sided p-value falls below alpha."""
        return self.p_value < alpha
