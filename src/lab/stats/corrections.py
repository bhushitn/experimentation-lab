"""Multiple-testing corrections, validated against statsmodels.

Benjamini, Hochberg, "Controlling the False Discovery Rate: A Practical
and Powerful Approach to Multiple Testing", JRSS-B 57(1), 1995,
pp. 289-300. DOI: 10.1111/j.2517-6161.1995.tb02031.x.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]


def bonferroni(p_values: ArrayLike, *, alpha: float = 0.05) -> BoolArray:
    """Reject where p < alpha / m. Controls the family-wise error rate."""
    p = np.asarray(p_values, dtype=np.float64)
    return np.asarray(p < alpha / p.size)


def benjamini_hochberg(p_values: ArrayLike, *, alpha: float = 0.05) -> BoolArray:
    """Benjamini-Hochberg step-up procedure. Controls the false discovery rate.

    Reject the hypotheses with the k smallest p-values, where k is the
    largest rank with p_(k) <= (k / m) * alpha.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    passes = ranked <= (np.arange(1, m + 1) / m) * alpha
    reject = np.zeros(m, dtype=bool)
    if passes.any():
        k = int(np.nonzero(passes)[0].max())
        reject[order[: k + 1]] = True
    return reject
