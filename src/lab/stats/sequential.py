"""Group-sequential boundaries via the Lan-DeMets alpha spending method.

Lan, DeMets, "Discrete sequential boundaries for clinical trials",
Biometrika 70(3), 1983, pp. 659-663. The O'Brien-Fleming-type spending
function is used throughout: it spends almost no alpha early, so early
looks need overwhelming evidence and the final boundary stays near the
fixed-horizon critical value.

Boundaries are computed with the standard recursive numerical integration
over the joint distribution of the partial-sum process, and validated in
tests/unit/test_sequential.py against published values from the R ldbounds
package vignette.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as sps
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]

_GRID_POINTS = 1001


def obf_spending(t: ArrayLike, *, alpha: float = 0.05) -> FloatArray:
    """Cumulative two-sided alpha spent by information fraction t.

    The O'Brien-Fleming-type spending function:
    A(t) = 4 - 4 * Phi(z_{1 - alpha/4} / sqrt(t)), so A(1) = alpha.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    z_quarter = float(sps.norm.ppf(1.0 - alpha / 4.0))
    return np.asarray(4.0 - 4.0 * sps.norm.cdf(z_quarter / np.sqrt(t_arr)), dtype=np.float64)


def obf_boundaries(info_fractions: ArrayLike, *, alpha: float = 0.05) -> FloatArray:
    """Symmetric two-sided z boundaries at the given information fractions.

    At look k the test rejects when |z_k| >= boundary[k]. The boundaries
    guarantee an overall type I error of alpha regardless of how many
    interim looks are taken, because each look is only allowed to spend
    the increment of the spending function.

    Parameters
    ----------
    info_fractions : array-like
        Increasing information fractions in (0, 1], e.g. day/k for equal
        daily looks. The last entry should be 1.0.
    alpha : float
        Total two-sided type I error.

    Returns
    -------
    NDArray[np.float64]
        One z boundary per look.
    """
    t = np.asarray(info_fractions, dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or np.any(np.diff(t) <= 0) or t[0] <= 0 or t[-1] > 1:
        raise ValueError("info_fractions must be increasing, in (0, 1]")

    cumulative = obf_spending(t, alpha=alpha)
    # Early OBF looks spend alpha below double precision; floor the
    # per-look spend so tail quantiles stay finite. The overspend this
    # introduces is ~1e-14 total, far below simulation resolution.
    spend = np.maximum(np.diff(cumulative, prepend=0.0), 1e-15)
    cumulative = np.cumsum(spend)

    boundaries: list[float] = []
    # Sub-density of the partial sum S_k over the continuation region,
    # carried on a grid; the S-scale boundary is c_k = z_k * sqrt(t_k).
    grid: FloatArray = np.zeros(0, dtype=np.float64)
    density: FloatArray = np.zeros(0, dtype=np.float64)

    for k in range(t.size):
        if k == 0:
            sd = float(np.sqrt(t[0]))
            # First look: S_1 ~ N(0, t_1), so the boundary is closed form.
            # isf on the tail keeps precision where ppf(1 - tiny) cannot.
            c = sd * float(sps.norm.isf(spend[0] / 2.0))
        else:
            sd = float(np.sqrt(t[k] - t[k - 1]))
            inside_target = 1.0 - float(cumulative[k])

            def inside_mass(
                c: float,
                sd: float = sd,
                grid: FloatArray = grid,
                density: FloatArray = density,
            ) -> float:
                xs = np.linspace(-c, c, _GRID_POINTS)
                transition = sps.norm.pdf((xs[:, None] - grid[None, :]) / sd) / sd
                g_next = np.trapezoid(transition * density[None, :], grid, axis=1)
                return float(np.trapezoid(g_next, xs))

            c = float(
                brentq(
                    lambda c, target=inside_target: inside_mass(c) - target,
                    1e-6,
                    15.0,
                    xtol=1e-10,
                )
            )

        boundaries.append(c / float(np.sqrt(t[k])))
        xs = np.linspace(-c, c, _GRID_POINTS)
        if k == 0:
            density = np.asarray(sps.norm.pdf(xs / sd) / sd, dtype=np.float64)
        else:
            transition = sps.norm.pdf((xs[:, None] - grid[None, :]) / sd) / sd
            density = np.trapezoid(transition * density[None, :], grid, axis=1)
        grid = xs

    return np.array(boundaries, dtype=np.float64)
