"""Network interference: a user's outcome depends on neighbors' assignments.

Outcome model (linear-in-means, the simplest interference model that
produces the qualitative failure):

    y_i = user_mean_i + direct * treated_i + spillover * exposure_i + noise

where exposure_i is the treated fraction of i's neighbors. The policy
question an experiment answers is the global treatment effect (GTE):
everyone treated vs no one treated, which here is exactly
direct + spillover.

Because the model is linear in exposure, the expected value of any
difference-in-means estimator is exactly

    direct + spillover * (mean exposure of treated - mean exposure of control)

so the bias of each design is checked against a closed form, not a
Monte Carlo hunch. Under user-level randomization both arms sit at
~50% treated neighbors, the contrast is ~0, and the estimator misses
the spillover entirely. Under cluster randomization the contrast is
close to 1 and the estimator approaches the GTE, degraded only by
between-cluster edges.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from lab.stats import welch_t_test


@dataclass(frozen=True)
class InterferenceResult:
    """Estimates under both randomization designs vs the global effect.

    Attributes
    ----------
    estimates : pd.Series
        user_randomized and cluster_randomized difference-in-means.
    expected : pd.Series
        Closed-form expected value of each estimator given the realized
        assignment (direct + spillover * exposure contrast).
    exposure_contrast : pd.Series
        Mean treated-neighbor fraction, treatment minus control arm,
        per design.
    global_treatment_effect : float
        direct + spillover, the all-treated vs none-treated truth.
    """

    estimates: pd.Series
    expected: pd.Series
    exposure_contrast: pd.Series
    global_treatment_effect: float


def _exposure(adjacency: sparse.csr_matrix, treated: np.ndarray) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    neighbor_treated = np.asarray(adjacency @ treated.astype(np.float64))
    return np.divide(
        neighbor_treated, degree, out=np.zeros_like(neighbor_treated), where=degree > 0
    )


def simulate_interference(
    users: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    user_treated: np.ndarray,
    cluster_treated: np.ndarray,
    *,
    direct: float = 0.5,
    spillover: float = 0.3,
    within_sd: float = 5.0,
    seed: int = 0,
) -> InterferenceResult:
    """Run the same interference model under both designs."""
    rng = np.random.default_rng(seed)
    baseline = users["user_mean"].to_numpy()

    rows_est, rows_exp, rows_contrast = {}, {}, {}
    for name, treated in (
        ("user_randomized", user_treated),
        ("cluster_randomized", cluster_treated),
    ):
        exposure = _exposure(adjacency, treated)
        noise = rng.normal(0.0, within_sd, len(users))
        y = baseline + direct * treated + spillover * exposure + noise
        contrast = float(exposure[treated].mean() - exposure[~treated].mean())
        baseline_gap = float(baseline[treated].mean() - baseline[~treated].mean())
        rows_est[name] = welch_t_test(y[treated], y[~treated]).estimate
        rows_exp[name] = direct + spillover * contrast + baseline_gap
        rows_contrast[name] = contrast

    return InterferenceResult(
        estimates=pd.Series(rows_est),
        expected=pd.Series(rows_exp),
        exposure_contrast=pd.Series(rows_contrast),
        global_treatment_effect=direct + spillover,
    )
