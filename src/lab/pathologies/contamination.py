"""Contamination / non-compliance: assignment is not receipt.

Some treatment users never receive the treatment (e.g. logged-out
sessions) and some control users are exposed (e.g. shared devices).
Receipt is deliberately correlated with the user's baseline: the
non-receiving treatment users are drawn from the low-baseline end and
the exposed control users from the high-baseline end, the usual
direction (power users find the feature). This is what breaks
per-protocol analysis.

Three estimators on the same data:

- intent_to_treat: compare by assignment. Unbiased for the ITT estimand
  but attenuated relative to the treatment effect by the compliance gap.
- per_protocol: compare by receipt. Selection-biased because receipt
  correlates with baseline.
- iv_wald: ITT effect divided by the compliance gap (the Wald/IV
  estimator). Recovers the true effect under constant effects.

Exact expectations for all three are computable from the population, so
every bias is verified against truth, not eyeballed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lab.stats import welch_t_test


@dataclass(frozen=True)
class ContaminationResult:
    """Estimates and exact expectations under contamination.

    Attributes
    ----------
    estimates : pd.Series
        intent_to_treat, per_protocol, iv_wald point estimates.
    expected : pd.Series
        Exact expected values of the three estimators for this
        population and contamination design.
    true_effect : float
        The effect received compliers experience.
    """

    estimates: pd.Series
    expected: pd.Series
    true_effect: float


def _receipt(
    users: pd.DataFrame,
    treated: np.ndarray,
    treat_noncompliance: float,
    control_exposure: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Who actually receives treatment, probabilistically tilted by baseline.

    Receipt probability is linear in the user's baseline rank u within
    their arm (u in [0, 1], 0 = lowest baseline):

    - treatment arm: (1 - nc) + nc * (u - 0.5), mean 1 - nc; low-baseline
      users are the likelier non-compliers.
    - control arm: ce + ce * (u - 0.5), mean ce; high-baseline power
      users are the likelier exposed.

    Linear-in-rank keeps the arm means exactly at the configured rates
    while inducing the baseline-receipt correlation that biases
    per-protocol analysis.
    """
    baseline = users["user_mean"].to_numpy()
    p = np.empty(len(users))

    t_idx = np.flatnonzero(treated)
    u_t = baseline[t_idx].argsort().argsort() / max(t_idx.size - 1, 1)
    p[t_idx] = (1.0 - treat_noncompliance) + treat_noncompliance * (u_t - 0.5)

    c_idx = np.flatnonzero(~treated)
    u_c = baseline[c_idx].argsort().argsort() / max(c_idx.size - 1, 1)
    p[c_idx] = control_exposure + control_exposure * (u_c - 0.5)

    return np.asarray(rng.random(len(users)) < p)


def simulate_contamination(
    users: pd.DataFrame,
    treated: np.ndarray,
    *,
    effect: float = 0.5,
    treat_noncompliance: float = 0.20,
    control_exposure: float = 0.10,
    within_sd: float = 5.0,
    seed: int = 0,
) -> ContaminationResult:
    """Run one contaminated experiment and all three estimators."""
    rng = np.random.default_rng(seed)
    received = _receipt(users, treated, treat_noncompliance, control_exposure, rng)
    baseline = users["user_mean"].to_numpy()
    y = baseline + effect * received + rng.normal(0.0, within_sd, len(users))

    itt = welch_t_test(y[treated], y[~treated]).estimate
    pp = welch_t_test(y[received], y[~received]).estimate
    compliance_gap = received[treated].mean() - received[~treated].mean()
    iv = itt / compliance_gap

    # Exact expectations from the latent baselines.
    e_itt = effect * compliance_gap + (
        baseline[treated].mean() - baseline[~treated].mean()
    )
    e_pp = effect + baseline[received].mean() - baseline[~received].mean()
    e_iv = e_itt / compliance_gap

    return ContaminationResult(
        estimates=pd.Series({"intent_to_treat": itt, "per_protocol": pp, "iv_wald": iv}),
        expected=pd.Series({"intent_to_treat": e_itt, "per_protocol": e_pp, "iv_wald": e_iv}),
        true_effect=effect,
    )
