"""Quasi-experiments: when you cannot randomize, and what that costs.

The setting real decision-makers face: a feature launched by city, and the
big cities got it first. No randomization, so three estimators compete:

- post_vs_pre: treated cities, after minus before. Absorbs the common time
  trend as if it were the effect.
- treated_vs_control: cross-section after launch. Absorbs the pre-existing
  gap between big and small cities.
- difference_in_differences: the change in treated cities minus the change
  in control cities. Correct exactly when the two groups would have moved
  in parallel without the launch.

The panel is simulated with every component explicit (city fixed effects
correlated with adoption, a common trend, an optional differential trend
for treated cities, the launch effect), so each estimator's bias is exact
arithmetic, not a claim. The honest part interviews probe at senior level:
when treated cities were already growing faster, DiD is biased by exactly
that differential trend, and the pre-period placebo (an "effect" estimated
at a fake launch date before the real one) is the test that catches it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuasiResult:
    """Estimates, exact expectations, and the placebo check.

    Attributes
    ----------
    estimates : pd.Series
        post_vs_pre, treated_vs_control, difference_in_differences.
    expected : pd.Series
        Exact expectations of the three estimators for this panel.
    placebo_did : float
        DiD estimated at a fake launch halfway through the pre-period;
        near zero when parallel trends holds, near the differential
        trend's accumulated gap when it does not.
    true_effect : float
        The constructed launch effect.
    panel : pd.DataFrame
        city, period, treated_city, post, outcome (for the explainer).
    """

    estimates: pd.Series
    expected: pd.Series
    placebo_did: float
    true_effect: float
    panel: pd.DataFrame


def simulate_quasi(
    *,
    n_cities: int = 60,
    treated_share: float = 0.4,
    n_pre: int = 8,
    n_post: int = 8,
    effect: float = 2.0,
    trend: float = 0.5,
    differential_trend: float = 0.0,
    city_sd: float = 4.0,
    adoption_gap: float = 5.0,
    noise_sd: float = 1.0,
    seed: int = 0,
) -> QuasiResult:
    """A city-by-period panel around a staggeredless single launch date.

    outcome[c, t] = alpha_c + trend * t + differential_trend * t * treated_c
                    + effect * treated_c * post_t + noise

    with alpha_c higher by adoption_gap for treated (big) cities: adoption
    correlates with size, which is what breaks the cross-section.
    """
    rng = np.random.default_rng(seed)
    n_treated = round(n_cities * treated_share)
    treated_city = np.arange(n_cities) < n_treated
    alpha = rng.normal(0.0, city_sd, n_cities) + adoption_gap * treated_city

    periods = np.arange(n_pre + n_post)
    post = periods >= n_pre

    city_idx, t_idx = np.meshgrid(np.arange(n_cities), periods, indexing="ij")
    treated_mat = treated_city[city_idx]
    post_mat = post[t_idx]
    outcome = (
        alpha[city_idx]
        + trend * t_idx
        + differential_trend * t_idx * treated_mat
        + effect * treated_mat * post_mat
        + rng.normal(0.0, noise_sd, city_idx.shape)
    )

    def mean(rows: np.ndarray, cols: np.ndarray) -> float:
        return float(outcome[np.ix_(rows, cols)].mean())

    tr = np.flatnonzero(treated_city)
    co = np.flatnonzero(~treated_city)
    pre_p = np.flatnonzero(~post)
    post_p = np.flatnonzero(post)

    post_vs_pre = mean(tr, post_p) - mean(tr, pre_p)
    cross_section = mean(tr, post_p) - mean(co, post_p)
    did = (mean(tr, post_p) - mean(tr, pre_p)) - (mean(co, post_p) - mean(co, pre_p))

    # Exact expectations, conditioning on the realized city effects (the
    # house style: expectations are computed from the latent draws, so the
    # match to the estimate is an identity plus noise, not an approximation).
    dt = float(periods[post].mean() - periods[~post].mean())
    alpha_gap = float(alpha[treated_city].mean() - alpha[~treated_city].mean())
    e_post_vs_pre = effect + trend * dt + differential_trend * dt
    e_cross = effect + alpha_gap + differential_trend * float(periods[post].mean())
    e_did = effect + differential_trend * dt

    # Placebo: fake launch halfway through the pre-period, pre data only.
    fake = n_pre // 2
    a = np.flatnonzero(periods < fake)
    b = np.flatnonzero((periods >= fake) & ~post)
    placebo = (mean(tr, b) - mean(tr, a)) - (mean(co, b) - mean(co, a))

    panel = pd.DataFrame(
        {
            "city": city_idx.ravel(),
            "period": t_idx.ravel(),
            "treated_city": treated_mat.ravel(),
            "post": post_mat.ravel(),
            "outcome": outcome.ravel(),
        }
    )
    return QuasiResult(
        estimates=pd.Series(
            {
                "post_vs_pre": post_vs_pre,
                "treated_vs_control": cross_section,
                "difference_in_differences": did,
            }
        ),
        expected=pd.Series(
            {
                "post_vs_pre": e_post_vs_pre,
                "treated_vs_control": e_cross,
                "difference_in_differences": e_did,
            }
        ),
        placebo_did=float(placebo),
        true_effect=effect,
        panel=panel,
    )
