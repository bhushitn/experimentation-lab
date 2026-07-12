"""The clean-experiment baseline behaves as theory says it must.

These are the calibration checks the pathology simulations lean on: if the
false positive rate, coverage, and power are right on a clean experiment,
deviations measured later are attributable to the pathology, not the engine.
"""

import numpy as np

from lab.assignment import assign_users
from lab.populations import continuous_users, draw_continuous
from lab.stats import power_two_sample, welch_t_test

N_USERS = 2_000
N_SIMS = 500


def _simulate_p_values_and_coverage(effect: float) -> tuple[np.ndarray, np.ndarray]:
    users = continuous_users(N_USERS, mean=10.0, between_sd=2.0, within_sd=5.0, seed=0)
    p_values, covered = [], []
    for sim in range(N_SIMS):
        treated = assign_users(users, seed=sim)
        y = draw_continuous(users, treated, effect=effect, within_sd=5.0, seed=10_000 + sim)
        res = welch_t_test(y[treated], y[~treated])
        p_values.append(res.p_value)
        covered.append(res.ci_low <= effect <= res.ci_high)
    return np.array(p_values), np.array(covered)


class TestCleanExperiment:
    def test_false_positive_rate_is_nominal(self) -> None:
        p_values, _ = _simulate_p_values_and_coverage(effect=0.0)
        fpr = (p_values < 0.05).mean()
        # 3 sigma binomial band around 0.05 with N_SIMS runs
        band = 3 * np.sqrt(0.05 * 0.95 / N_SIMS)
        assert abs(fpr - 0.05) < band

    def test_ci_coverage_is_nominal(self) -> None:
        _, covered = _simulate_p_values_and_coverage(effect=0.5)
        band = 3 * np.sqrt(0.95 * 0.05 / N_SIMS)
        assert abs(covered.mean() - 0.95) < band

    def test_measured_power_matches_calculator(self) -> None:
        effect = 0.5
        # total outcome sd: between + within variance
        sd = np.sqrt(2.0**2 + 5.0**2)
        predicted = power_two_sample(N_USERS // 2, effect=effect, sd=sd)
        p_values, _ = _simulate_p_values_and_coverage(effect=effect)
        measured = (p_values < 0.05).mean()
        band = 3 * np.sqrt(predicted * (1 - predicted) / N_SIMS)
        assert abs(measured - predicted) < band

    def test_null_p_values_are_uniform(self) -> None:
        p_values, _ = _simulate_p_values_and_coverage(effect=0.0)
        from scipy import stats as sps

        ks = sps.kstest(p_values, "uniform")
        assert ks.pvalue > 0.01
