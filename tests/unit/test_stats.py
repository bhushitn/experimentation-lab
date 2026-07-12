"""Every implemented test is checked against a scipy/statsmodels reference.

Where no deterministic reference exists (bootstrap), the check is against
analytic properties instead.
"""

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

from lab.stats import (
    bootstrap_mean_diff,
    chi_square_test,
    cuped_t_test,
    mann_whitney_u_test,
    minimum_detectable_effect,
    power_two_sample,
    sample_size_per_arm,
    two_proportion_z_test,
    welch_t_test,
)

RNG = np.random.default_rng(42)
X_CONT = RNG.normal(10.0, 3.0, 500)
Y_CONT = RNG.normal(9.5, 4.0, 450)
X_BIN = RNG.binomial(1, 0.12, 2000).astype(float)
Y_BIN = RNG.binomial(1, 0.10, 2100).astype(float)


class TestWelch:
    def test_matches_scipy(self) -> None:
        ours = welch_t_test(X_CONT, Y_CONT)
        ref = sps.ttest_ind(X_CONT, Y_CONT, equal_var=False)
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)
        assert ours.df == pytest.approx(ref.df)

    def test_ci_matches_scipy(self) -> None:
        ours = welch_t_test(X_CONT, Y_CONT)
        ref_ci = sps.ttest_ind(X_CONT, Y_CONT, equal_var=False).confidence_interval()
        assert ours.ci_low == pytest.approx(ref_ci.low)
        assert ours.ci_high == pytest.approx(ref_ci.high)


class TestProportions:
    def test_z_matches_statsmodels(self) -> None:
        ours = two_proportion_z_test(X_BIN, Y_BIN)
        ref_z, ref_p = proportions_ztest(
            count=[X_BIN.sum(), Y_BIN.sum()], nobs=[X_BIN.size, Y_BIN.size]
        )
        assert ours.statistic == pytest.approx(ref_z)
        assert ours.p_value == pytest.approx(ref_p)

    def test_chi_square_matches_scipy(self) -> None:
        ours = chi_square_test(X_BIN, Y_BIN)
        table = np.array(
            [
                [X_BIN.sum(), X_BIN.size - X_BIN.sum()],
                [Y_BIN.sum(), Y_BIN.size - Y_BIN.sum()],
            ]
        )
        ref = sps.chi2_contingency(table, correction=False)
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)

    def test_chi_square_equals_z_squared(self) -> None:
        z = two_proportion_z_test(X_BIN, Y_BIN).statistic
        chi2 = chi_square_test(X_BIN, Y_BIN).statistic
        assert chi2 == pytest.approx(z**2)


class TestMannWhitney:
    def test_matches_scipy_continuous(self) -> None:
        x = RNG.lognormal(1.0, 1.0, 300)
        y = RNG.lognormal(1.1, 1.0, 320)
        ours = mann_whitney_u_test(x, y)
        ref = sps.mannwhitneyu(x, y, method="asymptotic")
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)

    def test_matches_scipy_with_ties(self) -> None:
        x = RNG.integers(0, 5, 200).astype(float)
        y = RNG.integers(0, 5, 180).astype(float)
        ours = mann_whitney_u_test(x, y)
        ref = sps.mannwhitneyu(x, y, method="asymptotic")
        assert ours.statistic == pytest.approx(ref.statistic)
        assert ours.p_value == pytest.approx(ref.pvalue)


class TestCuped:
    def test_variance_reduction_is_rho_squared(self) -> None:
        rho = 0.6
        n = 20000
        pre = RNG.normal(0.0, 1.0, 2 * n)
        y = rho * pre + np.sqrt(1 - rho**2) * RNG.normal(0.0, 1.0, 2 * n)
        res = cuped_t_test(y[:n], pre[:n], y[n:], pre[n:])
        assert res.variance_reduction == pytest.approx(rho**2, abs=0.02)

    def test_estimate_unbiased_and_tighter(self) -> None:
        effect = 0.2
        n = 5000
        estimates, raw_widths, adj_widths = [], [], []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            pre = rng.normal(0.0, 1.0, 2 * n)
            noise = rng.normal(0.0, 1.0, 2 * n)
            y = 0.7 * pre + noise
            y[:n] += effect
            res = cuped_t_test(y[:n], pre[:n], y[n:], pre[n:])
            raw = welch_t_test(y[:n], y[n:])
            estimates.append(res.test.estimate)
            raw_widths.append(raw.ci_high - raw.ci_low)
            adj_widths.append(res.test.ci_high - res.test.ci_low)
        assert np.mean(estimates) == pytest.approx(effect, abs=0.01)
        # theory: CI width shrinks by sqrt(1 - rho^2); here rho^2 = 0.49/1.49
        expected_ratio = np.sqrt(1 - 0.49 / 1.49)
        assert np.mean(adj_widths) / np.mean(raw_widths) == pytest.approx(expected_ratio, abs=0.01)


class TestBootstrap:
    def test_ci_covers_truth_and_matches_analytic_width(self) -> None:
        x = RNG.normal(10.0, 3.0, 800)
        y = RNG.normal(9.0, 3.0, 800)
        res = bootstrap_mean_diff(x, y, n_boot=4000, seed=7)
        assert res.ci_low < 1.0 < res.ci_high
        welch = welch_t_test(x, y)
        boot_width = res.ci_high - res.ci_low
        welch_width = welch.ci_high - welch.ci_low
        assert boot_width == pytest.approx(welch_width, rel=0.10)


class TestPower:
    def test_power_matches_statsmodels(self) -> None:
        effect, sd, n = 0.5, 2.0, 400
        ours = power_two_sample(n, effect=effect, sd=sd)
        ref = NormalIndPower().power(effect_size=effect / sd, nobs1=n, alpha=0.05, ratio=1.0)
        assert ours == pytest.approx(ref, rel=1e-6)

    def test_sample_size_matches_statsmodels(self) -> None:
        effect, sd = 0.3, 2.0
        ours = sample_size_per_arm(effect=effect, sd=sd, power=0.80)
        ref = NormalIndPower().solve_power(
            effect_size=effect / sd, alpha=0.05, power=0.80, ratio=1.0
        )
        assert ours == pytest.approx(ref, rel=0.01)

    def test_mde_inverts_power(self) -> None:
        n, sd = 1000, 3.0
        mde = minimum_detectable_effect(n, sd=sd, power=0.80)
        assert power_two_sample(n, effect=mde, sd=sd) == pytest.approx(0.80, abs=0.005)
